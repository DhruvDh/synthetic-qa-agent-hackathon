#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune GPT-OSS-120B with DPO using Harmony chat templates end-to-end.

- Applies tokenizer.apply_chat_template for prompts and targets so training matches inference.
- Targets GPT-OSS-120B MXFP4 weights (single MI300X) with a LoRA adapter; no bitsandbytes required.
- max_seq_length = 8192, with optional float8 KV cache if available in your Unsloth build.
- Expects DPO preference pairs in outputs/eval/dpo_answers.jsonl and outputs/eval/dpo_questions.jsonl.
"""

from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

import json
import os
import random
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer


def ensure_mxfp4_activation_shim() -> None:
    """Align Unsloth's MXFP4 fused activation call with current triton kernels."""
    try:
        from transformers.integrations import mxfp4 as mxfp4_mod
        from triton_kernels import matmul_ogs, swiglu
    except Exception:
        return

    fused_activation = getattr(matmul_ogs, "FusedActivation", None)
    fn_specs = getattr(matmul_ogs, "FnSpecs", None)
    swiglu_fn = getattr(swiglu, "swiglu_fn", None)
    experts_cls = getattr(mxfp4_mod, "Mxfp4GptOssExperts", None)
    RoutingData = getattr(matmul_ogs, "RoutingData", None)
    InnerRoutingData = getattr(matmul_ogs, "InnerRoutingData", None)
    if None in (fused_activation, fn_specs, swiglu_fn, experts_cls):
        return

    def _ensure_act(module):
        if hasattr(module, "act"):
            return
        specs = fn_specs("swiglu", swiglu_fn, ("alpha", "limit"))
        try:
            module.act = fused_activation(specs, (module.alpha, module.limit), 2)
        except TypeError:
            module.act = fused_activation(specs, (module.alpha, module.limit))

    def _coerce_routing_data(data):
        if data is None or RoutingData is None:
            return data
        if isinstance(data, RoutingData):
            return data
        if InnerRoutingData is not None and isinstance(data, InnerRoutingData):
            base = _coerce_routing_data(getattr(data, "base", None))
            return InnerRoutingData(
                base=base,
                block_k=getattr(data, "block_k", None),
                x_is_padded=getattr(data, "x_is_padded", False),
                w_is_padded=getattr(data, "w_is_padded", False),
            )
        try:
            expt_data = getattr(data, "expt_data", None)
            if expt_data is not None:
                slice_sizes = getattr(expt_data, "slice_sizes", None)
                if slice_sizes is None and hasattr(expt_data, "slice_hist"):
                    slice_sizes = expt_data.slice_hist
                if slice_sizes is None and hasattr(expt_data, "hist"):
                    slice_sizes = expt_data.hist

                slice_offs = getattr(expt_data, "slice_offs", None)
                if slice_offs is None and hasattr(expt_data, "slice_offsets"):
                    slice_offs = expt_data.slice_offsets
                if slice_offs is None and hasattr(expt_data, "offsets"):
                    slice_offs = expt_data.offsets

                block_schedule = getattr(expt_data, "block_schedule", None)
                block_offs = getattr(expt_data, "block_offs", None)
                if block_schedule is None and hasattr(expt_data, "block_schedule_fn"):
                    block_schedule = expt_data.block_schedule_fn
                if block_offs is None and hasattr(expt_data, "block_offs_fn"):
                    block_offs = expt_data.block_offs_fn

                if block_schedule is None and slice_offs is not None:
                    block_schedule = lambda block: slice_offs
                if block_offs is None and slice_offs is not None:
                    block_offs = lambda block: slice_offs

                expt_data = SimpleNamespace(
                    slice_sizes=slice_sizes,
                    slice_offs=slice_offs,
                    block_schedule=block_schedule,
                    block_offs=block_offs,
                )
            return RoutingData(
                gate_scal=getattr(data, "gate_scal"),
                expt_hist=getattr(data, "expt_hist"),
                n_expts_tot=getattr(data, "n_expts_tot"),
                n_expts_act=getattr(data, "n_expts_act"),
                expt_data=expt_data,
                expected_tokens_per_expt=getattr(data, "expected_tokens_per_expt", None),
            )
        except AttributeError:
            return data

    original_forward = experts_cls.forward
    if getattr(original_forward, "_unsloth_shimmed", False):
        return

    def forward(self, hidden_states, routing_data, gather_idx, scatter_idx):
        _ensure_act(self)
        coerced = _coerce_routing_data(routing_data)
        return original_forward(self, hidden_states, coerced, gather_idx, scatter_idx)

    forward._unsloth_shimmed = True
    experts_cls.forward = forward


# =========================================================
# 0) AMD / MI300X assumptions (no bitsandbytes on AMD)
# =========================================================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =========================================================
# 1) IO helpers
# =========================================================
def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield JSONL records lazily."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_final_text(obj: Dict[str, Any]) -> str:
    """Return cleaned assistant final content."""
    norm = obj.get("normalized")
    if isinstance(norm, str) and norm.strip():
        return norm.strip()
    message = ((obj.get("response", {}).get("choices") or [{}])[0]).get("message", {})
    return (message.get("content") or "").strip()


def extract_reasoning_text(obj: Dict[str, Any]) -> str:
    """Ensure the reasoning (analysis channel) is always non-empty."""
    message = ((obj.get("response", {}).get("choices") or [{}])[0]).get("message", {})
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        return reasoning
    return "(analysis omitted in original sample)"


def make_prompt(messages: List[Dict[str, Any]], tokenizer) -> str:
    """Harmony prompt that ends with an open assistant turn."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )


def make_assistant_suffix(
    messages: List[Dict[str, Any]],
    final_text: str,
    reasoning_text: str,
    tokenizer,
) -> str:
    """Build analysis+final assistant suffix and strip the prompt prefix."""
    full_transcript = tokenizer.apply_chat_template(
        messages
        + [
            {
                "role": "assistant",
                "thinking": reasoning_text,
                "content": final_text,
            }
        ],
        tokenize=False,
        add_generation_prompt=False,
        reasoning_effort="low",
    )
    prompt = make_prompt(messages, tokenizer)
    if full_transcript.startswith(prompt):
        return full_transcript[len(prompt) :]
    splitter = "<|start|>assistant"
    position = full_transcript.rfind(splitter)
    return full_transcript[position:] if position != -1 else full_transcript


def build_pairs_from_file(
    path: str,
    tokenizer,
    *, validate: bool = False
) -> List[Dict[str, str]]:
    """Convert JSONL preference records into DPO pairs with Harmony templates."""

    pairs: List[Dict[str, str]] = []
    for record in iter_jsonl(path):
        messages = record.get("messages") or []
        if not messages:
            continue

        prompt = make_prompt(messages, tokenizer)

        chosen_payload = record.get("chosen", {}) or {}
        rejected_payload = record.get("rejected", {}) or {}
        chosen_final = extract_final_text(chosen_payload)
        rejected_final = extract_final_text(rejected_payload)
        if not chosen_final or not rejected_final:
            continue

        chosen_reason = extract_reasoning_text(chosen_payload)
        rejected_reason = extract_reasoning_text(rejected_payload)

        chosen_suffix = make_assistant_suffix(
            messages, chosen_final, chosen_reason, tokenizer
        )
        rejected_suffix = make_assistant_suffix(
            messages, rejected_final, rejected_reason, tokenizer
        )

        if validate:
            def _check_suffix(label: str, suffix: str) -> None:
                if "<|start|>assistant" not in suffix:
                    print(f"[validator] Missing assistant start in {path} ({label})")
                if "<|channel|>analysis" not in suffix:
                    print(f"[validator] Missing analysis channel in {path} ({label})")
                if "<|channel|>final" not in suffix:
                    print(f"[validator] Missing final channel in {path} ({label})")
                if not (prompt + suffix).startswith(prompt):
                    print(f"[validator] Prompt/suffix alignment issue in {path} ({label})")

            _check_suffix("chosen", chosen_suffix)
            _check_suffix("rejected", rejected_suffix)

        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen_suffix,
                "rejected": rejected_suffix,
            }
        )

    return pairs


# =========================================================
# 3) Dataset sources
# =========================================================
ANSWERS_PATH = "outputs/eval/dpo_answers.jsonl"
QUESTIONS_PATH = "outputs/eval/dpo_questions.jsonl"


# =========================================================
# 4) Load GPT-OSS-120B (MXFP4 base) + LoRA, 8192 ctx
# =========================================================
max_seq_length = 8192
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-120b",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    use_exact_model_name=True,
    fast_inference=False,
    # float8_kv_cache=True,  # Uncomment if supported to shrink KV memory footprint.
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    lora_alpha=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# =========================================================
# 5) Build DPO dataset using Harmony templates
# =========================================================
enable_validation = True
pairs = build_pairs_from_file(
    ANSWERS_PATH,
    tokenizer,
    validate=enable_validation,
)
pairs += build_pairs_from_file(
    QUESTIONS_PATH,
    tokenizer,
    validate=enable_validation,
)
random.shuffle(pairs)
dpo_dataset = Dataset.from_list(pairs)

print(f"[data] DPO pairs: {len(dpo_dataset)}")
if len(dpo_dataset) > 0:
    print("[sample prompt] ===\n", dpo_dataset[0]["prompt"][:400], "...\n===")
    print("[sample chosen suffix] ===\n", dpo_dataset[0]["chosen"][:200], "...\n===")


# =========================================================
# 6) DPO training
# =========================================================
ensure_mxfp4_activation_shim()
PatchDPOTrainer()

args = TrainingArguments(
    output_dir="dpo_gptoss120b_lora_harmony8192",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    bf16=is_bfloat16_supported(),
    fp16=False,
    optim="adamw_torch_fused",
)

# --- Unsloth DPOTrainer compat shim (one place for all extras) ---
def patch_unsloth_args(args, tokenizer, model):
    defaults = dict(
        padding_value=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        remove_unused_columns=False,
        truncation_mode="keep_end",
        packing=False,
        padding_free=False,
        model_init_kwargs=None,
        tokenizer_init_kwargs=None,
        ref_model_init_kwargs=None,
        ref_tokenizer_init_kwargs=None,
        force_use_ref_model=False,
        evaluation_strategy="no",
        generate_during_eval=False,
        predict_with_generate=False,
        generation_max_length=256,
        generation_num_beams=1,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_train_batch_size,
        model_adapter_name=None,
        adapter_name=None,
        ref_model_adapter_name=None,
        ref_adapter_name=None,
        reference_free=True,
        disable_dropout=True,
        use_liger_loss=False,
        use_logits_to_keep=False,
        precompute_ref_log_probs=False,
        precompute_ref_batch_size=None,
        tools=None,
        base_model_attribute_name="model",
        beta=0.1,
        f_divergence_type="kl",
        f_alpha_divergence_coef=1.0,
        label_smoothing=0.0,
        loss_type="sigmoid",
        loss_weights=None,
        use_weighting=False,
        rpo_alpha=None,
        ld_alpha=None,
        discopop_tau=0.05,
        sync_ref_model=False,
        ref_model_mixup_alpha=0.6,
        ref_model_sync_steps=512,
        vllm_sampling_params=None,
        unsloth_num_chunks=-1,
        dataset_num_proc=None,
        max_seq_length=max_seq_length,
        max_length=max_seq_length,
        max_prompt_length=7000,
        max_completion_length=1024,
    )
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Enforce reference-free setup and disabled dropout per run configuration.
    args.reference_free = True
    args.disable_dropout = True

    if args.model_adapter_name is None or args.adapter_name is None:
        adapter = "default"
        peft_config = getattr(model, "peft_config", None)
        if isinstance(peft_config, dict) and peft_config:
            adapter = next(iter(peft_config.keys()))
        if args.model_adapter_name is None:
            args.model_adapter_name = adapter
        if args.adapter_name is None:
            args.adapter_name = adapter

    if args.ref_model_adapter_name is None:
        args.ref_model_adapter_name = None
    if args.ref_adapter_name is None:
        args.ref_adapter_name = None


patch_unsloth_args(args, tokenizer, model)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    beta=0.1,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=7000,
    max_target_length=1024,
)

trainer.train()


# =========================================================
# 7) Save LoRA adapter for single-GPU inference
# =========================================================
model.save_lora("lora_dpo_gptoss120b_harmony8192")

# For multi-GPU serving with larger memory budgets, you can merge:
# model.save_pretrained_merged("merged_dpo_16bit", tokenizer, save_method="merged_16bit")


# =========================================================
# Smoke test example (run separately if desired)
# =========================================================
if __name__ == "__main__":
    print("Training complete. LoRA weights saved to lora_dpo_gptoss120b_harmony8192")
