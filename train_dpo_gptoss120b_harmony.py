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
from typing import Any, Dict, Iterator, List

from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer


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
    fast_inference=True,
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
PatchDPOTrainer()

training_args = TrainingArguments(
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

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
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
