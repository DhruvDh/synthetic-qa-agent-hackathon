#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune GPT-OSS-20B with DPO using Harmony chat templates in bf16.

- Applies tokenizer.apply_chat_template for prompts and targets so training matches inference.
- Loads `openai/gpt-oss-20b` with MXFP4 weights dequantized to bf16 (no Unsloth patches required).
- Adds a LoRA adapter and trains with TRL's reference DPOTrainer.
- Expects DPO preference pairs in outputs/eval/dpo_answers.jsonl and outputs/eval/dpo_questions.jsonl.
"""

import json
import os
import random
from typing import Any, Dict, Iterator, List

import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer


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
# 4) Load GPT-OSS-20B bf16 + LoRA
# =========================================================
max_seq_length = 8192
lora_rank = 32

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True),
    device_map="auto",
)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    init_lora_weights="gaussian",
)
model = get_peft_model(model, peft_config)


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
dpo_args = DPOConfig(
    output_dir="dpo_gptoss20b_lora_bf16",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    beta=0.1,
    report_to=[],
    max_length=max_seq_length,
    max_prompt_length=7000,
    max_completion_length=1024,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)

trainer.train()


# =========================================================
# 7) Save LoRA adapter for inference
# =========================================================
model.save_pretrained("lora_dpo_gptoss20b_bf16")
tokenizer.save_pretrained("lora_dpo_gptoss20b_bf16")

if __name__ == "__main__":
    print("Training complete. LoRA weights saved to lora_dpo_gptoss20b_bf16")
