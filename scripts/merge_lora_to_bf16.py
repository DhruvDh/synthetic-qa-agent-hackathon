#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge a LoRA adapter fine-tuned on GPT-OSS-20B into bf16 base weights.

Usage (default paths assume training artifacts live in the repo root):
    python scripts/merge_lora_to_bf16.py \
        --base-model openai/gpt-oss-20b \
        --lora-dir lora_dpo_gptoss20b_bf16 \
        --output-dir merged_gptoss20b_bf16
"""

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GPT-OSS LoRA adapter into bf16 weights.")
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Hugging Face model id or path for the base GPT-OSS-20B checkpoint.",
    )
    parser.add_argument(
        "--lora-dir",
        default="lora_dpo_gptoss20b_bf16",
        help="Directory containing the saved LoRA adapter (from training).",
    )
    parser.add_argument(
        "--output-dir",
        default="merged_gptoss20b_bf16",
        help="Destination directory for the merged bf16 weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="bfloat16",
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
    )

    merged_model = PeftModel.from_pretrained(base_model, args.lora_dir)
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"[merge] Saved merged bf16 model to {output_path.resolve()}")


if __name__ == "__main__":
    main()
