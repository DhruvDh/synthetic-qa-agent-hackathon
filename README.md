# GPT-OSS DPO Playground

This repository fine-tunes OpenAI’s public **gpt-oss-20b** reasoning model with **Direct Preference Optimization (DPO)** and wires the resulting model into a two-agent tournament loop (question writer + answerer) that runs on top of **vLLM**.

The workflow is designed for ROCm/MI300X hardware (bf16) but also runs on CUDA with minor adjustments.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `train_dpo_gptoss120b_harmony.py` | DPO fine-tuning script (bf16, Harmony prompts, LoRA) |
| `scripts/merge_lora_to_bf16.py` | One-off utility to merge the LoRA adapter into bf16 base weights |
| `agents/` | Question and answer agents (prompt logic lives here) |
| `utils/vllm_utils.py` | vLLM launcher + OpenAI-compatible client wrapper with sentinel management |
| `utils/build_prompt.py` | Small helpers for repairing / extracting JSON answers |
| `outputs/` | Default location for JSONL preference datasets (`dpo_questions.jsonl`, `dpo_answers.jsonl`) |
| `merged_gptoss20b_bf16/` | Sample output directory containing the merged full-model checkpoint after running the merge script |

> ⚠️ **Agent guard rails** — `agents/answer_agent.py` and `agents/question_agent.py` may only have prompt text edited (see `AGENTS.md`). All structural logic must remain untouched.

---

## Prerequisites

* Python 3.11+ (tested with 3.12)
* ROCm 7.1+ on MI300X (or CUDA 12.x on NVIDIA if you adjust the environment variables)
* [`uv`](https://github.com/astral-sh/uv) for reproducible installs (recommended)  
  Install with `pip install uv` or follow the upstream instructions.

GPU drivers must expose bf16; the finetune loads `openai/gpt-oss-20b` with bf16 weights via `Mxfp4Config(dequantize=True)`.

---

## Set Up Python Dependencies

```bash
# Optional: create an isolated venv first
python -m venv .venv
source .venv/bin/activate

# Install core libraries (transformers, trl, vllm, peft, etc.)
uv pip install -r default_requirements.txt
```

If you need the latest TRL/Transformers features, the DPO script installs them automatically on first run; otherwise use the pinned versions in `default_requirements.txt`.

---

## Data Preparation

The DPO script expects two JSONL files containing pairwise preferences:

```
outputs/eval/dpo_questions.jsonl
outputs/eval/dpo_answers.jsonl
```

Each record must include:

```json
{
  "messages": [...],    // Harmony-format chat transcript
  "chosen": {...},
  "rejected": {...}
}
```

The helper functions in `train_dpo_gptoss120b_harmony.py` strip prompts/suffixes using the tokenizer’s built-in Harmony template, preserving analysis/final channels.

---

## Fine-Tune GPT-OSS-20B with DPO

```bash
uv run python train_dpo_gptoss120b_harmony.py
```

What the script does:

1. Loads `openai/gpt-oss-20b` with MXFP4 weights dequantized to bf16.
2. Adds a LoRA adapter (`r=32`, attention + MoE projection modules).
3. Builds a DPO dataset from the two JSONL files (with optional validation prints).
4. Trains for one epoch using TRL’s `DPOTrainer` (bf16, cosine LR).
5. Saves the LoRA adapter + tokenizer in `lora_dpo_gptoss20b_bf16/`.

Warnings you can safely ignore:

* `bitsandbytes` ROCm binary not found – we deliberately run without bnb.
* TF32 deprecation messages – PyTorch 2.10 reminders.
* Gradient checkpointing “inputs do not require grad” – expected behavior.

Training produces the adapter folder only; the base weights are *not* modified.

---

## Merge the LoRA Adapter (Optional but Recommended)

To serve the model with vLLM (which, for GPT-OSS today, does not accept LoRA adapters), merge the adapter into the base bf16 weights:

```bash
uv run python scripts/merge_lora_to_bf16.py \
  --base-model openai/gpt-oss-20b \
  --lora-dir lora_dpo_gptoss20b_bf16 \
  --output-dir merged_gptoss20b_bf16
```

The script writes a standard Hugging Face model directory with the tokenizer copied alongside. You can safely rerun it; existing files will be overwritten.

---

## Serving with vLLM

`utils/vllm_utils.py` manages a background vLLM server and exposes a thin OpenAI-compatible `chat_completion` wrapper.

**Defaults** (see `VLLMConfig`):

* Model directory: `merged_gptoss20b_bf16`
* Host/Port: `127.0.0.1:4200`
* Tensor Parallelism: 1 (override with `VLLM_TENSOR_PARALLEL_SIZE`)
* Extra flags: `--dtype bfloat16`, `--trust-remote-code`, OpenAI tool-call parser

To override the model or launch command, export environment variables before calling any agent:

```bash
export VLLM_MODEL=/absolute/path/to/merged_gptoss20b_bf16
export VLLM_TENSOR_PARALLEL_SIZE=2   # optional
```

### Quick Launch / Teardown

```bash
# Start or reuse the server (writes runtime/vllm_server.json sentinel)
uv run python -m utils.vllm_utils --launch

# Check health
uv run python -m utils.vllm_utils --status

# Stop the managed server
uv run python -m utils.vllm_utils --stop
```

Once running, your agents will hit `http://127.0.0.1:4200/v1/chat/completions` automatically.

---

## Agents Overview

* `agents/answer_agent.py` – consumes puzzles, enforces a strict JSON schema, and can optionally self-reflect to repair malformed outputs. Uses `AnsweringAgent.answer_question`.
* `agents/question_agent.py` – generates puzzles (topics: seating or blood relations), again backing off to JSON repair heuristics if needed.
* `agents/answer_model.py` / `agents/question_model.py` – wrap vLLM requests, handle concurrency, and manage raw response caches.

Prompts live in the agent files and are the only editable sections per hackathon rules.

---

## Typical End-to-End Flow

1. **Prepare data** → `outputs/eval/dpo_*.jsonl`.
2. **Fine-tune** → `uv run python train_dpo_gptoss120b_harmony.py`.
3. **Merge adapter** → `uv run python scripts/merge_lora_to_bf16.py`.
4. **Start vLLM** → `uv run python -m utils.vllm_utils --launch`.
5. **Run agents** → import `AnsweringAgent` or `QuestioningAgent` and call `answer_question(...)` / `build_prompt(...)`.

Each agent call will ensure the vLLM server is up (launching it if necessary) and route requests through the OpenAI-compatible client.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `bitsandbytes` missing ROCm library | Safe to ignore; we do not rely on bnb kernels. |
| `torch.utils.checkpoint` warning | Expected with gradient checkpointing. |
| vLLM fails to load LoRA | Merge the adapter first (`scripts/merge_lora_to_bf16.py`). |
| Need to switch models | Override `VLLM_MODEL` or pass `VLLMConfig(model=...)` when creating agents. |
| Server already running | Sentinel file lives at `runtime/vllm_server.json`; delete it if a stale PID sticks around. |

---

## Contributing

*Respect the agent modification constraints* (prompt text only).
Before committing, run `git diff` to confirm the two agent files are unchanged aside from allowed prompt edits.

For any new training experiments, consider parameterizing `train_dpo_gptoss120b_harmony.py` or adding scripts under `scripts/` to keep the root uncluttered.

---

Happy fine-tuning! 🎯 If you run into issues with ROCm + vLLM, AMD’s official guides are a good reference, and the vLLM docs have a dedicated section for GPT-OSS integration.
