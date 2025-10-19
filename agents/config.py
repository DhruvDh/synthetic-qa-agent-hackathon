"""
Centralized configuration helpers for inference backends and sampling params.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def _read_prompt(path: Path) -> "PromptBlocks":
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {path} not found.")
    system, developer, user = [], [], []
    current = None
    with path.open("r") as handle:
        for line in handle:
            header = line.strip()
            if header == "# System":
                current = system
                continue
            if header == "# Developer":
                current = developer
                continue
            if header == "# User Template":
                current = user
                continue
            if current is not None:
                current.append(line)
    return PromptBlocks(
        system="".join(system).strip(),
        developer="".join(developer).strip(),
        user="".join(user).strip(),
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "config" / "inference.yaml"
QGEN_FILE = REPO_ROOT / "qgen.yaml"
AGEN_FILE = REPO_ROOT / "agen.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts"
Q_PROMPT_FILE = PROMPTS_DIR / "q_agent.md"
A_PROMPT_FILE = PROMPTS_DIR / "a_agent.md"

_DEFAULT_PROVIDER: Dict[str, Any] = {
    "type": "openai",
    "hf_model": "Qwen/Qwen3-4B",
    "openai": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "sk-noop",
        "model": "local",
    },
}

_DEFAULT_SAMPLING: Dict[str, Dict[str, Any]] = {
    "question": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 1.0,
        "repetition_penalty": 0,
        "top_k": 0,
        "do_sample": True,
    },
    "answer": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 1.0,
        "repetition_penalty": 0,
        "top_k": 0,
        "do_sample": True,
    },
}


@dataclass(frozen=True)
class PromptBlocks:
    system: str
    developer: str
    user: str


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    hf_model: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache()
def load_inference_config() -> Dict[str, Any]:
    config = {"provider": copy.deepcopy(_DEFAULT_PROVIDER)}
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            config = _deep_merge(config, data)
    return config


def get_provider_settings() -> ProviderSettings:
    config = load_inference_config()
    provider_cfg = config.get("provider", {})
    default_provider = _DEFAULT_PROVIDER

    openai_defaults = default_provider["openai"]
    openai_cfg = provider_cfg.get("openai", {})

    return ProviderSettings(
        provider=str(provider_cfg.get("type", default_provider["type"])).lower(),
        hf_model=str(provider_cfg.get("hf_model", default_provider["hf_model"])),
        openai_base_url=str(openai_cfg.get("base_url", openai_defaults["base_url"])),
        openai_api_key=str(openai_cfg.get("api_key", openai_defaults["api_key"])),
        openai_model=str(openai_cfg.get("model", openai_defaults["model"])),
    )


def get_sampling_settings(agent_key: str) -> Dict[str, Any]:
    agent_key = agent_key.lower()
    if agent_key not in _DEFAULT_SAMPLING:
        raise ValueError(f"Unknown agent key: {agent_key}")

    defaults = _DEFAULT_SAMPLING[agent_key]
    yaml_path = QGEN_FILE if agent_key == "question" else AGEN_FILE
    if yaml_path.exists():
        with yaml_path.open("r") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            merged = dict(defaults)
            for key, value in data.items():
                if key in merged:
                    merged[key] = value
            return merged
    return dict(defaults)


@lru_cache()
def get_question_prompt_blocks() -> PromptBlocks:
    return _read_prompt(Q_PROMPT_FILE)


@lru_cache()
def get_answer_prompt_blocks() -> PromptBlocks:
    return _read_prompt(A_PROMPT_FILE)
