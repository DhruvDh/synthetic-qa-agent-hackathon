"""
Shared inference provider abstraction for HF and OpenAI-compatible backends.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import ProviderSettings, get_provider_settings, get_sampling_settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai optional
    OpenAI = None  # type: ignore

from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceProvider:
    def __init__(
        self,
        agent_key: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_key = agent_key.lower()
        self.provider_settings: ProviderSettings = get_provider_settings()
        self.base_sampling: Dict[str, Any] = get_sampling_settings(self.agent_key)
        if sampling_overrides:
            for key, value in sampling_overrides.items():
                if key in self.base_sampling:
                    self.base_sampling[key] = value

        self._client: Optional[Any] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    # ------------------
    # Public API
    # ------------------
    def generate(
        self,
        messages: List[str],
        system_prompt: str,
        tgps_show: bool,
        overrides: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str] | str, Optional[int], Optional[float]]:
        sampling = self._merge_sampling(overrides)
        provider = self.provider_settings.provider
        if provider == "openai":
            return self._generate_openai(
                messages, system_prompt, tgps_show, sampling, extra or {}
            )
        return self._generate_hf(
            messages, system_prompt, tgps_show, sampling, extra or {}
        )

    def count_tokens(self, text: str) -> int:
        tokenizer = self._tokenizer
        if tokenizer is None and self.provider_settings.provider != "openai":
            self._ensure_hf_backend()
            tokenizer = self._tokenizer
        if tokenizer is None:
            return len(text.split())
        return len(tokenizer.encode(text, add_special_tokens=False))

    # ------------------
    # Internal helpers
    # ------------------
    def _merge_sampling(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self.base_sampling)
        if overrides:
            for key, value in overrides.items():
                if key in merged:
                    merged[key] = value
        return merged

    @staticmethod
    def _extract_openai_content(message: Any) -> str:
        """
        Normalise OpenAI-style message content:
        - handles str, list of content parts, dicts, or pydantic objects.
        - returns empty string when no textual content is present.
        """
        def _parts_to_text(parts: Any) -> str:
            texts: List[str] = []
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text")
                    else:
                        text = getattr(part, "text", None)
                    if text:
                        texts.append(str(text))
            return "".join(texts)

        content = getattr(message, "content", None)
        if isinstance(content, list):
            return _parts_to_text(content).strip()
        if isinstance(content, str):
            return content.strip()
        if content is None:
            # Try dictionary-like representations
            if hasattr(message, "model_dump"):
                data = message.model_dump()
            elif hasattr(message, "to_dict"):
                data = message.to_dict()
            elif isinstance(message, dict):
                data = message
            else:
                data = None
            if isinstance(data, dict):
                raw = data.get("content")
                if isinstance(raw, list):
                    return _parts_to_text(raw).strip()
                if raw is not None:
                    return str(raw).strip()
        return str(content).strip() if content is not None else ""

    def _ensure_openai_client(self) -> None:
        if self._client is None:
            assert OpenAI is not None, (
                "pip install openai>=1.0 to use OpenAI-compatible provider"
            )
            self._client = OpenAI(
                base_url=self.provider_settings.openai_base_url,
                api_key=self.provider_settings.openai_api_key,
            )

    def _ensure_hf_backend(self) -> None:
        if self._tokenizer is None or self._model is None:
            model_name = self.provider_settings.hf_model
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )

    def _generate_openai(
        self,
        messages: List[str],
        system_prompt: str,
        tgps_show: bool,
        sampling: Dict[str, Any],
        _: Dict[str, Any],
    ) -> Tuple[List[str] | str, Optional[int], Optional[float]]:
        self._ensure_openai_client()
        outs: List[str] = []
        total_tokens = 0
        start = time.time() if tgps_show else None
        for msg in messages:
            resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.provider_settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg},
                ],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                max_tokens=sampling["max_new_tokens"],
            )
            choice = resp.choices[0]
            message = getattr(choice, "message", None)
            content = self._extract_openai_content(message) if message else ""
            outs.append(content)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                total_tokens += getattr(usage, "completion_tokens", 0) or 0
        elapsed = (time.time() - start) if tgps_show else None
        payload: List[str] | str = outs if len(outs) > 1 else outs[0]
        return payload, (total_tokens if tgps_show else None), elapsed

    def _generate_hf(
        self,
        messages: List[str],
        system_prompt: str,
        tgps_show: bool,
        sampling: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> Tuple[List[str] | str, Optional[int], Optional[float]]:
        self._ensure_hf_backend()
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msg in messages
        ]
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        start = time.time() if tgps_show else None
        gen_ids = model.generate(
            **model_inputs,
            max_new_tokens=sampling["max_new_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            do_sample=sampling["do_sample"],
            repetition_penalty=sampling["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            **extra,
        )
        elapsed = (time.time() - start) if tgps_show else None

        outputs: List[str] = []
        token_len = 0
        for input_ids, seq in zip(model_inputs.input_ids, gen_ids):
            new_ids = seq[len(input_ids) :].tolist()
            token_len += len(new_ids)
            outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
        payload: List[str] | str = outputs if len(outputs) > 1 else outputs[0]
        return payload, (token_len if tgps_show else None), elapsed
