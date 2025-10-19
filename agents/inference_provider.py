"""
Inference provider that targets OpenAI-compatible backends with Harmony completions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import ProviderSettings, get_provider_settings, get_sampling_settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai optional
    OpenAI = None  # type: ignore


class InferenceProvider:
    def __init__(
        self,
        agent_key: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_key = agent_key.lower()
        self.provider_settings: ProviderSettings = get_provider_settings()
        if self.provider_settings.provider != "openai":
            raise ValueError(
                "Only OpenAI-compatible completions are supported by this provider."
            )
        self.base_sampling: Dict[str, Any] = get_sampling_settings(self.agent_key)
        if sampling_overrides:
            for key, value in sampling_overrides.items():
                if key in self.base_sampling:
                    self.base_sampling[key] = value

        self._client: Optional[Any] = None

    def generate(
        self,
        messages: List[str],
        system_prompt: str,
        tgps_show: bool,
        overrides: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str] | str, Optional[int], Optional[float]]:
        sampling = self._merge_sampling(overrides)
        extra = extra or {}

        if extra.get("use_completions") and extra.get("raw_harmony"):
            completions: List[str] = []
            total_tokens: Optional[int] = 0
            total_time: Optional[float] = 0.0
            for prompt in messages:
                text, tokens, elapsed = self._generate_openai_completions_raw(
                    prompt, tgps_show, sampling, extra.get("stop")
                )
                completions.append(text)
                if tokens is None:
                    total_tokens = None
                elif total_tokens is not None:
                    total_tokens += tokens
                if elapsed is None:
                    total_time = None
                elif total_time is not None:
                    total_time += elapsed
            payload: List[str] | str = (
                completions if len(completions) > 1 else completions[0]
            )
            return payload, total_tokens, total_time

        return self._generate_openai(messages, system_prompt, tgps_show, sampling)

    def count_tokens(self, text: str) -> int:
        # Without a tokenizer, fall back to a whitespace approximation.
        return len(text.split())

    def _merge_sampling(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self.base_sampling)
        if overrides:
            for key, value in overrides.items():
                if key in merged:
                    merged[key] = value
        return merged

    def _ensure_openai_client(self) -> None:
        if self._client is None:
            assert OpenAI is not None, (
                "pip install openai>=1.0 to use OpenAI-compatible provider"
            )
            self._client = OpenAI(
                base_url=self.provider_settings.openai_base_url,
                api_key=self.provider_settings.openai_api_key,
            )

    def _generate_openai(
        self,
        messages: List[str],
        system_prompt: str,
        tgps_show: bool,
        sampling: Dict[str, Any],
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
                temperature=sampling.get("temperature"),
                top_p=sampling.get("top_p"),
                max_tokens=sampling.get("max_new_tokens"),
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

    def _generate_openai_completions_raw(
        self,
        prompt: str,
        tgps_show: bool,
        sampling: Dict[str, Any],
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[int], Optional[float]]:
        self._ensure_openai_client()
        start = time.time() if tgps_show else None
        resp = self._client.completions.create(  # type: ignore[union-attr]
            model=self.provider_settings.openai_model,
            prompt=prompt,
            temperature=sampling.get("temperature"),
            top_p=sampling.get("top_p"),
            max_tokens=sampling.get("max_new_tokens"),
            repetition_penalty=sampling.get("repetition_penalty"),
            stop=stop,
        )
        text = resp.choices[0].text if resp.choices else ""
        usage = getattr(resp, "usage", None)
        total_tokens = getattr(usage, "completion_tokens", None) if usage else None
        elapsed = (time.time() - start) if tgps_show else None
        return text, total_tokens, elapsed

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
