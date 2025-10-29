# Qwen3-4B in action.
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from utils.vllm_utils import VLLMConfig, chat_completion, ensure_vllm_server_running


class AAgent(object):
    def __init__(self, config: Optional[VLLMConfig] = None, **kwargs):
        self.config = ensure_vllm_server_running(config or VLLMConfig())
        tokenizer_name = kwargs.get("tokenizer", self.config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left"
        )
        self._local = threading.local()

    @property
    def last_raw_responses(self) -> List[Dict[str, Any]]:
        stack = getattr(self._local, "raw_responses_stack", None)
        if stack:
            try:
                latest = stack.pop()
            except IndexError:
                latest = []
        else:
            latest = getattr(self._local, "raw_responses", [])
        return list(latest)

    def _record_raw_responses(self, responses: List[Dict[str, Any]]) -> None:
        self._local.raw_responses = responses
        stack = getattr(self._local, "raw_responses_stack", None)
        if stack is None:
            stack = []
            self._local.raw_responses_stack = stack
        else:
            stack.clear()
        stack.append(responses)
    @staticmethod
    def _split_sys_dev(system_prompt: str) -> tuple[str, Optional[str]]:
        marker = "<|DEVELOPER|>"
        if marker in system_prompt:
            sys_text, dev_text = system_prompt.split(marker, 1)
            return sys_text.strip(), dev_text.strip()
        return system_prompt.strip(), None

    def _translate_generation_kwargs(self, kwargs: dict) -> tuple[dict, float]:
        params = {}
        timeout = kwargs.pop("timeout", 60.0)
        mapping = {"max_new_tokens": "max_tokens"}
        for key, value in kwargs.items():
            if key in mapping:
                params[mapping[key]] = value
            elif key != "tgps_show":
                params[key] = value
        return params, timeout

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        message_list = [message] if isinstance(message, str) else list(message)

        kwargs = dict(kwargs)
        return_raw = kwargs.pop("return_raw", False)
        tgps_show_var = kwargs.pop("tgps_show", False)
        concurrency_raw = kwargs.pop("concurrency", 128)
        try:
            concurrency = int(concurrency_raw)
        except (TypeError, ValueError):
            concurrency = 1
        concurrency = max(1, min(concurrency, 128))
        params, timeout = self._translate_generation_kwargs(dict(kwargs))

        outputs: List[str] = [""] * len(message_list)
        raw_responses: List[Dict[str, Any]] = [{}] * len(message_list)
        token_len = 0
        start_time = time.time() if tgps_show_var else None
        sys_text, dev_text = self._split_sys_dev(system_prompt)

        def handle_single(idx: int, prompt_text: str) -> tuple[int, str, int, Dict[str, Any]]:
            local_messages = [{"role": "system", "content": sys_text}]
            if dev_text:
                local_messages.append({"role": "developer", "content": dev_text})
            local_messages.append({"role": "user", "content": prompt_text})
            try:
                response = chat_completion(
                    local_messages, self.config, timeout=timeout, **params
                )
            except RuntimeError as exc:
                if dev_text:
                    fallback_messages = [
                        {
                            "role": "system",
                            "content": f"{sys_text}\n\n# Developer\n{dev_text}",
                        },
                        {"role": "user", "content": prompt_text},
                    ]
                    response = chat_completion(
                        fallback_messages, self.config, timeout=timeout, **params
                    )
                else:
                    raise
            if not response.get("choices"):
                return idx, "", 0, response
            content = self._extract_message_content(response["choices"][0])
            usage = response.get("usage") or {}
            return idx, content, usage.get("completion_tokens", 0), response

        if concurrency > 1 and len(message_list) > 1:
            with ThreadPoolExecutor(
                max_workers=min(concurrency, len(message_list))
            ) as executor:
                futures = {
                    executor.submit(handle_single, idx, msg): idx
                    for idx, msg in enumerate(message_list)
                }
                for future in as_completed(futures):
                    idx, content, tokens, raw = future.result()
                    outputs[idx] = content
                    raw_responses[idx] = raw
                    token_len += tokens
        else:
            for idx, msg in enumerate(message_list):
                _, content, tokens, raw = handle_single(idx, msg)
                outputs[idx] = content
                raw_responses[idx] = raw
                token_len += tokens

        results = outputs[0] if len(outputs) == 1 else outputs
        self._record_raw_responses(raw_responses)

        if tgps_show_var and start_time is not None:
            generation_time = time.time() - start_time
            base_return = (
                results,
                token_len,
                generation_time,
            )
        else:
            base_return = (results, None, None)

        if return_raw:
            return (*base_return, list(raw_responses))
        return base_return

    @staticmethod
    def _extract_message_content(choice: dict) -> str:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        if isinstance(content, list):
            fragments = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type in {"text", "output_text", None}:
                        fragments.append(part.get("text", ""))
            text = "".join(fragments).strip()
        else:
            text = str(content).strip() if content is not None else ""

        if not text:
            fallback = AAgent._extract_reasoning(choice, message)
            if fallback:
                return fallback

        if content is None:
            tool_calls = message.get("tool_calls")
            if tool_calls:
                try:
                    return json.dumps(tool_calls)
                except (TypeError, ValueError):
                    return str(tool_calls)
            refusal = message.get("refusal")
            if refusal:
                return refusal.strip()
        return text

    @staticmethod
    def _extract_reasoning(choice: dict, message: dict) -> str:
        def _collect_text(source: Any) -> str:
            if isinstance(source, str):
                return source.strip()
            if isinstance(source, list):
                parts = []
                for item in source:
                    if isinstance(item, dict):
                        text = item.get("content") or item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts).strip()
            if isinstance(source, dict):
                text = source.get("content") or source.get("text")
                if isinstance(text, str):
                    return text.strip()
            return ""

        candidates = [
            message.get("reasoning_content"),
            message.get("reasoning"),
            choice.get("reasoning_content"),
            choice.get("reasoning"),
        ]
        for candidate in candidates:
            text = _collect_text(candidate)
            if text:
                return text
        return ""


if __name__ == "__main__":
    ensure_vllm_server_running()
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response(
        "Solve: 2x + 5 = 15",
        system_prompt="You are a math tutor.",
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        repetition_penalty=0.0,
    )
    print(f"Single response: {response}")
    print(
        f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}"
    )
    print("-----------------------------------------------------------")

    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(
        messages,
        max_new_tokens=512,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        repetition_penalty=0.0,
        tgps_show=True,
    )
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(
        f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}"
    )
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story",
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        repetition_penalty=0.0,
        max_new_tokens=512,
    )
    print(f"Custom response: {response}")
