# Qwen3-4B in action.
import json
import time
from typing import List, Optional

from transformers import AutoTokenizer

from utils.vllm_utils import VLLMConfig, chat_completion, ensure_vllm_server_running


class AAgent(object):
    def __init__(self, config: Optional[VLLMConfig] = None, **kwargs):
        self.config = ensure_vllm_server_running(config or VLLMConfig())
        tokenizer_name = kwargs.get("tokenizer", self.config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left"
        )

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
        if isinstance(message, str):
            message = [message]

        kwargs = dict(kwargs)
        tgps_show_var = kwargs.pop("tgps_show", False)
        params, timeout = self._translate_generation_kwargs(dict(kwargs))

        outputs: List[str] = []
        token_len = 0
        start_time = time.time() if tgps_show_var else None

        for msg in message:
            sys_text, dev_text = self._split_sys_dev(system_prompt)
            messages = [{"role": "system", "content": sys_text}]
            if dev_text:
                messages.append({"role": "developer", "content": dev_text})
            messages.append({"role": "user", "content": msg})
            try:
                response = chat_completion(messages, self.config, timeout=timeout, **params)
            except RuntimeError as exc:
                if dev_text:
                    fallback_messages = [
                        {
                            "role": "system",
                            "content": f"{sys_text}\n\n# Developer\n{dev_text}",
                        },
                        {"role": "user", "content": msg},
                    ]
                    response = chat_completion(
                        fallback_messages, self.config, timeout=timeout, **params
                    )
                else:
                    raise
            if not response.get("choices"):
                outputs.append("")
                continue
            content = self._extract_message_content(response["choices"][0])
            outputs.append(content)
            usage = response.get("usage") or {}
            token_len += usage.get("completion_tokens", 0)

        if tgps_show_var and start_time is not None:
            generation_time = time.time() - start_time
            return (
                outputs[0] if len(outputs) == 1 else outputs,
                token_len,
                generation_time,
            )
        return outputs[0] if len(outputs) == 1 else outputs, None, None

    @staticmethod
    def _extract_message_content(choice: dict) -> str:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            fragments = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type in {"text", "output_text", None}:
                        fragments.append(part.get("text", ""))
            return "".join(fragments).strip()
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
        return str(content).strip() if content is not None else ""


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
