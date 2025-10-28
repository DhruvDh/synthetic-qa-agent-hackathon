# Starting with Qwen3-4B
import json
import time
from typing import List, Optional

from transformers import AutoTokenizer

from utils.vllm_utils import (
    VLLMConfig,
    chat_completion,
    ensure_vllm_server_running,
)


class QAgent(object):
    def __init__(self, config: Optional[VLLMConfig] = None, **kwargs):
        self.config = ensure_vllm_server_running(config or VLLMConfig())
        tokenizer_name = kwargs.get("tokenizer", self.config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left"
        )

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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            response = chat_completion(messages, self.config, timeout=timeout, **params)
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
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """

    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        repetition_penalty=0.0,
    )
    print("Single example response:")
    print("Response: ", response)
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(
        prompts,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        repetition_penalty=0.0,
    )
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
