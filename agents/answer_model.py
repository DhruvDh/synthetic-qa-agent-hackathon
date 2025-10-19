"""Answer model wrapper using shared inference provider."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .inference_provider import InferenceProvider


class AAgent(object):
    _SAMPLING_KEYS = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "do_sample",
        "repetition_penalty",
    }

    # WARNING: public contract – signature/return type must match initial commit.
    def __init__(self, **kwargs):
        sampling_overrides = {
            key: kwargs[key] for key in self._SAMPLING_KEYS if key in kwargs
        }
        self._provider = InferenceProvider(
            "answer", sampling_overrides=sampling_overrides
        )

    # WARNING: public contract – signature/return type must match initial commit.
    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> Tuple[List[str] | str, Optional[int], Optional[float]]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        prompts = [message] if isinstance(message, str) else list(message)
        tgps_show = kwargs.get("tgps_show", False)
        overrides = {key: kwargs[key] for key in self._SAMPLING_KEYS if key in kwargs}
        extra_args = {
            key: value
            for key, value in kwargs.items()
            if key not in self._SAMPLING_KEYS and key != "tgps_show"
        }
        return self._provider.generate(
            prompts, system_prompt, tgps_show, overrides, extra_args
        )

    def count_tokens(self, text: str) -> int:
        return self._provider.count_tokens(text)

    def generate_completion_raw(
        self,
        harmony_prompt: str,
        **kwargs,
    ) -> Tuple[str, Optional[int], Optional[float]]:
        # NOTE: optional helper for experimental Harmony flows; core contract still uses generate_response.
        tgps_show = kwargs.get("tgps_show", False)
        overrides = {key: kwargs[key] for key in self._SAMPLING_KEYS if key in kwargs}
        extra_args = {
            key: value
            for key, value in kwargs.items()
            if key not in self._SAMPLING_KEYS and key != "tgps_show"
        }
        extra_args.update({"use_completions": True, "raw_harmony": True})
        extra_args.setdefault("stop", ["<|end|>", "<|assistant", "<|start|>"])
        return self._provider.generate(
            [harmony_prompt],
            system_prompt="",
            tgps_show=tgps_show,
            overrides=overrides,
            extra=extra_args,
        )


if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response(
        "Solve: 2x + 5 = 15",
        system_prompt="You are a math tutor.",
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print(f"Single response: {response}")
    if tl is not None and gt:
        print(
            f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl / gt:.2f}"
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
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        tgps_show=True,
    )
    print("Responses:")
    if isinstance(responses, list):
        for i, resp in enumerate(responses):
            print(f"Message {i + 1}: {resp}")
    else:
        print(responses)
    if tl is not None and gt:
        print(
            f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl / gt:.2f}"
        )
    print("-----------------------------------------------------------")

    # Custom parameters
    response, tl, gt = ans_agent.generate_response(
        "Write a story",
        temperature=0.8,
        max_new_tokens=512,
        tgps_show=True,
    )
    print(f"Custom response: {response}")
    if tl is not None and gt:
        print(
            f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl / gt:.2f}"
        )
