"""Question model wrapper using shared inference provider."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .inference_provider import InferenceProvider


class QAgent(object):
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
            "question", sampling_overrides=sampling_overrides
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

    # WARNING: public contract – signature/return type must match initial commit.
    def count_tokens(self, text: str) -> int:
        return self._provider.count_tokens(text)

    def generate_completion_raw(
        self,
        harmony_prompt: str,
        **kwargs,
    ) -> Tuple[str, Optional[int], Optional[float]]:
        # Optional helper for future Harmony experiments; primary API remains generate_response.
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
    # Single example generation
    model = QAgent()
    prompt = """
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {
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
        }
    """

    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("Single example response:")
    print("Response: ", response)
    if tl is not None and tm:
        print(
            f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl / tm:.2f} tokens/sec"
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
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("\nMulti example responses:")
    if isinstance(responses, list):
        for i, resp in enumerate(responses):
            print(f"Response {i + 1}: {resp}")
    else:
        print(responses)
    if tl is not None and tm:
        print(
            f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl / tm:.2f} tokens/sec"
        )
