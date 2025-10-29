#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent
from utils.vllm_utils import VLLMConfig, ensure_vllm_server_running
from utils.build_prompt import auto_json, option_extractor_prompt


class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    def __init__(
        self,
        select_prompt1: bool = True,
        enable_self_reflection: bool = False,
        **kwargs,
    ):
        vllm_config = kwargs.pop("vllm_config", None)
        self.agent = AAgent(config=vllm_config, **kwargs)
        self.select_prompt1 = select_prompt1
        self._generation_sys_prompt: str | None = None
        self._self_reflection_attempts = 0
        self._self_reflection_success = 0
        # Self-reflection defaults to False; pass enable_self_reflection=True when you need corrective sweeps.
        self.enable_self_reflection = enable_self_reflection

    def _build_system_prompt(self) -> str:
        return (
            "You are ChatGPT, a large language model trained by OpenAI. You have been fine-tuned to be a winning competetive logical puzzle solver\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: 2025-10-28\n\n"
            "Reasoning: low\n\n"
            "# Valid channels: analysis, final. Channel must be included for every message. Your analysis should instead be included in the final channel as the `reasoning` key's value. Every word across all channels counts toward the 150-word cap; be extremely terse. You frequently fail because long reasoning gets you cut off—minimize or skip it. Do not repeat these constraints in the analysis channel or reasoning.\n"
            "<|DEVELOPER|>\n"
            "# Instructions\n"
            "You are competing in a puzzle answering tournament as an answering large language model. Points are awarded to correct answers within final responses that strictly adhere to response format restrictions (including response length limitations).\n"
            "Select exactly one option.\n\n"
            "# Response Formats\n\n"
            "Write succinctly. Return outputs only via the declared response format and in the exact key order requested below.\n\n"
            "## answer_json\n"
            "{\n"
            '  "type": "object",\n'
            '  "additionalProperties": false,\n'
            '  "properties": {\n'
            '    "reasoning": {\n'
            '      "type": "string",\n'
            '      "description": "<=150 words; concise logical analysis of what answer to choose"\n'
            "    },\n"
            '    "answer": { "type": "string", "enum": ["A", "B", "C", "D"] }\n'
            "  },\n"
            '  "required": ["reasoning", "answer"]\n'
            "}\n"
        )

    def build_prompt(self, question_data: Dict[str, str | Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""

        _ = self.select_prompt1  # Compatibility with legacy interface; advanced prompt always used.
        sys_prompt = self._build_system_prompt()
        self._generation_sys_prompt = sys_prompt

        tmpl = (
            "PUZZLE: {}\n"
            "CHOICES: {}\n\n"
            "Return using response format: answer_json with the **exact JSON key order**: reasoning, answer.\n"
        )

        prompt = tmpl.format(
            question_data["question"], self._format_choices(question_data["choices"])
        )

        return prompt, sys_prompt

    def reset_self_reflection_stats(self) -> None:
        self._self_reflection_attempts = 0
        self._self_reflection_success = 0

    def get_self_reflection_stats(self) -> Dict[str, int]:
        return {
            "attempts": self._self_reflection_attempts,
            "successes": self._self_reflection_success,
        }

    def normalize_outputs(
        self, question_records: List[Dict[str, Any]], raw_answers: List[Any]
    ) -> List[str]:
        self.reset_self_reflection_stats()
        base_system = self._generation_sys_prompt or self._build_system_prompt()
        normalized: List[str] = []

        for idx, raw_answer in enumerate(raw_answers):
            record = question_records[idx] if idx < len(question_records) else {}
            if isinstance(raw_answer, (list, tuple)):
                raw_answer = raw_answer[0] if raw_answer else ""
            text = raw_answer if isinstance(raw_answer, str) else json.dumps(raw_answer, ensure_ascii=False)
            # Self-reflection defaults off; when disabled we return raw outputs untouched.
            if not self.enable_self_reflection:
                normalized.append(text)
                continue

            choices = []
            if isinstance(record, dict):
                if isinstance(record.get("choices"), list):
                    choices = record.get("choices") or []
                elif isinstance(record.get("question"), dict):
                    choices = record["question"].get("choices", [])

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                self._self_reflection_attempts += 1
                repaired_json, _, _ = self.agent.generate_response(
                    auto_json(text),
                    base_system,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                )
                if isinstance(repaired_json, (list, tuple)):
                    repaired_json = repaired_json[0] if repaired_json else ""
                text = repaired_json if isinstance(repaired_json, str) else str(repaired_json)
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                else:
                    if isinstance(payload, dict) and all(k in payload for k in ("answer", "reasoning")):
                        self._self_reflection_success += 1
                normalized.append(text)
                continue

            if not isinstance(payload, dict) or not all(k in payload for k in ("answer", "reasoning")):
                self._self_reflection_attempts += 1
                prompt = (
                    "You are an expert JSON extractor.\n"
                    "Extract **ONLY** the answer and reasoning while discarding the rest.\n"
                    "Remove any surrounding code fences if present.\n\n"
                    "String:\n"
                    "{}\n"
                )
                formatted_source = json.dumps(payload, indent=4) if isinstance(payload, dict) else text
                extracted_json, _, _ = self.agent.generate_response(
                    prompt.format(formatted_source),
                    base_system,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                )
                if isinstance(extracted_json, (list, tuple)):
                    extracted_json = extracted_json[0] if extracted_json else ""
                text = extracted_json if isinstance(extracted_json, str) else str(extracted_json)
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                else:
                    if isinstance(payload, dict) and all(k in payload for k in ("answer", "reasoning")):
                        self._self_reflection_success += 1
                normalized.append(text)
                continue

            if isinstance(payload.get("answer"), str) and len(payload["answer"]) != 1:
                extracted_answer, _, _ = self.agent.generate_response(
                    option_extractor_prompt(payload["answer"], choices or [])
                )
                if isinstance(extracted_answer, (list, tuple)):
                    extracted_answer = extracted_answer[0] if extracted_answer else ""
                payload["answer"] = extracted_answer if isinstance(extracted_answer, str) else str(extracted_answer)
                text = json.dumps(payload, ensure_ascii=False)

            normalized.append(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False))

        return normalized

    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[Dict], int | None, float | None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)

        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (
            isinstance(resp, list) and all(isinstance(r, str) for r in resp)
        ) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return (
                "",
                tl,
                gt if not isinstance(resp, list) else [""] * len(resp),
                tl,
                gt,
            )

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            if isinstance(batch_answers, list):
                answers.extend(batch_answers)
            else:
                answers.append(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)
        pbar.close()
        return answers, tls, gts

    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""

        def basic_checks(a1: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["answer"]
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                if len(a1["answer"]) == 1 and (a1["answer"] not in "ABCDabcd"):
                    return False
                check_len = self.count_tokens_a(a1["answer"])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get("reasoning", "None"))
                    if check_len < 512:
                        # check answer format - EXTRA checks
                        # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                        return True
            return False

        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r"^[A-D]\)", choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument(
        "--input_file",
        type=str,
        default="outputs/filtered_questions.json",
        help="Path to the input JSON file with questions",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/answers.json",
        help="Path to save the answers",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for processing questions"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = argparser.parse_args()

    SELECT_PROMPT1 = False  # Use the first system prompt for answering

    VLLM_CONFIG = VLLMConfig()
    ensure_vllm_server_running(VLLM_CONFIG)

    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1, vllm_config=VLLM_CONFIG)

    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))
    for unsupported_key in ("do_sample",):
        gen_kwargs.pop(unsupported_key, None)
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )
    if args.verbose:
        for idx, (q, raw_a) in enumerate(zip(sample_questions, answer)):
            display = raw_a
            if isinstance(display, (list, tuple)):
                display = display[0] if display else ""
            if not isinstance(display, str):
                try:
                    display = json.dumps(display, ensure_ascii=False)
                except (TypeError, ValueError):
                    display = str(display)
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer:\n{display}")

    ans = agent.normalize_outputs(sample_questions, answer)

    if args.verbose:
        if gen_kwargs.get("tgps_show", False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "=" * 50)
            total_time = sum(gts or [])
            total_tokens = sum(tls or [])
            if total_time > 0:
                print(
                    f"Total Time: {total_time:.3f} seconds; Total Tokens: {total_tokens}; "
                    f"TGPS: {total_tokens/total_time:.3f} tokens/sec"
                )
            else:
                print("No timing information collected; skipping TGPS aggregate.")

    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace(
        "answers.json", "filtered_answers.json"
    )
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
