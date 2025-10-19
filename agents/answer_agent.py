#!/usr/bin/python3

import re
import json

from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

from .answer_model import AAgent
from .config import (
    get_sampling_settings,
    get_answer_prompt_blocks,
)
from utils.harmony_prompt import render_harmony_prompt, extract_final

A_RESPONSE_FORMAT_NAME = "mcq_answer"
A_RESPONSE_FORMAT_SCHEMA = r"""
{"type":"object","additionalProperties":false,"required":["reasoning","answer"],
"properties":{"reasoning":{"type":"string","maxLength":480},
"answer":{"type":"string","enum":["A","B","C","D"]}}}
"""


class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    # WARNING: Public method signatures/return values must remain identical to the
    # initial commit to preserve compatibility with the evaluation pipeline.

    # WARNING: public contract – do not modify signature or return type.
    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1
        prompt_blocks = get_answer_prompt_blocks()
        self.prompt_system = prompt_blocks.system
        self.prompt_developer = prompt_blocks.developer
        self.prompt_user_template = prompt_blocks.user

    # WARNING: public contract – do not modify signature or return type.
    def build_prompt(self, question_data: Dict[str, str | Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""

        choices_formatted = self._format_choices(question_data["choices"])
        user_prompt = self.prompt_user_template.format(
            question=question_data["question"],
            choices=choices_formatted,
        )

        return user_prompt, self.prompt_system

    # WARNING: public contract – do not modify signature or return type.
    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[Dict], int | None, float | None]:
        """Generate answer(s) for the given question(s)"""
        dataset = question_data if isinstance(question_data, list) else [question_data]
        outputs: List[str] = []
        total_tokens: Optional[int] = 0
        total_time: Optional[float] = 0.0

        developer_text = self.prompt_developer
        for entry in dataset:
            prompt_text, sys_prompt = self.build_prompt(entry)
            user_prompt = prompt_text.strip()
            harmony_prompt = render_harmony_prompt(
                developer_instructions=developer_text,
                response_format_name=A_RESPONSE_FORMAT_NAME,
                response_format_json_schema=A_RESPONSE_FORMAT_SCHEMA,
                user_prompt=user_prompt,
                reasoning="medium" if self.select_prompt1 else "high",
                system_extra=sys_prompt,
            )
            resp_text, tokens, elapsed = self.agent.generate_completion_raw(
                harmony_prompt, **kwargs
            )
            final = extract_final(resp_text) or resp_text

            structured = self._reorder_answer_json(final.strip())
            outputs.append(structured)

            if tokens is None:
                total_tokens = None
            elif total_tokens is not None:
                total_tokens += tokens

            if elapsed is None:
                total_time = None
            elif total_time is not None:
                total_time += elapsed

        payload: List[str] | str = (
            outputs if isinstance(question_data, list) else outputs[0]
        )
        return payload, total_tokens, total_time

    # WARNING: public contract – do not modify signature or return type.
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
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)
        pbar.close()
        return answers, tls, gts

    # WARNING: public contract – do not modify signature or return type.
    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the active backend."""
        return self.agent.count_tokens(text)

    # WARNING: public contract – do not modify signature or return type.
    def filter_answers(self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""

        def basic_checks(a1: Dict[str, Any]) -> Optional[Dict[str, str]]:
            if "answer" not in a1:
                return None
            reasoning = str(a1.get("reasoning", "")).strip()
            answer = self._normalize_answer_letter(a1.get("answer", ""))
            if answer not in {"A", "B", "C", "D"}:
                return None
            ordered = OrderedDict()
            ordered["reasoning"] = reasoning
            ordered["answer"] = answer
            return ordered

        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                candidate = basic_checks(a)
                if candidate:
                    filtered_answers.append(candidate)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    candidate = basic_checks(a1)
                    if candidate:
                        filtered_answers.append(candidate)
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

    # WARNING: public contract – do not modify signature or return type.
    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    # WARNING: public contract – do not modify signature or return type.
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
        return "\n".join(formatted)

    @staticmethod
    def _normalize_answer_letter(value: str) -> str:
        """Ensure the answer field is a single uppercase letter A-D."""
        letter = value.strip().upper()
        for opt in ("A", "B", "C", "D"):
            if letter == opt:
                return opt
            if letter.startswith(opt + ")") or letter.startswith(opt + " "):
                return opt
        return ""

    def _reorder_answer_json(self, text: str) -> str:
        """
        Ensure output JSON keys follow the mandated order: reasoning -> answer.
        If parsing fails, return the original text.
        """
        try:
            data = json.loads(text)
            reasoning = " ".join(data.get("reasoning", "").strip().split())
            words = reasoning.split()
            if len(words) > 60:
                reasoning = " ".join(words[:60])

            answer = self._normalize_answer_letter(data.get("answer", ""))
            ordered = OrderedDict()
            ordered["reasoning"] = reasoning
            ordered["answer"] = answer
            return json.dumps(ordered, ensure_ascii=False)
        except json.JSONDecodeError:
            return text


# Example usage
if __name__ == "__main__":
    import json
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt

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

    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1)

    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    gen_kwargs.update(get_sampling_settings("answer"))
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )
    ans = []
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        if args.verbose:
            print(f"\n=== Question {idx + 1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer:\n{a}")
        try:
            a = json.loads(a)
            if all(k in a for k in ["answer", "reasoning"]):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if len(a["answer"]) != 1:
                    a["answer"] = agent.agent.generate_response(
                        option_extractor_prompt(a["answer"], q["choices"])
                    )
                # ++++++++++++++++++++++++++
            else:
                # the dictionary is not as expected. So extract it using the same model: Self-Reflection
                prompt = (
                    "Extract **ONLY** the answer and reasoning while discarding the rest.\n\n"
                    "String:\n"
                    "{}\n\n"
                    "Given Format:\n"
                    "{{\n"
                    '    "answer": "Only the option letter (A, B, C, or D)",\n'
                    '    "reasoning": "..."\n'
                    "}}"
                )
                a = agent.agent.generate_response(
                    prompt.format(json.dumps(a, indent=4))
                )
        except json.JSONDecodeError:
            a = agent.agent.generate_response(auto_json(a))
        ans.append(a)

    if args.verbose:
        if gen_kwargs.get("tgps_show", False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl / gt:.3f} seconds")
            print("\n" + "=" * 50)
            print(
                f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls) / sum(gts):.3f} seconds"
            )

    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace(
        "answers.json", "filtered_answers.json"
    )
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
