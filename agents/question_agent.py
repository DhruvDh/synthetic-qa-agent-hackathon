#!/usr/bin/python3

import random
import json

from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .config import (
    get_sampling_settings,
    get_question_prompt_blocks,
)
from .question_model import QAgent
from utils.harmony_prompt import render_harmony_prompt, extract_final


class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    # WARNING: The method signatures and return types in this class are part of the
    # official submission contract. Do not change them; they must match the initial
    # repository version so downstream tooling continues to work.

    # WARNING: public contract – do not modify signature or return type.
    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)
        prompt_blocks = get_question_prompt_blocks()
        self.prompt_system = prompt_blocks.system
        self.prompt_developer = prompt_blocks.developer
        self.prompt_user_template = prompt_blocks.user

    # WARNING: public contract – do not modify signature or return type.
    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            "EXAMPLE: {}\n"
            "{{\n"
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "explanation": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}"\n'
            "}}"
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = self._format_choices_array(sample.get("choices", []))
            answer = self._normalize_answer_letter(sample.get("answer", ""))
            explanation = sample.get("explanation", "")
            choice_bodies = []
            for choice in choices:
                parts = choice.split(")", 1)
                body = parts[1].strip() if len(parts) == 2 else choice.strip()
                choice_bodies.append(body)
            sample_str += (
                fmt.format(
                    topic,
                    topic.split("/")[-1],
                    question,
                    explanation,
                    *choice_bodies,
                    answer,
                )
                + "\n\n"
            )
        return sample_str.strip()

    # WARNING: public contract – do not modify signature or return type.
    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""

        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join(
            opt for opt in ["A", "B", "C", "D"] if opt != correct_option
        )

        samples_block = ""
        if wicl and inc_samples:
            examples = self.build_inc_samples(inc_samples, topic)
            if examples:
                samples_block = f"\nREFERENCE EXAMPLES:\n{examples}"

        user_prompt = self.prompt_user_template.format(
            topic=topic,
            answer_letter=correct_option,
            distractors=distractors,
            samples_section=samples_block,
        )

        return user_prompt, self.prompt_system

    # WARNING: public contract – do not modify signature or return type.
    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate a question prompt for the LLM"""
        prompts: List[Tuple[str, str]] = []
        if isinstance(topic, list):
            for t in topic:
                prompt_text, sys_prompt = self.build_prompt(
                    f"{t[0]}/{t[1]}",
                    wadvsys,
                    wicl,
                    inc_samples.get(t[1]) if inc_samples else None,
                )
                prompts.append((prompt_text, sys_prompt))
        else:
            prompt_text, sys_prompt = self.build_prompt(
                f"{topic[0]}/{topic[1]}",
                wadvsys,
                wicl,
                inc_samples.get(topic[1]) if inc_samples else None,
            )
            prompts.append((prompt_text, sys_prompt))

        outputs: List[str] = []
        total_tokens: Optional[int] = 0
        total_time: Optional[float] = 0.0

        developer_text = self.prompt_developer
        for prompt_text, sys_prompt in prompts:
            combined_user = prompt_text.strip()
            harmony_prompt = render_harmony_prompt(
                system_prompt=sys_prompt,
                developer_prompt=developer_text,
                user_prompt=combined_user,
            )
            resp_text, tokens, elapsed = self.agent.generate_completion_raw(
                harmony_prompt, **gen_kwargs
            )
            final = extract_final(resp_text) or resp_text
            structured = self._reorder_question_json(final.strip())
            outputs.append(structured)

            if tokens is None:
                total_tokens = None
            elif total_tokens is not None:
                total_tokens += tokens

            if elapsed is None:
                total_time = None
            elif total_time is not None:
                total_time += elapsed

        payload: List[str] | str = outputs if len(outputs) > 1 else outputs[0]
        return payload, total_tokens, total_time

    # WARNING: public contract – do not modify signature or return type.
    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            payload = batch_questions[0]
            if isinstance(payload, list):
                questions.extend(payload)
            else:
                questions.append(payload)
            tls.append(batch_questions[1])
            gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    # WARNING: public contract – do not modify signature or return type.
    def count_tokens_q(self, text: str) -> int:
        """Count tokens for the provided text using the active inference backend."""
        return self.agent.count_tokens(text)

    # WARNING: public contract – do not modify signature or return type.
    def filter_questions(
        self, questions: List[str | Dict[str, str | Any]]
    ) -> List[Dict[str, str | Any]]:
        def basic_checks(q2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            required_keys = {"topic", "question", "explanation", "choices", "answer"}
            if not required_keys.issubset(q2.keys()):
                return None

            topic = str(q2.get("topic", "")).strip()
            question = str(q2.get("question", "")).strip()
            explanation = str(q2.get("explanation", "")).strip()
            if not topic or not question:
                return None

            raw_choices = q2.get("choices", [])
            if not isinstance(raw_choices, list) or len(raw_choices) != 4:
                return None
            choices = self._format_choices_array(raw_choices)
            bodies = [
                c.split(")", 1)[1].strip().lower() if ")" in c else c.strip().lower()
                for c in choices
            ]
            if len(set(bodies)) != 4:
                return None

            answer = self._normalize_answer_letter(q2.get("answer", ""))
            if answer not in {"A", "B", "C", "D"}:
                return None

            ordered = OrderedDict()
            ordered["topic"] = topic
            ordered["question"] = question
            ordered["explanation"] = explanation
            ordered["choices"] = choices
            ordered["answer"] = answer
            return ordered

        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                candidate = basic_checks(q)
                if candidate:
                    correct_format_question.append(candidate)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    candidate = basic_checks(q1)
                    if candidate:
                        correct_format_question.append(candidate)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()

    @staticmethod
    def _normalize_answer_letter(value: str) -> str:
        """Normalize answer strings to a single uppercase letter A-D."""
        letter = value.strip().upper()
        for opt in ("A", "B", "C", "D"):
            if letter == opt:
                return opt
            if letter.startswith(opt + ")") or letter.startswith(opt + " "):
                return opt
        return ""

    def _format_choices_array(self, choices: List[str]) -> List[str]:
        """Ensure there are four choices labeled A)-D) with trimmed bodies."""
        formatted: List[str] = []
        for idx in range(4):
            label = chr(65 + idx)
            raw = choices[idx] if idx < len(choices) else ""
            text = raw.strip()
            body = text
            if text.startswith(f"{label})"):
                body = text[len(f"{label})") :].strip()
            elif text and text[0].upper() == label and text[1:2] in {")", "."}:
                body = text[2:].strip()
            elif text.startswith(f"{label} "):
                body = text[len(f"{label} ") :].strip()
            else:
                body = text.strip()
            formatted.append(f"{label}) {body}".strip())
        return formatted

    def _reorder_question_json(self, text: str) -> str:
        """
        Ensure output JSON keys follow mandated order:
        topic -> question -> explanation -> choices -> answer.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text

        topic = data.get("topic", "").strip()
        question = data.get("question", "").strip()
        explanation = " ".join(data.get("explanation", "").strip().split())
        words = explanation.split()
        if len(words) > 90:
            explanation = " ".join(words[:90])
        choices = self._format_choices_array(data.get("choices", []))
        answer = self._normalize_answer_letter(data.get("answer", ""))

        ordered = OrderedDict()
        ordered["topic"] = topic
        ordered["question"] = question
        ordered["explanation"] = explanation
        ordered["choices"] = choices
        ordered["answer"] = answer
        return json.dumps(ordered, ensure_ascii=False)

    # WARNING: public contract – do not modify signature or return type.
    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    # WARNING: public contract – do not modify signature or return type.
    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError(
                "Topics must be a dictionary with topic names as keys and lists of subtopics as values."
            )

        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")

        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    # WARNING: public contract – do not modify signature or return type.
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples


# Example usage
if __name__ == "__main__":
    import argparse

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    gen_kwargs.update(get_sampling_settings("question"))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls) / sum(gts):.3f} seconds\n\n"
            )
        print("\n" + "+" * 50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # TODO: IMPROVE THE FOLLOWING
            prompt = (
                "Extract **ONLY** the topic, question, explanation, choices, and answer while discarding the rest.\n"
                "Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n"
                "String:\n"
                "{}\n\n"
                "Given Format:\n"
                "{{\n"
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "explanation": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)"\n'
                "}}"
            )
            raw_resp, _, _ = agent.agent.generate_response(
                prompt.format(q),
                "You are an expert JSON extractor.",
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
            )
            q = raw_resp[0] if isinstance(raw_resp, list) else raw_resp
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
