#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent
from utils.vllm_utils import VLLMConfig, ensure_vllm_server_running

import random
import json


class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, vllm_config: VLLMConfig | None = None, **kwargs):
        self.agent = QAgent(config=vllm_config, **kwargs)

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
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            "}}"
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += (
                fmt.format(
                    topic, topic.split("/")[-1], question, *choices, answer, explanation
                )
                + "\n\n"
            )
        return sample_str.strip()

    def build_prompt(
        self,
        topic: str,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""

        sys_prompt = (
            "You are ChatGPT, a large language model trained by OpenAI. You have been fine-tuned to be a winning competetive logical puzzle creator.\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: 2025-10-28\n\n"
            "Reasoning: low\n\n"
            "# Valid channels: analysis, final. Channel must be included for every message.\n"
            "<|DEVELOPER|>\n"
            "# Instructions\n"
            "You are competing in a puzzle answering tournament as an puzzle cearting large language model. Points are awarded if (a) you generate correct questions that strictly adhere to response format restrictions and (b) your opponent is unable to answer your puzzle correctly. You have been trained to generate tricky, hard to understand and even harder to solve puzzles in the domain of:\n"
            "1. Seating Arrangements (Circular and Linear). Don’t include any numeric style seating arrangements questions, e.g., how many permutations such arrangements possible, etc.\n"
            "2. Blood relations and family trees\n\n"
            "Points are not awarded if the answer to your own question is not correct or your explanation is not sufficient.\n\n"
            "# Response Formats\n\n"
            "## question_json\n"
            "{\n"
            '  "type": "object",\n'
            '  "additionalProperties": false,\n'
            '  "properties": {\n'
            '    "topic": { "type": "string" },\n'
            '    "question": {\n'
            '      "type": "string",\n'
            '      "description": "Prompt for the puzzle."\n'
            "    },\n"
            '    "explanation": {\n'
            '      "type": "string",\n'
            '      "description": "<=150 words; key lines that make the single option correct."\n'
            "    },\n"
            '    "answer": { "type": "string", "enum": ["A", "B", "C", "D"] },\n'
            '    "choices": {\n'
            '      "type": "array",\n'
            '      "minItems": 4,\n'
            '      "maxItems": 4,\n'
            '      "items": {\n'
            '        "type": "string",\n'
            '        "description": "Must begin with \'A) \', \'B) \', \'C) \', \'D) \' and be mutually exclusive."\n'
            "      }\n"
            "    }\n"
            "  },\n"
            '  "required": ["topic", "question", "explanation", "answer", "choices"]\n'
            "}\n"
        )
        tmpl = (
            "Respond with one **winning** puzzle JSON (using the question_json schema) in this domain: {0}\n"
            "- Provide exactly four options labeled 'A) ...', 'B) ...', 'C) ...', 'D) ...'\n"
            "- The only correct answer must be: {2} (others: {3})\n"
            "- Keep the explanation <= 90 words\n"
            "{5}\n\n"
            "Return using response format: question_json with the **exact JSON key order**: \"topic\", \"question\", \"explanation\", \"answer\", \"choices\".\n"
            'Set "topic" = "{7}" and place the correct answer at "{8}".'
        )
        # Remove model's preferential bias for options
        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join(
            [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
        )

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
        prompt = tmpl.format(
            topic,
            topic,
            correct_option,
            distractors,
            correct_option,
            inc_samples_ex,
            topic,
            topic.split("/")[-1],
            correct_option,
            correct_option,
        )

        return prompt, sys_prompt

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                inc_topic_samples = inc_samples[t[1]] if inc_samples else None
                p, sp = self.build_prompt(f"{t[0]}/{t[1]}", wicl, inc_topic_samples)
                prompt.append(p)
        else:
            inc_topic_samples = inc_samples[topic[1]] if inc_samples else None
            prompt, sp = self.build_prompt(
                f"{topic[0]}/{topic[1]}", wicl, inc_topic_samples
            )

        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

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

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
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
                batch_topics, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        # for last batch with less than batch_size
        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size) :]
            batch_questions = self.generate_question(
                batch_topics, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(
        self, questions: List[str | Dict[str, str | Any]]
    ) -> List[Dict[str, str | Any]]:
        def basic_checks(q2: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["topic", "question", "choices", "answer"]
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(
                    isinstance(choice, str)
                    and len(choice) > 2
                    and choice[0].upper() in "ABCD"
                    for choice in q2["choices"]
                )
                if (
                    isinstance(q2["choices"], list)
                    and len(q2["choices"]) == 4
                    and checks
                ):
                    # check answer format
                    # Check token length
                    check_len = sum(
                        self.count_tokens_q(q2[k]) for k in ["question", "answer"]
                    )
                    check_len += (
                        sum(self.count_tokens_q(choice) for choice in q2["choices"])
                        - 15
                    )
                    if check_len < 130:
                        if (
                            check_len
                            + self.count_tokens_q(q2.get("explanation", "None"))
                            <= 1024
                        ):
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2["answer"], str):
                                return True
            return False

        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

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
    import yaml

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

    VLLM_CONFIG = VLLMConfig()
    ensure_vllm_server_running(VLLM_CONFIG)

    agent = QuestioningAgent(vllm_config=VLLM_CONFIG)
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))
    for unsupported_key in ("do_sample",):
        gen_kwargs.pop(unsupported_key, None)

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
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
            total_time = sum(gts or [])
            total_tokens = sum(tls or [])
            if total_time > 0:
                print(
                    f"Total Time Taken: {total_time:.3f} seconds; Total Tokens: {total_tokens}; "
                    f"TGPS: {total_tokens/total_time:.3f} tokens/sec\n\n"
                )
            else:
                print("No timing information collected; skipping TGPS aggregate.\n\n")
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
                "Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n"
                "Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n"
                "String:\n"
                "{}\n\n"
                "Given Format:\n"
                "{{\n"
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                "}}"
            )
            q = agent.agent.generate_response(
                prompt.format(q),
                "You are an expert JSON extractor.",
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
            )
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
