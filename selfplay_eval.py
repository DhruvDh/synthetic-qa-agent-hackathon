#!/usr/bin/env python3
"""
Offline self-play evaluation and manual review helpers.

Usage:
    python selfplay_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class QARecord:
    topic: str
    question: str
    correct_answer: str
    model_answer: Optional[str]
    reasoning: str
    duplicate_choices: bool
    long_reasoning: bool
    is_correct: Optional[bool]
    invalid_answer: bool


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r") as fp:
        return json.load(fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Q/A self-play outputs and export review artifacts."
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing questions/answers artifacts.",
    )
    return parser.parse_args()


def to_dict_list(items: Iterable[Any]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for entry in items:
        if isinstance(entry, dict):
            parsed.append(entry)
        elif isinstance(entry, str):
            try:
                parsed.append(json.loads(entry))
            except json.JSONDecodeError:
                continue
    return parsed


def choice_has_duplicates(choices: Iterable[str]) -> bool:
    cleaned: List[str] = []
    for choice in choices:
        if not isinstance(choice, str):
            continue
        parts = choice.split(")", 1)
        cleaned_text = parts[1] if len(parts) == 2 else parts[0]
        cleaned.append(cleaned_text.strip().lower())
    return len(cleaned) != len(set(cleaned))


def word_count(text: str) -> int:
    return len(text.split())


def build_records(
    questions: List[Dict[str, Any]], answers: List[Any]
) -> Tuple[List[QARecord], Counter]:
    records: List[QARecord] = []
    letter_counter: Counter = Counter()
    for idx, question in enumerate(questions):
        answer_entry = answers[idx] if idx < len(answers) else None
        answer_dict: Optional[Dict[str, Any]]
        if isinstance(answer_entry, dict):
            answer_dict = answer_entry
        elif isinstance(answer_entry, str):
            try:
                answer_dict = json.loads(answer_entry)
            except json.JSONDecodeError:
                answer_dict = None
        else:
            answer_dict = None

        model_answer = None
        reasoning = ""
        is_correct = None
        invalid_answer = answer_dict is None
        if answer_dict is not None:
            model_answer = str(answer_dict.get("answer", "")).strip().upper() or None
            reasoning = str(answer_dict.get("reasoning", "")).strip()
            if model_answer:
                letter_counter[model_answer] += 1

        correct_answer = str(question.get("answer", "")).strip().upper()
        if model_answer and correct_answer:
            is_correct = model_answer == correct_answer

        duplicate_choices = choice_has_duplicates(question.get("choices", []))
        long_reasoning = bool(reasoning) and word_count(reasoning) > 100

        records.append(
            QARecord(
                topic=str(question.get("topic", "")),
                question=str(question.get("question", "")),
                correct_answer=correct_answer,
                model_answer=model_answer,
                reasoning=reasoning,
                duplicate_choices=duplicate_choices,
                long_reasoning=long_reasoning,
                is_correct=is_correct,
                invalid_answer=invalid_answer,
            )
        )
    return records, letter_counter


def compute_metrics(
    raw_questions: List[Any],
    filtered_questions: List[Dict[str, Any]],
    raw_answers: List[Any],
    filtered_answers: List[Any],
    records: List[QARecord],
    letter_counter: Counter,
) -> Dict[str, Any]:
    total_raw_q = len(raw_questions)
    total_filtered_q = len(filtered_questions)
    total_raw_a = len(raw_answers)
    valid_answers = [a for a in filtered_answers if isinstance(a, dict)]

    considered_records = records[: len(filtered_answers)]
    valid_records = [rec for rec in considered_records if rec.model_answer]
    correct = sum(1 for rec in valid_records if rec.is_correct)

    duplicate_rate = (
        sum(1 for rec in records if rec.duplicate_choices) / total_filtered_q
        if total_filtered_q
        else 0.0
    )
    long_reasoning_rate = (
        sum(1 for rec in valid_records if rec.long_reasoning) / len(valid_records)
        if valid_records
        else 0.0
    )

    metrics = {
        "n_questions_raw": total_raw_q,
        "n_questions_filtered": total_filtered_q,
        "n_answers_raw": total_raw_a,
        "n_answers_valid": len(valid_answers),
        "a_agent_accuracy": round(correct / len(valid_records), 4)
        if valid_records
        else 0.0,
        "q_agent_score": round((total_filtered_q / total_raw_q) * 100, 2)
        if total_raw_q
        else 0.0,
        "question_valid_rate": round(total_filtered_q / total_raw_q, 4)
        if total_raw_q
        else 0.0,
        "answer_valid_rate": round(len(valid_answers) / total_raw_a, 4)
        if total_raw_a
        else 0.0,
        "duplicate_choices_rate": round(duplicate_rate, 4),
        "long_reasoning_rate": round(long_reasoning_rate, 4),
        "answer_letter_distribution": dict(letter_counter),
    }
    return metrics


def write_summary(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(metrics, fp, indent=4)


def write_review_csv(path: Path, records: List[QARecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "index",
                "topic",
                "question",
                "correct_answer",
                "model_answer",
                "is_correct",
                "duplicate_choices",
                "long_reasoning",
                "invalid_answer",
                "reasoning",
            ],
        )
        writer.writeheader()
        for idx, rec in enumerate(records):
            writer.writerow(
                {
                    "index": idx,
                    "topic": rec.topic,
                    "question": rec.question,
                    "correct_answer": rec.correct_answer,
                    "model_answer": rec.model_answer or "",
                    "is_correct": rec.is_correct,
                    "duplicate_choices": rec.duplicate_choices,
                    "long_reasoning": rec.long_reasoning,
                    "invalid_answer": rec.invalid_answer,
                    "reasoning": rec.reasoning,
                }
            )


def write_review_html(path: Path, records: List[QARecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, rec in enumerate(records):
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{rec.topic}</td>"
            f"<td>{rec.question}</td>"
            f"<td>{rec.correct_answer}</td>"
            f"<td>{rec.model_answer or ''}</td>"
            f"<td>{rec.is_correct}</td>"
            f"<td>{rec.duplicate_choices}</td>"
            f"<td>{rec.long_reasoning}</td>"
            f"<td>{rec.invalid_answer}</td>"
            f"<td>{rec.reasoning}</td>"
            "</tr>"
        )
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Self-Play Review</title>
<style>
body { font-family: Arial, sans-serif; margin: 1.5rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.5rem; vertical-align: top; }
th { background-color: #f5f5f5; }
tr:nth-child(even) { background-color: #fafafa; }
</style>
</head>
<body>
<h1>Self-Play Review</h1>
<table>
<thead>
<tr>
<th>#</th>
<th>Topic</th>
<th>Question</th>
<th>Correct</th>
<th>Model</th>
<th>Match</th>
<th>Dup Choices</th>
<th>Reasoning &gt;100 words</th>
<th>Invalid Answer</th>
<th>Reasoning</th>
</tr>
</thead>
<tbody>
"""
    html += "\n".join(rows)
    html += """
</tbody>
</table>
</body>
</html>
"""
    with path.open("w") as fp:
        fp.write(html)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)

    questions_raw_path = outputs_dir / "questions.json"
    questions_filtered_path = outputs_dir / "filtered_questions.json"
    answers_raw_path = outputs_dir / "answers.json"
    answers_filtered_path = outputs_dir / "filtered_answers.json"

    raw_questions = load_json(questions_raw_path, default=[])
    filtered_questions = load_json(questions_filtered_path, default=[])
    raw_answers = load_json(answers_raw_path, default=[])
    filtered_answers = load_json(answers_filtered_path, default=[])

    parsed_filtered_questions = to_dict_list(filtered_questions)
    parsed_filtered_answers = [
        entry if isinstance(entry, (dict, type(None))) else entry
        for entry in filtered_answers
    ]

    records, letter_counter = build_records(
        parsed_filtered_questions, parsed_filtered_answers
    )
    metrics = compute_metrics(
        raw_questions,
        parsed_filtered_questions,
        raw_answers,
        filtered_answers,
        records,
        letter_counter,
    )

    summary_path = outputs_dir / "selfplay_summary.json"
    csv_path = outputs_dir / "selfplay_review.csv"
    html_path = outputs_dir / "selfplay_review.html"

    write_summary(summary_path, metrics)
    write_review_csv(csv_path, records)
    write_review_html(html_path, records)

    print("=== Self-Play Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSummary written to: {summary_path}")
    print(f"Review CSV: {csv_path}")
    print(f"Review HTML: {html_path}")


if __name__ == "__main__":
    main()
