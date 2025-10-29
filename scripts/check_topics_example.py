#!/usr/bin/env python3
"""Quick check that topics_example.json entries satisfy QuestioningAgent.filter_questions."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_questions(data: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(data, dict):
        for section, records in data.items():
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict):
                        yield section, record
    elif isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                yield "<list>", record


def _count_tokens(text: str) -> int:
    return len((text or "").split())


def _passes_filter(record: Dict[str, Any]) -> bool:
    required_keys = {"topic", "question", "choices", "answer"}
    if not required_keys.issubset(record.keys()):
        return False
    choices = record.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        return False
    checks = all(
        isinstance(choice, str)
        and len(choice) > 2
        and choice[0].upper() in "ABCD"
        for choice in choices
    )
    if not checks:
        return False
    answer = record.get("answer")
    if not isinstance(answer, str):
        return False
    check_len = _count_tokens(record.get("question", "")) + _count_tokens(answer)
    check_len += sum(_count_tokens(choice) for choice in choices) - 15
    if check_len >= 130:
        return False
    if check_len + _count_tokens(record.get("explanation", "None")) > 1024:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if topics_example.json entries satisfy QuestioningAgent.filter_questions logic."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("assets/topics_example.json"),
        help="Path to the topics example JSON file.",
    )
    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f"File not found: {args.file}")

    data = json.loads(args.file.read_text())

    results: List[Tuple[str, bool, Dict[str, Any]]] = []
    for section, record in _iter_questions(data):
        passed = _passes_filter(record)
        results.append((section, passed, record))

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)

    print(f"Total entries inspected: {total}")
    print(f"Passed filter: {passed}")
    print(f"Failed filter: {total - passed}")

    for idx, (section, ok, record) in enumerate(results, start=1):
        status = "PASS" if ok else "FAIL"
        question = record.get("question", "").strip().replace("\n", " ")
        print(f"[{idx:03d}] {status} :: {section} :: {question[:100]}")


if __name__ == "__main__":
    main()
