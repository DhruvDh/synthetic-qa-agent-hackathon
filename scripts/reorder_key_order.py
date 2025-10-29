#!/usr/bin/env python3
"""Utility to reorder QA JSON records so keys match agent expectations."""
import json
from collections import OrderedDict
from pathlib import Path
import argparse

QUESTION_KEYS = ["topic", "question", "explanation", "answer", "choices"]
ANSWER_KEYS = ["reasoning", "answer"]


def _normalise_question(entry: dict) -> OrderedDict:
    ordered = OrderedDict()
    for key in QUESTION_KEYS:
        if key in entry:
            ordered[key] = entry[key]
    for key, value in entry.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _normalise_answer(entry: dict) -> OrderedDict:
    ordered = OrderedDict()
    for key in ANSWER_KEYS:
        if key in entry:
            ordered[key] = entry[key]
    for key, value in entry.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def process_file(path: Path) -> None:
    data = json.loads(path.read_text())

    if isinstance(data, dict):
        new_data = OrderedDict()
        for section, records in data.items():
            new_data[section] = [_normalise_question(dict(record)) for record in records]
    elif isinstance(data, list):
        if not data:
            new_data = []
        else:
            first_non_null = next((item for item in data if item is not None), None)
            if isinstance(first_non_null, dict):
                exemplar = first_non_null
            else:
                try:
                    exemplar = json.loads(first_non_null)
                except (TypeError, json.JSONDecodeError):
                    exemplar = None
            if exemplar and set(ANSWER_KEYS).issubset(exemplar.keys()):
                formatter = _normalise_answer
            else:
                formatter = _normalise_question

            new_data = []
            for item in data:
                if isinstance(item, dict):
                    new_data.append(formatter(dict(item)))
                else:
                    try:
                        parsed = json.loads(item)
                    except (TypeError, json.JSONDecodeError):
                        new_data.append(item)
                        continue
                    normalised = formatter(dict(parsed))
                    new_data.append(json.dumps(normalised, indent=4))
    else:
        raise TypeError(f"Unsupported JSON root type in {path}: {type(data)!r}")

    path.write_text(json.dumps(new_data, indent=4) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder JSON keys for question/answer records")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    for file_path in args.files:
        process_file(file_path)


if __name__ == "__main__":
    main()
