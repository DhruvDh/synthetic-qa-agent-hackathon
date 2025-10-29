#!/usr/bin/env python3

import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import yaml

if TYPE_CHECKING:
    from agents.answer_agent import AnsweringAgent
    from agents.question_agent import QuestioningAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the question/answer pipeline with detailed diagnostics."
    )
    parser.add_argument("--num_questions", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--answer_batch_size", type=int, default=128)
    parser.add_argument("--topics_file", type=str, default="assets/topics.json")
    parser.add_argument(
        "--icl_file",
        type=str,
        default="assets/topics_example.json",
        help="Optional ICL sample file; set to '' to disable.",
    )
    parser.add_argument("--question_config", type=str, default="qgen.yaml")
    parser.add_argument("--answer_config", type=str, default="agen.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--skip_answering", action="store_true")
    parser.add_argument(
        "--question_concurrency",
        type=int,
        default=128,
        help="Maximum concurrent question-generation requests.",
    )
    parser.add_argument(
        "--answer_concurrency",
        type=int,
        default=128,
        help="Maximum concurrent answer-generation requests.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_yaml(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_choice(choice: Any, idx: int) -> Tuple[bool, str]:
    if not isinstance(choice, str):
        return False, "choice_not_string"
    text = choice.strip()
    expected_prefix = f"{chr(65 + idx)})"
    if not text:
        return False, "choice_empty"
    if not text[:2].upper().startswith(expected_prefix):
        return False, "choice_prefix_mismatch"
    return True, ""


def validate_question(obj: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    required = ["topic", "question", "explanation", "answer", "choices"]
    missing = [key for key in required if key not in obj]
    if missing:
        return False, f"missing_keys:{','.join(missing)}", {}
    choices = obj["choices"]
    if not isinstance(choices, list):
        return False, "choices_not_list", {}
    if len(choices) != 4:
        return False, f"choices_count_{len(choices)}", {}
    for idx, choice in enumerate(choices):
        ok, reason = normalize_choice(choice, idx)
        if not ok:
            return False, f"{reason}:{idx}", {}
    answer = obj["answer"]
    if not isinstance(answer, str):
        return False, "answer_not_string", {}
    answer_clean = answer.strip().upper()
    if answer_clean not in {"A", "B", "C", "D"}:
        return False, "answer_invalid_value", {}
    question_text = obj["question"]
    explanation_text = obj["explanation"]
    if not isinstance(question_text, str) or not question_text.strip():
        return False, "question_invalid", {}
    if not isinstance(explanation_text, str) or not explanation_text.strip():
        return False, "explanation_invalid", {}
    word_limit = 150
    if len(explanation_text.split()) > word_limit:
        return False, "explanation_too_long", {}
    cleaned = {
        "topic": obj["topic"],
        "question": question_text.strip(),
        "explanation": explanation_text.strip(),
        "answer": answer_clean,
        "choices": [str(choice).strip() for choice in choices],
    }
    return True, "", cleaned


def evaluate_questions(raw_entries: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()

    for idx, entry in enumerate(raw_entries):
        if entry is None:
            reason_counter["none_entry"] += 1
            invalid.append(
                {"index": idx, "reason": "none_entry", "detail": "", "raw": None}
            )
            continue
        if isinstance(entry, dict):
            payload = entry
            raw_text = json.dumps(entry, ensure_ascii=True)
        else:
            raw_text = str(entry)
            if raw_text.strip() == "":
                reason_counter["empty_output"] += 1
                invalid.append(
                    {"index": idx, "reason": "empty_output", "detail": "", "raw": raw_text}
                )
                continue
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                reason_counter["json_decode_error"] += 1
                invalid.append(
                    {
                        "index": idx,
                        "reason": "json_decode_error",
                        "detail": str(exc),
                        "raw": raw_text,
                    }
                )
                continue
        if not isinstance(payload, dict):
            reason_counter["not_dict"] += 1
            invalid.append(
                {"index": idx, "reason": "not_dict", "detail": "", "raw": raw_text}
            )
            continue
        ok, reason, cleaned = validate_question(payload)
        if not ok:
            reason_counter[reason] += 1
            invalid.append(
                {"index": idx, "reason": reason, "detail": "", "raw": raw_text}
            )
            continue
        valid.append({"index": idx, "question": cleaned, "raw": raw_text})

    metrics = {
        "total_generated": len(raw_entries),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "invalid_reasons": dict(reason_counter),
    }
    return valid, invalid, metrics


def ensure_module(module: str, package: str | None = None) -> None:
    try:
        importlib.import_module(module)
        return
    except ImportError:
        pkg = package or module
        print(f"[eval] Attempting to install missing dependency '{pkg}' via pip...")
        cmd = [sys.executable, "-m", "pip", "install", "--user", pkg]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Failed to install package '{pkg}'. Install manually and rerun.\n{detail}"
            )
    importlib.import_module(module)


def run_answer_agent(batch_size: int, questions: List[Dict[str, Any]], agent: "AnsweringAgent", kwargs: Dict[str, Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    if not questions:
        return [], [], []
    payloads = [item["question"] for item in questions]
    answers, token_lengths, gen_times = agent.answer_batches(
        payloads, batch_size=batch_size, **kwargs
    )
    answers_list = answers if isinstance(answers, list) else [answers]
    return answers_list, token_lengths or [], gen_times or []


def evaluate_answers(
    questions: List[Dict[str, Any]], raw_answers: List[Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    correct = 0

    for pos, item in enumerate(questions):
        idx = item["index"]
        question = item["question"]
        raw_answer = raw_answers[pos] if pos < len(raw_answers) else None
        if raw_answer is None:
            reason_counter["missing_answer"] += 1
            invalid.append(
                {"index": idx, "reason": "missing_answer", "detail": "", "raw": None}
            )
            continue
        raw_text = (
            raw_answer
            if isinstance(raw_answer, str)
            else json.dumps(raw_answer, ensure_ascii=True)
        )
        if not isinstance(raw_answer, (str, dict)):
            reason_counter["unsupported_answer_type"] += 1
            invalid.append(
                {"index": idx, "reason": "unsupported_answer_type", "detail": "", "raw": raw_text}
            )
            continue
        if isinstance(raw_answer, str):
            if raw_answer.strip() == "":
                reason_counter["empty_output"] += 1
                invalid.append(
                    {"index": idx, "reason": "empty_output", "detail": "", "raw": raw_answer}
                )
                continue
            try:
                payload = json.loads(raw_answer)
            except json.JSONDecodeError as exc:
                reason_counter["json_decode_error"] += 1
                invalid.append(
                    {
                        "index": idx,
                        "reason": "json_decode_error",
                        "detail": str(exc),
                        "raw": raw_answer,
                    }
                )
                continue
        else:
            payload = raw_answer
        if not isinstance(payload, dict):
            reason_counter["not_dict"] += 1
            invalid.append(
                {"index": idx, "reason": "not_dict", "detail": "", "raw": raw_text}
            )
            continue
        if "answer" not in payload:
            reason_counter["missing_answer_key"] += 1
            invalid.append(
                {"index": idx, "reason": "missing_answer_key", "detail": "", "raw": raw_text}
            )
            continue
        model_answer = payload["answer"]
        if not isinstance(model_answer, str):
            reason_counter["answer_not_string"] += 1
            invalid.append(
                {"index": idx, "reason": "answer_not_string", "detail": "", "raw": raw_text}
            )
            continue
        normalized_answer = model_answer.strip().upper()
        if normalized_answer not in {"A", "B", "C", "D"}:
            reason_counter["answer_invalid_value"] += 1
            invalid.append(
                {"index": idx, "reason": "answer_invalid_value", "detail": "", "raw": raw_text}
            )
            continue
        reasoning_text = payload.get("reasoning", "")
        if reasoning_text and not isinstance(reasoning_text, str):
            reason_counter["reasoning_not_string"] += 1
            invalid.append(
                {"index": idx, "reason": "reasoning_not_string", "detail": "", "raw": raw_text}
            )
            continue
        is_correct = normalized_answer == question["answer"]
        if is_correct:
            correct += 1
        valid.append(
            {
                "index": idx,
                "model_answer": normalized_answer,
                "expected_answer": question["answer"],
                "is_correct": is_correct,
                "reasoning": reasoning_text.strip() if isinstance(reasoning_text, str) else "",
            }
        )

    metrics = {
        "questions_answered": len(questions),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "invalid_reasons": dict(reason_counter),
        "correct_count": correct,
        "accuracy": (correct / len(valid)) if valid else None,
    }
    return valid, invalid, metrics


def collect_timing_metrics(token_lengths: List[Any], generation_times: List[Any]) -> Dict[str, Any]:
    tokens = [t for t in token_lengths if isinstance(t, (int, float))]
    times = [t for t in generation_times if isinstance(t, (int, float))]
    total_tokens = sum(tokens)
    total_time = sum(times)
    return {
        "batch_count": len(token_lengths),
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "tokens_per_sec": (total_tokens / total_time) if total_time else None,
    }


def main() -> None:
    args = parse_args()

    ensure_module("tqdm")
    from agents.question_agent import QuestioningAgent
    from agents.answer_agent import AnsweringAgent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topics_path = Path(args.topics_file)
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")
    with topics_path.open("r") as handle:
        topics = json.load(handle)

    icl_samples = None
    if args.icl_file:
        icl_path = Path(args.icl_file)
        if not icl_path.exists():
            raise FileNotFoundError(f"ICL sample file not found: {icl_path}")
        icl_samples = QuestioningAgent.load_icl_samples(str(icl_path))

    question_concurrency = max(1, min(args.question_concurrency, 128))
    answer_concurrency = max(1, min(args.answer_concurrency, 128))

    question_kwargs = {"tgps_show": True}
    question_kwargs.update(load_yaml(args.question_config))
    question_kwargs["concurrency"] = question_concurrency
    question_kwargs.pop("do_sample", None)

    answer_kwargs = {"tgps_show": True}
    answer_kwargs.update(load_yaml(args.answer_config))
    answer_kwargs["concurrency"] = answer_concurrency
    answer_kwargs.pop("do_sample", None)

    q_agent = QuestioningAgent()
    questions_raw, q_token_lengths, q_generation_times = q_agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=bool(icl_samples),
        inc_samples=icl_samples,
        **question_kwargs,
    )

    normalized_questions = []
    for item in questions_raw:
        if isinstance(item, tuple):
            item = item[0] if item else ""
        normalized_questions.append(item if isinstance(item, (dict, list)) else str(item))

    valid_questions, invalid_questions, question_metrics = evaluate_questions(
        normalized_questions
    )
    question_timing = collect_timing_metrics(q_token_lengths, q_generation_times)

    raw_questions_path = output_dir / "questions_raw.json"
    ensure_dir(raw_questions_path)
    with raw_questions_path.open("w") as handle:
        json.dump(
            [{"index": idx, "raw": raw} for idx, raw in enumerate(normalized_questions)],
            handle,
            indent=2,
        )

    valid_questions_path = output_dir / "questions_valid.json"
    ensure_dir(valid_questions_path)
    with valid_questions_path.open("w") as handle:
        json.dump([item["question"] | {"index": item["index"]} for item in valid_questions], handle, indent=2)

    invalid_questions_path = output_dir / "questions_invalid.json"
    ensure_dir(invalid_questions_path)
    with invalid_questions_path.open("w") as handle:
        json.dump(invalid_questions, handle, indent=2)

    answers_raw: List[Any] = []
    answer_token_lengths: List[int] = []
    answer_generation_times: List[float] = []
    valid_answers: List[Dict[str, Any]] = []
    invalid_answers: List[Dict[str, Any]] = []
    answer_metrics: Dict[str, Any] = {
        "questions_answered": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "invalid_reasons": {},
        "correct_count": 0,
        "accuracy": None,
    }
    answer_timing = {}

    if not args.skip_answering:
        ans_agent = AnsweringAgent()
        answer_outputs, answer_token_lengths, answer_generation_times = run_answer_agent(
            args.answer_batch_size, valid_questions, ans_agent, answer_kwargs
        )
        answers_raw = answer_outputs
        valid_answers, invalid_answers, answer_metrics = evaluate_answers(
            valid_questions, answer_outputs
        )
        answer_timing = collect_timing_metrics(
            answer_token_lengths, answer_generation_times
        )

        raw_answers_path = output_dir / "answers_raw.json"
        ensure_dir(raw_answers_path)
        with raw_answers_path.open("w") as handle:
            json.dump(
                [
                    {
                        "index": valid_questions[pos]["index"],
                        "raw": ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=True),
                    }
                    for pos, ans in enumerate(answers_raw)
                ],
                handle,
                indent=2,
            )

        valid_answers_path = output_dir / "answers_valid.json"
        ensure_dir(valid_answers_path)
        with valid_answers_path.open("w") as handle:
            json.dump(valid_answers, handle, indent=2)

        invalid_answers_path = output_dir / "answers_invalid.json"
        ensure_dir(invalid_answers_path)
        with invalid_answers_path.open("w") as handle:
            json.dump(invalid_answers, handle, indent=2)

    metrics = {
        "question_metrics": question_metrics,
        "question_timing": question_timing,
        "answer_metrics": answer_metrics,
        "answer_timing": answer_timing,
        "config": {
            "num_questions": args.num_questions,
            "batch_size": args.batch_size,
            "answer_batch_size": args.answer_batch_size,
            "topics_file": str(topics_path),
            "icl_file": str(args.icl_file) if args.icl_file else "",
            "skip_answering": args.skip_answering,
            "question_concurrency": question_concurrency,
            "answer_concurrency": answer_concurrency,
        },
    }

    metrics_path = output_dir / "metrics.json"
    ensure_dir(metrics_path)
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print("=== Question Generation ===")
    print(json.dumps({**question_metrics, **question_timing}, indent=2))
    if invalid_questions and args.verbose:
        print("Sample invalid question reasons:")
        for item in invalid_questions[:5]:
            print(f"- idx {item['index']}: {item['reason']}")
    if not args.skip_answering:
        print("\n=== Answering ===")
        print(json.dumps({**answer_metrics, **answer_timing}, indent=2))
        if invalid_answers and args.verbose:
            print("Sample invalid answers:")
            for item in invalid_answers[:5]:
                print(f"- idx {item['index']}: {item['reason']}")


if __name__ == "__main__":
    main()
