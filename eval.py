#!/usr/bin/env python3

import argparse
import importlib
import json
import subprocess
import sys
import threading
import time
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


def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            filtered[key] = value
    return filtered


class ModeStats:
    __slots__ = (
        "total_calls",
        "finish_counts",
        "empty_reason_counts",
        "empty_records",
        "error_counts",
        "error_samples",
        "prompt_example",
        "completion_tokens",
        "prompt_tokens",
        "total_latency",
        "max_latency",
        "min_latency",
    )

    def __init__(self) -> None:
        self.total_calls = 0
        self.finish_counts: Counter[str] = Counter()
        self.empty_reason_counts: Counter[str] = Counter()
        self.empty_records: List[Dict[str, Any]] = []
        self.error_counts: Counter[str] = Counter()
        self.error_samples: List[Dict[str, Any]] = []
        self.prompt_example: str | None = None
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_latency = 0.0
        self.max_latency = 0.0
        self.min_latency: float | None = None

    def record_latency(self, elapsed: float) -> None:
        self.total_latency += elapsed
        if elapsed > self.max_latency:
            self.max_latency = elapsed
        if self.min_latency is None or elapsed < self.min_latency:
            self.min_latency = elapsed

    def summary(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_latency / self.total_calls if self.total_calls else None
        )
        return {
            "total_calls": self.total_calls,
            "finish_reasons": dict(self.finish_counts),
            "empty_reason_counts": dict(self.empty_reason_counts),
            "errors": dict(self.error_counts),
            "error_samples": self.error_samples,
            "prompt_example": self.prompt_example[:400]
            if isinstance(self.prompt_example, str)
            else None,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "avg_latency_sec": avg_latency,
            "max_latency_sec": self.max_latency if self.total_calls else None,
            "min_latency_sec": self.min_latency,
        }


class CompletionTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode_stats: Dict[str, ModeStats] = {
            "question": ModeStats(),
            "answer": ModeStats(),
            "other": ModeStats(),
        }
        self.total_calls = 0
        self.total_errors = 0
        self.total_latency = 0.0

    def _detect_mode(self, messages: List[Dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content", "") or ""
            lowered = content.lower()
            if "puzzle creator" in lowered:
                return "question"
            if "puzzle solver" in lowered:
                return "answer"
        return "other"

    def _append_empty_record(
        self,
        stats: ModeStats,
        kwargs: Dict[str, Any],
        response: Dict[str, Any],
        reason: str,
    ) -> None:
        stats.empty_reason_counts[reason] += 1
        stats.empty_records.append(
            {
                "reason": reason,
                "kwargs": _filter_kwargs(kwargs),
                "response": response,
            }
        )

    def record_success(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        response: Dict[str, Any],
        elapsed: float,
    ) -> None:
        mode = self._detect_mode(messages)
        stats = self.mode_stats[mode]
        with self.lock:
            stats.total_calls += 1
            self.total_calls += 1
            stats.record_latency(elapsed)
            self.total_latency += elapsed

            if stats.prompt_example is None:
                user_content = next(
                    (
                        msg.get("content", "")
                        for msg in messages
                        if msg.get("role") == "user"
                    ),
                    "",
                )
                stats.prompt_example = user_content

            choices = response.get("choices") or []
            if not choices:
                self._append_empty_record(stats, kwargs, response, "no_choices")
                return

            logged_empty = False
            for choice in choices:
                finish_reason = (choice.get("finish_reason") or "unknown").lower()
                stats.finish_counts[finish_reason] += 1

                content = ""
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content", "") or ""
                elif isinstance(message, str):
                    content = message
                if not isinstance(content, str):
                    content = str(content)

                if not content.strip() and not logged_empty:
                    self._append_empty_record(
                        stats, kwargs, response, "empty_content"
                    )
                    logged_empty = True

            usage = response.get("usage") or {}
            stats.completion_tokens += usage.get("completion_tokens", 0)
            stats.prompt_tokens += usage.get("prompt_tokens", 0)

    def record_error(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        exc: Exception,
        elapsed: float,
    ) -> None:
        mode = self._detect_mode(messages)
        stats = self.mode_stats[mode]
        error_category = classify_error(exc)
        with self.lock:
            stats.total_calls += 1
            self.total_calls += 1
            stats.error_counts[error_category] += 1
            if len(stats.error_samples) < 50:
                stats.error_samples.append(
                    {
                        "category": error_category,
                        "error": str(exc),
                        "kwargs": _filter_kwargs(kwargs),
                    }
                )
            stats.record_latency(elapsed)
            self.total_errors += 1
            self.total_latency += elapsed

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_calls": self.total_calls,
                "total_errors": self.total_errors,
                "total_latency_sec": self.total_latency,
                "modes": {
                    mode: stats.summary() for mode, stats in self.mode_stats.items()
                },
            }

    def get_empty_records(self, mode: str) -> List[Dict[str, Any]]:
        return list(self.mode_stats[mode].empty_records)

    def get_prompt_example(self, mode: str) -> str | None:
        return self.mode_stats[mode].prompt_example


def classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "connection" in message or "network" in message:
        return "connection"
    if "failed to reach" in message or "503" in message or "502" in message:
        return "backend"
    return "other"


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


def summarize_self_reflection(stats: Dict[str, int]) -> Dict[str, Any]:
    attempts = stats.get("attempts", 0)
    successes = stats.get("successes", 0)
    return {
        "attempts": attempts,
        "successes": successes,
        "success_rate": (successes / attempts) if attempts else None,
        "failures": attempts - successes,
    }


def main() -> None:
    args = parse_args()

    ensure_module("tqdm")
    from agents.question_agent import QuestioningAgent
    from agents.answer_agent import AnsweringAgent
    from agents import question_model as qm
    from agents import answer_model as am
    from utils import vllm_utils

    tracker = CompletionTracker()
    original_chat_completion = vllm_utils.chat_completion
    original_qm_chat = qm.chat_completion
    original_am_chat = am.chat_completion

    def tracking_chat_completion(
        messages: List[Dict[str, Any]],
        config: Any = None,
        timeout: float = 60.0,
        **params: Any,
    ) -> Dict[str, Any]:
        params_copy = dict(params)
        kwargs_for_log = {**params_copy, "timeout": timeout}
        start_time = time.time()
        try:
            response = original_chat_completion(
                messages, config=config, timeout=timeout, **params
            )
        except Exception as exc:  # noqa: BLE001
            tracker.record_error(messages, kwargs_for_log, exc, time.time() - start_time)
            raise
        tracker.record_success(
            messages, kwargs_for_log, response, time.time() - start_time
        )
        return response

    vllm_utils.chat_completion = tracking_chat_completion
    qm.chat_completion = tracking_chat_completion  # type: ignore[attr-defined]
    am.chat_completion = tracking_chat_completion  # type: ignore[attr-defined]
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

    normalized_questions = q_agent.normalize_outputs(questions_raw)
    q_self_stats = summarize_self_reflection(q_agent.get_self_reflection_stats())

    question_truncations = None
    question_max_tokens = question_kwargs.get("max_new_tokens")
    if question_max_tokens and hasattr(q_agent, "agent") and hasattr(
        q_agent.agent, "tokenizer"
    ):
        tokenizer = q_agent.agent.tokenizer
        question_truncations = 0
        for raw in normalized_questions:
            if not raw:
                continue
            token_count = len(tokenizer.encode(raw, add_special_tokens=False))
            if token_count >= question_max_tokens:
                question_truncations += 1

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

    question_empty_records = tracker.get_empty_records("question")
    question_empty_path = output_dir / "question_empty_debug.json"
    ensure_dir(question_empty_path)
    with question_empty_path.open("w") as handle:
        json.dump(
            {
                "prompt_example": tracker.get_prompt_example("question"),
                "event_count": len(question_empty_records),
                "events": question_empty_records,
            },
            handle,
            indent=2,
        )

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
    answer_truncations = None
    answer_empty_records: List[Dict[str, Any]] = []
    answer_self_reflection_attempts_total = 0
    answer_self_reflection_success_total = 0
    answer_self_reflection_stats = {"attempts": 0, "successes": 0}

    if not args.skip_answering:
        ans_agent = AnsweringAgent()
        answer_outputs, answer_token_lengths, answer_generation_times = run_answer_agent(
            args.answer_batch_size, valid_questions, ans_agent, answer_kwargs
        )
        normalized_answers = ans_agent.normalize_outputs(valid_questions, answer_outputs)
        stats_answer = summarize_self_reflection(ans_agent.get_self_reflection_stats())
        answer_self_reflection_attempts_total = stats_answer["attempts"]
        answer_self_reflection_success_total = stats_answer["successes"]
        answer_self_reflection_stats = stats_answer
        answers_raw = normalized_answers
        valid_answers, invalid_answers, answer_metrics = evaluate_answers(
            valid_questions, normalized_answers
        )
        answer_timing = collect_timing_metrics(
            answer_token_lengths, answer_generation_times
        )

        answer_max_tokens = answer_kwargs.get("max_new_tokens")
        if answer_max_tokens and hasattr(ans_agent, "agent") and hasattr(
            ans_agent.agent, "tokenizer"
        ):
            tokenizer = ans_agent.agent.tokenizer
            answer_truncations = 0
            for raw in answers_raw:
                if isinstance(raw, str):
                    raw_text = raw
                else:
                    raw_text = json.dumps(raw, ensure_ascii=True)
                if not raw_text:
                    continue
                token_count = len(tokenizer.encode(raw_text, add_special_tokens=False))
                if token_count >= answer_max_tokens:
                    answer_truncations += 1

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

        answer_empty_records = tracker.get_empty_records("answer")
        answer_empty_path = output_dir / "answer_empty_debug.json"
        ensure_dir(answer_empty_path)
        with answer_empty_path.open("w") as handle:
            json.dump(
                {
                    "prompt_example": tracker.get_prompt_example("answer"),
                    "event_count": len(answer_empty_records),
                    "events": answer_empty_records,
                },
                handle,
                indent=2,
            )

    metrics = {
        "question_metrics": question_metrics,
        "question_timing": question_timing,
        "question_truncations": question_truncations,
        "question_empty_events": len(question_empty_records),
        "question_self_reflection": q_self_stats,
        "answer_metrics": answer_metrics,
        "answer_timing": answer_timing,
        "answer_truncations": answer_truncations,
        "answer_empty_events": len(answer_empty_records) if not args.skip_answering else 0,
        "answer_self_reflection": answer_self_reflection_stats,
        "self_reflection_totals": {
            "attempts": q_self_stats["attempts"] + answer_self_reflection_stats["attempts"],
            "successes": q_self_stats["successes"] + answer_self_reflection_stats["successes"],
            "failures": q_self_stats["failures"] + answer_self_reflection_stats["failures"],
        },
        "completion_monitor": tracker.summary(),
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

    vllm_utils.chat_completion = original_chat_completion
    qm.chat_completion = original_qm_chat  # type: ignore[attr-defined]
    am.chat_completion = original_am_chat  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
