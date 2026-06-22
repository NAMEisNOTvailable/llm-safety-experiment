#!/usr/bin/env python3
"""Validate prompt/result JSONL files used by the smoke test."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 1500
LABEL_VERSION = "2026-06-22"
PROMPT_FILES = {
    "Chinese": ROOT / "data" / "prompts" / "1500_Chinese_prompt.jsonl",
    "English": ROOT / "data" / "prompts" / "1500_English_prompt.jsonl",
}
RESULT_FILES = {
    "glm3_results_Chinese.jsonl": "Chinese",
    "glm3_results_English.jsonl": "English",
    "glm4_results_Chinese.jsonl": "Chinese",
    "glm4_results_English.jsonl": "English",
    "llama2_results_Chinese.jsonl": "Chinese",
    "llama2_results_English.jsonl": "English",
}
LABEL_ORDER = ("Complete refusal", "Partial compliance", "Full compliance")
LABELS = set(LABEL_ORDER)
PROMPT_KEYS = {"prompt", "evaluation", "attack", "goal", "category", "idx"}
RESULT_KEYS = {
    "idx",
    "category",
    "model_answer",
    "evaluation_raw",
    "evaluation_result_gatekeeper_v2",
    "evaluation_result",
    "adjudication_reason",
    "meta",
}


class ValidationError(Exception):
    """Raised when a dataset invariant is broken."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValidationError(f"Missing file: {path.relative_to(ROOT)}")

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{path.relative_to(ROOT)}:{line_no} is not valid JSONL: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValidationError(f"{path.relative_to(ROOT)}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def require_keys(path: Path, rows: list[dict[str, Any]], keys: set[str]) -> None:
    for index, row in enumerate(rows, 1):
        missing = keys - row.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} missing: {missing_list}")


def require_malicious_categories(path: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, 1):
        category = row.get("category")
        if not isinstance(category, str):
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} has non-string category")
        if category != category.strip():
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} has untrimmed category")
        if category.strip() != "malicious":
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} category is {category!r}")


def require_unique_idx(path: Path, rows: list[dict[str, Any]]) -> set[int]:
    idx_values: list[int] = []
    for index, row in enumerate(rows, 1):
        idx = row.get("idx")
        if not isinstance(idx, int):
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} has non-integer idx")
        idx_values.append(idx)

    duplicates = [idx for idx, count in Counter(idx_values).items() if count > 1]
    if duplicates:
        sample = ", ".join(str(idx) for idx in sorted(duplicates)[:10])
        raise ValidationError(f"{path.relative_to(ROOT)} has duplicate idx values: {sample}")
    return set(idx_values)


def validate_prompt_file(language: str, path: Path) -> set[int]:
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_ROWS:
        raise ValidationError(f"{path.relative_to(ROOT)} has {len(rows)} rows, expected {EXPECTED_ROWS}")
    require_keys(path, rows, PROMPT_KEYS)
    require_malicious_categories(path, rows)

    for index, row in enumerate(rows, 1):
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, dict) or not evaluation:
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} has invalid evaluation metadata")

    idxs = require_unique_idx(path, rows)
    print(f"{language} prompts: {len(rows)} rows, {len(idxs)} unique idx values")
    return idxs


def validate_result_file(path: Path, expected_idx: set[int]) -> None:
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_ROWS:
        raise ValidationError(f"{path.relative_to(ROOT)} has {len(rows)} rows, expected {EXPECTED_ROWS}")
    require_keys(path, rows, RESULT_KEYS)
    require_malicious_categories(path, rows)
    idxs = require_unique_idx(path, rows)

    missing = expected_idx - idxs
    extra = idxs - expected_idx
    if missing or extra:
        raise ValidationError(
            f"{path.relative_to(ROOT)} idx coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )

    final_labels = Counter()
    gatekeeper_labels = Counter()
    changed_labels = 0
    for index, row in enumerate(rows, 1):
        if "prompt" in row:
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} should not duplicate prompt text")

        gatekeeper_label = row.get("evaluation_result_gatekeeper_v2")
        if gatekeeper_label not in LABELS:
            raise ValidationError(
                f"{path.relative_to(ROOT)} row {index} invalid Gatekeeper label: {gatekeeper_label!r}"
            )

        label = row.get("evaluation_result")
        if label not in LABELS:
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} invalid final label: {label!r}")

        reason = row.get("adjudication_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValidationError(
                f"{path.relative_to(ROOT)} row {index} has invalid adjudication_reason"
            )

        meta = row.get("meta")
        if not isinstance(meta, dict):
            raise ValidationError(f"{path.relative_to(ROOT)} row {index} has invalid meta")
        if meta.get("label_source") != "human_review":
            raise ValidationError(
                f"{path.relative_to(ROOT)} row {index} invalid meta.label_source: "
                f"{meta.get('label_source')!r}"
            )
        if meta.get("label_version") != LABEL_VERSION:
            raise ValidationError(
                f"{path.relative_to(ROOT)} row {index} invalid meta.label_version: "
                f"{meta.get('label_version')!r}"
            )

        final_labels[label] += 1
        gatekeeper_labels[gatekeeper_label] += 1
        if label != gatekeeper_label:
            changed_labels += 1

    label_summary = ", ".join(f"{label}={final_labels[label]}" for label in LABEL_ORDER)
    gatekeeper_summary = ", ".join(
        f"{label}={gatekeeper_labels[label]}" for label in LABEL_ORDER
    )
    print(
        f"{path.relative_to(ROOT)}: {len(rows)} rows, "
        f"final labels: {label_summary}; "
        f"Gatekeeper labels: {gatekeeper_summary}; "
        f"changed={changed_labels}"
    )


def main() -> None:
    prompt_idx_by_language = {
        language: validate_prompt_file(language, path)
        for language, path in PROMPT_FILES.items()
    }

    if prompt_idx_by_language["Chinese"] != prompt_idx_by_language["English"]:
        raise ValidationError("Chinese and English prompt idx sets do not match")
    print("Prompt pairing: Chinese and English idx sets match")

    for filename, language in RESULT_FILES.items():
        validate_result_file(ROOT / "data" / "results" / filename, prompt_idx_by_language[language])

    print("Dataset validation passed.")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc
