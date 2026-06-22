#!/usr/bin/env python3
"""Merge raw model outputs with human-reviewed final labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 1500
LABEL_ORDER = ("Complete refusal", "Partial compliance", "Full compliance")
LABELS = set(LABEL_ORDER)
FILE_PAIRS = (
    ("glm3_results_Chinese.jsonl", "glm3_chinese_classification.json"),
    ("glm3_results_English.jsonl", "glm3_english_classification.json"),
    ("glm4_results_Chinese.jsonl", "glm4_chinese_classification.json"),
    ("glm4_results_English.jsonl", "glm4_english_classification.json"),
    ("llama2_results_Chinese.jsonl", "llama2_chinese_classification.json"),
    ("llama2_results_English.jsonl", "llama2_english_classification.json"),
)


class MergeError(Exception):
    """Raised when raw output and human review data cannot be merged safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join raw model-output JSONL files with human-reviewed classification "
            "JSON arrays and write final result JSONL files."
        )
    )
    parser.add_argument(
        "--raw-dir",
        default=ROOT / "build" / "raw_results",
        type=Path,
        help="Directory containing raw runner outputs named like *_results_*.jsonl.",
    )
    parser.add_argument(
        "--manual-dir",
        required=True,
        type=Path,
        help="Directory containing *_classification.json human-review files.",
    )
    parser.add_argument(
        "--out-dir",
        default=ROOT / "data" / "results",
        type=Path,
        help="Directory where final merged JSONL files will be written.",
    )
    parser.add_argument(
        "--label-version",
        default="2026-06-22",
        help="Value written to meta.label_version for final labels.",
    )
    parser.add_argument(
        "--expected-rows",
        default=EXPECTED_ROWS,
        type=int,
        help="Expected row count for each raw and human-review file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks and print summaries without writing output files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_jsonl(path: Path) -> list[OrderedDict[str, Any]]:
    if not path.exists():
        raise MergeError(f"Missing raw result file: {path}")

    rows: list[OrderedDict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=OrderedDict)
        except json.JSONDecodeError as exc:
            raise MergeError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
        if not isinstance(value, dict):
            raise MergeError(f"{path}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def load_manual(path: Path) -> list[OrderedDict[str, Any]]:
    if not path.exists():
        raise MergeError(f"Missing human-review file: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as exc:
        raise MergeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise MergeError(f"{path} must contain a JSON array")
    if not all(isinstance(row, dict) for row in value):
        raise MergeError(f"{path} contains a non-object row")
    return value


def require_unique(rows: list[dict[str, Any]], key: str, path: Path) -> dict[int, dict[str, Any]]:
    by_idx: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for row_number, row in enumerate(rows, 1):
        idx = row.get(key)
        if not isinstance(idx, int):
            raise MergeError(f"{path} row {row_number} has non-integer {key}")
        if idx in by_idx:
            duplicates.append(idx)
        by_idx[idx] = row

    if duplicates:
        sample = ", ".join(str(idx) for idx in sorted(set(duplicates))[:10])
        raise MergeError(f"{path} has duplicate {key} values: {sample}")
    return by_idx


def require_row_count(path: Path, rows: list[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise MergeError(f"{path} has {len(rows)} rows, expected {expected_rows}")


def merge_file(
    raw_path: Path,
    manual_path: Path,
    output_path: Path,
    label_version: str,
    expected_rows: int,
    dry_run: bool,
) -> None:
    raw_rows = load_jsonl(raw_path)
    manual_rows = load_manual(manual_path)
    require_row_count(raw_path, raw_rows, expected_rows)
    require_row_count(manual_path, manual_rows, expected_rows)

    raw_by_idx = require_unique(raw_rows, "idx", raw_path)
    manual_by_idx = require_unique(manual_rows, "index", manual_path)
    if set(raw_by_idx) != set(manual_by_idx):
        missing = set(raw_by_idx) - set(manual_by_idx)
        extra = set(manual_by_idx) - set(raw_by_idx)
        raise MergeError(
            f"{raw_path.name} and {manual_path.name} idx coverage mismatch: "
            f"missing_manual={len(missing)} extra_manual={len(extra)}"
        )

    final_labels: Counter[str] = Counter()
    gatekeeper_labels: Counter[str] = Counter()
    changed_labels = 0
    merged_rows: list[OrderedDict[str, Any]] = []

    for raw in raw_rows:
        idx = raw["idx"]
        manual = manual_by_idx[idx]

        if raw.get("model_answer") != manual.get("model_answer"):
            raise MergeError(f"{raw_path.name}: model_answer mismatch at idx {idx}")

        gatekeeper_label = raw.get("evaluation_result_gatekeeper_v2", raw.get("evaluation_result"))
        if gatekeeper_label not in LABELS:
            raise MergeError(f"{raw_path.name}: invalid Gatekeeper label at idx {idx}: {gatekeeper_label!r}")

        final_label = manual.get("classification")
        if final_label not in LABELS:
            raise MergeError(f"{manual_path.name}: invalid classification at idx {idx}: {final_label!r}")

        reason = manual.get("classification_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise MergeError(f"{manual_path.name}: missing classification_reason at idx {idx}")

        for key in ("category", "model_answer", "evaluation_raw"):
            if key not in raw:
                raise MergeError(f"{raw_path.name}: missing {key} at idx {idx}")

        meta = raw.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta = OrderedDict(meta)
        meta["label_source"] = "human_review"
        meta["label_version"] = label_version

        merged = OrderedDict()
        merged["idx"] = raw["idx"]
        merged["category"] = raw["category"]
        merged["model_answer"] = raw["model_answer"]
        merged["evaluation_raw"] = raw["evaluation_raw"]
        merged["evaluation_result_gatekeeper_v2"] = gatekeeper_label
        merged["evaluation_result"] = final_label
        merged["adjudication_reason"] = reason.strip()
        merged["meta"] = meta
        merged_rows.append(merged)

        final_labels[final_label] += 1
        gatekeeper_labels[gatekeeper_label] += 1
        if final_label != gatekeeper_label:
            changed_labels += 1

    final_summary = ", ".join(f"{label}={final_labels[label]}" for label in LABEL_ORDER)
    gatekeeper_summary = ", ".join(
        f"{label}={gatekeeper_labels[label]}" for label in LABEL_ORDER
    )
    print(
        f"{output_path.relative_to(ROOT)}: checked {len(merged_rows)} rows, "
        f"final labels: {final_summary}; "
        f"Gatekeeper labels: {gatekeeper_summary}; "
        f"changed={changed_labels}"
    )

    if dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    raw_dir = resolve_path(args.raw_dir)
    manual_dir = resolve_path(args.manual_dir)
    out_dir = resolve_path(args.out_dir)

    for raw_name, manual_name in FILE_PAIRS:
        merge_file(
            raw_path=raw_dir / raw_name,
            manual_path=manual_dir / manual_name,
            output_path=out_dir / raw_name,
            label_version=args.label_version,
            expected_rows=args.expected_rows,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("Merge dry run passed.")
    else:
        print("Human-review merge complete.")


if __name__ == "__main__":
    try:
        main()
    except MergeError as exc:
        raise SystemExit(f"Human-review merge failed: {exc}") from exc
