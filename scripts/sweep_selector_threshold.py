#!/usr/bin/env python3
"""Sweep the selector P(YES) threshold on saved selector predictions.

Run two-stage inference on the dev split first so ``selector_predictions.jsonl`` holds
``p_yes``, then pick the threshold here and pass it to ``infer_two_stage.py
--selector-threshold`` for the test split. Do not tune the threshold on test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_thresholds(value: str) -> list[float]:
    """Parse ``start:stop:step`` (inclusive) or a comma-separated list."""

    if ":" in value:
        start, stop, step = (float(part) for part in value.split(":"))
        if step <= 0 or stop < start:
            raise ValueError("--thresholds start:stop:step needs step > 0 and stop >= start")
        count = int(round((stop - start) / step))
        return [round(start + index * step, 6) for index in range(count + 1)]
    return [float(part) for part in value.split(",") if part.strip()]


def load_rows(predictions: Path, gold: Path) -> list[dict[str, Any]]:
    """Join predictions with gold labels by unit id, keeping the gold file order."""

    by_id = {row["id"]: row for row in iter_jsonl(predictions)}
    rows: list[dict[str, Any]] = []
    for source in iter_jsonl(gold):
        prediction = by_id.get(source["id"])
        if prediction is None:
            raise ValueError(f"No prediction for gold row {source['id']!r}")
        if "p_yes" not in prediction:
            raise ValueError(
                f"Prediction {source['id']!r} has no p_yes; rerun inference with a build that saves it"
            )
        rows.append(
            {
                "example_id": source["example_id"],
                "gold": int(source["label"]) == 1,
                "p_yes": float(prediction["p_yes"]),
                "greedy": prediction["predicted_label"] == "YES",
            }
        )
    if len(rows) != len(by_id):
        raise ValueError(f"{len(by_id)} predictions but {len(rows)} gold rows; the files do not match")
    return rows


def score(rows: Iterable[dict[str, Any]], predict: Any) -> dict[str, float]:
    """Unit-level precision/recall/F1 plus per-question extra and missed units."""

    tp = fp = fn = 0
    questions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        predicted = predict(row)
        tp += predicted and row["gold"]
        fp += predicted and not row["gold"]
        fn += (not predicted) and row["gold"]
        counts = questions[row["example_id"]]
        counts[0] += predicted and not row["gold"]
        counts[1] += (not predicted) and row["gold"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(questions)
    return {
        "precision": 100 * precision,
        "recall": 100 * recall,
        "f1": 100 * f1,
        "extra_per_question": sum(extra for extra, _ in questions.values()) / total,
        "questions_missing_gold": 100 * sum(missed > 0 for _, missed in questions.values()) / total,
        "questions_exact": 100 * sum(extra == 0 and missed == 0 for extra, missed in questions.values()) / total,
    }


def sweep(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    results = [{"threshold": "greedy", **score(rows, lambda row: row["greedy"])}]
    for threshold in thresholds:
        results.append({"threshold": threshold, **score(rows, lambda row, t=threshold: row["p_yes"] >= t)})
    return results


def format_table(results: list[dict[str, Any]]) -> str:
    header = f"{'threshold':>9} {'prec':>6} {'recall':>6} {'F1':>6} {'extra/q':>8} {'miss%':>6} {'exact%':>7}"
    lines = [header]
    for result in results:
        threshold = result["threshold"] if isinstance(result["threshold"], str) else f"{result['threshold']:.2f}"
        lines.append(
            f"{threshold:>9} {result['precision']:6.1f} {result['recall']:6.1f} {result['f1']:6.1f} "
            f"{result['extra_per_question']:8.2f} {result['questions_missing_gold']:6.1f} "
            f"{result['questions_exact']:7.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="selector_predictions.jsonl with p_yes")
    parser.add_argument("--gold", type=Path, required=True, help="Matching labelled selection_*.jsonl")
    parser.add_argument(
        "--thresholds",
        default="0.30:0.95:0.05",
        help="start:stop:step (inclusive) or a comma-separated list. Default: 0.30:0.95:0.05",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optionally save the table as JSON.")
    args = parser.parse_args()

    results = sweep(load_rows(args.predictions, args.gold), parse_thresholds(args.thresholds))
    print(format_table(results))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
