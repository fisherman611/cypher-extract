"""Build a compact, balanced selector-training corpus from a pipeline output.

The filter keeps every observed ``(schema_id, unit_id, label)`` combination,
then fills equal per-graph and per-label quotas with one row per question.
Generation data and evaluation splits are copied unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=3200,
        help="Total selector-train rows. Must divide evenly across graphs and labels. Default: 3200.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} in {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _one_row_per_question(
    rows: Iterable[dict[str, Any]], used_questions: set[str], rng: random.Random
) -> list[dict[str, Any]]:
    """Choose one randomized row for each as-yet-unused question."""

    candidates = list(rows)
    rng.shuffle(candidates)
    result: dict[str, dict[str, Any]] = {}
    for row in candidates:
        question_id = str(row["example_id"])
        if question_id not in used_questions:
            result.setdefault(question_id, row)
    return list(result.values())


def select_stage_one_rows(
    rows: list[dict[str, Any]], target_rows: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select balanced rows while covering every observed unit and label pair."""

    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        required = {"example_id", "graph", "schema_id", "unit_id", "label"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Selection row missing required fields: {sorted(missing)}")
        if row["label"] not in (0, 1):
            raise ValueError(f"Selection label must be 0 or 1: {row['label']!r}")
        by_graph[str(row["graph"])].append(row)

    if not by_graph:
        raise ValueError("No selection rows found")
    if target_rows <= 0 or target_rows % len(by_graph) != 0:
        raise ValueError("target_rows must be positive and divide evenly across graphs")
    per_graph = target_rows // len(by_graph)
    if per_graph % 2 != 0:
        raise ValueError("Each graph quota must be even so labels can be balanced")
    per_label = per_graph // 2

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for graph, graph_rows in sorted(by_graph.items()):
        by_pair: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in graph_rows:
            by_pair[(str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))].append(row)
        if any(sum(pair[2] == label for pair in by_pair) > per_label for label in (0, 1)):
            raise ValueError(f"{graph}: coverage requires more rows than its label quota")

        used_questions: set[str] = set()
        graph_selected: list[dict[str, Any]] = []
        # Rarest unit-label pairs first keeps coverage feasible without reusing a question.
        required_pairs = sorted(
            by_pair,
            key=lambda pair: (len({str(row['example_id']) for row in by_pair[pair]}), pair),
        )
        for pair in required_pairs:
            candidates = _one_row_per_question(by_pair[pair], used_questions, rng)
            if not candidates:
                raise ValueError(
                    f"{graph}: cannot cover {pair} without reusing a question; "
                    "increase the question-repeat allowance or relax coverage."
                )
            row = candidates[0]
            graph_selected.append(row)
            used_questions.add(str(row["example_id"]))

        for label in (0, 1):
            needed = per_label - sum(row["label"] == label for row in graph_selected)
            candidates = _one_row_per_question(
                (row for row in graph_rows if row["label"] == label), used_questions, rng
            )
            if len(candidates) < needed:
                raise ValueError(
                    f"{graph}: only {len(candidates)} unused questions available for label {label}; need {needed}"
                )
            rng.shuffle(candidates)
            additions = candidates[:needed]
            graph_selected.extend(additions)
            used_questions.update(str(row["example_id"]) for row in additions)

        label_counts = Counter(int(row["label"]) for row in graph_selected)
        covered_pairs = {
            (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
            for row in graph_selected
        }
        if len(graph_selected) != per_graph or label_counts != Counter({0: per_label, 1: per_label}):
            raise AssertionError(f"{graph}: quota validation failed")
        if covered_pairs != set(by_pair):
            raise AssertionError(f"{graph}: unit-label coverage validation failed")
        if len(used_questions) != len(graph_selected):
            raise AssertionError(f"{graph}: question uniqueness validation failed")

        selected.extend(graph_selected)
        summaries[graph] = {
            "rows": len(graph_selected),
            "unique_questions": len(used_questions),
            "selection_positive": per_label,
            "selection_negative": per_label,
            "schema_unit_label_pairs_covered": len(covered_pairs),
        }

    rng.shuffle(selected)
    if len({str(row["example_id"]) for row in selected}) != len(selected):
        raise AssertionError("Questions overlap across graph groups")
    return selected, summaries


def prepare_output_directory(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input-dir and output-dir must differ")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    selection_path = input_dir / "selection_train.jsonl"
    manifest_path = input_dir / "manifest.json"
    if not selection_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("input-dir must contain selection_train.jsonl and manifest.json")

    selection_rows = read_jsonl(selection_path)
    filtered_rows, graph_summaries = select_stage_one_rows(
        selection_rows, args.target_rows, args.seed
    )
    prepare_output_directory(input_dir, output_dir, args.overwrite)

    for path in input_dir.iterdir():
        if path.name not in {"selection_train.jsonl", "manifest.json"}:
            shutil.copy2(path, output_dir / path.name)
    write_jsonl(output_dir / "selection_train.jsonl", filtered_rows)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_counts = manifest["counts"]["cypherbench/train"]
    train_counts["selection_examples"] = len(filtered_rows)
    train_counts["selection_positive"] = sum(int(row["label"]) for row in filtered_rows)
    train_counts["selection_negative"] = len(filtered_rows) - train_counts["selection_positive"]
    manifest["selector_stage1"] = {
        "source_directory": str(input_dir),
        "sampling_seed": args.seed,
        "rows": len(filtered_rows),
        "unique_questions": len({str(row["example_id"]) for row in filtered_rows}),
        "policy": "One row per question; equal per-graph and label quotas; cover every observed schema unit and label pair.",
        "graphs": graph_summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["selector_stage1"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
