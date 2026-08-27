"""Build a compact, paired selector-training corpus from a pipeline output.

Every selected question contributes one positive and one negative schema unit,
so the model sees a same-question contrast instead of unrelated examples from
different questions. The filter also keeps every observed
``(schema_id, unit_id, label)`` combination and balances pair counts by graph.
Generation data and evaluation splits are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=3200,
        help=(
            "Total selector-train rows. Every two rows form a same-question "
            "positive/negative pair. Must divide evenly across graphs. Default: 3200."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
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


def _choose_row(
    rows: list[dict[str, Any]],
    uncovered_pairs: set[tuple[str, str, int]],
    pair_frequencies: Counter[tuple[str, str, int]],
    rng: random.Random,
) -> dict[str, Any]:
    """Prefer an uncovered, rare unit-label pair with randomized tie-breaking."""

    candidates = list(rows)
    rng.shuffle(candidates)

    def priority(row: dict[str, Any]) -> tuple[int, int]:
        pair = (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
        return (0 if pair in uncovered_pairs else 1, pair_frequencies[pair])

    return min(candidates, key=priority)


def select_stage_one_rows(
    rows: list[dict[str, Any]], target_rows: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select same-question positive/negative pairs with full unit-label coverage."""

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
        raise ValueError("Each graph quota must be even so rows can form contrast pairs")
    pairs_per_graph = per_graph // 2

    rng = random.Random(seed)
    selected_pair_groups: list[list[dict[str, Any]]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for graph, graph_rows in sorted(by_graph.items()):
        by_question: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_pair: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in graph_rows:
            label = int(row["label"])
            by_question[str(row["example_id"])][label].append(row)
            by_pair[(str(row["schema_id"]), str(row["unit_id"]), label)].append(row)

        eligible_questions = {
            question_id
            for question_id, label_rows in by_question.items()
            if label_rows[0] and label_rows[1]
        }
        if len(eligible_questions) < pairs_per_graph:
            raise ValueError(
                f"{graph}: only {len(eligible_questions)} questions contain both labels; "
                f"need {pairs_per_graph} contrast pairs"
            )
        if len(by_pair) > pairs_per_graph * 2:
            raise ValueError(f"{graph}: contrast-pair quota is too small for unit-label coverage")

        used_questions: set[str] = set()
        selected_pairs: list[list[dict[str, Any]]] = []
        covered_pairs: set[tuple[str, str, int]] = set()
        pair_frequencies = Counter(
            {
                pair: len(
                    {
                        str(row["example_id"])
                        for row in pair_rows
                        if str(row["example_id"]) in eligible_questions
                    }
                )
                for pair, pair_rows in by_pair.items()
            }
        )
        impossible_pairs = [pair for pair, frequency in pair_frequencies.items() if frequency == 0]
        if impossible_pairs:
            raise ValueError(
                f"{graph}: unit-label pairs cannot be placed in same-question contrasts: {impossible_pairs}"
            )

        # Cover rare unit-label pairs first. Each chosen question contributes
        # the requested row plus one row with the opposite label.
        required_pairs = sorted(
            by_pair,
            key=lambda pair: (pair_frequencies[pair], pair),
        )
        for pair in required_pairs:
            if pair in covered_pairs:
                continue
            candidates = [
                row
                for row in by_pair[pair]
                if str(row["example_id"]) in eligible_questions
                and str(row["example_id"]) not in used_questions
            ]
            if not candidates:
                raise ValueError(
                    f"{graph}: cannot cover {pair} without reusing a contrast question; "
                    "increase --target-rows or relax unit-label coverage."
                )
            rng.shuffle(candidates)
            anchor = candidates[0]
            question_id = str(anchor["example_id"])
            opposite_label = 1 - int(anchor["label"])
            opposite = _choose_row(
                by_question[question_id][opposite_label],
                set(by_pair).difference(covered_pairs),
                pair_frequencies,
                rng,
            )
            contrast = [dict(anchor), dict(opposite)]
            rng.shuffle(contrast)
            for row in contrast:
                row["contrast_pair_id"] = question_id
            selected_pairs.append(contrast)
            used_questions.add(question_id)
            covered_pairs.update(
                (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
                for row in contrast
            )

        remaining_questions = list(eligible_questions.difference(used_questions))
        rng.shuffle(remaining_questions)
        needed_pairs = pairs_per_graph - len(selected_pairs)
        if len(remaining_questions) < needed_pairs:
            raise ValueError(f"{graph}: insufficient unused questions to fill contrast-pair quota")
        for question_id in remaining_questions[:needed_pairs]:
            contrast = [
                dict(
                    _choose_row(
                        by_question[question_id][label],
                        set(by_pair).difference(covered_pairs),
                        pair_frequencies,
                        rng,
                    )
                )
                for label in (0, 1)
            ]
            rng.shuffle(contrast)
            for row in contrast:
                row["contrast_pair_id"] = question_id
            selected_pairs.append(contrast)
            used_questions.add(question_id)
            covered_pairs.update(
                (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
                for row in contrast
            )

        rng.shuffle(selected_pairs)
        graph_selected = [row for contrast in selected_pairs for row in contrast]
        label_counts = Counter(int(row["label"]) for row in graph_selected)
        if len(graph_selected) != per_graph or label_counts != Counter({0: pairs_per_graph, 1: pairs_per_graph}):
            raise AssertionError(f"{graph}: quota validation failed")
        if covered_pairs != set(by_pair):
            raise AssertionError(f"{graph}: unit-label coverage validation failed")
        if len(used_questions) != len(selected_pairs):
            raise AssertionError(f"{graph}: contrast-question uniqueness validation failed")
        if any(
            first["contrast_pair_id"] != second["contrast_pair_id"]
            or {int(first["label"]), int(second["label"])} != {0, 1}
            for first, second in selected_pairs
        ):
            raise AssertionError(f"{graph}: malformed contrast pair")

        # Keep the two members adjacent; shuffle complete pairs globally below.
        selected_pair_groups.extend(selected_pairs)
        summaries[graph] = {
            "rows": len(graph_selected),
            "contrast_pairs": len(selected_pairs),
            "unique_questions": len(used_questions),
            "selection_positive": pairs_per_graph,
            "selection_negative": pairs_per_graph,
            "schema_unit_label_pairs_covered": len(covered_pairs),
        }

    rng.shuffle(selected_pair_groups)
    flattened = [row for contrast in selected_pair_groups for row in contrast]
    if len({str(row["example_id"]) for row in flattened}) * 2 != len(flattened):
        raise AssertionError("Contrast questions overlap across graph groups")
    return flattened, summaries


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
        "contrast_pairs": len(filtered_rows) // 2,
        "policy": (
            "One YES and one NO schema unit per selected question; equal per-graph pair quotas; "
            "cover every observed schema unit and label pair. Pair members are adjacent."
        ),
        "graphs": graph_summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["selector_stage1"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
