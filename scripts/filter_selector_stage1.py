"""Build a compact selector-training corpus from a pipeline output.

Every selected positive belongs to a same-question YES/NO contrast pair. Extra
negative-only questions are then sampled so the final label prior matches the
combined selector-test distribution. The filter also keeps every observed
``(schema_id, unit_id, label)`` combination and balances row counts by graph.
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_MANIFESTS = (
    ROOT / "data" / "cypherbench_schema_grounding_full" / "manifest.json",
    ROOT / "data" / "mind_the_query_schema_grounding_full" / "manifest.json",
    ROOT / "data" / "neo4j_text2cypher_schema_grounding_full" / "manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help=(
            "Total selector-train rows. Positive rows retain same-question "
            "positive/negative contrasts. By default, match generation_train.jsonl."
        ),
    )
    parser.add_argument(
        "--test-manifests",
        type=Path,
        nargs="+",
        default=list(DEFAULT_TEST_MANIFESTS),
        help=(
            "Manifests whose */test selector counts define the row-weighted target "
            "label distribution. Defaults to the three local benchmark manifests."
        ),
    )
    parser.add_argument(
        "--target-positive-ratio",
        type=float,
        default=None,
        help=(
            "Optional explicit YES ratio in (0, 0.5]. By default it is derived from "
            "--test-manifests."
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


def load_test_distribution(
    manifest_paths: Iterable[Path],
) -> tuple[float, list[dict[str, Any]]]:
    """Return the row-weighted selector YES ratio from benchmark test manifests."""

    distributions: list[dict[str, Any]] = []
    total_positive = 0
    total_negative = 0
    for manifest_path in manifest_paths:
        resolved = manifest_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Test-distribution manifest does not exist: {resolved}")
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
        test_counts = [
            (name, counts)
            for name, counts in manifest.get("counts", {}).items()
            if str(name).endswith("/test")
        ]
        if len(test_counts) != 1:
            raise ValueError(
                f"Expected exactly one */test count entry in {resolved}, found {len(test_counts)}"
            )
        benchmark, counts = test_counts[0]
        positive = int(counts.get("selection_positive", 0))
        negative = int(counts.get("selection_negative", 0))
        if positive <= 0 or negative <= 0:
            raise ValueError(f"Invalid selector test counts in {resolved}: {counts}")
        total = positive + negative
        distributions.append(
            {
                "benchmark": benchmark,
                "manifest": str(resolved),
                "selection_rows": total,
                "selection_positive": positive,
                "selection_negative": negative,
                "positive_ratio": positive / total,
            }
        )
        total_positive += positive
        total_negative += negative

    if not distributions:
        raise ValueError("At least one test-distribution manifest is required")
    return total_positive / (total_positive + total_negative), distributions


def _proportional_quotas(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Allocate an integer total proportionally using largest remainders."""

    if total < 0 or not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("Quota total and weights must be valid positive values")
    weight_sum = sum(weights.values())
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in priority[:remainder]:
        quotas[key] += 1
    return quotas


def select_stage_one_rows(
    rows: list[dict[str, Any]],
    target_rows: int,
    seed: int,
    positive_ratio: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select contrast pairs plus unique-question negatives with full coverage."""

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
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    if not 0 < positive_ratio <= 0.5:
        raise ValueError("positive_ratio must be in (0, 0.5]")

    graph_names = sorted(by_graph)
    question_counts = {
        graph: len({str(row["example_id"]) for row in by_graph[graph]})
        for graph in graph_names
    }
    target_positive = round(target_rows * positive_ratio)
    if target_positive < len(graph_names):
        raise ValueError("Positive quota is too small to retain every graph")
    row_quotas = _proportional_quotas(target_rows, question_counts)
    positive_quotas = _proportional_quotas(target_positive, row_quotas)
    for graph in graph_names:
        if positive_quotas[graph] * 2 > row_quotas[graph]:
            raise ValueError(f"{graph}: positive quota leaves no room for contrast negatives")

    rng = random.Random(seed)
    selected_pair_groups: list[list[dict[str, Any]]] = []
    summaries: dict[str, dict[str, Any]] = {}

    selected_negative_rows: list[dict[str, Any]] = []
    for graph in graph_names:
        graph_rows = by_graph[graph]
        per_graph = row_quotas[graph]
        pairs_per_graph = positive_quotas[graph]
        extra_negatives_per_graph = per_graph - 2 * pairs_per_graph
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
        positive_unit_pairs = {pair for pair in by_pair if pair[2] == 1}
        if len(positive_unit_pairs) > pairs_per_graph:
            raise ValueError(f"{graph}: contrast-pair quota is too small for positive-unit coverage")

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
        impossible_positive_pairs = [
            pair for pair in positive_unit_pairs if pair_frequencies[pair] == 0
        ]
        if impossible_positive_pairs:
            raise ValueError(
                f"{graph}: positive units cannot be placed in same-question contrasts: "
                f"{impossible_positive_pairs}"
            )

        # Cover rare positive units first. Every positive row remains paired
        # with a negative unit from the same question.
        required_pairs = sorted(
            positive_unit_pairs,
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

            def anchor_priority(row: dict[str, Any]) -> tuple[int, int]:
                negative_rows = by_question[str(row["example_id"])][0]
                negative_keys = [
                    (str(candidate["schema_id"]), str(candidate["unit_id"]), 0)
                    for candidate in negative_rows
                ]
                return (
                    0 if any(key not in covered_pairs for key in negative_keys) else 1,
                    min(pair_frequencies[key] for key in negative_keys),
                )

            anchor = min(candidates, key=anchor_priority)
            question_id = str(anchor["example_id"])
            opposite = _choose_row(
                by_question[question_id][0],
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
                for label in (1, 0)
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

        # Cover any negative units not already represented by the contrast
        # members, then fill the remaining negative quota from unused questions.
        selected_negatives: list[dict[str, Any]] = []
        negative_pairs = {pair for pair in by_pair if pair[2] == 0}
        for pair in sorted(
            negative_pairs.difference(covered_pairs),
            key=lambda item: (pair_frequencies[item], item),
        ):
            candidates = [
                row
                for row in by_pair[pair]
                if str(row["example_id"]) not in used_questions
            ]
            if not candidates:
                raise ValueError(
                    f"{graph}: cannot cover negative unit {pair} without reusing a question"
                )
            rng.shuffle(candidates)
            chosen = dict(candidates[0])
            selected_negatives.append(chosen)
            used_questions.add(str(chosen["example_id"]))
            covered_pairs.add(pair)

        needed_negatives = extra_negatives_per_graph - len(selected_negatives)
        if needed_negatives < 0:
            raise ValueError(f"{graph}: negative-only quota is too small for unit coverage")
        remaining_negative_questions = [
            question_id
            for question_id, label_rows in by_question.items()
            if label_rows[0] and question_id not in used_questions
        ]
        rng.shuffle(remaining_negative_questions)
        if len(remaining_negative_questions) < needed_negatives:
            raise ValueError(f"{graph}: insufficient unused questions to fill negative-only quota")
        for question_id in remaining_negative_questions[:needed_negatives]:
            chosen = dict(
                _choose_row(
                    by_question[question_id][0],
                    set(by_pair).difference(covered_pairs),
                    pair_frequencies,
                    rng,
                )
            )
            selected_negatives.append(chosen)
            used_questions.add(question_id)
            covered_pairs.add((str(chosen["schema_id"]), str(chosen["unit_id"]), 0))

        rng.shuffle(selected_pairs)
        rng.shuffle(selected_negatives)
        graph_selected = [row for contrast in selected_pairs for row in contrast] + selected_negatives
        label_counts = Counter(int(row["label"]) for row in graph_selected)
        expected_counts = Counter({0: per_graph - pairs_per_graph, 1: pairs_per_graph})
        if len(graph_selected) != per_graph or label_counts != expected_counts:
            raise AssertionError(f"{graph}: quota validation failed")
        if covered_pairs != set(by_pair):
            raise AssertionError(f"{graph}: unit-label coverage validation failed")
        if len(used_questions) != len(selected_pairs) + len(selected_negatives):
            raise AssertionError(f"{graph}: selected-question uniqueness validation failed")
        if any(
            first["contrast_pair_id"] != second["contrast_pair_id"]
            or {int(first["label"]), int(second["label"])} != {0, 1}
            for first, second in selected_pairs
        ):
            raise AssertionError(f"{graph}: malformed contrast pair")

        # Keep the two members adjacent; shuffle complete pairs globally below.
        selected_pair_groups.extend(selected_pairs)
        selected_negative_rows.extend(selected_negatives)
        summaries[graph] = {
            "rows": len(graph_selected),
            "contrast_pairs": len(selected_pairs),
            "unpaired_negative_rows": len(selected_negatives),
            "unique_questions": len(used_questions),
            "selection_positive": pairs_per_graph,
            "selection_negative": per_graph - pairs_per_graph,
            "schema_unit_label_pairs_covered": len(covered_pairs),
        }

    rng.shuffle(selected_pair_groups)
    rng.shuffle(selected_negative_rows)
    flattened = [row for contrast in selected_pair_groups for row in contrast] + selected_negative_rows
    contrast_pairs = len(selected_pair_groups)
    expected_unique_questions = len(flattened) - contrast_pairs
    if len({str(row["example_id"]) for row in flattened}) != expected_unique_questions:
        raise AssertionError("Selected questions overlap across graph groups or sampling roles")
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
    generation_path = input_dir / "generation_train.jsonl"
    manifest_path = input_dir / "manifest.json"
    if not selection_path.exists() or not generation_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "input-dir must contain selection_train.jsonl, generation_train.jsonl, and manifest.json"
        )

    selection_rows = read_jsonl(selection_path)
    target_rows = (
        args.target_rows
        if args.target_rows is not None
        else len(read_jsonl(generation_path))
    )
    derived_positive_ratio, test_distributions = load_test_distribution(args.test_manifests)
    target_positive_ratio = (
        args.target_positive_ratio
        if args.target_positive_ratio is not None
        else derived_positive_ratio
    )
    filtered_rows, graph_summaries = select_stage_one_rows(
        selection_rows, target_rows, args.seed, target_positive_ratio
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
    contrast_pairs = sum(1 for row in filtered_rows if "contrast_pair_id" in row) // 2
    actual_positive_ratio = train_counts["selection_positive"] / len(filtered_rows)
    manifest["selector_stage1"] = {
        "source_directory": str(input_dir),
        "sampling_seed": args.seed,
        "rows": len(filtered_rows),
        "target_rows_source": (
            "explicit --target-rows"
            if args.target_rows is not None
            else "generation_train.jsonl row count"
        ),
        "unique_questions": len({str(row["example_id"]) for row in filtered_rows}),
        "contrast_pairs": contrast_pairs,
        "unpaired_negative_rows": len(filtered_rows) - 2 * contrast_pairs,
        "target_positive_ratio": target_positive_ratio,
        "actual_positive_ratio": actual_positive_ratio,
        "test_distribution_weighting": "row_weighted",
        "test_distributions": test_distributions,
        "policy": (
            "Every YES belongs to a same-question YES/NO contrast pair; unique-question NO rows "
            "are added to match the row-weighted three-test label prior; per-graph quotas follow "
            "the source question distribution; cover every observed schema unit and label pair. "
            "Pair members are adjacent and precede unpaired negatives."
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
