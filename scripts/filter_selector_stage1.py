"""Build a compact selector-training corpus from a pipeline output.

Every selected positive belongs to a same-question YES/NO contrast pair. Extra
negative-only questions are then sampled to reach either an explicit positive
ratio or the combined selector-test prior. The filter also keeps every observed
``(schema_id, unit_id, label)`` combination and balances row counts by graph.
Generation data and evaluation splits are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema_grounding.inference.prompting import competing_unit_ids  # noqa: E402

PAIR_STRATEGIES = ("random-coverage", "hard-competitor")
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
    parser.add_argument(
        "--pair-strategy",
        choices=PAIR_STRATEGIES,
        default="random-coverage",
        help=(
            "How the NO row of each same-question contrast is chosen. random-coverage keeps the original "
            "rare-unit coverage preference; hard-competitor additionally swaps same-type rows inside pairs so "
            "the NO unit competes with the YES unit (same node labels or relationship type, or similar "
            "node properties) whenever the question offers such a unit, preserving every label x unit-type "
            "count and the schema-unit coverage."
        ),
    )
    parser.add_argument(
        "--cover-all-questions",
        action="store_true",
        help=(
            "Use every source question exactly once (a contrast pair or one negative), so the selector sees "
            "every generator-train question. The row count becomes questions + contrast pairs, which exceeds "
            "generation_train.jsonl; incompatible with --target-rows."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cover_all_questions and args.target_rows is not None:
        parser.error("--cover-all-questions derives the row count; do not pass --target-rows")
    return args


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


def _unit_type(row: dict[str, Any]) -> str:
    unit_type = row.get("unit_type")
    if unit_type is None and isinstance(row.get("unit"), dict):
        unit_type = row["unit"].get("kind")
    if unit_type is None:
        unit_type = str(row.get("unit_id", "")).split(":", 1)[0]
    if unit_type not in {"node", "relation"}:
        raise ValueError(f"Unsupported selector unit type: {unit_type!r}")
    return str(unit_type)


def load_test_distribution(
    manifest_paths: Iterable[Path],
) -> tuple[float, dict[int, dict[str, float]], list[dict[str, Any]]]:
    """Return row-weighted label and unit-type priors from benchmark tests."""

    distributions: list[dict[str, Any]] = []
    total_positive = 0
    total_negative = 0
    total_type_labels: Counter[tuple[str, int]] = Counter()
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
        selection_filename = manifest.get("files", {}).get("selection", {}).get("test")
        if not isinstance(selection_filename, str):
            raise ValueError(f"Missing files.selection.test in {resolved}")
        selection_path = resolved.parent / selection_filename
        type_labels = Counter(
            (_unit_type(row), int(row["label"])) for row in read_jsonl(selection_path)
        )
        if sum(type_labels.values()) != total:
            raise ValueError(
                f"Selector test row count in {selection_path} does not match {resolved}"
            )
        if sum(count for (__, label), count in type_labels.items() if label == 1) != positive:
            raise ValueError(f"Selector positive count in {selection_path} does not match {resolved}")
        distributions.append(
            {
                "benchmark": benchmark,
                "manifest": str(resolved),
                "selection_rows": total,
                "selection_positive": positive,
                "selection_negative": negative,
                "positive_ratio": positive / total,
                "unit_type_labels": {
                    f"{unit_type}:{label}": type_labels[(unit_type, label)]
                    for label in (0, 1)
                    for unit_type in ("node", "relation")
                },
            }
        )
        total_positive += positive
        total_negative += negative
        total_type_labels.update(type_labels)

    if not distributions:
        raise ValueError("At least one test-distribution manifest is required")
    label_type_ratios = {
        label: {
            unit_type: total_type_labels[(unit_type, label)]
            / sum(total_type_labels[(kind, label)] for kind in ("node", "relation"))
            for unit_type in ("node", "relation")
        }
        for label in (0, 1)
    }
    return (
        total_positive / (total_positive + total_negative),
        label_type_ratios,
        distributions,
    )


def _proportional_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate an integer total proportionally using largest remainders."""

    if total < 0 or not weights or any(weight < 0 for weight in weights.values()):
        raise ValueError("Quota total and weights must be non-negative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("At least one quota weight must be positive")
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in priority[:remainder]:
        quotas[key] += 1
    return quotas


def _rebalance_selected_unit_types(
    selected_pair_groups: list[list[dict[str, Any]]],
    selected_negative_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    label_type_ratios: dict[int, dict[str, float]],
    rng: random.Random,
) -> tuple[
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Counter[tuple[str, int]]],
]:
    """Reselect units on the chosen questions to match label × type quotas."""

    for label in (0, 1):
        ratios = label_type_ratios.get(label, {})
        if set(ratios) != {"node", "relation"}:
            raise ValueError("label_type_ratios must define node and relation for labels 0 and 1")

    source_by_question: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_frequencies: Counter[tuple[str, str, int]] = Counter()
    for row in source_rows:
        question_id = str(row["example_id"])
        label = int(row["label"])
        source_by_question[question_id][label].append(row)
        pair_frequencies[(str(row["schema_id"]), str(row["unit_id"]), label)] += 1

    selected_by_graph_label: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for contrast in selected_pair_groups:
        for row in contrast:
            selected_by_graph_label[str(row["graph"])][int(row["label"])].append(row)
    for row in selected_negative_rows:
        selected_by_graph_label[str(row["graph"])][0].append(row)

    graph_names = sorted(selected_by_graph_label)
    target_type_labels: dict[int, dict[str, int]] = {}
    graph_type_quotas: dict[str, dict[int, dict[str, int]]] = {
        graph: {0: {}, 1: {}} for graph in graph_names
    }
    for label in (0, 1):
        label_total = sum(len(selected_by_graph_label[graph][label]) for graph in graph_names)
        target_type_labels[label] = _proportional_quotas(label_total, label_type_ratios[label])
        graph_label_totals = {
            graph: float(len(selected_by_graph_label[graph][label])) for graph in graph_names
        }
        node_quotas = _proportional_quotas(
            target_type_labels[label]["node"], graph_label_totals
        )
        for graph in graph_names:
            graph_type_quotas[graph][label] = {
                "node": node_quotas[graph],
                "relation": len(selected_by_graph_label[graph][label]) - node_quotas[graph],
            }

    replacements: dict[tuple[str, int], dict[str, Any]] = {}
    type_summaries: dict[str, Counter[tuple[str, int]]] = {}
    for graph in graph_names:
        graph_counts: Counter[tuple[str, int]] = Counter()
        for label in (0, 1):
            selected_rows = selected_by_graph_label[graph][label]
            selected_question_ids = {str(row["example_id"]) for row in selected_rows}
            if len(selected_question_ids) != len(selected_rows):
                raise AssertionError(f"{graph}: selected label {label} reuses a question")

            # Preserve one currently selected row for every schema-unit×label
            # pair. The original sampler already guarantees full coverage and
            # one selected row per question and label.
            assigned_questions: set[str] = set()
            covered: set[tuple[str, str, int]] = set()
            for row in selected_rows:
                pair = (str(row["schema_id"]), str(row["unit_id"]), label)
                question_id = str(row["example_id"])
                if pair not in covered and question_id not in assigned_questions:
                    replacement = dict(row)
                    replacement.pop("contrast_pair_id", None)
                    replacements[(question_id, label)] = replacement
                    assigned_questions.add(question_id)
                    covered.add(pair)
                    graph_counts[(_unit_type(replacement), label)] += 1

            quota = graph_type_quotas[graph][label]
            for unit_type in ("node", "relation"):
                if graph_counts[(unit_type, label)] > quota[unit_type]:
                    raise ValueError(
                        f"{graph}: {unit_type}:{label} quota is too small for unit coverage"
                    )

            remaining_questions = selected_question_ids.difference(assigned_questions)
            unit_type_order = sorted(
                ("node", "relation"),
                key=lambda unit_type: sum(
                    any(_unit_type(row) == unit_type for row in source_by_question[qid][label])
                    for qid in remaining_questions
                ),
            )
            for unit_type in unit_type_order:
                needed = quota[unit_type] - graph_counts[(unit_type, label)]
                candidates = [
                    question_id
                    for question_id in sorted(remaining_questions)
                    if any(
                        _unit_type(row) == unit_type
                        for row in source_by_question[question_id][label]
                    )
                ]
                rng.shuffle(candidates)
                if len(candidates) < needed:
                    raise ValueError(
                        f"{graph}: insufficient selected questions for {unit_type}:{label} quota"
                    )
                for question_id in candidates[:needed]:
                    replacement = dict(
                        _choose_row(
                            [
                                row
                                for row in source_by_question[question_id][label]
                                if _unit_type(row) == unit_type
                            ],
                            set(),
                            pair_frequencies,
                            rng,
                        )
                    )
                    replacement.pop("contrast_pair_id", None)
                    replacements[(question_id, label)] = replacement
                    remaining_questions.remove(question_id)
                    graph_counts[(unit_type, label)] += 1
            if remaining_questions:
                raise AssertionError(f"{graph}: not every selected label {label} row was assigned")

        expected = Counter(
            {
                (unit_type, label): graph_type_quotas[graph][label][unit_type]
                for label in (0, 1)
                for unit_type in ("node", "relation")
            }
        )
        if graph_counts != expected:
            raise AssertionError(f"{graph}: unit-type quota validation failed")
        type_summaries[graph] = graph_counts

    rebalanced_pairs: list[list[dict[str, Any]]] = []
    for contrast in selected_pair_groups:
        question_id = str(contrast[0]["example_id"])
        rebuilt = [dict(replacements[(question_id, label)]) for label in (0, 1)]
        for row in rebuilt:
            row["contrast_pair_id"] = question_id
        rng.shuffle(rebuilt)
        rebalanced_pairs.append(rebuilt)

    rebalanced_negatives = [
        dict(replacements[(str(row["example_id"]), 0)]) for row in selected_negative_rows
    ]

    # Coverage must remain identical after changing the chosen unit on a
    # question; this catches any accidental loss of a rare schema unit.
    source_coverage = {
        (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
        for row in source_rows
    }
    selected_coverage = {
        (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
        for contrast in rebalanced_pairs
        for row in contrast
    }
    selected_coverage.update(
        (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))
        for row in rebalanced_negatives
    )
    if selected_coverage != source_coverage:
        raise AssertionError("Unit-type rebalancing changed schema-unit×label coverage")
    return rebalanced_pairs, rebalanced_negatives, type_summaries


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["schema_id"]), str(row["unit_id"]), int(row["label"]))


def _competitor_ids(
    question_rows: list[dict[str, Any]], row: dict[str, Any], cache: dict[tuple[str, str], set[str]]
) -> set[str]:
    """Ids of the units of ``row``'s question that compete with ``row``'s unit."""

    cache_key = (str(row["example_id"]), str(row["unit_id"]))
    if cache_key not in cache:
        units = [
            {**item["unit"], "id": str(item["unit_id"])}
            for item in question_rows
            if isinstance(item.get("unit"), dict) and "schema" in item["unit"]
        ]
        unit = row.get("unit")
        cache[cache_key] = (
            competing_unit_ids({**unit, "id": str(row["unit_id"])}, units)
            if isinstance(unit, dict) and "schema" in unit
            else set()
        )
    return cache[cache_key]


def apply_competitor_pairing(
    pair_groups: list[list[dict[str, Any]]],
    unpaired_negative_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    """Make the NO row of each contrast pair compete with its YES row when possible.

    Only same-type swaps inside a pair's own question are made, so every
    graph x label x unit-type count is unchanged. A swap must also leave every
    schema-unit x label combination covered at least once. Pairs whose question
    offers no suitable competing negative keep their rows.
    """

    by_question: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_question[str(row["example_id"])][int(row["label"])].append(row)
        all_rows[str(row["example_id"])].append(row)
    coverage = Counter(_row_key(row) for pair in pair_groups for row in pair)
    coverage.update(_row_key(row) for row in unpaired_negative_rows)
    cache: dict[tuple[str, str], set[str]] = {}
    stats = {"pairs": len(pair_groups), "already_competing": 0, "swapped": 0, "no_option": 0}

    result: list[list[dict[str, Any]]] = []
    for pair in pair_groups:
        question_id = str(pair[0]["example_id"])
        yes_index = next(index for index, row in enumerate(pair) if int(row["label"]) == 1)
        yes, no = pair[yes_index], pair[1 - yes_index]
        if str(no["unit_id"]) in _competitor_ids(all_rows[question_id], yes, cache):
            stats["already_competing"] += 1
            result.append(pair)
            continue
        options = [
            (candidate_yes, candidate_no)
            for candidate_yes in by_question[question_id][1]
            if _unit_type(candidate_yes) == _unit_type(yes)
            for candidate_no in by_question[question_id][0]
            if _unit_type(candidate_no) == _unit_type(no)
            and str(candidate_no["unit_id"]) in _competitor_ids(all_rows[question_id], candidate_yes, cache)
        ]
        if not options:
            stats["no_option"] += 1
            result.append(pair)
            continue
        rng.shuffle(options)
        # Prefer keeping the YES row: it changes the least coverage.
        options.sort(key=lambda option: _row_key(option[0]) != _row_key(yes))
        old_keys = (_row_key(yes), _row_key(no))
        for candidate_yes, candidate_no in options:
            new_keys = (_row_key(candidate_yes), _row_key(candidate_no))
            delta: Counter[tuple[str, str, int]] = Counter()
            for old, new in zip(old_keys, new_keys, strict=True):
                if old != new:
                    delta[old] -= 1
                    delta[new] += 1
            if any(coverage[key] + change <= 0 for key, change in delta.items() if change < 0):
                continue
            coverage.update(delta)
            rebuilt = [dict(candidate_yes), dict(candidate_no)]
            for row in rebuilt:
                row["contrast_pair_id"] = question_id
            result.append(rebuilt if yes_index == 0 else rebuilt[::-1])
            stats["swapped"] += 1
            break
        else:
            stats["no_option"] += 1
            result.append(pair)
    stats["competing_pairs"] = stats["already_competing"] + stats["swapped"]
    return result, stats


def select_stage_one_rows(
    rows: list[dict[str, Any]],
    target_rows: int,
    seed: int,
    positive_ratio: float = 0.5,
    label_type_ratios: dict[int, dict[str, float]] | None = None,
    pair_strategy: str = "random-coverage",
    cover_all_questions: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select contrast pairs plus unique-question negatives with full coverage.

    With ``cover_all_questions`` every question contributes exactly once (a
    contrast pair or one negative). ``target_rows`` is then ignored: each graph
    gets ``round(questions * ratio / (1 - ratio))`` pairs, so the row count is
    the question count plus the pair count.
    """

    if pair_strategy not in PAIR_STRATEGIES:
        raise ValueError(f"pair_strategy must be one of {PAIR_STRATEGIES}: {pair_strategy!r}")

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
    if cover_all_questions:
        positive_quotas = {
            graph: round(question_counts[graph] * positive_ratio / (1 - positive_ratio))
            for graph in graph_names
        }
        row_quotas = {graph: question_counts[graph] + positive_quotas[graph] for graph in graph_names}
        if min(positive_quotas.values()) < 1:
            raise ValueError("Positive quota is too small to retain every graph")
    else:
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

        remaining_questions = sorted(eligible_questions.difference(used_questions))
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
        if cover_all_questions and used_questions != set(by_question):
            raise AssertionError(f"{graph}: question coverage validation failed")
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
    if label_type_ratios is not None:
        selected_pair_groups, selected_negative_rows, type_summaries = (
            _rebalance_selected_unit_types(
                selected_pair_groups,
                selected_negative_rows,
                rows,
                label_type_ratios,
                rng,
            )
        )
    else:
        type_summaries = {
            graph: Counter(
                (_unit_type(row), int(row["label"]))
                for contrast in selected_pair_groups
                for row in contrast
                if str(row["graph"]) == graph
            )
            for graph in graph_names
        }
        for row in selected_negative_rows:
            type_summaries[str(row["graph"])][(_unit_type(row), 0)] += 1
    if pair_strategy == "hard-competitor":
        before_coverage = {_row_key(row) for contrast in selected_pair_groups for row in contrast}
        before_coverage.update(_row_key(row) for row in selected_negative_rows)
        before_types = Counter(
            (str(row["graph"]), _unit_type(row), int(row["label"]))
            for row in [row for contrast in selected_pair_groups for row in contrast] + selected_negative_rows
        )
        selected_pair_groups, _ = apply_competitor_pairing(
            selected_pair_groups, selected_negative_rows, rows, rng
        )
        paired = [row for contrast in selected_pair_groups for row in contrast]
        after_coverage = {_row_key(row) for row in paired + selected_negative_rows}
        after_types = Counter(
            (str(row["graph"]), _unit_type(row), int(row["label"])) for row in paired + selected_negative_rows
        )
        if after_coverage != before_coverage or after_types != before_types:
            raise AssertionError("Competitor pairing changed schema-unit coverage or label x unit-type counts")
    for graph in graph_names:
        summaries[graph]["unit_type_labels"] = {
            f"{unit_type}:{label}": type_summaries[graph][(unit_type, label)]
            for label in (0, 1)
            for unit_type in ("node", "relation")
        }
    flattened = [row for contrast in selected_pair_groups for row in contrast] + selected_negative_rows
    contrast_pairs = len(selected_pair_groups)
    expected_unique_questions = len(flattened) - contrast_pairs
    if len({str(row["example_id"]) for row in flattened}) != expected_unique_questions:
        raise AssertionError("Selected questions overlap across graph groups or sampling roles")
    return flattened, summaries


def pairing_summary(
    filtered_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> dict[str, int]:
    """Count contrast pairs whose NO unit competes with the YES unit, and YES units with any competitor."""

    all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        all_rows[str(row["example_id"])].append(row)
    cache: dict[tuple[str, str], set[str]] = {}
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered_rows:
        if "contrast_pair_id" in row:
            pairs[str(row["contrast_pair_id"])].append(row)
    competing = 0
    for members in pairs.values():
        yes = next(row for row in members if int(row["label"]) == 1)
        no = next(row for row in members if int(row["label"]) == 0)
        competing += str(no["unit_id"]) in _competitor_ids(all_rows[str(yes["example_id"])], yes, cache)
    yes_with_competitor = sum(
        1
        for row in filtered_rows
        if int(row["label"]) == 1 and _competitor_ids(all_rows[str(row["example_id"])], row, cache)
    )
    return {
        "contrast_pairs": len(pairs),
        "pairs_with_competing_no": competing,
        "yes_rows_with_any_competing_unit_in_question": yes_with_competitor,
    }


def prepare_output_directory(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input-dir and output-dir must differ")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _copy_unchanged_files(input_dir: Path, output_dir: Path) -> None:
    """Copy regular files that are not rewritten by the stage-one filter."""

    for path in input_dir.iterdir():
        if path.is_file() and path.name not in {"selection_train.jsonl", "manifest.json"}:
            shutil.copy2(path, output_dir / path.name)


def _update_train_selection_counts(
    manifest: dict[str, Any], filtered_rows: Iterable[dict[str, Any]]
) -> tuple[int, int]:
    """Update every source-specific ``*/train`` manifest entry."""

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Manifest must contain a counts object")

    train_entries: dict[str, dict[str, Any]] = {}
    for key, entry in counts.items():
        source, separator, split = str(key).rpartition("/")
        if separator and split == "train":
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest count entry {key!r} must be an object")
            train_entries[source] = entry
    if not train_entries:
        raise ValueError("Manifest contains no */train count entries")

    totals: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    for row in filtered_rows:
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("Filtered selector row is missing a source")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Selection label must be 0 or 1: {row['label']!r}")
        totals[source] += 1
        positives[source] += label

    unknown_sources = set(totals) - set(train_entries)
    if unknown_sources:
        raise ValueError(
            f"Filtered rows contain sources absent from manifest train counts: {sorted(unknown_sources)}"
        )

    for source, entry in train_entries.items():
        entry["selection_examples"] = totals[source]
        entry["selection_positive"] = positives[source]
        entry["selection_negative"] = totals[source] - positives[source]
    return sum(positives.values()), sum(totals.values()) - sum(positives.values())


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
    derived_positive_ratio, label_type_ratios, test_distributions = load_test_distribution(
        args.test_manifests
    )
    target_positive_ratio = (
        args.target_positive_ratio
        if args.target_positive_ratio is not None
        else derived_positive_ratio
    )
    filtered_rows, graph_summaries = select_stage_one_rows(
        selection_rows,
        target_rows,
        args.seed,
        target_positive_ratio,
        label_type_ratios,
        pair_strategy=args.pair_strategy,
        cover_all_questions=args.cover_all_questions,
    )
    prepare_output_directory(input_dir, output_dir, args.overwrite)

    _copy_unchanged_files(input_dir, output_dir)
    write_jsonl(output_dir / "selection_train.jsonl", filtered_rows)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filtered_positive, filtered_negative = _update_train_selection_counts(
        manifest, filtered_rows
    )
    contrast_pairs = sum(1 for row in filtered_rows if "contrast_pair_id" in row) // 2
    actual_positive_ratio = filtered_positive / (filtered_positive + filtered_negative)
    actual_type_labels = Counter(
        (_unit_type(row), int(row["label"])) for row in filtered_rows
    )
    manifest["selector_stage1"] = {
        "source_directory": str(input_dir),
        "sampling_seed": args.seed,
        "rows": len(filtered_rows),
        "target_rows_source": (
            "every source question once (--cover-all-questions)"
            if args.cover_all_questions
            else "explicit --target-rows"
            if args.target_rows is not None
            else "generation_train.jsonl row count"
        ),
        "cover_all_questions": args.cover_all_questions,
        "unique_questions": len({str(row["example_id"]) for row in filtered_rows}),
        "contrast_pairs": contrast_pairs,
        "unpaired_negative_rows": len(filtered_rows) - 2 * contrast_pairs,
        "target_positive_ratio": target_positive_ratio,
        "actual_positive_ratio": actual_positive_ratio,
        "target_label_type_ratios": {
            str(label): label_type_ratios[label] for label in (0, 1)
        },
        "actual_unit_type_labels": {
            f"{unit_type}:{label}": actual_type_labels[(unit_type, label)]
            for label in (0, 1)
            for unit_type in ("node", "relation")
        },
        "pair_strategy": args.pair_strategy,
        "pairing": pairing_summary(filtered_rows, selection_rows),
        "test_distribution_weighting": "row_weighted",
        "test_distributions": test_distributions,
        "policy": (
            "Every YES belongs to a same-question YES/NO contrast pair; unique-question NO rows "
            f"are added to reach the configured positive ratio {target_positive_ratio:.6f}; "
            "node/relation priors within each label are row-weighted across the three test sets; "
            "per-graph quotas follow the source question distribution; cover every observed "
            "schema unit and label pair. Pair members are adjacent and precede unpaired negatives."
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
