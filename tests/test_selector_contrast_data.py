import pytest

from distillation.sampling import selector_droppable_batch_range
from scripts.filter_selector_stage1 import (
    _copy_unchanged_files,
    _update_train_selection_counts,
    select_stage_one_rows,
)
from scripts.prepare_multitask_prompts import interleave_without_replacement, parse_args


def _source_row(graph: str, question: int, unit: str, label: int) -> dict[str, object]:
    return {
        "example_id": f"{graph}:q{question}",
        "graph": graph,
        "schema_id": f"schema:{graph}",
        "unit_id": unit,
        "label": label,
    }


def test_prepare_default_batch_size_matches_training(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prepare_multitask_prompts.py"])
    assert parse_args().batch_size == 2


def test_manifest_counts_support_single_non_cypherbench_source() -> None:
    manifest = {
        "counts": {
            "neo4j_text2cypher/train": {
                "selection_examples": 100,
                "selection_positive": 25,
                "selection_negative": 75,
            },
            "neo4j_text2cypher/test": {"selection_examples": 10},
        }
    }
    rows = [
        {"source": "neo4j_text2cypher", "label": 1},
        {"source": "neo4j_text2cypher", "label": 0},
        {"source": "neo4j_text2cypher", "label": 0},
    ]

    positive, negative = _update_train_selection_counts(manifest, rows)

    assert (positive, negative) == (1, 2)
    assert manifest["counts"]["neo4j_text2cypher/train"] == {
        "selection_examples": 3,
        "selection_positive": 1,
        "selection_negative": 2,
    }
    assert manifest["counts"]["neo4j_text2cypher/test"] == {"selection_examples": 10}


def test_manifest_counts_are_updated_per_source() -> None:
    manifest = {
        "counts": {
            "cypherbench/train": {},
            "mind_the_query/train": {},
            "neo4j_text2cypher/train": {},
        }
    }
    rows = [
        {"source": "cypherbench", "label": 1},
        {"source": "cypherbench", "label": 0},
        {"source": "mind_the_query", "label": 0},
    ]

    _update_train_selection_counts(manifest, rows)

    assert manifest["counts"]["cypherbench/train"] == {
        "selection_examples": 2,
        "selection_positive": 1,
        "selection_negative": 1,
    }
    assert manifest["counts"]["mind_the_query/train"] == {
        "selection_examples": 1,
        "selection_positive": 0,
        "selection_negative": 1,
    }
    assert manifest["counts"]["neo4j_text2cypher/train"] == {
        "selection_examples": 0,
        "selection_positive": 0,
        "selection_negative": 0,
    }


def test_copy_unchanged_files_skips_subdirectories_and_rewritten_files(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "generation_train.jsonl").write_text("generation\n", encoding="utf-8")
    (input_dir / "selection_train.jsonl").write_text("selection\n", encoding="utf-8")
    (input_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    nested = input_dir / "nested"
    nested.mkdir()
    (nested / "artifact.txt").write_text("nested\n", encoding="utf-8")

    _copy_unchanged_files(input_dir, output_dir)

    assert (output_dir / "generation_train.jsonl").read_text(encoding="utf-8") == "generation\n"
    assert not (output_dir / "selection_train.jsonl").exists()
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "nested").exists()


def test_stage_one_builds_adjacent_same_question_contrasts_with_coverage() -> None:
    rows = []
    for graph in ("art", "soccer"):
        rows.extend(
            [
                _source_row(graph, 1, "node:A", 1),
                _source_row(graph, 1, "node:B", 0),
                _source_row(graph, 2, "node:B", 1),
                _source_row(graph, 2, "node:A", 0),
                _source_row(graph, 3, "node:A", 1),
                _source_row(graph, 3, "node:A", 0),
            ]
        )

    selected, summaries = select_stage_one_rows(
        rows, target_rows=8, seed=42, positive_ratio=0.5
    )

    assert len(selected) == 8
    assert all(summary["contrast_pairs"] == 2 for summary in summaries.values())
    for offset in range(0, len(selected), 2):
        first, second = selected[offset : offset + 2]
        assert first["example_id"] == second["example_id"]
        assert first["contrast_pair_id"] == second["contrast_pair_id"]
        assert {first["label"], second["label"]} == {0, 1}

    expected_coverage = {
        (row["schema_id"], row["unit_id"], row["label"]) for row in rows
    }
    actual_coverage = {
        (row["schema_id"], row["unit_id"], row["label"]) for row in selected
    }
    assert actual_coverage == expected_coverage


def test_stage_one_matches_target_prior_without_breaking_contrast_pairs() -> None:
    rows = []
    for graph in ("art", "soccer"):
        for question in range(1, 7):
            rows.extend(
                [
                    _source_row(graph, question, "node:A", 1),
                    _source_row(graph, question, "node:B", 0),
                ]
            )

    selected, summaries = select_stage_one_rows(
        rows, target_rows=12, seed=42, positive_ratio=1 / 3
    )

    assert sum(row["label"] == 1 for row in selected) == 4
    assert sum(row["label"] == 0 for row in selected) == 8
    assert sum("contrast_pair_id" not in row for row in selected) == 4
    assert len({row["example_id"] for row in selected}) == 8
    assert all(summary["contrast_pairs"] == 2 for summary in summaries.values())
    assert all(summary["unpaired_negative_rows"] == 2 for summary in summaries.values())

    paired = [row for row in selected if "contrast_pair_id" in row]
    for offset in range(0, len(paired), 2):
        first, second = paired[offset : offset + 2]
        assert first["contrast_pair_id"] == second["contrast_pair_id"]
        assert {first["label"], second["label"]} == {0, 1}


def test_stage_one_cover_all_questions_uses_every_question_once() -> None:
    rows = []
    for graph in ("art", "soccer"):
        for question in range(1, 8):
            rows.extend(
                [
                    _source_row(graph, question, "node:A", 1),
                    _source_row(graph, question, "node:B", 0),
                ]
            )

    selected, summaries = select_stage_one_rows(
        rows, target_rows=1, seed=42, positive_ratio=0.3, cover_all_questions=True
    )

    # round(7 * 0.3 / 0.7) = 3 pairs per graph; rows = 7 questions + 3 pairs.
    assert all(summary["contrast_pairs"] == 3 for summary in summaries.values())
    assert all(summary["rows"] == 10 for summary in summaries.values())
    assert {row["example_id"] for row in selected} == {row["example_id"] for row in rows}
    assert sum(row["label"] == 1 for row in selected) == 6
    assert len(selected) == 20


def test_train_interleave_puts_selector_surplus_in_full_trailing_batches() -> None:
    import random

    def selector_rows(pairs: int, negatives: int) -> list[dict[str, object]]:
        rows = [
            {"task": "selector", "example_id": f"q{pair}", "contrast_pair_id": f"q{pair}", "label": label}
            for pair in range(pairs)
            for label in ("YES", "NO")
        ]
        rows.extend({"task": "selector", "example_id": f"n{index}", "label": "NO"} for index in range(negatives))
        return rows

    generators = [{"task": "generator", "example_id": f"g{index}"} for index in range(7)]
    selectors = selector_rows(pairs=3, negatives=7)
    for batch_size in (2, 4):
        result = interleave_without_replacement(generators, selectors, batch_size=batch_size, rng=random.Random(42))
        assert len(result) == len(generators) + len(selectors)
        assert len({id(row) for row in result}) == len(result)
        capacity = batch_size // 2
        mixed_batches = len(generators) // capacity
        batches = [result[start : start + batch_size] for start in range(0, len(result), batch_size)]
        assert all(len(batch) == batch_size for batch in batches[:-1])
        for batch in batches[:mixed_batches]:
            assert sum(row["task"] == "generator" for row in batch) == capacity
        contrast_positions = [position for position, row in enumerate(result) if "contrast_pair_id" in row]
        assert max(contrast_positions) // batch_size < mixed_batches

        start, count = selector_droppable_batch_range(
            batch_size=batch_size, contrast_pairs=3, unpaired_negatives=7, total_rows=len(result)
        )
        # Droppable batches start right after the last contrast batch and run to the last full batch.
        assert start == max(contrast_positions) // batch_size + 1
        assert start + count == len(result) // batch_size
        assert all("contrast_pair_id" not in row for batch in batches[start:] for row in batch)

    with pytest.raises(ValueError, match="fit in the mixed batches"):
        interleave_without_replacement(
            generators[:3], selector_rows(pairs=2, negatives=5), batch_size=2, rng=random.Random(42)
        )


def test_train_interleave_keeps_contrast_pair_order_without_duplication() -> None:
    generators = [{"task": "generator", "example_id": f"g{index}"} for index in range(6)]
    selectors = [
        {
            "task": "selector",
            "example_id": f"q{pair}",
            "contrast_pair_id": f"q{pair}",
            "label": label,
        }
        for pair in range(2)
        for label in ("YES", "NO")
    ]
    selectors.extend(
        {"task": "selector", "example_id": f"negative{index}", "label": "NO"}
        for index in range(2)
    )

    import random

    result = interleave_without_replacement(generators, selectors, batch_size=2, rng=random.Random(42))
    selector_result = [row for row in result if row["task"] == "selector"]
    assert len(result) == 12
    assert len({id(row) for row in result}) == len(result)
    for offset in range(0, 4, 2):
        first, second = selector_result[offset : offset + 2]
        assert first["contrast_pair_id"] == second["contrast_pair_id"]
        assert {first["label"], second["label"]} == {"YES", "NO"}
    assert all("contrast_pair_id" not in row for row in selector_result[4:])
    assert all(row["label"] == "NO" for row in selector_result[4:])


def test_train_interleave_keeps_pairs_inside_larger_prepared_batches() -> None:
    import random

    generators = [{"task": "generator", "example_id": f"g{index}"} for index in range(16)]
    selectors = [
        {
            "task": "selector",
            "example_id": f"q{pair}",
            "contrast_pair_id": f"q{pair}",
            "label": label,
        }
        for pair in range(4)
        for label in ("YES", "NO")
    ]

    for batch_size in (4, 8):
        result = interleave_without_replacement(
            generators,
            selectors,
            batch_size=batch_size,
            rng=random.Random(42),
        )
        pair_batches: dict[str, set[int]] = {}
        for position, row in enumerate(result):
            pair_id = row.get("contrast_pair_id")
            if pair_id is not None:
                pair_batches.setdefault(str(pair_id), set()).add(position // batch_size)
        assert pair_batches
        assert all(len(batch_indices) == 1 for batch_indices in pair_batches.values())


def test_train_interleave_handles_equal_task_counts_with_partial_larger_batch() -> None:
    import random

    generators = [{"task": "generator", "example_id": f"g{index}"} for index in range(7)]
    selectors = [
        {
            "task": "selector",
            "example_id": f"q{pair}",
            "contrast_pair_id": f"q{pair}",
            "label": label,
        }
        for pair in range(2)
        for label in ("YES", "NO")
    ]
    selectors.extend(
        {"task": "selector", "example_id": f"negative{index}", "label": "NO"}
        for index in range(3)
    )

    for batch_size in (4, 8):
        result = interleave_without_replacement(
            generators,
            selectors,
            batch_size=batch_size,
            rng=random.Random(42),
        )

        assert len(result) == len(generators) + len(selectors)
        assert {row["example_id"] for row in result} == {
            row["example_id"] for row in generators + selectors
        }


def test_stage_one_graph_quotas_follow_source_question_distribution() -> None:
    rows = []
    for graph, questions in (("art", 6), ("soccer", 2)):
        for question in range(questions):
            rows.extend(
                [
                    _source_row(graph, question, "node:A", 1),
                    _source_row(graph, question, "node:B", 0),
                ]
            )

    selected, summaries = select_stage_one_rows(
        rows, target_rows=16, seed=42, positive_ratio=0.5
    )

    assert len(selected) == 16
    assert summaries["art"]["rows"] == 12
    assert summaries["soccer"]["rows"] == 4


def test_stage_one_matches_joint_label_and_unit_type_distribution() -> None:
    rows = []
    for graph in ("art", "soccer"):
        for question in range(10):
            rows.extend(
                [
                    _source_row(graph, question, "node:A", 1),
                    _source_row(graph, question, "relation:A|r|B", 1),
                    _source_row(graph, question, "node:B", 0),
                    _source_row(graph, question, "relation:B|s|A", 0),
                ]
            )

    selected, summaries = select_stage_one_rows(
        rows,
        target_rows=16,
        seed=42,
        positive_ratio=0.25,
        label_type_ratios={
            1: {"node": 0.5, "relation": 0.5},
            0: {"node": 0.25, "relation": 0.75},
        },
    )

    counts = {}
    for unit_type in ("node", "relation"):
        for label in (0, 1):
            counts[(unit_type, label)] = sum(
                row["unit_id"].startswith(f"{unit_type}:") and row["label"] == label
                for row in selected
            )
    assert counts == {
        ("node", 1): 2,
        ("relation", 1): 2,
        ("node", 0): 3,
        ("relation", 0): 9,
    }
    assert all(summary["schema_unit_label_pairs_covered"] == 4 for summary in summaries.values())


def _unit_row(graph: str, question: int, unit_id: str, label: int, schema: dict, kind: str) -> dict[str, object]:
    row = _source_row(graph, question, unit_id, label)
    row["unit"] = {"id": unit_id, "kind": kind, "schema": schema, "text": unit_id}
    return row


def _rel(source: str, name: str, target: str) -> dict[str, str]:
    return {"source": source, "type": name, "target": target, "properties": {}}


def _relation_row(graph: str, question: int, label: int, source: str, name: str, target: str) -> dict[str, object]:
    return _unit_row(
        graph, question, f"relation:{source}|{name}|{target}", label, _rel(source, name, target), "relation"
    )


def _competitor_pool(graph: str, questions: int) -> list[dict[str, object]]:
    """Each question keeps rel:foundedBy (YES); hasCEO competes with it, basedIn does not."""

    rows = []
    for question in range(questions):
        rows.extend(
            [
                _relation_row(graph, question, 1, "Company", "foundedBy", "Person"),
                _relation_row(graph, question, 0, "Company", "hasCEO", "Person"),
                _relation_row(graph, question, 0, "Company", "basedIn", "Country"),
                _relation_row(graph, question, 0, "Company", "operatesIn", "Industry"),
            ]
        )
    return rows


def _pairs(selected: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    paired = [row for row in selected if "contrast_pair_id" in row]
    return [paired[offset : offset + 2] for offset in range(0, len(paired), 2)]


def _competing_pairs(selected: list[dict[str, object]]) -> int:
    return sum(
        1
        for first, second in _pairs(selected)
        if (first if first["label"] == 0 else second)["unit_id"] == "relation:Company|hasCEO|Person"
    )


def test_hard_competitor_pairing_makes_most_no_rows_compete_with_the_yes_row() -> None:
    rows = _competitor_pool("company", 30)

    baseline, _ = select_stage_one_rows(rows, target_rows=30, seed=3, positive_ratio=1 / 3)
    hard, _ = select_stage_one_rows(
        rows, target_rows=30, seed=3, positive_ratio=1 / 3, pair_strategy="hard-competitor"
    )

    assert len(_pairs(hard)) == 10
    assert _competing_pairs(hard) > _competing_pairs(baseline)
    # Coverage still needs basedIn and operatesIn as NO rows somewhere, so not every pair can switch.
    assert _competing_pairs(hard) >= 7


def test_hard_competitor_pairing_preserves_counts_and_coverage() -> None:
    rows = []
    for graph in ("company", "politics"):
        rows.extend(_competitor_pool(graph, 12))
    label_type_ratios = {1: {"node": 0.0, "relation": 1.0}, 0: {"node": 0.0, "relation": 1.0}}

    baseline, _ = select_stage_one_rows(
        rows, target_rows=24, seed=5, positive_ratio=0.5, label_type_ratios=label_type_ratios
    )
    hard, _ = select_stage_one_rows(
        rows,
        target_rows=24,
        seed=5,
        positive_ratio=0.5,
        label_type_ratios=label_type_ratios,
        pair_strategy="hard-competitor",
    )

    def counts(selected: list[dict[str, object]]) -> dict:
        return {
            (row["graph"], row["label"], row["unit_id"].split(":")[0]): 1
            for row in selected
        } | {"rows": len(selected), "labels": sum(row["label"] for row in selected)}

    assert counts(hard) == counts(baseline)
    coverage = lambda selected: {(r["schema_id"], r["unit_id"], r["label"]) for r in selected}  # noqa: E731
    assert coverage(hard) == coverage(baseline) == {(r["schema_id"], r["unit_id"], r["label"]) for r in rows}
    assert len({row["example_id"] for row in hard}) == len({row["example_id"] for row in baseline})


def test_hard_competitor_pairing_leaves_pairs_without_a_competitor_unchanged() -> None:
    rows = []
    for question in range(6):
        rows.extend(
            [
                _relation_row("art", question, 1, "Company", "foundedBy", "Person"),
                _relation_row("art", question, 0, "Company", "basedIn", "Country"),
            ]
        )

    random_pairs, _ = select_stage_one_rows(rows, target_rows=6, seed=1, positive_ratio=0.5)
    hard_pairs, _ = select_stage_one_rows(
        rows, target_rows=6, seed=1, positive_ratio=0.5, pair_strategy="hard-competitor"
    )

    assert hard_pairs == random_pairs


def test_default_pair_strategy_is_random_coverage_and_rejects_unknown() -> None:
    rows = _competitor_pool("company", 6)
    default, _ = select_stage_one_rows(rows, target_rows=6, seed=2, positive_ratio=0.5)
    explicit, _ = select_stage_one_rows(
        rows, target_rows=6, seed=2, positive_ratio=0.5, pair_strategy="random-coverage"
    )
    assert default == explicit
    with pytest.raises(ValueError, match="pair_strategy"):
        select_stage_one_rows(rows, target_rows=6, seed=2, positive_ratio=0.5, pair_strategy="nope")


@pytest.mark.parametrize("pair_strategy", ["random-coverage", "hard-competitor"])
def test_stage_one_is_reproducible_across_hash_seeds(pair_strategy: str) -> None:
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    program = (
        "import json, sys\n"
        "from scripts.filter_selector_stage1 import select_stage_one_rows\n"
        "from tests.test_selector_contrast_data import _competitor_pool\n"
        "rows = _competitor_pool('company', 30) + _competitor_pool('politics', 30)\n"
        "selected, _ = select_stage_one_rows(rows, 60, 7, 1 / 3, pair_strategy=sys.argv[1])\n"
        "print(json.dumps([[r['example_id'], r['unit_id'], r['label']] for r in selected]))\n"
    )
    root = Path(__file__).resolve().parents[1]
    outputs = []
    for hash_seed in ("0", "1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": f"{root}{os.pathsep}{root / 'src'}"}
        completed = subprocess.run(
            [sys.executable, "-c", program, pair_strategy], cwd=root, env=env, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert outputs[0] == outputs[1] == outputs[2]
