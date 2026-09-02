from scripts.filter_selector_stage1 import select_stage_one_rows
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
