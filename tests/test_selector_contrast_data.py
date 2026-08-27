from scripts.filter_selector_stage1 import select_stage_one_rows
from scripts.prepare_multitask_prompts import interleave_without_replacement


def _source_row(graph: str, question: int, unit: str, label: int) -> dict[str, object]:
    return {
        "example_id": f"{graph}:q{question}",
        "graph": graph,
        "schema_id": f"schema:{graph}",
        "unit_id": unit,
        "label": label,
    }


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

    selected, summaries = select_stage_one_rows(rows, target_rows=8, seed=42)

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

    import random

    result = interleave_without_replacement(generators, selectors, batch_size=2, rng=random.Random(42))
    selector_result = [row for row in result if row["task"] == "selector"]
    assert len(result) == 10
    assert len({id(row) for row in result}) == len(result)
    for offset in range(0, len(selector_result), 2):
        first, second = selector_result[offset : offset + 2]
        assert first["contrast_pair_id"] == second["contrast_pair_id"]
        assert {first["label"], second["label"]} == {"YES", "NO"}
