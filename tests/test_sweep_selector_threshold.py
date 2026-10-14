import json
from pathlib import Path

import pytest

from scripts.sweep_selector_threshold import format_table, load_rows, parse_thresholds, sweep


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_files(tmp_path: Path, *, with_p_yes: bool = True) -> tuple[Path, Path]:
    # q1: gold {a}; q2: gold {c}. Units b and d are negatives.
    units = [("q1", "a", 1, 0.9), ("q1", "b", 0, 0.6), ("q2", "c", 1, 0.45), ("q2", "d", 0, 0.1)]
    gold = [{"id": f"{q}:{u}", "example_id": q, "label": label} for q, u, label, _ in units]
    predictions = [
        {
            "id": f"{q}:{u}",
            "predicted_label": "YES" if p >= 0.5 else "NO",
            **({"p_yes": p} if with_p_yes else {}),
        }
        for q, u, _, p in units
    ]
    write_jsonl(tmp_path / "gold.jsonl", gold)
    write_jsonl(tmp_path / "pred.jsonl", predictions)
    return tmp_path / "pred.jsonl", tmp_path / "gold.jsonl"


def test_parse_thresholds_supports_ranges_and_lists() -> None:
    assert parse_thresholds("0.5:0.7:0.1") == [0.5, 0.6, 0.7]
    assert parse_thresholds("0.4, 0.8") == [0.4, 0.8]
    with pytest.raises(ValueError, match="step > 0"):
        parse_thresholds("0.5:0.4:0.1")


def test_sweep_trades_precision_for_recall_and_includes_greedy_row(tmp_path: Path) -> None:
    predictions, gold = fixture_files(tmp_path)
    results = {row["threshold"]: row for row in sweep(load_rows(predictions, gold), [0.4, 0.7])}

    greedy = results["greedy"]  # predicts a, b -> TP 1, FP 1, FN 1
    assert greedy["precision"] == pytest.approx(50.0)
    assert greedy["recall"] == pytest.approx(50.0)
    assert greedy["extra_per_question"] == pytest.approx(0.5)
    assert greedy["questions_missing_gold"] == pytest.approx(50.0)
    assert greedy["questions_exact"] == pytest.approx(0.0)

    loose = results[0.4]  # predicts a, b, c -> TP 2, FP 1, FN 0
    assert loose["recall"] == pytest.approx(100.0)
    assert loose["precision"] == pytest.approx(200 / 3)
    assert loose["questions_exact"] == pytest.approx(50.0)

    strict = results[0.7]  # predicts a only -> TP 1, FP 0, FN 1
    assert strict["precision"] == pytest.approx(100.0)
    assert strict["recall"] == pytest.approx(50.0)
    assert strict["extra_per_question"] == pytest.approx(0.0)
    assert "greedy" in format_table(list(results.values()))


def test_load_rows_requires_p_yes_and_matching_files(tmp_path: Path) -> None:
    predictions, gold = fixture_files(tmp_path, with_p_yes=False)
    with pytest.raises(ValueError, match="no p_yes"):
        load_rows(predictions, gold)

    predictions, gold = fixture_files(tmp_path)
    kept = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()][:3]
    write_jsonl(predictions, kept)
    with pytest.raises(ValueError, match="No prediction for gold row"):
        load_rows(predictions, gold)
