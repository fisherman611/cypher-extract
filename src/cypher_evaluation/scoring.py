from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from distillation.metrics import extract_cypher

from .metrics import METRICS, QueryRunner

DEFAULT_METRICS = tuple(METRICS)


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array or JSONL input, got {type(payload).__name__}")
    return payload


def score_records(
    records: Iterable[dict[str, Any]],
    connector: QueryRunner,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    predicted_key: str = "predicted_cypher",
    target_key: str = "reference_cypher",
    timeout: int = 120,
    desc: str = "Evaluating Cypher",
) -> list[dict[str, Any]]:
    unknown = set(metrics) - METRICS.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    scored: list[dict[str, Any]] = []
    rows = list(records)
    for record in tqdm(rows, desc=desc, unit="query"):
        predicted = extract_cypher(str(record.get(predicted_key) or "")).removesuffix("<end_of_turn>").strip()
        target = str(record.get(target_key) or "").strip()
        if not target:
            raise ValueError(f"Record {record.get('id', '<unknown>')!r} has no target Cypher in {target_key!r}")
        values = {
            name: METRICS[name](predicted, target, connector, timeout=timeout)
            for name in metrics
        }
        scored.append({**record, predicted_key: predicted, "metrics": values})
    return scored


def aggregate_scores(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    metric_rows = [row.get("metrics", row.get("cypher_metrics", {})) for row in rows]
    names = sorted({name for metrics in metric_rows for name in metrics})
    return {
        "count": len(rows),
        "overall": {
            name: (sum(float(metrics[name]) for metrics in metric_rows if name in metrics) /
                   sum(name in metrics for metrics in metric_rows))
            if any(name in metrics for metrics in metric_rows) else math.nan
            for name in names
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
