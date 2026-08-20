"""Adapters that expose the structured training records of each benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .schema import (
    CanonicalSchema,
    canonical_schema,
    from_cypherbench,
    from_mind_the_query,
    from_neo4j_schema_text,
)


SUPPORTED_SOURCES = ("cypherbench", "mind_the_query", "neo4j_text2cypher")


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    source: str
    split: str
    graph: str
    question: str
    cypher: str
    schema: CanonicalSchema
    schema_reference: str
    raw_schema: str | None = None
    normalization_issues: tuple[str, ...] = ()
    normalization_error: str | None = None


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _split_file(root: Path, source_dir: str, split: str) -> Path:
    mapping = {
        "train": "train.json",
        "dev": "dev.json",
        "val": "dev.json",
        "test": "test.json",
    }
    if source_dir == "Mind_the_query" and split == "train":
        if not (root / source_dir / "train.json").exists() and (root / source_dir / "train_val.json").exists():
            mapping["train"] = "train_val.json"
    if split not in mapping:
        raise ValueError(
            f"{source_dir} has no structured '{split}' split. Supported splits: train, dev (or val), test."
        )
    path = root / source_dir / mapping[split]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalise_graph(value: Any) -> str:
    return str(value) if value not in (None, "") else "unknown"


def _load_normalized_schema(
    cache: dict[str, tuple[CanonicalSchema, tuple[str, ...], str | None]],
    cache_key: str,
    source: str,
    graph: str,
    loader: Callable[[], CanonicalSchema],
) -> tuple[CanonicalSchema, tuple[str, ...], str | None]:
    """Load a schema once, retaining normalization failures for later audit."""

    if cache_key in cache:
        return cache[cache_key]

    try:
        schema = loader()
    except Exception as error:  # malformed released schema must not abort a corpus build
        result = (
            canonical_schema(source, graph, (), ()),
            ("normalization_error",),
            f"{type(error).__name__}: {error}",
        )
    else:
        issues = () if schema.units else ("empty_normalized_schema",)
        result = (schema, issues, None)
    cache[cache_key] = result
    return result


def iter_benchmark_examples(
    benchmarks_root: Path, source: str, split: str
) -> Iterator[BenchmarkExample]:
    """Yield examples with a normalized schema, retaining their original split."""

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source '{source}'. Expected one of {SUPPORTED_SOURCES}.")

    directory = {
        "cypherbench": "Cypherbench",
        "mind_the_query": "Mind_the_query",
        "neo4j_text2cypher": "Neo4j_Text2Cypher",
    }[source]
    records_path = _split_file(benchmarks_root, directory, split)
    records = _load_json(records_path)
    schema_cache: dict[str, tuple[CanonicalSchema, tuple[str, ...], str | None]] = {}

    for index, record in enumerate(records):
        graph = _normalise_graph(record.get("graph"))
        raw_schema: str | None = None
        if source == "cypherbench":
            cache_key = graph
            schema_path = benchmarks_root / directory / "graphs" / "schemas" / f"{graph}_schema.json"
            schema, issues, normalization_error = _load_normalized_schema(
                schema_cache,
                cache_key,
                source,
                graph,
                lambda: from_cypherbench(_load_json_object(schema_path), graph),
            )
            question = str(record.get("nl_question", ""))
            schema_reference = str(schema_path.relative_to(benchmarks_root))
        elif source == "mind_the_query":
            cache_key = graph
            schema_path = benchmarks_root / directory / "graphs" / "schemas" / f"{graph}.json"
            schema, issues, normalization_error = _load_normalized_schema(
                schema_cache,
                cache_key,
                source,
                graph,
                lambda: from_mind_the_query(_load_json_object(schema_path), graph),
            )
            question = str(record.get("question", record.get("nl_question", "")))
            schema_reference = str(schema_path.relative_to(benchmarks_root))
        else:
            raw_schema = str(record.get("schema", ""))
            # Identical text can occur under different graph names. The graph
            # remains part of a canonical schema ID, so it must also scope the
            # cache entry.
            cache_key = f"{graph}\0{raw_schema}"
            schema, issues, normalization_error = _load_normalized_schema(
                schema_cache,
                cache_key,
                source,
                graph,
                lambda: from_neo4j_schema_text(raw_schema, graph),
            )
            question = str(record.get("nl_question", record.get("question", "")))
            schema_reference = f"{records_path.name}[{index}].schema"

        example_key = record.get("qid", record.get("instance_id", index))
        yield BenchmarkExample(
            # Mind-the-Query reuses numeric qids across graphs, so graph must
            # be part of the corpus key. Including it for every source also
            # makes IDs self-describing when datasets are mixed.
            example_id=f"{source}:{split}:{graph}:{example_key}",
            source=source,
            split=split,
            graph=graph,
            question=question.strip(),
            cypher=str(record.get("gold_cypher", "")).strip(),
            schema=schema,
            schema_reference=schema_reference,
            raw_schema=raw_schema,
            normalization_issues=issues,
            normalization_error=normalization_error,
        )
