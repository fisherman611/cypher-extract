from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Protocol


class QueryRunner(Protocol):
    def run_query(self, cypher: str, *, timeout: int | None = None, **parameters: Any) -> list[dict[str, Any]]: ...


def _hashable(value: Any) -> Any:
    """Normalize Neo4j/Python values before comparing query denotations."""
    if value is None or isinstance(value, str | int | float | bool | bytes):
        return value

    # Neo4j temporal and spatial values expose stable public representations.
    if value.__class__.__module__.startswith("neo4j.time"):
        iso_format = getattr(value, "iso_format", None)
        return iso_format() if callable(iso_format) else str(value)
    if value.__class__.__module__.startswith("neo4j.spatial"):
        return tuple(value)

    # Node and Relationship are Mapping-like in the driver, while Path is not.
    if value.__class__.__module__.startswith("neo4j.graph"):
        if hasattr(value, "nodes") and hasattr(value, "relationships"):
            return (
                "path",
                tuple(_hashable(item) for item in value.nodes),
                tuple(_hashable(item) for item in value.relationships),
            )
        properties = tuple(sorted((str(key), _hashable(item)) for key, item in dict(value).items()))
        return (
            value.__class__.__name__,
            getattr(value, "element_id", None),
            tuple(sorted(getattr(value, "labels", ()))),
            properties,
        )
    if isinstance(value, Mapping):
        entries = ((_hashable(key), _hashable(item)) for key, item in value.items())
        return tuple(sorted(entries, key=repr))
    if isinstance(value, list | set | frozenset):
        # Match CypherKD: collection-valued cells are compared without order.
        return tuple(sorted((_hashable(item) for item in value), key=repr))
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"Unhashable result value: {type(value)!r}") from error
    return value


def _rows(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    if not records:
        return []
    keys = tuple(records[0])
    if any(set(record) != set(keys) for record in records):
        raise ValueError("A query returned inconsistent columns across records")
    return [tuple(_hashable(record[key]) for key in keys) for record in records]


def _unordered_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(sorted(row, key=lambda item: (type(item).__name__, repr(item))))


def _equivalent_results(
    predicted: list[dict[str, Any]],
    target: list[dict[str, Any]],
    *,
    order_matters: bool,
) -> bool:
    left, right = _rows(target), _rows(predicted)
    if not left or not right:
        return left == right
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        return False
    if [*_map_rows(left, _unordered_row, order_matters)] != [*_map_rows(right, _unordered_row, order_matters)]:
        return False

    width = len(left[0])
    for permutation in itertools.permutations(range(width)):
        permuted = [tuple(row[index] for index in permutation) for row in right]
        if order_matters:
            if left == permuted:
                return True
        elif Counter(left) == Counter(permuted):
            return True
    return False


def _map_rows(rows: list[tuple[Any, ...]], transform, order_matters: bool):
    normalized = [transform(row) for row in rows]
    return normalized if order_matters else sorted(normalized, key=repr)


def execution_accuracy(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: QueryRunner,
    timeout: int = 120,
) -> float:
    """Return 1 when predicted and gold queries have equivalent results."""
    if pred_cypher.strip() == target_cypher.strip():
        return 1.0
    try:
        target = neo4j_connector.run_query(target_cypher, timeout=timeout)
        predicted = neo4j_connector.run_query(pred_cypher, timeout=timeout)
        equivalent = _equivalent_results(
            predicted,
            target,
            order_matters=bool(re.search(r"\bORDER\s+BY\b", target_cypher, re.IGNORECASE)),
        )
    except Exception:
        return 0.0
    return float(equivalent)


def executable(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: QueryRunner,
    timeout: int = 120,
) -> float:
    """Return 1 when Neo4j can execute the predicted query."""
    del target_cypher
    try:
        neo4j_connector.run_query(pred_cypher, timeout=timeout)
    except Exception:
        return 0.0
    return 1.0


_CLAUSE = re.compile(
    r"\b(OPTIONAL\s+MATCH|ORDER\s+BY|MATCH|WHERE|RETURN|UNION|WITH|CREATE|SET|DELETE|MERGE|UNWIND|LIMIT|SKIP|FOREACH|CALL|YIELD)\b",
    re.IGNORECASE,
)


def _clauses(query: str) -> list[str]:
    matches = list(_CLAUSE.finditer(query))
    return [query[item.start() : matches[index + 1].start() if index + 1 < len(matches) else len(query)].strip()
            for index, item in enumerate(matches)]


def _match_prefix(query: str) -> str | None:
    if not re.match(r"^\s*MATCH\b", query, re.IGNORECASE):
        return None
    selected: list[str] = []
    for clause in _clauses(query):
        keyword = clause.upper()
        if keyword.startswith(("MATCH", "OPTIONAL MATCH", "WHERE")):
            selected.append(clause)
        elif keyword.startswith("WITH") and not re.search(r"\sAS\s", clause, re.IGNORECASE):
            selected.append("WITH *")
        else:
            break
    while selected and selected[-1].upper().startswith("WITH"):
        selected.pop()
    return " ".join(selected) or None


def _name_anonymous_patterns(query: str) -> str:
    node_number = relationship_number = 0

    def node(match: re.Match[str]) -> str:
        nonlocal node_number
        result = f"(ps_n{node_number}:{match.group(1)}{match.group(2) or ''})"
        node_number += 1
        return result

    def relationship(match: re.Match[str]) -> str:
        nonlocal relationship_number
        result = f"[ps_r{relationship_number}{match.group(1)}]"
        relationship_number += 1
        return result

    query = re.sub(r"\[(:[^\]]+)\]", relationship, query)
    return re.sub(r"\(:([A-Za-z_][A-Za-z0-9_]*)(\s*\{.*?\})?\)", node, query)


def _split_by_union(query: str) -> list[str]:
    """Split top-level UNIONs and UNIONs inside the CALL shape used by the benchmark."""
    stripped = query.strip()
    if re.match(r"^CALL\b", stripped, re.IGNORECASE):
        inner = re.search(
            r"CALL\s*\{(.*?)\}\s*(?:WITH|RETURN|WHERE|UNWIND)\b",
            stripped,
            re.IGNORECASE | re.DOTALL,
        )
        if inner:
            stripped = inner.group(1)
    return [part.strip() for part in re.split(r"\bUNION(?:\s+ALL)?\b", stripped, flags=re.IGNORECASE)]


def _provenance_query(query: str, return_name: str) -> str:
    parts: list[str] = []
    for union_part in _split_by_union(query):
        prefix = _match_prefix(union_part.strip())
        if not prefix:
            continue
        prefix = _name_anonymous_patterns(prefix)
        nodes = sorted(set(re.findall(r"\(([A-Za-z_]\w*)", re.sub(r"\{[^}]*\}", "{}", prefix))))
        expression = " + ".join(f"collect(DISTINCT elementId({name}))" for name in nodes) or "[]"
        parts.append(f"{prefix} WITH {expression} AS ids UNWIND ids AS id RETURN id AS {return_name}")
    return " UNION ".join(parts) if parts else f"UNWIND [] AS id RETURN id AS {return_name}"


def provenance_subgraph_jaccard_similarity(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: QueryRunner,
    timeout: int = 120,
) -> float:
    """Jaccard similarity between nodes touched by predicted and gold MATCH clauses."""
    if pred_cypher.strip() == target_cypher.strip():
        return 1.0
    try:
        target = neo4j_connector.run_query(_provenance_query(target_cypher, "target_id"), timeout=timeout)
        predicted = neo4j_connector.run_query(_provenance_query(pred_cypher, "predicted_id"), timeout=timeout)
        target_ids = {row["target_id"] for row in target}
        predicted_ids = {row["predicted_id"] for row in predicted}
    except Exception:
        return 0.0
    union = target_ids | predicted_ids
    return len(target_ids & predicted_ids) / len(union) if union else 0.0


METRICS = {
    "execution_accuracy": execution_accuracy,
    "psjs": provenance_subgraph_jaccard_similarity,
    "executable": executable,
}
