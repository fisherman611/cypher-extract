from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .metrics import METRICS
from .neo4j import Neo4jConnector
from .scoring import aggregate_scores, read_records, score_records, write_jsonl

CONNECTOR_NAMES = ("cypherbench-db", "mind-the-query-db")
DEFAULT_CONNECTOR_NAME = "cypherbench-db"
DEFAULT_GRAPH = "nba"


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate generated Cypher against a Neo4j database")
    parser.add_argument("--input", type=Path, required=True, help="JSON array or generator_predictions.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path; defaults to the matching results/evaluation/.../<graph>/ folder",
    )
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    parser.add_argument("--username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument(
        "--name",
        choices=CONNECTOR_NAMES,
        default=DEFAULT_CONNECTOR_NAME,
        help=f"Logical connector/dataset name (default: {DEFAULT_CONNECTOR_NAME})",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database; defaults to --graph with underscores replaced by dots",
    )
    parser.add_argument("--predicted-key", default="predicted_cypher")
    parser.add_argument("--target-key", default="reference_cypher")
    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH,
        help=f"Only score records whose graph field has this value (default: {DEFAULT_GRAPH})",
    )
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRICS), default=tuple(METRICS))
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def resolve_database(database: str | None, graph: str) -> str:
    return database or graph.replace("_", ".")


def resolve_output_path(input_path: Path, output_path: Path | None, graph: str) -> Path:
    if output_path is not None:
        return output_path

    parts = input_path.parts
    try:
        inference_index = parts.index("inference")
    except ValueError:
        return input_path.parent / "evaluation" / graph / "cypher_scores.jsonl"
    return Path(
        *parts[:inference_index],
        "evaluation",
        *parts[inference_index + 1 : -1],
        graph,
        "cypher_scores.jsonl",
    )


def main() -> None:
    args = parse_args()
    if not args.password:
        raise SystemExit("Set NEO4J_PASSWORD in .env or pass --password")
    records = read_records(args.input)
    if args.graph:
        records = [row for row in records if row.get("graph") == args.graph]
    database = resolve_database(args.database, args.graph)
    output_path = resolve_output_path(args.input, args.output, args.graph)
    with Neo4jConnector(args.uri, args.username, args.password, database=database) as connector:
        connector.verify_connectivity(timeout=args.timeout)
        scored = score_records(
            records,
            connector,
            metrics=args.metrics,
            predicted_key=args.predicted_key,
            target_key=args.target_key,
            timeout=args.timeout,
            desc=f"Evaluating {args.name}/{database}",
        )
    write_jsonl(output_path, scored)
    summary = aggregate_scores(scored)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
