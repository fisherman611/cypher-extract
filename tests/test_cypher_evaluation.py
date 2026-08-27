import json
import sys
from pathlib import Path

import pytest

import cypher_evaluation.cli as evaluation_cli
from cypher_evaluation.cli import parse_args, resolve_database, resolve_output_path
from cypher_evaluation.merge import merge_graph_evaluations
from cypher_evaluation.metrics import (
    _provenance_query,
    executable,
    execution_accuracy,
    provenance_subgraph_jaccard_similarity,
)
from cypher_evaluation.scoring import aggregate_scores, score_records


class FakeConnector:
    def __init__(self, responses):
        self.responses = responses

    def run_query(self, cypher, *, timeout=None, **parameters):
        response = self.responses[cypher]
        if isinstance(response, Exception):
            raise response
        return response


def write_score_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_defaults_to_cypherbench_nba(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate-cypher", "--input", "input.jsonl", "--output", "output.jsonl"])
    args = parse_args()
    assert args.name == "cypherbench-db"
    assert args.database is None
    assert args.graph == "nba"
    assert resolve_database(args.database, args.graph) == "nba"


def test_database_defaults_to_dotted_graph_name():
    assert resolve_database(None, "flight_accident") == "flight.accident"
    assert resolve_database("custom-db", "flight_accident") == "custom-db"


def test_cli_uses_configured_timeout_for_connectivity(monkeypatch, tmp_path: Path):
    observed = {}

    class Connector:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def verify_connectivity(self, *, timeout):
            observed["timeout"] = timeout

    input_path = tmp_path / "input.jsonl"
    input_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(evaluation_cli, "Neo4jConnector", Connector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-cypher",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "scores.jsonl"),
            "--password",
            "secret",
            "--timeout",
            "45",
        ],
    )

    evaluation_cli.main()

    assert observed["timeout"] == 45


def test_output_path_is_derived_from_input_and_graph():
    input_path = Path("results/inference/qwen3/sft/cypherbench/generator_predictions.jsonl")
    assert resolve_output_path(input_path, None, "flight_accident") == Path(
        "results/evaluation/qwen3/sft/cypherbench/flight_accident/cypher_scores.jsonl"
    )


def test_explicit_output_path_overrides_automatic_path():
    output_path = Path("custom/scores.jsonl")
    assert resolve_output_path(Path("input.jsonl"), output_path, "nba") == output_path


def test_merge_graph_evaluations_builds_per_graph_and_overall_summary(tmp_path: Path):
    write_score_file(
        tmp_path / "geography/cypher_scores.jsonl",
        [{"id": "geo-1", "graph": "geography", "metrics": {"executable": 1.0}}],
    )
    write_score_file(
        tmp_path / "nba/cypher_scores.jsonl",
        [{"id": "nba-1", "graph": "nba", "cypher_metrics": {"executable": 0.0}}],
    )
    merged, summary = merge_graph_evaluations(tmp_path, expected_graphs=("geography", "nba"))
    assert [row["id"] for row in merged] == ["geo-1", "nba-1"]
    assert all("metrics" in row and "cypher_metrics" not in row for row in merged)
    assert summary == {
        "count": 2,
        "graphs": {
            "geography": {"count": 1, "overall": {"executable": 1.0}},
            "nba": {"count": 1, "overall": {"executable": 0.0}},
        },
        "overall": {"executable": 0.5},
    }


def test_merge_graph_evaluations_rejects_missing_graph(tmp_path: Path):
    write_score_file(
        tmp_path / "nba/cypher_scores.jsonl",
        [{"id": "nba-1", "graph": "nba", "metrics": {"executable": 1.0}}],
    )
    with pytest.raises(ValueError, match="missing graphs: geography"):
        merge_graph_evaluations(tmp_path, expected_graphs=("geography", "nba"))


def test_merge_graph_evaluations_ignores_empty_unexpected_artifact(tmp_path: Path):
    write_score_file(
        tmp_path / "nba/cypher_scores.jsonl",
        [{"id": "nba-1", "graph": "nba", "metrics": {"executable": 1.0}}],
    )
    write_score_file(tmp_path / "flight.accident/cypher_scores.jsonl", [])
    merged, summary = merge_graph_evaluations(tmp_path, expected_graphs=("nba",))
    assert [row["id"] for row in merged] == ["nba-1"]
    assert summary["count"] == 1


def test_execution_accuracy_ignores_row_and_column_order_without_order_by():
    connector = FakeConnector({
        "gold": [{"name": "A", "score": 1}, {"name": "B", "score": 2}],
        "pred": [{"x": 2, "y": "B"}, {"x": 1, "y": "A"}],
    })
    assert execution_accuracy("pred", "gold", connector) == 1.0


def test_execution_accuracy_respects_gold_order_by():
    connector = FakeConnector({
        "gold ORDER BY n.name": [{"name": "A"}, {"name": "B"}],
        "pred": [{"name": "B"}, {"name": "A"}],
    })
    assert execution_accuracy("pred", "gold ORDER BY n.name", connector) == 0.0


def test_execution_accuracy_preserves_duplicate_rows():
    connector = FakeConnector({
        "gold": [{"name": "A"}, {"name": "A"}],
        "pred": [{"name": "A"}, {"name": "B"}],
    })
    assert execution_accuracy("pred", "gold", connector) == 0.0


def test_execution_accuracy_treats_list_cells_as_unordered_like_reference():
    connector = FakeConnector({
        "gold": [{"names": ["A", "B"]}],
        "pred": [{"names": ["B", "A"]}],
    })
    assert execution_accuracy("pred", "gold", connector) == 1.0


def test_execution_accuracy_constrains_wide_column_permutations():
    gold = {f"gold_{index}": index for index in range(10)}
    predicted = {f"pred_{index}": 9 - index for index in range(10)}
    connector = FakeConnector({"gold": [gold], "pred": [predicted]})
    assert execution_accuracy("pred", "gold", connector) == 1.0


def test_provenance_query_handles_call_with_union():
    query = """CALL {
        MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a
        UNION
        MATCH (c:Company) RETURN c
    } RETURN *"""
    provenance = _provenance_query(query, "element_id")
    assert "elementId(a)" in provenance
    assert "elementId(b)" in provenance
    assert "elementId(c)" in provenance
    assert " UNION " in provenance


def test_provenance_parser_ignores_keywords_and_variables_outside_match():
    provenance = _provenance_query(
        "MATCH (n:Movie {title: 'MATCH UNION'}) WHERE any(x IN n.tags WHERE x = 'UNION') RETURN n",
        "element_id",
    )
    assert "elementId(n)" in provenance
    assert "elementId(x)" not in provenance
    assert provenance.count(" UNION ") == 0


def test_provenance_parser_does_not_treat_relationship_type_as_where_clause():
    provenance = _provenance_query(
        "MATCH (n:Location)<-[r:where]-(m:Character) RETURN n",
        "element_id",
    )
    assert "elementId(n)" in provenance
    assert "elementId(m)" in provenance


def test_incomplete_call_has_empty_provenance():
    provenance = _provenance_query(
        "CALL { MATCH (n:A) RETURN n UNION MATCH (m:B) RETURN m",
        "element_id",
    )
    assert provenance == "UNWIND [] AS id RETURN id AS element_id"


def test_psjs_computes_node_set_jaccard():
    class ProvenanceConnector:
        def run_query(self, cypher, *, timeout=None, **parameters):
            if "target_id" in cypher:
                return [{"target_id": "1"}, {"target_id": "2"}]
            return [{"predicted_id": "2"}, {"predicted_id": "3"}]

    score = provenance_subgraph_jaccard_similarity(
        "MATCH (n:B) RETURN n",
        "MATCH (n:A) RETURN n",
        ProvenanceConnector(),
    )
    assert score == 1 / 3


def test_bad_query_is_not_executable():
    connector = FakeConnector({"bad": RuntimeError("syntax error")})
    assert executable("bad", "gold", connector) == 0.0


def test_score_records_matches_inference_output_fields():
    connector = FakeConnector({"RETURN 1": [{"value": 1}]})
    rows = score_records(
        [{"id": "one", "predicted_cypher": "RETURN 1", "reference_cypher": "RETURN 1"}],
        connector,
        metrics=("execution_accuracy", "executable"),
    )
    assert rows[0]["metrics"] == {"execution_accuracy": 1.0, "executable": 1.0}
    assert aggregate_scores(rows)["overall"] == {"executable": 1.0, "execution_accuracy": 1.0}


def test_score_records_recovers_cypher_from_malformed_model_json():
    connector = FakeConnector({"RETURN 1": [{"value": 1}]})
    rows = score_records(
        [{"id": "one", "predicted_cypher": '{"cypher": "RETURN 1"', "reference_cypher": "RETURN 1"}],
        connector,
        metrics=("executable",),
    )
    assert rows[0]["predicted_cypher"] == "RETURN 1"
    assert rows[0]["metrics"] == {"executable": 1.0}


def test_aggregate_scores_accepts_legacy_cypher_metrics_key():
    summary = aggregate_scores([{"cypher_metrics": {"executable": 1.0}}])
    assert summary == {"count": 1, "overall": {"executable": 1.0}}
