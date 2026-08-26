import sys

from cypher_evaluation.cli import parse_args, resolve_database
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
    assert rows[0]["cypher_metrics"] == {"execution_accuracy": 1.0, "executable": 1.0}
    assert aggregate_scores(rows)["metrics"] == {"executable": 1.0, "execution_accuracy": 1.0}


def test_score_records_recovers_cypher_from_malformed_model_json():
    connector = FakeConnector({"RETURN 1": [{"value": 1}]})
    rows = score_records(
        [{"id": "one", "predicted_cypher": '{"cypher": "RETURN 1"', "reference_cypher": "RETURN 1"}],
        connector,
        metrics=("executable",),
    )
    assert rows[0]["predicted_cypher"] == "RETURN 1"
    assert rows[0]["cypher_metrics"] == {"executable": 1.0}
