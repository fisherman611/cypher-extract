import json
import math
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

import cypher_evaluation.cli as evaluation_cli
from cypher_evaluation.cli import parse_args, resolve_connection, resolve_database, resolve_output_path
from cypher_evaluation.merge import merge_graph_evaluations
from cypher_evaluation.metrics import (
    executable,
    execution_accuracy,
    get_ps_cypher,
    provenance_subgraph_jaccard_similarity,
)
from cypher_evaluation.neo4j import Neo4jConnector
from cypher_evaluation.scoring import (
    aggregate_scores,
    clean_pred_cypher,
    metric_error_counts,
    safe_compute,
    score_records,
)


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


def test_neo4j_text2cypher_uses_reference_connection_defaults():
    args = evaluation_cli.argparse.Namespace(
        name="neo4j_text2cypher_db",
        graph="movies",
        uri=None,
        username=None,
        password=None,
        database=None,
    )
    uri, username, password, database, config = resolve_connection(args)
    assert uri == "bolt+s://demo.neo4jlabs.com:7687"
    assert (username, password, database) == ("movies", "movies", "movies")
    assert config.debug is True


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


def test_connector_executes_queries_in_managed_read_transaction(monkeypatch):
    observed = {"execute_read": 0, "auto_commit": 0}

    class Result:
        def data(self):
            return [{"value": 1}]

    class Transaction:
        def run(self, query, **parameters):
            observed["query"] = query
            observed["parameters"] = parameters
            return Result()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def execute_read(self, callback):
            observed["execute_read"] += 1
            return callback(Transaction())

        def run(self, *_args, **_kwargs):
            observed["auto_commit"] += 1
            raise AssertionError("Evaluation must not use an auto-commit session")

    class Driver:
        def session(self, **kwargs):
            observed["session"] = kwargs
            return Session()

    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(Query=lambda text, timeout: (text, timeout)))
    connector = Neo4jConnector.__new__(Neo4jConnector)
    connector.database = "demo"
    connector.debug = False
    connector.driver = Driver()

    assert connector.run_query("CREATE (:Unsafe)", timeout=17, value=3) == [{"value": 1}]
    assert observed == {
        "execute_read": 1,
        "auto_commit": 0,
        "session": {"database": "demo"},
        "query": ("CREATE (:Unsafe)", 17),
        "parameters": {"value": 3},
    }


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
            "geography": {
                "count": 1,
                "overall": {"execution_accuracy": math.nan, "psjs": math.nan, "executable": 1.0},
            },
            "nba": {
                "count": 1,
                "overall": {"execution_accuracy": math.nan, "psjs": math.nan, "executable": 0.0},
            },
        },
        "overall": {"execution_accuracy": math.nan, "psjs": math.nan, "executable": 0.5},
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


def test_merge_graph_evaluations_uses_selected_metrics_only(tmp_path: Path):
    write_score_file(
        tmp_path / "nba/cypher_scores.jsonl",
        [{"id": "nba-1", "graph": "nba", "metrics": {"executable": 1.0}}],
    )
    _, summary = merge_graph_evaluations(
        tmp_path,
        expected_graphs=("nba",),
        metrics=("executable",),
    )
    assert summary == {
        "count": 1,
        "graphs": {"nba": {"count": 1, "overall": {"executable": 1.0}}},
        "overall": {"executable": 1.0},
    }


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


def test_reference_provenance_query_handles_call_with_union():
    query = """CALL {
        MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a
        UNION
        MATCH (c:Company) RETURN c
    } RETURN *"""
    provenance = get_ps_cypher(query, "element_id", node_element_id_only=True)
    assert "elementId(a)" in provenance
    assert "elementId(b)" in provenance
    assert "elementId(c)" in provenance
    assert " UNION " in provenance


def test_reference_provenance_query_adds_anonymous_variables_exactly():
    provenance = get_ps_cypher(
        "MATCH (n:A)-[:R]->(:B {name: 'x'}) RETURN n",
        "rid",
        node_element_id_only=True,
    )
    assert provenance == (
        "MATCH (n:A)-[rtmp0:R]->(ntmp0:B {name: 'x'}) "
        "WITH collect(distinct elementId(n)) + collect(distinct elementId(ntmp0)) AS elemIds "
        "UNWIND elemIds AS elemId RETURN elemId AS rid"
    )


def test_reference_provenance_parser_is_case_sensitive():
    provenance = get_ps_cypher("match (n:A) return n", "rid", node_element_id_only=True)
    assert provenance == "UNWIND [] AS elemId RETURN elemId AS rid"


def test_incomplete_call_has_empty_provenance():
    provenance = get_ps_cypher(
        "CALL { MATCH (n:A) RETURN n UNION MATCH (m:B) RETURN m",
        "element_id",
        node_element_id_only=True,
    )
    assert provenance == "UNWIND [] AS elemId RETURN elemId AS element_id"


def test_psjs_computes_node_set_jaccard():
    class ProvenanceConnector:
        def run_query(self, cypher, *, timeout=None, **parameters):
            if "elemId1" in cypher:
                return [{"elemId1": "1"}, {"elemId1": "2"}]
            return [{"elemId2": "2"}, {"elemId2": "3"}]

    score = provenance_subgraph_jaccard_similarity(
        "MATCH (n:B) RETURN n",
        "MATCH (n:A) RETURN n",
        ProvenanceConnector(),
    )
    assert score == 1 / 3


def test_executable_preserves_infrastructure_error_for_safe_compute():
    connector = FakeConnector({"bad": RuntimeError("syntax error")})
    with pytest.raises(RuntimeError, match="syntax error"):
        executable("bad", "gold", connector)
    assert safe_compute("executable", "bad", "gold", connector) == {
        "error": "executable failed: syntax error"
    }


def test_score_records_matches_inference_output_fields():
    connector = FakeConnector({"RETURN 1": [{"value": 1}]})
    rows = score_records(
        [{"id": "one", "predicted_cypher": "RETURN 1", "reference_cypher": "RETURN 1"}],
        connector,
        metrics=("execution_accuracy", "executable"),
    )
    assert rows[0]["metrics"] == {"execution_accuracy": 1.0, "executable": 1.0}
    assert aggregate_scores(rows, metrics=("execution_accuracy", "executable"))["overall"] == {
        "execution_accuracy": 1.0,
        "executable": 1.0,
    }


def test_score_records_scores_rejected_inference_example_normally():
    connector = FakeConnector({"RETURN 1": [{"value": 1}]})
    rows = score_records(
        [
            {
                "id": "rejected",
                "predicted_cypher": "RETURN 1",
                "reference_cypher": "RETURN 1",
                "gold_subschema_available": False,
            }
        ],
        connector,
        metrics=("execution_accuracy", "executable"),
    )

    assert rows[0]["metrics"] == {"execution_accuracy": 1.0, "executable": 1.0}


def test_score_records_does_not_recover_malformed_model_json_like_reference(monkeypatch):
    class QueryError(Exception):
        pass

    malformed = '{"cypher": "RETURN 1"'
    connector = FakeConnector({malformed: QueryError("syntax error")})
    executable_module = import_module("cypher_evaluation.executable")
    monkeypatch.setattr(executable_module, "_neo4j_query_errors", lambda: (QueryError,))
    rows = score_records(
        [{"id": "one", "predicted_cypher": malformed, "reference_cypher": "RETURN 1"}],
        connector,
        metrics=("executable",),
    )
    assert rows[0]["predicted_cypher"] == malformed
    assert rows[0]["metrics"] == {"executable": 0.0}


def test_aggregate_scores_accepts_legacy_cypher_metrics_key():
    summary = aggregate_scores([{"cypher_metrics": {"executable": 1.0}}], metrics=("executable",))
    assert summary == {"overall": {"executable": 1.0}}


def test_reference_cleanup_only_strips_end_of_turn_suffix():
    assert clean_pred_cypher(" RETURN 1 ") == " RETURN 1 "
    assert clean_pred_cypher(" RETURN 1 <end_of_turn>") == "RETURN 1"


def test_reference_aggregate_turns_metric_errors_into_zero_and_rounds():
    rows = [
        {"metrics": {"execution_accuracy": 1.0}},
        {"metrics": {"execution_accuracy": {"error": "gold failed"}}},
        {"metrics": {"execution_accuracy": 1.0}},
    ]
    assert aggregate_scores(rows, metrics=("execution_accuracy",)) == {
        "overall": {"execution_accuracy": 0.6667}
    }
    assert metric_error_counts(rows, metrics=("execution_accuracy", "executable")) == {
        "execution_accuracy": 1,
        "executable": 0,
    }


def test_cli_writes_error_tally_and_exits_nonzero(monkeypatch, tmp_path: Path):
    class Connector:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def verify_connectivity(self, *, timeout):
            pass

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "scores.jsonl"
    input_path.write_text(
        json.dumps({"graph": "nba", "predicted_cypher": "RETURN 1", "reference_cypher": "RETURN 1"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_cli, "Neo4jConnector", Connector)
    monkeypatch.setattr(
        evaluation_cli,
        "score_records",
        lambda *_args, **_kwargs: [
            {"metrics": {"execution_accuracy": {"error": "database unavailable"}}}
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-cypher",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--password",
            "secret",
            "--metrics",
            "execution_accuracy",
        ],
    )

    with pytest.raises(SystemExit) as error:
        evaluation_cli.main()

    assert error.value.code == 1
    summary = json.loads((tmp_path / "scores_summary.json").read_text(encoding="utf-8"))
    assert summary["errors"] == {"execution_accuracy": 1}


def test_reference_target_execution_error_is_preserved_by_safe_compute():
    connector = FakeConnector({"gold": RuntimeError("database unavailable")})
    assert safe_compute("execution_accuracy", "pred", "gold", connector) == {
        "error": "execution_accuracy failed: database unavailable"
    }
