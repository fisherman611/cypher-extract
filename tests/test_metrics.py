from distillation.metrics import compute_task_metrics, extract_cypher


def test_extract_cypher_from_json_and_fence() -> None:
    assert extract_cypher('{"cypher": "MATCH (n) RETURN n"}') == "MATCH (n) RETURN n"
    assert extract_cypher("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"


def test_compute_mixed_task_metrics() -> None:
    metrics = compute_task_metrics(
        [
            "RELATED",
            "RELATED",
            '{"cypher":"MATCH (n) RETURN n"}',
            '{"cypher":"MATCH (n) RETURN n.name"}',
        ],
        [
            "RELATED",
            "UNRELATED",
            '{"cypher":"MATCH (n)  RETURN n;"}',
            '{"cypher":"MATCH (n) RETURN n.age"}',
        ],
    )

    assert metrics["selector_count"] == 2
    assert metrics["selector_accuracy"] == 50.0
    assert metrics["generator_count"] == 2
    assert metrics["generator_exact_match"] == 50.0
    assert 0.0 < metrics["generator_rouge1"] < 100.0
    assert 0.0 < metrics["generator_rouge2"] < 100.0
    assert 0.0 < metrics["generator_rougeL"] < 100.0
