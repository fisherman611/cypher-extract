"""Executable-query metric with query errors separated from infrastructure failures."""

from .execution_accuracy import _neo4j_query_errors


def executable(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector,
    timeout: int = 120,
) -> float:
    """Whether the predicted Cypher query is executable."""

    del target_cypher
    try:
        neo4j_connector.run_query(pred_cypher, timeout=timeout)
    except _neo4j_query_errors():
        return 0.0
    except Exception:
        # Connectivity, authentication, and session failures do not measure
        # model quality. Let safe_compute retain them as explicit errors.
        raise
    return 1.0
