import json
import random

from schema_grounding.augmentation import schema_units
from schema_grounding.inference.merge import merge_schema_units
from schema_grounding.inference.prompting import generator_other_schema
from scripts.prepare_multitask_prompts import format_generator_rows, load_prompt, simulate_selector_miss

SCHEMA = {
    "schema_id": "fixture:company",
    "nodes": [
        {"label": "Company", "properties": {"launch_year": "INTEGER"}},
        {"label": "Person", "properties": {"date_of_death": "DATE"}},
        {"label": "Country", "properties": {}},
    ],
    "relationships": [
        {"source": "Company", "type": "foundedBy", "target": "Person", "properties": {}},
        {"source": "Company", "type": "hasBoardMember", "target": "Person", "properties": {}},
        {"source": "Company", "type": "basedIn", "target": "Country", "properties": {}},
    ],
}
UNITS = schema_units(SCHEMA)
UNITS_BY_ID = {unit["id"]: unit for unit in UNITS}
GOLD = ["node:Company", "node:Person", "relation:Company|foundedBy|Person", "relation:Company|hasBoardMember|Person"]


def _graph_valid(candidate_ids: set[str]) -> bool:
    nodes = {unit_id for unit_id in candidate_ids if unit_id.startswith("node:")}
    for unit_id in candidate_ids:
        if unit_id.startswith("relation:"):
            schema = UNITS_BY_ID[unit_id]["schema"]
            if {f"node:{schema['source']}", f"node:{schema['target']}"} - nodes:
                return False
    return True


def test_simulated_miss_drops_a_gold_relationship_and_keeps_shared_endpoints() -> None:
    removed = simulate_selector_miss(GOLD, GOLD, UNITS_BY_ID, random.Random(0))

    # Both gold relationships join Company and Person, so the endpoints stay.
    assert len(removed) == 1 and removed[0].startswith("relation:")
    assert _graph_valid(set(GOLD) - set(removed))


def test_simulated_miss_drops_endpoints_no_remaining_relationship_uses() -> None:
    candidate = [
        "node:Company",
        "node:Person",
        "node:Country",
        "relation:Company|foundedBy|Person",
        "relation:Company|basedIn|Country",
    ]
    gold = ["node:Company", "node:Country", "relation:Company|basedIn|Country"]

    removed = simulate_selector_miss(candidate, gold, UNITS_BY_ID, random.Random(0))

    # Company is still used by foundedBy, Country only by the dropped basedIn.
    assert removed == ["relation:Company|basedIn|Country", "node:Country"]
    assert _graph_valid(set(candidate) - set(removed))


def test_simulated_miss_never_empties_the_candidates() -> None:
    candidate = ["node:Company", "node:Country", "relation:Company|basedIn|Country"]

    assert simulate_selector_miss(candidate, candidate, UNITS_BY_ID, random.Random(0)) == []


def test_simulated_miss_on_node_only_question_keeps_one_candidate() -> None:
    assert simulate_selector_miss(["node:Person"], ["node:Person"], UNITS_BY_ID, random.Random(0)) == []
    removed = simulate_selector_miss(
        ["node:Country", "node:Person"], ["node:Country", "node:Person"], UNITS_BY_ID, random.Random(1)
    )
    assert len(removed) == 1 and removed[0] in {"node:Country", "node:Person"}


def _rows(count: int) -> list[dict]:
    sub_schema = merge_schema_units(UNITS, GOLD).sub_schema
    return [
        {
            "id": f"fixture:{index}",
            "question": "Who founded and sat on the board of a company?",
            "cypher": "MATCH (n:Person)<-[:foundedBy]-(m:Company) RETURN n.name",
            "sub_schema": sub_schema,
            "schema_id": SCHEMA["schema_id"],
            "source": "fixture",
            "split": "train",
            "graph": "company",
        }
        for index in range(count)
    ]


def _format(rows: list[dict], miss_ratio: float) -> list[dict]:
    return format_generator_rows(
        rows,
        load_prompt("generator/system_prompt.txt"),
        load_prompt("generator/user_prompt.txt"),
        {SCHEMA["schema_id"]: UNITS},
        miss_ratio=miss_ratio,
        seed=42,
    )


def test_miss_simulation_is_seeded_and_moves_units_to_other() -> None:
    first = _format(_rows(400), miss_ratio=0.1)
    second = _format(_rows(400), miss_ratio=0.1)

    assert first == second
    missed = [row for row in first if "candidate_missing_unit_ids" in row]
    assert 20 <= len(missed) <= 60
    for row in missed:
        candidate_text, other_text = row["user_prompt"].split("OTHER SCHEMA:\n", 1)
        other = json.loads(other_text.split("\n\nGenerate a Cypher query", 1)[0])
        other_ids = {f"node:{node['label']}" for node in other["nodes"]} | {
            f"relation:{rel['source']}|{rel['type']}|{rel['target']}" for rel in other["relationships"]
        }
        assert set(row["candidate_missing_unit_ids"]) <= other_ids


def test_without_miss_ratio_the_candidate_is_the_row_sub_schema() -> None:
    rows = _rows(50)
    prepared = _format(rows, miss_ratio=0.0)

    other = generator_other_schema(rows[0]["sub_schema"], UNITS)
    assert all("candidate_missing_unit_ids" not in row for row in prepared)
    assert all(json.dumps(other, ensure_ascii=False, indent=2) in row["user_prompt"] for row in prepared)
