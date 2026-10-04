import json
import random

import pytest

from schema_grounding.augmentation import (
    AugmentationConfig,
    augment_generation_row,
    distractor_pools,
    schema_units,
)
from scripts import augment_generator_schema

SCHEMA = {
    "schema_id": "cypherbench:movie:abc",
    "benchmark": "cypherbench",
    "graph": "movie",
    "nodes": [
        {"label": "Award", "properties": {"name": "STRING"}},
        {"label": "City", "properties": {}},
        {"label": "Genre", "properties": {}},
        {"label": "Movie", "properties": {"release_year": "INTEGER", "title": "STRING"}},
        {"label": "Person", "properties": {"age": "INTEGER", "name": "STRING"}},
        {"label": "Studio", "properties": {}},
    ],
    "relationships": [
        {"properties": {}, "source": "Movie", "target": "Award", "type": "won"},
        {"properties": {}, "source": "Movie", "target": "Genre", "type": "inGenre"},
        {"properties": {"role": "STRING"}, "source": "Person", "target": "Movie", "type": "actedIn"},
        {"properties": {}, "source": "Person", "target": "Movie", "type": "directed"},
        {"properties": {}, "source": "Studio", "target": "City", "type": "locatedIn"},
    ],
}
GOLD_SUB_SCHEMA = {
    "nodes": [SCHEMA["nodes"][3], SCHEMA["nodes"][4]],
    "relationships": [SCHEMA["relationships"][2]],
}
GOLD_IDS = ["node:Movie", "node:Person", "relation:Person|actedIn|Movie"]


def _row(index: int = 0, question: str = "Which movies won an award in 2000?") -> dict:
    return {
        "cypher": "MATCH (p:Person)-[:actedIn]->(m:Movie) RETURN m.title",
        "graph": "movie",
        "id": f"cypherbench:train:movie:{index}",
        "question": question,
        "schema_id": SCHEMA["schema_id"],
        "source": "cypherbench",
        "split": "train",
        "sub_schema": json.loads(json.dumps(GOLD_SUB_SCHEMA)),
    }


def _canonical_ids(sub_schema: dict) -> list[str]:
    ids = [f"node:{node['label']}" for node in sub_schema["nodes"]]
    ids.extend(
        f"relation:{rel['source']}|{rel['type']}|{rel['target']}" for rel in sub_schema["relationships"]
    )
    return ids


def test_gold_mode_keeps_the_rendered_sub_schema_byte_identical() -> None:
    row = _row()
    augmented = augment_generation_row(row, SCHEMA, AugmentationConfig(gold_ratio=1.0, full_ratio=0.0))

    assert augmented["schema_augmentation"] == {
        "mode": "gold",
        "gold_unit_ids": GOLD_IDS,
        "distractor_unit_ids": [],
    }
    assert json.dumps(augmented["sub_schema"], indent=2) == json.dumps(row["sub_schema"], indent=2)


def test_full_mode_renders_the_complete_schema() -> None:
    augmented = augment_generation_row(_row(), SCHEMA, AugmentationConfig(gold_ratio=0.0, full_ratio=1.0))

    assert augmented["schema_augmentation"]["mode"] == "full"
    assert augmented["sub_schema"] == {"nodes": SCHEMA["nodes"], "relationships": SCHEMA["relationships"]}


@pytest.mark.parametrize("max_distractors", [1, 3, 6])
def test_noisy_rows_keep_gold_and_add_bounded_graph_valid_distractors(max_distractors: int) -> None:
    config = AugmentationConfig(gold_ratio=0.0, full_ratio=0.0, max_distractors=max_distractors)
    canonical_order = [unit["id"] for unit in schema_units(SCHEMA)]

    for index in range(60):
        row = _row(index)
        augmented = augment_generation_row(row, SCHEMA, config)
        info = augmented["schema_augmentation"]
        ids = _canonical_ids(augmented["sub_schema"])

        assert info["mode"] == "noisy"
        assert set(GOLD_IDS) <= set(ids)
        assert not set(info["distractor_unit_ids"]) & set(GOLD_IDS)
        assert set(ids) == set(GOLD_IDS) | set(info["distractor_unit_ids"])
        assert 1 <= len(info["distractor_unit_ids"]) <= max_distractors
        assert ids == [unit_id for unit_id in canonical_order if unit_id in set(ids)]
        labels = {node["label"] for node in augmented["sub_schema"]["nodes"]}
        for relationship in augmented["sub_schema"]["relationships"]:
            assert {relationship["source"], relationship["target"]} <= labels
        for node in augmented["sub_schema"]["nodes"]:
            assert node in SCHEMA["nodes"]
        for key in ("question", "cypher", "id"):
            assert augmented[key] == row[key]


def test_augmentation_is_deterministic_and_independent_of_row_order() -> None:
    config = AugmentationConfig(gold_ratio=0.2, full_ratio=0.1, max_distractors=4, seed=7)
    rows = [_row(index) for index in range(40)]
    first = {row["id"]: augment_generation_row(row, SCHEMA, config) for row in rows}
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)
    second = {row["id"]: augment_generation_row(row, SCHEMA, config) for row in shuffled}

    assert first == second
    assert {item["schema_augmentation"]["mode"] for item in first.values()} == {"gold", "noisy", "full"}


def test_distractor_pools_target_graph_neighbors_and_question_words() -> None:
    pools = distractor_pools(schema_units(SCHEMA), GOLD_IDS, "Which movies won an award in 2000?")

    assert set(pools["neighbor"]) == {
        "node:Award",
        "node:Genre",
        "relation:Movie|won|Award",
        "relation:Movie|inGenre|Genre",
        "relation:Person|directed|Movie",
    }
    assert set(pools["lexical"]) == {"node:Award", "relation:Movie|won|Award"}
    assert set(pools["random"]) == {unit["id"] for unit in schema_units(SCHEMA)} - set(GOLD_IDS)
    assert set(pools["hard"]) == {"relation:Person|directed|Movie"}


HARD_SCHEMA = {
    "schema_id": "cypherbench:media:def",
    "benchmark": "cypherbench",
    "graph": "media",
    "nodes": [
        {"label": "Book", "properties": {"pages": "INTEGER", "title": "STRING"}},
        {"label": "Genre", "properties": {}},
        {"label": "Movie", "properties": {"country": "STRING", "release_year": "INTEGER", "title": "STRING"}},
        {"label": "Person", "properties": {"age": "INTEGER", "name": "STRING"}},
        {"label": "Series", "properties": {"release_year": "INTEGER", "seasons": "INTEGER", "title": "STRING"}},
    ],
    "relationships": [
        {"properties": {}, "source": "Movie", "target": "Genre", "type": "inGenre"},
        {"properties": {}, "source": "Movie", "target": "Person", "type": "features"},
        {"properties": {}, "source": "Person", "target": "Movie", "type": "actedIn"},
        {"properties": {}, "source": "Person", "target": "Series", "type": "actedIn"},
        {"properties": {}, "source": "Person", "target": "Movie", "type": "directed"},
        {"properties": {}, "source": "Person", "target": "Book", "type": "wrote"},
    ],
}
HARD_GOLD_IDS = ["node:Movie", "node:Person", "relation:Person|actedIn|Movie"]
EXPECTED_HARD = {
    "node:Series",  # shares release_year and title with gold Movie (Jaccard 2/4)
    "relation:Person|actedIn|Series",  # gold relationship type, other endpoint
    "relation:Movie|features|Person",  # gold endpoint pair, reversed
    "relation:Person|directed|Movie",  # gold endpoint pair, other type
}


def _hard_row(index: int) -> dict:
    return {
        "cypher": "MATCH (p:Person)-[:actedIn]->(m:Movie) RETURN m.title",
        "graph": "media",
        "id": f"cypherbench:train:media:{index}",
        "question": "Which movies did Tom Hanks act in?",
        "schema_id": HARD_SCHEMA["schema_id"],
        "source": "cypherbench",
        "split": "train",
        "sub_schema": {
            "nodes": [HARD_SCHEMA["nodes"][2], HARD_SCHEMA["nodes"][3]],
            "relationships": [HARD_SCHEMA["relationships"][2]],
        },
    }


def test_hard_pool_keeps_only_units_that_imitate_gold() -> None:
    pools = distractor_pools(schema_units(HARD_SCHEMA), HARD_GOLD_IDS, "Which movies did Tom Hanks act in?")

    # Book shares a single property and inGenre/wrote touch gold nodes only as neighbors.
    assert set(pools["hard"]) == EXPECTED_HARD


def test_hard_only_sampling_adds_only_hard_distractors() -> None:
    config = AugmentationConfig(
        gold_ratio=0.0,
        full_ratio=0.0,
        max_distractors=6,
        hard_weight=1.0,
        neighbor_weight=0.0,
        lexical_weight=0.0,
        random_weight=0.0,
    )

    for index in range(30):
        info = augment_generation_row(_hard_row(index), HARD_SCHEMA, config)["schema_augmentation"]
        assert info["distractor_unit_ids"]
        assert set(info["distractor_unit_ids"]) <= EXPECTED_HARD


def test_rejects_negative_pool_weights() -> None:
    with pytest.raises(ValueError, match="pool weights"):
        AugmentationConfig(hard_weight=-0.1)


def test_rejects_gold_entries_that_do_not_match_the_schema() -> None:
    row = _row()
    row["sub_schema"]["nodes"][0] = {"label": "Movie", "properties": {}}

    with pytest.raises(ValueError, match="does not match"):
        augment_generation_row(row, SCHEMA, AugmentationConfig())


def test_script_rewrites_only_generator_train_rows(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "grounding"
    input_dir.mkdir()
    rows = [_row(index) for index in range(10)]
    augment_generator_schema.write_jsonl(input_dir / "generation_train.jsonl", rows)
    augment_generator_schema.write_jsonl(input_dir / "schemas.jsonl", [SCHEMA])
    unchanged = {
        "generation_dev.jsonl": b'{"id": "dev"}\n',
        "selection_train.jsonl": b'{"id": "sel"}\n',
        "manifest.json": b"{}\n",
    }
    for name, content in unchanged.items():
        (input_dir / name).write_bytes(content)
    output_dir = tmp_path / "augmented"
    monkeypatch.setattr(
        "sys.argv",
        ["augment_generator_schema.py", "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
    )

    augment_generator_schema.main()

    for name, content in unchanged.items():
        assert (output_dir / name).read_bytes() == content
    assert (output_dir / "schemas.jsonl").read_bytes() == (input_dir / "schemas.jsonl").read_bytes()
    written = augment_generator_schema.read_jsonl(output_dir / "generation_train.jsonl")
    assert [row["id"] for row in written] == [row["id"] for row in rows]
    assert all(set(GOLD_IDS) <= set(_canonical_ids(row["sub_schema"])) for row in written)
    manifest = json.loads((output_dir / "augmentation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["train"]["rows"] == 10
    assert manifest["rewritten_files"] == ["generation_train.jsonl"]


def _added_kinds(config: AugmentationConfig, rows: int = 300) -> tuple[int, int]:
    relations = nodes = 0
    for index in range(rows):
        augmented = augment_generation_row(_row(index), SCHEMA, config)
        for unit_id in augmented["schema_augmentation"]["distractor_unit_ids"]:
            if unit_id.startswith("relation:"):
                relations += 1
            else:
                nodes += 1
    return relations, nodes


def test_new_options_default_to_the_original_sampling() -> None:
    legacy = AugmentationConfig(hard_weight=0.5, neighbor_weight=0.0, lexical_weight=0.2, random_weight=0.3)
    explicit = AugmentationConfig(
        hard_weight=0.5,
        neighbor_weight=0.0,
        lexical_weight=0.2,
        random_weight=0.3,
        relation_share=None,
        count_weights=None,
    )
    rows = [_row(index) for index in range(60)]

    assert [augment_generation_row(row, SCHEMA, legacy) for row in rows] == [
        augment_generation_row(row, SCHEMA, explicit) for row in rows
    ]


def test_relation_share_shifts_distractors_toward_relationships() -> None:
    base = {"gold_ratio": 0.0, "full_ratio": 0.0, "max_distractors": 4, "seed": 11}
    relations_default, nodes_default = _added_kinds(AugmentationConfig(**base))
    relations_biased, nodes_biased = _added_kinds(AugmentationConfig(relation_share=0.9, **base))

    share_default = relations_default / (relations_default + nodes_default)
    share_biased = relations_biased / (relations_biased + nodes_biased)
    assert share_biased > share_default
    only_nodes = _added_kinds(AugmentationConfig(relation_share=0.0, **base))
    assert only_nodes[1] > only_nodes[0]


def test_count_weights_control_the_number_of_added_units() -> None:
    config = AugmentationConfig(
        gold_ratio=0.0, full_ratio=0.0, max_distractors=6, count_weights=(1.0, 0.0, 0.0, 0.0), seed=3
    )

    for index in range(80):
        augmented = augment_generation_row(_row(index), SCHEMA, config)
        assert len(augmented["schema_augmentation"]["distractor_unit_ids"]) == 1

    heavy_tail = AugmentationConfig(
        gold_ratio=0.0, full_ratio=0.0, max_distractors=8, count_weights=(1, 1, 1, 1, 1, 1, 1, 1), seed=3
    )
    sizes = {
        len(augment_generation_row(_row(index), SCHEMA, heavy_tail)["schema_augmentation"]["distractor_unit_ids"])
        for index in range(200)
    }
    # The schema only has 12 other units, so counts above that are impossible; several sizes occur.
    assert max(sizes) >= 5 and min(sizes) <= 2


def test_new_options_are_validated() -> None:
    invalid = (
        {"relation_share": 1.5},
        {"count_weights": ()},
        {"count_weights": (0.0, 1.0)},
        {"count_weights": (1.0, -1.0)},
    )
    for bad in invalid:
        with pytest.raises(ValueError):
            AugmentationConfig(**bad)


def test_count_weights_cli_parser_requires_consecutive_counts() -> None:
    assert augment_generator_schema.parse_count_weights("1:0.5,2:0.3,3:0.2") == (0.5, 0.3, 0.2)
    with pytest.raises(Exception, match="consecutive"):
        augment_generator_schema.parse_count_weights("1:0.5,3:0.5")
    with pytest.raises(Exception, match="count:weight"):
        augment_generator_schema.parse_count_weights("1")
