from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_grounding.cypher import extract_subschema
from schema_grounding.pipeline import build_dataset
from schema_grounding.schema import (
    canonical_schema,
    from_cypherbench,
    from_mind_the_query,
    from_neo4j_schema_text,
)


class SchemaNormalizationTest(unittest.TestCase):
    def test_cypherbench_adapter(self) -> None:
        schema = from_cypherbench(
            {
                "entities": [{"label": "Person", "properties": {"name": "str"}}],
                "relations": [
                    {
                        "label": "KNOWS",
                        "subj_label": "Person",
                        "obj_label": "Person",
                        "properties": {"since": "int"},
                    }
                ],
            },
            "social",
        )
        self.assertEqual(schema.nodes[0].properties, (("name", "STRING"),))
        self.assertEqual(schema.relations[0].id, "relation:Person|KNOWS|Person")

    def test_cypherbench_adapter_does_not_invent_properties(self) -> None:
        schema = from_cypherbench(
            {
                "entities": [{"label": "Person", "properties": {"id": "int"}}],
                "relations": [],
            },
            "social",
        )
        self.assertEqual(schema.nodes[0].properties, (("id", "INTEGER"),))

    def test_mind_the_query_adapter(self) -> None:
        schema = from_mind_the_query(
            {
                "demo": {
                    "node_props": {"Person": [{"property": "id", "datatype": "STRING"}]},
                    "rel_props": {"VISITS": [{"property": "duration", "datatype": "DURATION"}]},
                    "relationships": [{"start": "Person", "type": "VISITS", "end": "Place"}],
                }
            },
            "demo",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"Person", "Place"})
        self.assertEqual(schema.relations[0].properties, (("duration", "DURATION"),))

    def test_neo4j_text_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """Node properties:
- **Person**
  - `name`: STRING
- **Movie**
  - `title`: STRING
Relationship properties:
- **ACTED_IN**
  - `role`: STRING
The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
""",
            "movies",
        )
        self.assertEqual(len(schema.nodes), 2)
        self.assertEqual(schema.relations[0].id, "relation:Person|ACTED_IN|Movie")
        self.assertEqual(schema.relations[0].properties, (("role", "STRING"),))

    def test_neo4j_relevant_text_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """Graph schema: Relevant node labels and their properties (with datatypes) are:
Article {abstract: STRING}
Keyword {}

Relevant relationships are:
{'start': Article, 'type': HAS_KEY, 'end': Keyword }
""",
            "functional",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"Article", "Keyword"})
        self.assertEqual(schema.relations[0].id, "relation:Article|HAS_KEY|Keyword")

    def test_neo4j_relevant_bare_labels_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """Graph schema: Relevant node labels and their properties are:
Author
DOI
""",
            "functional",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"Author", "DOI"})

    def test_neo4j_repr_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """[<Record nodes=[<Node labels=frozenset({'author'}) properties={}>,
<Node labels=frozenset({'paper'}) properties={}>]
relationships=[<Relationship nodes=(<Node labels=frozenset({'author'}) properties={}>,
<Node labels=frozenset({'paper'}) properties={}>) type='WROTE' properties={}>]>]""",
            "inspection",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"author", "paper"})
        self.assertEqual(schema.relations[0].id, "relation:author|WROTE|paper")

    def test_neo4j_json_schema_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            '{"Person": {"type": "node", "properties": {"name": {"type": "STRING"}}, '
            '"relationships": {"KNOWS": {"direction": "out", "labels": ["Person"], "properties": {}}}}, '
            '"KNOWS": {"type": "relationship", "properties": {"since": {"type": "INTEGER"}}}}',
            "inspected",
        )
        self.assertEqual(schema.nodes[0].properties, (("name", "STRING"),))
        self.assertEqual(schema.relations[0].properties, (("since", "INTEGER"),))


class SubSchemaExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = canonical_schema(
            "test",
            "graph",
            [("Person", {"name": "STRING"}), ("Movie", {"title": "STRING"}), ("City", {})],
            [
                ("Person", "ACTED_IN", "Movie", {}),
                ("Person", "LIVES_IN", "City", {}),
            ],
        )

    def test_direction_and_variable_resolution(self) -> None:
        result = extract_subschema(
            "MATCH (m:Movie)<-[:ACTED_IN]-(p:Person) RETURN p.name", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(
            result.relation_unit_ids, ("relation:Person|ACTED_IN|Movie",)
        )
        self.assertEqual(set(result.node_unit_ids), {"node:Person", "node:Movie"})

    def test_label_predicate_resolves_unlabelled_node(self) -> None:
        result = extract_subschema(
            "MATCH (p) WHERE p:Person RETURN p.name", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))


    def test_unknown_relation_is_reported(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN m", self.schema
        )
        self.assertIn("DIRECTED", result.unmapped_relation_types)
        self.assertFalse(result.complete)

    def test_relation_endpoint_mismatch_is_reported_separately(self) -> None:
        result = extract_subschema(
            "MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN p", self.schema
        )
        self.assertFalse(result.unmapped_relation_types)
        self.assertEqual(
            result.unmatched_relationship_patterns,
            ("(:Movie)-[:ACTED_IN]->(:Person)",),
        )

    def test_case_expression_is_not_a_node_pattern(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person) WITH p, (CASE WHEN 1 > 0 THEN 1 ELSE 0 END) AS score RETURN score",
            self.schema,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))

    def test_relation_propagates_variable_label(self) -> None:
        result = extract_subschema(
            "MATCH (person)-[:LIVES_IN]->(:City) RETURN person", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.relation_unit_ids, ("relation:Person|LIVES_IN|City",))
        self.assertEqual(set(result.node_unit_ids), {"node:Person", "node:City"})

    def test_parentheses_inside_string_literal_are_not_patterns(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person {name: 'Example (alias)'}) RETURN p", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))


class PipelineAuditTest(unittest.TestCase):
    def test_unrecognized_schema_is_written_to_normalization_audit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Neo4j_Text2Cypher"
            source_directory.mkdir(parents=True)
            (source_directory / "train.json").write_text(
                json.dumps(
                    [
                        {
                            "qid": "invalid-schema",
                            "graph": "demo",
                            "nl_question": "List every item.",
                            "gold_cypher": "MATCH (item) RETURN item",
                            "schema": "CREATE TABLE items (id INTEGER);",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_directory = Path(temporary_directory) / "output"
            manifest = build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("neo4j_text2cypher",),
            )

            issue = json.loads(
                (output_directory / "normalization_issues_train.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            rejected = json.loads(
                (output_directory / "rejected_train.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(issue["issues"], ["empty_normalized_schema"])
            self.assertEqual(issue["raw_schema"], "CREATE TABLE items (id INTEGER);")
            self.assertEqual(rejected["reason"], "empty_normalized_schema")
            self.assertEqual(
                manifest["files"]["normalization_issues"]["train"],
                "normalization_issues_train.jsonl",
            )

    def test_dev_split_build(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Cypherbench"
            source_directory.mkdir(parents=True)
            (source_directory / "graphs" / "schemas").mkdir(parents=True)
            (source_directory / "graphs" / "schemas" / "demo_schema.json").write_text(
                json.dumps({"entities": [{"label": "Person", "properties": {}}], "relations": []}),
                encoding="utf-8",
            )
            (source_directory / "dev.json").write_text(
                json.dumps(
                    [
                        {
                            "qid": "dev-sample",
                            "graph": "demo",
                            "nl_question": "List all people.",
                            "gold_cypher": "MATCH (p:Person) RETURN p",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_directory = Path(temporary_directory) / "output"
            manifest = build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("cypherbench",),
                splits=("dev",),
            )
            self.assertEqual(manifest["counts"]["cypherbench/dev"]["generation_examples"], 1)
            self.assertTrue((output_directory / "generation_dev.jsonl").exists())
            self.assertTrue((output_directory / "selection_dev.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
