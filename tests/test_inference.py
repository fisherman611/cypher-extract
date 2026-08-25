import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from schema_grounding.inference import checkpoints
from schema_grounding.inference.checkpoints import (
    DEFAULT_METHODS,
    LastCheckpoint,
    download_inference_checkpoint,
    resolve_last_checkpoint,
    select_last_checkpoint,
)
from schema_grounding.inference.data import DatasetSpec, iter_jsonl
from schema_grounding.inference.merge import merge_schema_units
from schema_grounding.inference.parsing import parse_selector_label
from schema_grounding.inference.pipeline import (
    InferenceOptions,
    run_dataset_pipeline,
    run_selector_stage,
)
from schema_grounding.inference.prompting import PromptTemplates

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def node(label: str) -> dict:
    return {
        "id": f"node:{label}",
        "kind": "node",
        "schema": {"label": label, "properties": {"name": "STRING"}},
        "text": f"(:{label} {{ name: STRING }})",
    }


def relation(source: str, relation_type: str, target: str) -> dict:
    return {
        "id": f"relation:{source}|{relation_type}|{target}",
        "kind": "relation",
        "schema": {
            "source": source,
            "type": relation_type,
            "target": target,
            "properties": {},
        },
        "text": f"(:{source})-[:{relation_type}]->(:{target})",
    }


def test_all_methods_include_teacher_and_exclude_da_kd() -> None:
    assert "teacher_lora" in DEFAULT_METHODS
    assert "da_kd" not in DEFAULT_METHODS
    assert len(DEFAULT_METHODS) == 13


def test_selector_label_parser_is_strict() -> None:
    assert parse_selector_label(" RELATED\n") == "RELATED"
    assert parse_selector_label("unrelated") == "UNRELATED"
    assert parse_selector_label("The answer is RELATED") is None


def test_merge_closes_relationship_endpoints_and_preserves_order() -> None:
    units = [node("A"), node("B"), node("C"), relation("A", "LINKS", "B")]
    result = merge_schema_units(units, ["relation:A|LINKS|B"])
    assert result.sub_schema == {
        "nodes": [units[0]["schema"], units[1]["schema"]],
        "relationships": [units[3]["schema"]],
    }
    assert result.directly_selected_unit_ids == ("relation:A|LINKS|B",)
    assert result.closure_added_node_ids == ("node:A", "node:B")


def test_merge_can_return_the_exact_empty_schema() -> None:
    result = merge_schema_units([node("A")], [])
    assert result.sub_schema == {"nodes": [], "relationships": []}


def test_select_and_resolve_last_checkpoint() -> None:
    prefix = "qwen3/sft"
    paths = [f"{prefix}/checkpoint-{step}" for step in (314, 1570, 628)]
    assert select_last_checkpoint(paths, prefix) == (1570, f"{prefix}/checkpoint-1570")

    class FakeApi:
        def list_repo_tree(self, **kwargs):
            assert kwargs["path_in_repo"] == prefix
            return [SimpleNamespace(path=path) for path in paths]

    checkpoint = resolve_last_checkpoint("sft", api=FakeApi(), token="test")
    assert checkpoint.step == 1570
    assert checkpoint.subfolder == "qwen3/sft/checkpoint-1570"


def test_checkpoint_download_excludes_training_state(monkeypatch, tmp_path: Path) -> None:
    checkpoint = LastCheckpoint("owner/repo", "main", "sft", 1570, "qwen3/sft/checkpoint-1570")
    adapter = tmp_path / checkpoint.subfolder
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    def fake_snapshot_download(**kwargs) -> str:
        patterns = kwargs["allow_patterns"]
        assert f"{checkpoint.subfolder}/adapter_model.safetensors" in patterns
        assert not any("global_step" in pattern or "optim" in pattern for pattern in patterns)
        return str(tmp_path)

    monkeypatch.setattr(checkpoints, "snapshot_download", fake_snapshot_download)
    assert download_inference_checkpoint(checkpoint, token="test") == adapter


def test_prompt_messages_match_training_format() -> None:
    templates = PromptTemplates.from_repository(REPOSITORY_ROOT)
    generator = templates.generator_messages(
        "Who?", {"nodes": [{"label": "Person", "properties": {}}], "relationships": []}
    )
    selector = templates.selector_messages("Who?", "(:Person)")
    assert (
        generator[0]["content"]
        == (REPOSITORY_ROOT / "prompts/generator/system_prompt.txt").read_text(encoding="utf-8").strip()
    )
    assert "QUESTION:\nWho?" in generator[1]["content"]
    assert '"relationships": []' in generator[1]["content"]
    assert (
        selector[0]["content"]
        == (REPOSITORY_ROOT / "prompts/selector/system_prompt.txt").read_text(encoding="utf-8").strip()
    )
    assert "SCHEMA UNIT:\n(:Person)" in selector[1]["content"]


class FakeRunner:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))

    def prompt_length(self, messages) -> int:
        return sum(len(message["content"].split()) for message in messages)

    def generate(self, conversations, *, max_new_tokens):
        outputs = []
        for messages in conversations:
            system = messages[0]["content"]
            user = messages[1]["content"]
            if "relevance classifier" in system:
                outputs.append("RELATED" if "(:A " in user or "[:LINKS]" in user else "UNRELATED")
            else:
                outputs.append('{"cypher": "MATCH (n:A) RETURN n.name"}')
        return outputs


class CountingRunner(FakeRunner):
    def __init__(self) -> None:
        self.generated = 0

    def generate(self, conversations, *, max_new_tokens):
        self.generated += len(conversations)
        return super().generate(conversations, max_new_tokens=max_new_tokens)


def test_end_to_end_pipeline_with_fake_model(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    units = [node("A"), node("B"), relation("A", "LINKS", "B")]
    generation = {
        "id": "example-1",
        "source": "fixture",
        "split": "test",
        "graph": "graph",
        "schema_id": "schema-1",
        "question": "Find A",
        "cypher": "MATCH (n:A) RETURN n.name",
        "sub_schema": {
            "nodes": [units[0]["schema"], units[1]["schema"]],
            "relationships": [units[2]["schema"]],
        },
    }
    selection = [
        {
            "id": f"example-1:{unit['id']}",
            "example_id": "example-1",
            "schema_id": "schema-1",
            "unit_id": unit["id"],
            "unit_type": unit["kind"],
            "unit": unit,
            "question": "Find A",
            "label": int(unit["id"] != "node:B"),
        }
        for unit in units
    ]
    write_jsonl(data / "generation_test.jsonl", [generation])
    write_jsonl(data / "selection_test.jsonl", selection)
    output = tmp_path / "output"
    checkpoint = LastCheckpoint("owner/repo", "main", "sft", 7, "qwen3/sft/checkpoint-7")
    manifest = run_dataset_pipeline(
        method="sft",
        checkpoint=checkpoint,
        spec=DatasetSpec("fixture", data),
        runner=FakeRunner(),
        templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
        output_directory=output,
        options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
    )
    assert manifest["method"] == "sft"
    prediction = list(iter_jsonl(output / "generator_predictions.jsonl"))[0]
    assert prediction["predicted_cypher"] == generation["cypher"]
    assert prediction["predicted_sub_schema"] == generation["sub_schema"]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selector"]["accuracy"] == pytest.approx(100.0)
    assert metrics["generator"]["generator_exact_match"] == pytest.approx(100.0)


def test_pipeline_reuses_completed_stages(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    output = tmp_path / "output"
    checkpoint = LastCheckpoint("owner/repo", "main", "sft", 7, "qwen3/sft/checkpoint-7")
    manifest = run_dataset_pipeline(
        method="sft",
        checkpoint=checkpoint,
        spec=DatasetSpec("fixture", tmp_path / "benchmark"),
        runner=FakeRunner(),
        templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
        output_directory=output,
        options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
    )
    assert all(stage["status"] == "reused" for stage in manifest["stages"].values())


def test_pipeline_rejects_stale_outputs_from_another_checkpoint(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    changed_checkpoint = LastCheckpoint("owner/repo", "main", "sft", 8, "qwen3/sft/checkpoint-8")
    with pytest.raises(ValueError, match="different checkpoint"):
        run_dataset_pipeline(
            method="sft",
            checkpoint=changed_checkpoint,
            spec=DatasetSpec("fixture", tmp_path / "benchmark"),
            runner=FakeRunner(),
            templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
            output_directory=tmp_path / "output",
            options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
        )


def test_selector_stage_resumes_a_contiguous_partial_file(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    units = [node("A"), node("B"), relation("A", "LINKS", "B")]
    selection = [
        {
            "id": f"example-1:{unit['id']}",
            "example_id": "example-1",
            "schema_id": "schema-1",
            "unit_id": unit["id"],
            "unit_type": unit["kind"],
            "unit": unit,
            "question": "Find A",
            "label": 1,
        }
        for unit in units
    ]
    write_jsonl(data / "selection_test.jsonl", selection)
    output = tmp_path / "selector_predictions.jsonl"
    partial = output.with_suffix(".jsonl.partial")
    write_jsonl(
        partial,
        [
            {
                "id": selection[0]["id"],
                "example_id": "example-1",
                "schema_id": "schema-1",
                "unit_id": "node:A",
                "unit_type": "node",
                "predicted_label": "RELATED",
                "valid": True,
                "raw_output": "RELATED",
            }
        ],
    )
    runner = CountingRunner()
    result = run_selector_stage(
        DatasetSpec("fixture", data),
        runner,
        PromptTemplates.from_repository(REPOSITORY_ROOT),
        output,
        InferenceOptions(selector_batch_size=2),
    )
    assert result["resumed_rows"] == 1
    assert runner.generated == 2
    assert len(list(iter_jsonl(output))) == 3
    assert not partial.exists()
