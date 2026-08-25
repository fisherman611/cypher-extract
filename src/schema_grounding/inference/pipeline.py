from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from distillation.metrics import compute_task_metrics, extract_cypher
from schema_grounding.inference.checkpoints import LastCheckpoint
from schema_grounding.inference.data import (
    DatasetSpec,
    grouped_paired_units,
    iter_jsonl,
    load_generation_index,
    paired_rows,
)
from schema_grounding.inference.merge import merge_schema_units
from schema_grounding.inference.outputs import ResumableJsonl, write_json_atomic, write_json_line
from schema_grounding.inference.parsing import parse_selector_label
from schema_grounding.inference.prompting import Message, PromptTemplates


class GenerationRunner(Protocol):
    model: Any

    def generate(self, conversations: Sequence[Sequence[Message]], *, max_new_tokens: int) -> list[str]: ...

    def prompt_length(self, messages: Sequence[Message]) -> int: ...


@dataclass(frozen=True)
class InferenceOptions:
    selector_batch_size: int = 128
    generator_batch_size: int = 16
    selector_max_new_tokens: int = 8
    generator_max_new_tokens: int = 256
    close_relation_endpoints: bool = True

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "close_relation_endpoints":
                continue
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _batches(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _rows_after_progress(
    source_path: Path, completed_rows: int, last_completed_id: str | None
) -> Iterator[dict[str, Any]]:
    observed = 0
    for observed, row in enumerate(iter_jsonl(source_path), 1):
        if observed <= completed_rows:
            if observed == completed_rows and row.get("id") != last_completed_id:
                raise ValueError(
                    f"Resume mismatch at row {observed}: source has {row.get('id')!r}, "
                    f"partial output has {last_completed_id!r}"
                )
            continue
        yield row
    if completed_rows > observed:
        raise ValueError(f"Partial output has {completed_rows} rows but source {source_path} has only {observed}")


def run_selector_stage(
    spec: DatasetSpec,
    runner: GenerationRunner,
    templates: PromptTemplates,
    output_path: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    output = ResumableJsonl(output_path)
    if output.complete:
        return {"status": "reused", "path": str(output_path)}
    completed_rows, last_id = output.progress()
    started = time.monotonic()
    generated_rows = 0
    invalid_rows = 0
    with output.open_append() as handle:
        source = _rows_after_progress(spec.selection_test, completed_rows, last_id)
        for batch in _batches(source, options.selector_batch_size):
            conversations = [
                templates.selector_messages(str(row["question"]), str(row["unit"]["text"])) for row in batch
            ]
            raw_outputs = runner.generate(conversations, max_new_tokens=options.selector_max_new_tokens)
            if len(raw_outputs) != len(batch):
                raise RuntimeError("Model returned the wrong number of selector outputs")
            for row, raw_output in zip(batch, raw_outputs, strict=True):
                parsed = parse_selector_label(raw_output)
                invalid_rows += parsed is None
                write_json_line(
                    handle,
                    {
                        "id": row["id"],
                        "example_id": row["example_id"],
                        "schema_id": row["schema_id"],
                        "unit_id": row["unit_id"],
                        "unit_type": row["unit_type"],
                        "predicted_label": parsed or "INVALID",
                        "valid": parsed is not None,
                        "raw_output": raw_output,
                    },
                )
                generated_rows += 1
            handle.flush()
            if generated_rows and generated_rows % (options.selector_batch_size * 100) == 0:
                print(f"[{spec.name}/selector] generated {completed_rows + generated_rows} unit labels", flush=True)
    output.publish()
    return {
        "status": "completed",
        "path": str(output_path),
        "resumed_rows": completed_rows,
        "generated_rows": generated_rows,
        "invalid_rows_in_current_run": invalid_rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_merge_stage(
    spec: DatasetSpec,
    selector_predictions: Path,
    output_path: Path,
    *,
    close_relation_endpoints: bool,
) -> dict[str, Any]:
    if output_path.is_file():
        return {"status": "reused", "path": str(output_path)}
    generation_order, generation_rows = load_generation_index(spec.generation_test)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_examples: set[str] = set()
    closure_count = 0
    empty_count = 0
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example_id, source_rows, prediction_rows in grouped_paired_units(
            spec.selection_test, selector_predictions
        ):
            generation = generation_rows.get(example_id)
            if generation is None:
                raise ValueError(f"Selector example {example_id!r} has no generator test row")
            units = [row["unit"] for row in source_rows]
            related_ids = [
                str(source["unit_id"])
                for source, prediction in zip(source_rows, prediction_rows, strict=True)
                if prediction["predicted_label"] == "RELATED"
            ]
            merged = merge_schema_units(
                units,
                related_ids,
                close_relation_endpoints=close_relation_endpoints,
            )
            closure_count += len(merged.closure_added_node_ids)
            empty_count += not merged.sub_schema["nodes"] and not merged.sub_schema["relationships"]
            write_json_line(
                handle,
                {
                    "id": example_id,
                    "source": generation["source"],
                    "split": generation["split"],
                    "graph": generation["graph"],
                    "schema_id": generation["schema_id"],
                    "question": generation["question"],
                    "predicted_sub_schema": merged.sub_schema,
                    "directly_selected_unit_ids": list(merged.directly_selected_unit_ids),
                    "closure_added_node_ids": list(merged.closure_added_node_ids),
                },
            )
            seen_examples.add(example_id)
    missing = [example_id for example_id in generation_order if example_id not in seen_examples]
    if missing:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Generator examples have no selector units: {missing[:5]}")
    temporary.replace(output_path)
    return {
        "status": "completed",
        "path": str(output_path),
        "examples": len(seen_examples),
        "closure_added_nodes": closure_count,
        "empty_sub_schemas": empty_count,
        "elapsed_seconds": time.monotonic() - started,
    }


def _context_limit(runner: GenerationRunner) -> int | None:
    model_limit = getattr(getattr(runner, "model", None), "config", None)
    value = getattr(model_limit, "max_position_embeddings", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def run_generator_stage(
    spec: DatasetSpec,
    runner: GenerationRunner,
    templates: PromptTemplates,
    sub_schema_path: Path,
    output_path: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    output = ResumableJsonl(output_path)
    if output.complete:
        return {"status": "reused", "path": str(output_path)}
    _, generation_rows = load_generation_index(spec.generation_test)
    completed_rows, last_id = output.progress()
    started = time.monotonic()
    generated_rows = 0
    max_prompt_length = 0
    context_limit = _context_limit(runner)
    with output.open_append() as handle:
        source = _rows_after_progress(sub_schema_path, completed_rows, last_id)
        for batch in _batches(source, options.generator_batch_size):
            conversations = [
                templates.generator_messages(str(row["question"]), row["predicted_sub_schema"]) for row in batch
            ]
            prompt_lengths = [runner.prompt_length(messages) for messages in conversations]
            max_prompt_length = max(max_prompt_length, *prompt_lengths)
            if context_limit is not None:
                overflowing = [
                    length for length in prompt_lengths if length + options.generator_max_new_tokens > context_limit
                ]
                if overflowing:
                    raise ValueError(
                        f"Generator prompt plus response budget exceeds context limit {context_limit}: "
                        f"prompt_length={max(overflowing)}"
                    )
            raw_outputs = runner.generate(conversations, max_new_tokens=options.generator_max_new_tokens)
            if len(raw_outputs) != len(batch):
                raise RuntimeError("Model returned the wrong number of generator outputs")
            for row, raw_output, prompt_length in zip(batch, raw_outputs, prompt_lengths, strict=True):
                # Gold data is attached only after generation and is never part of the prompt.
                reference = generation_rows[str(row["id"])]
                write_json_line(
                    handle,
                    {
                        **row,
                        "raw_output": raw_output,
                        "predicted_cypher": extract_cypher(raw_output),
                        "prompt_length": prompt_length,
                        "reference_cypher": reference["cypher"],
                        "reference_sub_schema": reference["sub_schema"],
                    },
                )
                generated_rows += 1
            handle.flush()
            if generated_rows and generated_rows % (options.generator_batch_size * 25) == 0:
                print(
                    f"[{spec.name}/generator] generated {completed_rows + generated_rows} Cypher queries",
                    flush=True,
                )
    output.publish()
    return {
        "status": "completed",
        "path": str(output_path),
        "resumed_rows": completed_rows,
        "generated_rows": generated_rows,
        "max_prompt_length_in_current_run": max_prompt_length,
        "elapsed_seconds": time.monotonic() - started,
    }


def compute_inference_metrics(
    spec: DatasetSpec,
    selector_predictions: Path,
    generator_predictions: Path,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = invalid = 0
    for source, prediction in paired_rows(spec.selection_test, selector_predictions):
        expected = "RELATED" if source["label"] == 1 else "UNRELATED"
        predicted = prediction["predicted_label"]
        invalid += predicted == "INVALID"
        if expected == "RELATED" and predicted == "RELATED":
            true_positive += 1
        elif expected == "UNRELATED" and predicted == "RELATED":
            false_positive += 1
        elif expected == "RELATED":
            false_negative += 1
        else:
            true_negative += 1
    selector_count = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    predicted_cypher: list[str] = []
    reference_cypher: list[str] = []
    closure_nodes = empty_schemas = 0
    for row in iter_jsonl(generator_predictions):
        predicted_cypher.append(str(row["predicted_cypher"]))
        reference_cypher.append(str(row["reference_cypher"]))
        closure_nodes += len(row["closure_added_node_ids"])
        schema = row["predicted_sub_schema"]
        empty_schemas += not schema["nodes"] and not schema["relationships"]
    generator_metrics = compute_task_metrics(predicted_cypher, reference_cypher)
    return {
        "selector": {
            "count": selector_count,
            "accuracy": 100.0 * (true_positive + true_negative) / selector_count,
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "f1": 100.0 * f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "invalid": invalid,
        },
        "generator": generator_metrics,
        "pipeline": {
            "closure_added_nodes": closure_nodes,
            "empty_sub_schemas": empty_schemas,
        },
    }


def run_dataset_pipeline(
    *,
    method: str,
    checkpoint: LastCheckpoint,
    spec: DatasetSpec,
    runner: GenerationRunner,
    templates: PromptTemplates,
    output_directory: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    options.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": method,
        "dataset": spec.name,
        "checkpoint": asdict(checkpoint),
        "options": asdict(options),
        "selection_test": str(spec.selection_test.resolve()),
        "generation_test": str(spec.generation_test.resolve()),
    }
    run_config_path = output_directory / "run_config.json"
    if run_config_path.is_file():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError(
                f"Existing inference outputs in {output_directory} were created with a different "
                "checkpoint, dataset, or option set. Choose a new --output-dir or remove that "
                "method/dataset directory."
            )
    else:
        write_json_atomic(run_config_path, run_config)
    selector_path = output_directory / "selector_predictions.jsonl"
    sub_schema_path = output_directory / "predicted_subschemas.jsonl"
    generator_path = output_directory / "generator_predictions.jsonl"
    stages = {
        "selector": run_selector_stage(spec, runner, templates, selector_path, options),
        "merge": run_merge_stage(
            spec,
            selector_path,
            sub_schema_path,
            close_relation_endpoints=options.close_relation_endpoints,
        ),
        "generator": run_generator_stage(spec, runner, templates, sub_schema_path, generator_path, options),
    }
    metrics = compute_inference_metrics(spec, selector_path, generator_path)
    write_json_atomic(output_directory / "metrics.json", metrics)
    manifest = {
        "method": method,
        "dataset": spec.name,
        "checkpoint": asdict(checkpoint),
        "options": asdict(options),
        "files": {
            "selector_predictions": selector_path.name,
            "predicted_subschemas": sub_schema_path.name,
            "generator_predictions": generator_path.name,
            "metrics": "metrics.json",
        },
        "stages": stages,
    }
    write_json_atomic(output_directory / "manifest.json", manifest)
    return manifest
