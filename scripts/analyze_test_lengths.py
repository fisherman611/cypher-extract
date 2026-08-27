#!/usr/bin/env python3
"""Measure prompt and full-sequence token lengths for all benchmark test sets."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "cypherbench": ROOT / "data" / "cypherbench_schema_grounding_full_final",
    "mind_the_query": ROOT / "data" / "mind_the_query_schema_grounding_full",
    "neo4j_text2cypher": ROOT / "data" / "neo4j_text2cypher_schema_grounding_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Tokenizer model name or local path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "qwen3_test_length_analysis.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per task, useful for a smoke test.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download the tokenizer; use the local Hugging Face cache only.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path, max_rows: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        emitted = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_rows is not None and emitted >= max_rows:
                return
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            emitted += 1
            yield line_number, row


def load_prompt(relative_path: str) -> str:
    return (ROOT / "prompts" / relative_path).read_text(encoding="utf-8").strip()


def build_messages(row: dict[str, Any], task: str) -> list[dict[str, str]]:
    if task == "generator":
        system = load_prompt("generator/system_prompt.txt")
        user_template = load_prompt("generator/user_prompt.txt")
        user = user_template.format(
            question=row["question"],
            schema=json.dumps(row["sub_schema"], ensure_ascii=False, indent=2),
        )
        response = json.dumps({"cypher": row["cypher"]}, ensure_ascii=False)
    elif task == "selector":
        system = load_prompt("selector/system_prompt.txt")
        user_template = load_prompt("selector/user_prompt.txt")
        user = user_template.format(question=row["question"], schema_unit=row["unit"]["text"])
        response = "YES" if row["label"] == 1 else "NO"
    else:
        raise ValueError(f"Unsupported task: {task}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]


def token_count(tokenizer: Any, messages: list[dict[str, str]], *, generation_prompt: bool) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
        enable_thinking=False,
    )
    if hasattr(token_ids, "keys") and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    return len(token_ids)


def analyze_file(path: Path, task: str, tokenizer: Any, max_rows: int | None) -> dict[str, int]:
    maxima = {"rows": 0, "max_prompt_length": 0, "max_length": 0, "max_target_length": 0}
    for line_number, row in iter_jsonl(path, max_rows):
        try:
            messages = build_messages(row, task)
            prompt_length = token_count(tokenizer, messages[:-1], generation_prompt=True)
            full_length = token_count(tokenizer, messages, generation_prompt=False)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid {task} row at {path}:{line_number}: {error}") from error
        target_length = max(full_length - prompt_length, 0)
        maxima["rows"] += 1
        maxima["max_prompt_length"] = max(maxima["max_prompt_length"], prompt_length)
        maxima["max_length"] = max(maxima["max_length"], full_length)
        maxima["max_target_length"] = max(maxima["max_target_length"], target_length)
    if maxima["rows"] == 0:
        raise ValueError(f"No rows found in {path}")
    return maxima


def analyze_all(tokenizer: Any, max_rows: int | None = None) -> dict[str, dict[str, dict[str, int]]]:
    results: dict[str, dict[str, dict[str, int]]] = {}
    for dataset_name, directory in DEFAULT_DATASETS.items():
        results[dataset_name] = {}
        for task, filename in (("generator", "generation_test.jsonl"), ("selector", "selection_test.jsonl")):
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing test data: {path}")
            results[dataset_name][task] = analyze_file(path, task, tokenizer, max_rows)
            values = results[dataset_name][task]
            print(
                f"{dataset_name:20} {task:9} rows={values['rows']:6} "
                f"max_prompt_length={values['max_prompt_length']:4} max_length={values['max_length']:4}"
            )
    return results


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.getenv("HF_TOKEN") or os.getenv("HF_READ_TOKEN"),
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.model_max_length = 10**9
    report = {
        "model": args.model,
        "template": "qwen3_nothink",
        "datasets": analyze_all(tokenizer, args.max_rows),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
