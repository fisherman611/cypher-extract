#!/usr/bin/env python3
"""Run selector -> predicted sub-schema -> Cypher generation inference."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from schema_grounding.inference.checkpoints import (  # noqa: E402
    DEFAULT_METHODS,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_REPO_ID,
    download_inference_checkpoint,
    resolve_last_checkpoint,
)
from schema_grounding.inference.data import default_dataset_specs  # noqa: E402
from schema_grounding.inference.model import ModelRunner  # noqa: E402
from schema_grounding.inference.pipeline import (  # noqa: E402
    InferenceOptions,
    model_runner_required,
    prepare_run_directory,
    run_dataset_pipeline,
)
from schema_grounding.inference.prompting import PromptTemplates  # noqa: E402


def comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    datasets = default_dataset_specs(REPOSITORY_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods or 'all'. 'all' includes teacher_lora and excludes da_kd.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(datasets),
        help=f"Comma-separated datasets. Choices: {', '.join(datasets)}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference/qwen3"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--selector-batch-size", type=int, default=128)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    parser.add_argument("--selector-max-new-tokens", type=int, default=8)
    parser.add_argument("--generator-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--no-merge-adapter",
        action="store_true",
        help="Keep LoRA modules separate instead of merging them into the base model.",
    )
    parser.add_argument(
        "--no-relation-endpoint-closure",
        action="store_true",
        help="Do not add endpoint nodes for selected relationships.",
    )
    return parser.parse_args()


def validate_choices(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    methods = list(DEFAULT_METHODS) if args.methods == "all" else comma_separated(args.methods)
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unknown_methods = sorted(set(methods).difference(DEFAULT_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")
    if len(methods) != len(set(methods)):
        raise ValueError("--methods contains duplicates")

    available_datasets = default_dataset_specs(REPOSITORY_ROOT)
    datasets = comma_separated(args.datasets)
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset")
    unknown_datasets = sorted(set(datasets).difference(available_datasets))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {', '.join(unknown_datasets)}")
    if len(datasets) != len(set(datasets)):
        raise ValueError("--datasets contains duplicates")
    return methods, datasets


def main() -> None:
    args = parse_args()
    methods, dataset_names = validate_choices(args)
    specs = default_dataset_specs(REPOSITORY_ROOT)
    templates = PromptTemplates.from_repository(REPOSITORY_ROOT)
    options = InferenceOptions(
        selector_batch_size=args.selector_batch_size,
        generator_batch_size=args.generator_batch_size,
        selector_max_new_tokens=args.selector_max_new_tokens,
        generator_max_new_tokens=args.generator_max_new_tokens,
        close_relation_endpoints=not args.no_relation_endpoint_closure,
    )
    options.validate()

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "methods": methods,
                "datasets": dataset_names,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for method in methods:
        checkpoint = resolve_last_checkpoint(
            method,
            repo_id=args.repo_id,
            model_family=args.model_family,
            revision=args.revision,
        )
        print(f"[{method}] resolved {checkpoint.uri} (step {checkpoint.step}, revision {checkpoint.revision})")
        planned_runs = [
            (dataset_name, args.output_dir.resolve() / method / dataset_name) for dataset_name in dataset_names
        ]
        for dataset_name, output_directory in planned_runs:
            prepare_run_directory(
                method=method,
                checkpoint=checkpoint,
                spec=specs[dataset_name],
                templates=templates,
                output_directory=output_directory,
                options=options,
            )

        runner = None
        if any(model_runner_required(output_directory) for _, output_directory in planned_runs):
            adapter_path = download_inference_checkpoint(checkpoint, cache_dir=args.cache_dir)
            runner = ModelRunner.from_adapter(
                adapter_path,
                dtype=args.dtype,
                device=args.device,
                merge_adapter=not args.no_merge_adapter,
            )
        else:
            print(f"[{method}] all model-backed stages are complete; skipping model load")
        try:
            for dataset_name, output_directory in planned_runs:
                print(f"[{method}/{dataset_name}] starting")
                run_dataset_pipeline(
                    method=method,
                    checkpoint=checkpoint,
                    spec=specs[dataset_name],
                    runner=runner,
                    templates=templates,
                    output_directory=output_directory,
                    options=options,
                )
                print(f"[{method}/{dataset_name}] completed")
        finally:
            if runner is not None:
                del runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
