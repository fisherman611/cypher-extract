#!/usr/bin/env python3
"""Run selector -> predicted sub-schema -> Cypher generation inference."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from distillation.utils import seed_everything  # noqa: E402
from schema_grounding.inference.checkpoints import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_METHODS,
    DEFAULT_MODEL_FAMILY,
    SUPPORTED_METHODS,
    LastCheckpoint,
    resolve_checkpoint_directory,
    resolve_last_checkpoint,
)
from schema_grounding.inference.data import DatasetSpec, default_dataset_specs  # noqa: E402
from schema_grounding.inference.model import ModelRunner  # noqa: E402
from schema_grounding.inference.pipeline import (  # noqa: E402
    InferenceOptions,
    inference_run_complete,
    model_runner_required,
    prepare_run_directory,
    run_dataset_pipeline,
)
from schema_grounding.inference.prompting import PromptTemplates, chat_template_metadata  # noqa: E402

DEFAULT_INFERENCE_SEEDS = (10, 42, 50, 100, 1234)
PlannedRun: TypeAlias = tuple[str, Path, InferenceOptions]
RunGroup: TypeAlias = tuple[int, str, list[PlannedRun]]


def comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    raw_seeds = comma_separated(value)
    if not raw_seeds:
        raise ValueError("--seeds must contain at least one seed")
    try:
        seeds = [int(seed) for seed in raw_seeds]
    except ValueError as error:
        raise ValueError("--seeds must be comma-separated integers") from error
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain only non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    return seeds


def parse_args() -> argparse.Namespace:
    datasets = default_dataset_specs(REPOSITORY_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Local root containing <model-family>/<method>/checkpoint-N directories.",
    )
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods or 'all'. 'all' preserves the default LoRA matrix; "
            "teacher_full can be requested explicitly."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=",".join(datasets),
        help=f"Comma-separated datasets. Choices: {', '.join(datasets)}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Inference output root. Defaults to results/inference/lora/<model-family>.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help=(
            "Number of visible CUDA devices used by parallel inference workers. "
            "Use CUDA_VISIBLE_DEVICES to select the physical GPUs."
        ),
    )
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--selector-batch-size", type=int, default=100)
    parser.add_argument("--generator-batch-size", type=int, default=100)
    parser.add_argument("--generator-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_INFERENCE_SEEDS),
        help="Comma-separated inference seeds; each seed is written to its own seed<value> folder.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing outputs of the selected seeds/methods/datasets and run them again "
            "instead of skipping completed runs."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("results/inference/lora") / args.model_family
    return args


def validate_choices(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    methods = list(DEFAULT_METHODS) if args.methods == "all" else comma_separated(args.methods)
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unknown_methods = sorted(set(methods).difference(SUPPORTED_METHODS))
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


def build_seed_first_run_groups(
    *,
    methods: list[str],
    dataset_names: list[str],
    seeds: list[int],
    output_root: Path,
    options: InferenceOptions,
) -> list[RunGroup]:
    """Plan every dataset run, completing one seed before moving to the next."""

    groups = []
    for seed in seeds:
        seed_options = replace(options, seed=seed)
        for method in methods:
            runs = [
                (
                    dataset_name,
                    output_root / f"seed{seed}" / method / dataset_name,
                    seed_options,
                )
                for dataset_name in dataset_names
            ]
            groups.append((seed, method, runs))
    return groups


def clear_planned_run_outputs(
    run_groups: list[RunGroup],
) -> None:
    """Remove every planned dataset run directory so it is inferred from scratch."""

    for seed, method, planned_runs in run_groups:
        for dataset_name, output_directory, _ in planned_runs:
            if output_directory.exists():
                print(f"[seed{seed}/{method}/{dataset_name}] --overwrite: removing {output_directory}")
                shutil.rmtree(output_directory)


def resolve_worker_devices(device: str, num_gpus: int, *, cuda_device_count: int | None = None) -> list[str]:
    """Resolve one logical CUDA device per inference worker."""

    if num_gpus <= 0:
        raise ValueError("--num-gpus must be a positive integer")
    if num_gpus == 1:
        return [device]

    target_device = torch.device(device)
    if target_device.type != "cuda" or target_device.index is not None:
        raise ValueError("--num-gpus greater than 1 requires --device cuda")
    available_devices = torch.cuda.device_count() if cuda_device_count is None else cuda_device_count
    if available_devices < num_gpus:
        raise RuntimeError(
            f"--num-gpus={num_gpus} requested, but only {available_devices} CUDA device(s) are visible"
        )
    return [f"cuda:{index}" for index in range(num_gpus)]


def distribute_run_groups(run_groups: list[RunGroup], num_workers: int) -> list[list[RunGroup]]:
    """Distribute work across workers, splitting datasets only when necessary."""

    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not run_groups:
        return []

    worker_count = min(num_workers, sum(len(planned_runs) for _, _, planned_runs in run_groups))
    expanded_groups = [(seed, method, list(planned_runs)) for seed, method, planned_runs in run_groups]
    while len(expanded_groups) < worker_count:
        split_index = max(range(len(expanded_groups)), key=lambda index: len(expanded_groups[index][2]))
        seed, method, planned_runs = expanded_groups[split_index]
        if len(planned_runs) < 2:
            break
        expanded_groups[split_index] = (seed, method, planned_runs[::2])
        expanded_groups.append((seed, method, planned_runs[1::2]))

    worker_count = min(worker_count, len(expanded_groups))
    return [expanded_groups[worker_index::worker_count] for worker_index in range(worker_count)]


def run_inference_groups(
    run_groups: list[RunGroup],
    *,
    device: str,
    checkpoints: dict[str, LastCheckpoint],
    checkpoint_paths: dict[str, Path],
    specs: dict[str, DatasetSpec],
    templates: PromptTemplates,
    dtype: str,
    merge_adapter: bool,
    model_family: str,
    worker_index: int = 0,
) -> None:
    """Run assigned seed/method groups on one device."""

    target_device = torch.device(device)
    if target_device.type == "cuda" and target_device.index is not None:
        torch.cuda.set_device(target_device)
    print(f"[worker{worker_index}/{device}] started with {len(run_groups)} group(s)", flush=True)
    for seed, method, planned_runs in run_groups:
        checkpoint = checkpoints[method]
        pending_runs = [
            planned_run
            for planned_run in planned_runs
            if not inference_run_complete(planned_run[1])
        ]
        if not pending_runs:
            print(f"[worker{worker_index}/{device}/seed{seed}/{method}] already completed; skipping", flush=True)
            continue
        runner = None
        if any(model_runner_required(output_directory) for _, output_directory, _ in pending_runs):
            runner = ModelRunner.from_checkpoint(
                checkpoint_paths[method],
                dtype=dtype,
                device=device,
                merge_adapter=merge_adapter,
                model_family=model_family,
            )
        else:
            print(
                f"[worker{worker_index}/{device}/seed{seed}/{method}] "
                "all model-backed stages are complete; skipping model load",
                flush=True,
            )
        try:
            for dataset_name, output_directory, seed_options in pending_runs:
                seed_everything(seed, rank_offset=False)
                prefix = f"[worker{worker_index}/{device}/seed{seed}/{method}/{dataset_name}]"
                print(f"{prefix} starting", flush=True)
                run_dataset_pipeline(
                    method=method,
                    checkpoint=checkpoint,
                    spec=specs[dataset_name],
                    runner=runner,
                    templates=templates,
                    output_directory=output_directory,
                    options=seed_options,
                )
                print(f"{prefix} completed", flush=True)
        finally:
            if runner is not None:
                del runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"[worker{worker_index}/{device}] completed", flush=True)


def run_parallel_inference(
    worker_run_groups: list[list[RunGroup]],
    *,
    devices: list[str],
    checkpoints: dict[str, LastCheckpoint],
    checkpoint_paths: dict[str, Path],
    specs: dict[str, DatasetSpec],
    templates: PromptTemplates,
    dtype: str,
    merge_adapter: bool,
    model_family: str,
) -> None:
    """Launch one spawn-based process per CUDA device and propagate failures."""

    context = mp.get_context("spawn")
    processes = []
    for worker_index, (device, assigned_groups) in enumerate(zip(devices, worker_run_groups, strict=True)):
        process = context.Process(
            target=run_inference_groups,
            kwargs={
                "run_groups": assigned_groups,
                "device": device,
                "checkpoints": checkpoints,
                "checkpoint_paths": checkpoint_paths,
                "specs": specs,
                "templates": templates,
                "dtype": dtype,
                "merge_adapter": merge_adapter,
                "model_family": model_family,
                "worker_index": worker_index,
            },
            name=f"inference-worker-{worker_index}",
        )
        process.start()
        processes.append(process)
    try:
        for process in processes:
            process.join()
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise

    failed_workers = [process for process in processes if process.exitcode != 0]
    if failed_workers:
        failures = ", ".join(f"{process.name} (exit code {process.exitcode})" for process in failed_workers)
        raise RuntimeError(f"Parallel inference failed: {failures}")


def main() -> None:
    args = parse_args()
    methods, dataset_names = validate_choices(args)
    seeds = parse_seeds(args.seeds)
    devices = resolve_worker_devices(args.device, args.num_gpus)
    specs = default_dataset_specs(REPOSITORY_ROOT)
    templates = PromptTemplates.from_repository(REPOSITORY_ROOT)
    options = InferenceOptions(
        selector_batch_size=args.selector_batch_size,
        generator_batch_size=args.generator_batch_size,
        generator_max_new_tokens=args.generator_max_new_tokens,
        close_relation_endpoints=not args.no_relation_endpoint_closure,
        generator_do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )
    options.validate()

    print(
        json.dumps(
            {
                "checkpoint_root": str(args.checkpoint_root.resolve()),
                "methods": methods,
                "datasets": dataset_names,
                "seeds": seeds,
                "devices": devices,
                "selector_decoding": {
                    "labels": ["YES", "NO"],
                    "output_format": {"label": "YES|NO"},
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": options.selector_max_new_tokens,
                },
                "generator_decoding": {
                    "do_sample": options.generator_do_sample,
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "top_k": options.top_k,
                    "num_beams": options.num_beams,
                },
                "chat_template": chat_template_metadata(args.model_family)["name"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    checkpoints = {}
    checkpoint_paths = {}
    for method in methods:
        checkpoints[method] = resolve_last_checkpoint(
            method,
            checkpoint_root=args.checkpoint_root,
            model_family=args.model_family,
        )
        checkpoint = checkpoints[method]
        print(
            f"[{method}] resolved {checkpoint.uri} "
            f"(step {checkpoint.step}, fingerprint {checkpoint.fingerprint[:12]})"
        )
        # Validate every local model/adapter before creating any resumable
        # output directory or run_config.json file.
        checkpoint_paths[method] = resolve_checkpoint_directory(checkpoint)

    run_groups = build_seed_first_run_groups(
        methods=methods,
        dataset_names=dataset_names,
        seeds=seeds,
        output_root=args.output_dir.resolve(),
        options=options,
    )
    if args.overwrite:
        # Checkpoints were validated above, so old outputs are only removed
        # once the replacement run is known to be loadable.
        clear_planned_run_outputs(run_groups)
    for _seed, method, planned_runs in run_groups:
        checkpoint = checkpoints[method]
        for dataset_name, output_directory, seed_options in planned_runs:
            prepare_run_directory(
                method=method,
                checkpoint=checkpoint,
                spec=specs[dataset_name],
                templates=templates,
                output_directory=output_directory,
                options=seed_options,
            )

    pending_run_groups = [
        group
        for group in run_groups
        if any(not inference_run_complete(output_directory) for _, output_directory, _ in group[2])
    ]
    if not pending_run_groups:
        print("Inference already completed; skipping")
        return

    worker_run_groups = distribute_run_groups(pending_run_groups, len(devices))
    active_devices = devices[: len(worker_run_groups)]
    if len(active_devices) == 1:
        run_inference_groups(
            worker_run_groups[0],
            device=active_devices[0],
            checkpoints=checkpoints,
            checkpoint_paths=checkpoint_paths,
            specs=specs,
            templates=templates,
            dtype=args.dtype,
            merge_adapter=not args.no_merge_adapter,
            model_family=args.model_family,
        )
        return

    run_parallel_inference(
        worker_run_groups,
        devices=active_devices,
        checkpoints=checkpoints,
        checkpoint_paths=checkpoint_paths,
        specs=specs,
        templates=templates,
        dtype=args.dtype,
        merge_adapter=not args.no_merge_adapter,
        model_family=args.model_family,
    )


if __name__ == "__main__":
    main()
