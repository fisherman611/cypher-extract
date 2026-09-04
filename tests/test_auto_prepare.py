from __future__ import annotations

import json
from pathlib import Path

from distillation.auto_prepare import (
    AutoPreparePlan,
    build_auto_prepare_plan,
    cache_is_ready,
    ensure_training_data,
)
from distillation.prepare_data import LAYOUT_FILE, SPLIT_FILES


def _write_config(path: Path, *, batch_size: int = 2, dataset: str = "cypher_prepared_train") -> None:
    path.write_text(
        "\n".join(
            [
                "do_train: true",
                f"dataset: {dataset}",
                "eval_dataset: cypher_prepared_eval",
                "dataset_dir: data/llamafactory",
                f"per_device_train_batch_size: {batch_size}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_plan_uses_batch_specific_cache_and_honors_cli_override(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    _write_config(config, batch_size=2)

    plan = build_auto_prepare_plan(
        config,
        ["per_device_train_batch_size=8"],
        project_root=tmp_path,
    )

    assert plan is not None
    assert plan.batch_size == 8
    assert plan.prepared_dir == tmp_path / "data" / "prepared" / "batch_8"
    assert plan.dataset_dir == tmp_path / "data" / "llamafactory" / "batch_8"
    assert plan.dataset_dir_override == "data/llamafactory/batch_8"


def test_plan_skips_non_managed_dataset(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    _write_config(config, dataset="external_train")

    assert build_auto_prepare_plan(config, project_root=tmp_path) is None


def _publish_ready_cache(plan: AutoPreparePlan) -> None:
    plan.dataset_dir.mkdir(parents=True, exist_ok=True)
    (plan.dataset_dir / "dataset_info.json").write_text("{}\n", encoding="utf-8")
    (plan.dataset_dir / LAYOUT_FILE).write_text(
        json.dumps({"batch_size": plan.batch_size}) + "\n",
        encoding="utf-8",
    )
    for split in SPLIT_FILES:
        (plan.dataset_dir / f"cypher_prepared_{split}.jsonl").write_text("{}\n", encoding="utf-8")


def test_cache_readiness_requires_matching_layout_and_all_files(tmp_path: Path) -> None:
    plan = AutoPreparePlan(
        batch_size=4,
        grounding_input_dir=tmp_path / "grounding",
        prepared_dir=tmp_path / "prepared",
        dataset_dir=tmp_path / "llamafactory",
        dataset_dir_override="llamafactory",
    )
    assert not cache_is_ready(plan)

    _publish_ready_cache(plan)
    assert cache_is_ready(plan)

    (plan.dataset_dir / LAYOUT_FILE).write_text('{"batch_size": 8}\n', encoding="utf-8")
    assert not cache_is_ready(plan)

    (plan.dataset_dir / LAYOUT_FILE).write_text("[]\n", encoding="utf-8")
    assert not cache_is_ready(plan)

    (plan.dataset_dir / LAYOUT_FILE).write_text('{"batch_size": null}\n', encoding="utf-8")
    assert not cache_is_ready(plan)


def test_ensure_builds_missing_cache_once(tmp_path: Path) -> None:
    plan = AutoPreparePlan(
        batch_size=4,
        grounding_input_dir=tmp_path / "grounding",
        prepared_dir=tmp_path / "prepared",
        dataset_dir=tmp_path / "llamafactory",
        dataset_dir_override="llamafactory",
    )
    plan.grounding_input_dir.mkdir()
    for task in ("generation", "selection"):
        for split in ("train", "dev", "test"):
            (plan.grounding_input_dir / f"{task}_{split}.jsonl").write_text("{}\n", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(command: list[str], _project_root: Path) -> None:
        commands.append(command)
        if "prepare_llamafactory_data.py" in command[1]:
            _publish_ready_cache(plan)

    assert ensure_training_data(plan, project_root=tmp_path, run_command=fake_run)
    assert [Path(command[1]).name for command in commands] == [
        "prepare_multitask_prompts.py",
        "prepare_llamafactory_data.py",
    ]
    assert not ensure_training_data(plan, project_root=tmp_path, run_command=fake_run)
    assert len(commands) == 2
