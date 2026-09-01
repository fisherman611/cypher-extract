import ast
from pathlib import Path

import pytest

from distillation.checkpointing import (
    CheckpointNotReadyError,
    resolve_automatic_resume_checkpoint,
    write_latest_checkpoint_pointer,
)


def test_kd_trainer_disables_transformers_loss_kwargs_contract() -> None:
    trainer_path = Path(__file__).parents[1] / "src" / "distillation" / "trainer.py"
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    kd_trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KDTrainer"
    )
    init = next(
        node for node in kd_trainer.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "model_accepts_loss_kwargs"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value is False


def test_automatic_resume_uses_pointer_instead_of_stale_higher_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-314").mkdir()
    (tmp_path / "checkpoint-1570").mkdir()
    write_latest_checkpoint_pointer(tmp_path, "checkpoint-314")

    assert resolve_automatic_resume_checkpoint(tmp_path) == str(tmp_path / "checkpoint-314")


def test_automatic_resume_rejects_run_without_a_new_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1570").mkdir()
    write_latest_checkpoint_pointer(tmp_path, "training-in-progress")

    with pytest.raises(CheckpointNotReadyError, match="latest fresh run stopped"):
        resolve_automatic_resume_checkpoint(tmp_path)
