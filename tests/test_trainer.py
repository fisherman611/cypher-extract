import ast
from pathlib import Path


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


def test_kd_trainer_saves_and_validates_custom_resume_state() -> None:
    trainer_path = Path(__file__).parents[1] / "src" / "distillation" / "trainer.py"
    source = trainer_path.read_text(encoding="utf-8")

    assert "self.trainer.save_resume_manifest(checkpoint, state.global_step)" in source
    assert "self.trainer.save_distillm_state(checkpoint)" in source
    assert "checkpoint_path = validate_resume_checkpoint(" in source
    assert "get_last_checkpoint" not in source


def test_eval_generation_splits_selector_and_generator_protocols() -> None:
    trainer_path = Path(__file__).parents[1] / "src" / "distillation" / "trainer.py"
    source = trainer_path.read_text(encoding="utf-8")

    assert "(generator_rows, gen_kwargs)" in source
    assert "(selector_rows, {**gen_kwargs, **selector_generation_kwargs()})" in source
    assert "generated_tokens[indices, : task_tokens.size(1)] = task_tokens" in source
