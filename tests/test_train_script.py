from pathlib import Path

import yaml

TRAIN_SCRIPT = Path("scripts/train.sh")


def _script() -> str:
    # Normalise line endings so the checks do not depend on the checkout's autocrlf setting.
    return TRAIN_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_wandb_logging_is_opt_in_and_defaults_to_the_project_name() -> None:
    script = _script()

    assert 'if [[ "${USE_WANDB:-0}" == "1" ]]; then' in script
    assert 'export WANDB_PROJECT="${WANDB_PROJECT:-cypher-extract}"' in script
    assert 'WANDB_OVERRIDES+=("report_to=wandb" "run_name=${WANDB_RUN_NAME:-${WANDB_DEFAULT_RUN_NAME}}")' in script
    assert "WANDB_OVERRIDES=()" in script


def test_wandb_overrides_precede_user_arguments_so_explicit_values_win() -> None:
    script = _script()

    launch = script[script.index("torchrun \\") :]
    assert launch.index('"${WANDB_OVERRIDES[@]}"') < launch.index('"$@"')


def test_every_config_keeps_wandb_off_by_default() -> None:
    configs = sorted(Path("configs").rglob("*.yaml"))
    assert configs
    for path in configs:
        assert yaml.safe_load(path.read_text(encoding="utf-8")).get("report_to") == "none", path
