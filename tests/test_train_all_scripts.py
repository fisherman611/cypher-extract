from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script_name", "family"),
    [
        ("train_all_qwen3.sh", "qwen3"),
        ("train_all_llama3.sh", "llama3"),
        ("train_all_qwen2_5_coder.sh", "qwen2.5_coder"),
    ],
)
def test_train_all_scripts_route_every_method_through_results_root(script_name: str, family: str) -> None:
    script = (Path("scripts") / script_name).read_text(encoding="utf-8")

    assert 'RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"' in script
    assert f'FAMILY_RESULTS="${{RESULTS_ROOT}}/{family}"' in script
    assert 'TEACHER_ADAPTER="${FAMILY_RESULTS}/teacher_lora"' in script
    assert '"output_dir=${TEACHER_ADAPTER}"' in script
    assert 'output_dir="${FAMILY_RESULTS}/${method}"' in script
    assert 'method_overrides+=("ref_model_adapters=${TEACHER_ADAPTER}")' in script
    assert '"$@" "${method_overrides[@]}"' in script
