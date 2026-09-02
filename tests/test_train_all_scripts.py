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
    assert 'TEACHER_MODEL="${FAMILY_RESULTS}/teacher_full"' in script
    assert '"output_dir=${TEACHER_MODEL}"' in script
    assert 'output_dir="${FAMILY_RESULTS}/${method}"' in script
    assert 'method_overrides+=("ref_model=${TEACHER_MODEL}")' in script
    assert '"$@" "${method_overrides[@]}"' in script
    assert 'if [[ "${override}" == resume_from_checkpoint=* ]]; then' in script
    assert "one checkpoint cannot be applied to every method" in script
    assert 'fresh_outputs=("${TEACHER_MODEL}")' in script
    assert 'for checkpoint in "${output_dir}"/checkpoint-*; do' in script
    assert "Choose a new RESULTS_ROOT" in script
