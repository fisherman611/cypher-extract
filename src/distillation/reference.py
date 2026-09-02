from __future__ import annotations

from typing import Any


def create_reference_model_at_revision(
    model_args: Any,
    finetuning_args: Any,
    ref_model_revision: str | None,
    factory: Any,
) -> Any:
    """Create the teacher with its own immutable revision and restore student args."""

    student_revision = model_args.model_revision
    try:
        if ref_model_revision is not None:
            model_args.model_revision = ref_model_revision
        return factory(model_args, finetuning_args)
    finally:
        model_args.model_revision = student_revision
