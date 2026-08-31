from types import SimpleNamespace

import pytest

from distillation.generation import generation_eos_value, resolve_eos_token_ids


def test_qwen2_5_generation_preserves_model_eos_list_without_other_special_tokens() -> None:
    tokenizer = SimpleNamespace(
        eos_token_id=151645,
        additional_special_tokens_ids=[151644, 151645, 151646, 151647],
    )
    generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

    eos_token_ids = resolve_eos_token_ids(tokenizer, generation_config)

    assert eos_token_ids == [151645, 151643]
    assert generation_eos_value(eos_token_ids) == [151645, 151643]


def test_generation_eos_falls_back_to_tokenizer_and_rejects_missing_id() -> None:
    assert resolve_eos_token_ids(SimpleNamespace(eos_token_id=2)) == [2]
    assert generation_eos_value([2]) == 2
    with pytest.raises(ValueError, match="EOS"):
        resolve_eos_token_ids(SimpleNamespace(eos_token_id=None))
