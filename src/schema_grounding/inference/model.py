from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_grounding.inference.prompting import Message

_DTYPES = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
)


@dataclass
class ModelRunner:
    model: Any
    tokenizer: Any
    device: torch.device
    safe_batch_sizes: dict[int, int] = field(default_factory=dict, repr=False)

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        merge_adapter: bool = True,
    ) -> ModelRunner:
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {', '.join(_DTYPES)}")
        adapter_path = str(adapter_path)
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_name = peft_config.base_model_name_or_path
        if not base_name:
            raise ValueError(f"Adapter {adapter_path} does not declare a base model")
        base_revision = peft_config.revision
        has_local_tokenizer = (Path(adapter_path) / "tokenizer_config.json").is_file() and any(
            (Path(adapter_path) / filename).is_file() for filename in _TOKENIZER_ASSETS
        )
        tokenizer_source = adapter_path if has_local_tokenizer else base_name
        tokenizer_revision = None if has_local_tokenizer else base_revision
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=tokenizer_revision, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        target_device = torch.device(device)
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            revision=base_revision,
            dtype=_DTYPES[dtype],
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
        model.to(target_device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=target_device)

    def prompt_length(self, messages: Sequence[Message]) -> int:
        return len(self._tokenize_chat(messages))

    def _tokenize_chat(self, messages: Sequence[Message]) -> list[int]:
        token_ids = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(token_ids, "keys") and "input_ids" in token_ids:
            token_ids = token_ids["input_ids"]
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()
        return token_ids

    @torch.inference_mode()
    def generate(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        if not conversations:
            return []
        safe_batch_size = self.safe_batch_sizes.get(max_new_tokens)
        if safe_batch_size is not None and len(conversations) > safe_batch_size:
            outputs: list[str] = []
            for start in range(0, len(conversations), safe_batch_size):
                outputs.extend(
                    self.generate(
                        conversations[start : start + safe_batch_size],
                        max_new_tokens=max_new_tokens,
                    )
                )
            return outputs
        try:
            return self._generate_batch(conversations, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            if len(conversations) == 1:
                raise
            torch.cuda.empty_cache()
            middle = len(conversations) // 2
            previous_safe_size = self.safe_batch_sizes.get(max_new_tokens, len(conversations))
            self.safe_batch_sizes[max_new_tokens] = min(previous_safe_size, middle)
            return [
                *self.generate(conversations[:middle], max_new_tokens=max_new_tokens),
                *self.generate(conversations[middle:], max_new_tokens=max_new_tokens),
            ]

    def _generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        features = []
        for messages in conversations:
            input_ids = self._tokenize_chat(messages)
            features.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        generated = self.model.generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_width = batch["input_ids"].shape[1]
        return self.tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
