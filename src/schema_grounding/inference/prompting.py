from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

Message = dict[str, str]


@dataclass(frozen=True)
class PromptTemplates:
    generator_system: str
    generator_user: str
    selector_system: str
    selector_user: str

    def fingerprints(self) -> dict[str, str]:
        """Return stable content hashes used to validate resumable outputs."""

        return {
            name: sha256(getattr(self, name).encode("utf-8")).hexdigest()
            for name in (
                "generator_system",
                "generator_user",
                "selector_system",
                "selector_user",
            )
        }

    @classmethod
    def from_repository(cls, repository_root: Path) -> PromptTemplates:
        prompt_root = repository_root / "prompts"

        def read(relative_path: str) -> str:
            return (prompt_root / relative_path).read_text(encoding="utf-8").strip()

        return cls(
            generator_system=read("generator/system_prompt.txt"),
            generator_user=read("generator/user_prompt.txt"),
            selector_system=read("selector/system_prompt.txt"),
            selector_user=read("selector/user_prompt.txt"),
        )

    def selector_messages(self, question: str, schema_unit: str) -> list[Message]:
        return [
            {"role": "system", "content": self.selector_system},
            {
                "role": "user",
                "content": self.selector_user.format(question=question, schema_unit=schema_unit),
            },
        ]

    def generator_messages(self, question: str, sub_schema: dict[str, Any]) -> list[Message]:
        schema = json.dumps(sub_schema, ensure_ascii=False, indent=2)
        return [
            {"role": "system", "content": self.generator_system},
            {
                "role": "user",
                "content": self.generator_user.format(question=question, schema=schema),
            },
        ]
