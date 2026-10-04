from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from schema_grounding.inference.merge import merge_schema_units

Message = dict[str, str]

_QWEN3_NOTHINK_MESSAGE = "<|im_start|>{role}\n{content}<|im_end|>\n"
_QWEN3_NOTHINK_GENERATION_PROMPT = "<|im_start|>assistant\n"
_LLAMA3_MESSAGE = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
_LLAMA3_GENERATION_PROMPT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
QWEN3_NOTHINK_TEMPLATE_NAME = "llamafactory:qwen3_nothink"
QWEN2_5_TEMPLATE_NAME = "llamafactory:qwen"
QWEN3_NOTHINK_TEMPLATE_FINGERPRINT = sha256(
    f"{_QWEN3_NOTHINK_MESSAGE}\0{_QWEN3_NOTHINK_GENERATION_PROMPT}".encode()
).hexdigest()
QWEN2_5_TEMPLATE_FINGERPRINT = QWEN3_NOTHINK_TEMPLATE_FINGERPRINT
LLAMA3_TEMPLATE_NAME = "llamafactory:llama3"
LLAMA3_TEMPLATE_FINGERPRINT = sha256(
    f"{_LLAMA3_MESSAGE}\0{_LLAMA3_GENERATION_PROMPT}".encode()
).hexdigest()


def qwen_template_metadata(model_family: str) -> dict[str, str]:
    if model_family == "qwen2.5_coder":
        return {"name": QWEN2_5_TEMPLATE_NAME, "fingerprint": QWEN2_5_TEMPLATE_FINGERPRINT}
    return {
        "name": QWEN3_NOTHINK_TEMPLATE_NAME,
        "fingerprint": QWEN3_NOTHINK_TEMPLATE_FINGERPRINT,
    }


def chat_template_metadata(model_family: str) -> dict[str, str]:
    if model_family == "llama3":
        return {"name": LLAMA3_TEMPLATE_NAME, "fingerprint": LLAMA3_TEMPLATE_FINGERPRINT}
    return qwen_template_metadata(model_family)


def render_qwen3_nothink(
    messages: list[Message],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render the exact non-reasoning ChatML format used by LlamaFactory."""

    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported qwen3_nothink message role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("qwen3_nothink message content must be a string")
        rendered.append(_QWEN3_NOTHINK_MESSAGE.format(role=role, content=content))
    if add_generation_prompt:
        rendered.append(_QWEN3_NOTHINK_GENERATION_PROMPT)
    return "".join(rendered)


def render_llama3(
    messages: list[Message],
    *,
    bos_token: str,
    add_generation_prompt: bool = True,
) -> str:
    """Render the exact LlamaFactory llama3 format without native date injection."""

    rendered = [bos_token]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported llama3 message role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("llama3 message content must be a string")
        rendered.append(_LLAMA3_MESSAGE.format(role=role, content=content))
    if add_generation_prompt:
        rendered.append(_LLAMA3_GENERATION_PROMPT)
    return "".join(rendered)


def _relationship_pattern(relationship: Mapping[str, Any]) -> str:
    return f"(:{relationship['source']})-[:{relationship['type']}]->(:{relationship['target']})"


# Same similarity rule as hard node distractors in schema_grounding.augmentation.
_SIMILAR_NODE_MIN_SHARED_PROPERTIES = 2
_SIMILAR_NODE_MIN_PROPERTY_JACCARD = 0.5


def _context_section(header: str, lines: Iterable[str]) -> str:
    body = "\n".join(f"- {line}" for line in sorted(set(lines))) or "- none"
    return f"{header}\n{body}"


def _similar_node_labels(node: Mapping[str, Any], nodes: Iterable[Mapping[str, Any]]) -> list[str]:
    properties = set(node.get("properties", {}))
    similar = []
    for other in nodes:
        if other["label"] == node["label"]:
            continue
        other_properties = set(other.get("properties", {}))
        shared = len(properties & other_properties)
        if shared >= _SIMILAR_NODE_MIN_SHARED_PROPERTIES and (
            shared / len(properties | other_properties) >= _SIMILAR_NODE_MIN_PROPERTY_JACCARD
        ):
            similar.append(f"(:{other['label']})")
    return similar


def selector_schema_context(unit: Mapping[str, Any], schema_units: Iterable[Mapping[str, Any]]) -> str:
    """Render the comparison context shown next to one selector unit.

    The context lists the schema units a selector could confuse with ``unit``
    (the hard-distractor rules of ``schema_grounding.augmentation``):

    * relationship ``(A)-[:R]->(B)``: other relationships joining the same two
      labels in either direction, and relationships of the same type ``R``
      between other labels;
    * node ``L``: the relationships connected to ``L`` (how it enters a query),
      and node labels whose properties largely duplicate ``L``'s.

    Lines are sorted so the text does not depend on unit order: training reads
    ``schemas.jsonl`` while inference reads the example's unit rows, and both
    must render the same prompt.
    """

    units = list(schema_units)
    relationships = [other["schema"] for other in units if other.get("kind") == "relation"]
    schema = unit["schema"]
    if unit.get("kind") == "relation":
        own = _relationship_pattern(schema)
        pair = sorted((schema["source"], schema["target"]))
        same_labels = [
            _relationship_pattern(other)
            for other in relationships
            if sorted((other["source"], other["target"])) == pair and _relationship_pattern(other) != own
        ]
        same_type = [
            _relationship_pattern(other)
            for other in relationships
            if other["type"] == schema["type"] and _relationship_pattern(other) != own
        ]
        return "\n".join(
            (
                _context_section("Other relationships between the same node labels:", same_labels),
                _context_section("Other relationships with the same type:", same_type),
            )
        )
    label = schema["label"]
    connected = [
        _relationship_pattern(other) for other in relationships if label in (other["source"], other["target"])
    ]
    nodes = [other["schema"] for other in units if other.get("kind") == "node"]
    return "\n".join(
        (
            _context_section("Relationships connected to this node label:", connected),
            _context_section("Node labels with similar properties:", _similar_node_labels(schema, nodes)),
        )
    )


def _sub_schema_unit_ids(sub_schema: Mapping[str, Any]) -> set[str]:
    ids = {f"node:{node['label']}" for node in sub_schema["nodes"]}
    ids.update(
        f"relation:{relationship['source']}|{relationship['type']}|{relationship['target']}"
        for relationship in sub_schema["relationships"]
    )
    return ids


def generator_other_schema(
    candidate_schema: Mapping[str, Any], schema_units: Iterable[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Return the schema units that are not part of the candidate sub-schema.

    Candidate and other schema together cover the full schema without overlap.
    Units keep the order of ``schema_units`` (the canonical schema order used
    by both ``schemas.jsonl`` and the inference unit rows).
    """

    units = [dict(unit) for unit in schema_units]
    chosen = _sub_schema_unit_ids(candidate_schema)
    rest = [str(unit["id"]) for unit in units if str(unit["id"]) not in chosen]
    return merge_schema_units(units, rest, close_relation_endpoints=False).sub_schema


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

    def selector_messages(self, question: str, schema_unit: str, schema_context: str) -> list[Message]:
        return [
            {"role": "system", "content": self.selector_system},
            {
                "role": "user",
                "content": self.selector_user.format(
                    question=question,
                    schema_unit=schema_unit,
                    schema_context=schema_context,
                ),
            },
        ]

    def generator_messages(
        self, question: str, sub_schema: dict[str, Any], other_schema: dict[str, Any]
    ) -> list[Message]:
        return [
            {"role": "system", "content": self.generator_system},
            {
                "role": "user",
                "content": self.generator_user.format(
                    question=question,
                    candidate_schema=json.dumps(sub_schema, ensure_ascii=False, indent=2),
                    other_schema=json.dumps(other_schema, ensure_ascii=False, indent=2),
                ),
            },
        ]
