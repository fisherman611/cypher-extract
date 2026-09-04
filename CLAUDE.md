# Project

Python ML repository for schema grounding, knowledge distillation, and
two-stage Text-to-Cypher generation.

The main inference flow is:

1. classify each `question + schema unit` with the selector;
2. merge selected units into a predicted sub-schema;
3. generate Cypher from `question + predicted sub-schema`.

Gold selector labels, gold sub-schemas, and gold Cypher must never enter the
inference prompt. They may only be attached after generation for evaluation.

# Environment

Conda environment: `myenv`
Supported Python: `>=3.11,<3.14`

# Rules

- Prefer minimal changes.
- Never modify datasets unless explicitly requested.
- Never scan `checkpoints/`, `results/`, `outputs/`, or `data/` unless required.
- Do not download models, run GPU training/inference, start Neo4j, or execute
  generated Cypher unless explicitly requested.
- Use `rg` or `rg --files` before reading files.
- Do not refactor unrelated code.
- Preserve selector/generator prompt, template, and decoding parity across
  training, evaluation, and inference.
- Preserve branch-specific training behavior when porting shared changes.
- Run targeted tests before full test suites.

# Python

- Use pathlib for paths.
- Preserve existing CLI interfaces.
- Follow existing formatting/style.
- Add or update focused regression tests for behavior changes.

# Context efficiency

- Search before reading.
- Never read large files in full unless required.
- Delegate large searches/log analysis to subagents.
