---
name: cheap
description: Handle a simple task with minimal reasoning and context.
disable-model-invocation: true
model: haiku
effort: low
---

$ARGUMENTS

Work directly.

Rules:
- Do not explore broadly.
- Read at most the minimum files required.
- Do not use subagents.
- Do not refactor unrelated code.
- Keep the response concise.