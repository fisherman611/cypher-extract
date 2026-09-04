---
name: explore
description: Quickly inspect a codebase or problem while minimizing context usage.
disable-model-invocation: true
context: fork
agent: Explore
model: haiku
effort: low
---

Investigate the following question:

$ARGUMENTS

Rules:
- Do not modify files.
- Do not scan the entire repository unless necessary.
- Start with grep/glob/file names.
- Read only files directly relevant to the question.
- Avoid reading generated files, logs, checkpoints, datasets, node_modules, .git, and build artifacts.
- Return only:
  1. Answer
  2. Relevant file paths
  3. Important symbols/functions
  4. Any uncertainty
- Keep the final response under 300 words.