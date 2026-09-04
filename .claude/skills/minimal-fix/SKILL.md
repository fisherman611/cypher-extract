---
name: minimal-fix
description: Make the smallest possible code change for a clearly defined bug.
disable-model-invocation: true
effort: medium
---

Fix:

$ARGUMENTS

Constraints:

- Make the smallest correct change.
- Do not refactor unrelated code.
- Do not browse unrelated files.
- Search before reading large files.
- Preserve existing APIs and behavior unless the task explicitly requires a change.
- Do not add abstractions unless required.
- Run only the tests/checks relevant to the modified code.

After editing, report only:

- files changed
- root cause
- change made
- verification result
