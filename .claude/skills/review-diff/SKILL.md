---
name: review-diff
description: Review only the current git changes for bugs and regressions.
disable-model-invocation: true
context: fork
model: sonnet
effort: medium
---

Review ONLY the current git changes.

First inspect:

!`git status --short`

!`git diff --stat`

Then inspect the actual diff and only the surrounding code necessary
to understand changed behavior.

Do not review unrelated files.

Prioritize:

1. correctness bugs
2. regressions
3. incorrect assumptions
4. edge cases
5. data loss
6. performance problems

Ignore minor style issues unless they cause a real problem.

Return at most 10 findings, ordered by severity.

If there are no significant problems, say so.
