---
name: debug-focused
description: Diagnose a specific error with minimal repository exploration.
disable-model-invocation: true
context: fork
model: sonnet
effort: medium
---

Debug this issue:

$ARGUMENTS

Use this procedure:

1. Identify the exact failing component.
2. Search for the error string, function, class, or config involved.
3. Read only directly relevant files.
4. Form one primary hypothesis.
5. Verify that hypothesis before exploring alternatives.
6. Do not perform broad repository exploration.
7. Do not modify code.

Return:

CAUSE:
<root cause>

EVIDENCE:
<files/functions/lines involved>

FIX:
<minimal recommended change>

VERIFY:
<command to verify the fix>