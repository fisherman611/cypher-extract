# Project

Python ML repository for full-duplex voice assistant research.

# Environment

Conda env: myenv
Python: 3.12

# Rules

- Prefer minimal changes.
- Never modify datasets unless explicitly requested.
- Never scan checkpoints/, outputs/, data/ unless required.
- Use rg/find before reading files.
- Do not refactor unrelated code.
- Run targeted tests before full test suites.

# Python

- Use pathlib for paths.
- Preserve existing CLI interfaces.
- Follow existing formatting/style.

# Context efficiency

- Search before reading.
- Never read large files in full unless required.
- Delegate large searches/log analysis to subagents.