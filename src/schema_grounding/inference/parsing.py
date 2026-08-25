from __future__ import annotations

SELECTOR_LABELS = frozenset({"RELATED", "UNRELATED"})


def parse_selector_label(text: str) -> str | None:
    """Accept only the exact label format used by selector supervision."""

    normalized = text.strip().upper()
    return normalized if normalized in SELECTOR_LABELS else None
