from __future__ import annotations

POSITIVE_SELECTOR_LABEL = "YES"
NEGATIVE_SELECTOR_LABEL = "NO"
SELECTOR_LABELS = frozenset({POSITIVE_SELECTOR_LABEL, NEGATIVE_SELECTOR_LABEL})


def selector_label_from_binary(label: int) -> str:
    """Map the source binary relevance label to its one-token target."""

    if label == 1:
        return POSITIVE_SELECTOR_LABEL
    if label == 0:
        return NEGATIVE_SELECTOR_LABEL
    raise ValueError(f"Selector label must be 0 or 1, got {label!r}")


def parse_selector_label(text: str) -> str | None:
    """Accept only the exact one-token labels used by selector supervision."""

    normalized = text.strip()
    return normalized if normalized in SELECTOR_LABELS else None
