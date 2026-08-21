from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def selection_ratio(epoch: float, total_epochs: float, schedule: str) -> float:
    """Return the DiffUp selection ratio for a zero-based training epoch.

    Epoch zero uses the full dataset. Later epochs follow the paper's linear
    or cosine decay. The caller is responsible for enforcing a usable minimum
    dataset size after the ratio reaches zero.
    """

    if schedule not in {"linear", "cosine"}:
        raise ValueError("DA-KD schedule must be linear or cosine.")
    if epoch <= 0 or total_epochs <= 0:
        return 1.0
    progress = min(max(float(epoch) / float(total_epochs), 0.0), 1.0)
    if schedule == "linear":
        return max(1.0 - progress, 0.0)
    if schedule == "cosine":
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.0)


def selection_size(
    total: int,
    *,
    ratio: float,
    min_size: int = 0,
    multiple: int = 1,
) -> int:
    """Return a valid active-set size for a selection ratio.

    ``multiple`` is useful for distributed training: selecting a multiple of
    the world size prevents the distributed sampler from silently discarding
    a large remainder of the selected subset.
    """

    if total < 0:
        raise ValueError("DA-KD dataset size must be non-negative.")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("DA-KD selection ratio must be in [0, 1].")
    if min_size < 0:
        raise ValueError("DA-KD minimum dataset size must be non-negative.")
    if multiple <= 0:
        raise ValueError("DA-KD selection multiple must be positive.")
    if total == 0:
        return 0

    target = min(total, max(int(min_size), int(round(total * ratio))))
    if multiple == 1 or target == total:
        return target

    # Round up so distributed sharding does not turn (for example) a desired
    # 31-sample subset into only 24 usable samples on an 8-rank job.  If the
    # dataset itself is not divisible by ``multiple``, use its largest valid
    # prefix instead.
    rounded_target = ((target + multiple - 1) // multiple) * multiple
    if rounded_target > total:
        rounded_target = (total // multiple) * multiple
    return min(total, rounded_target)


def stratified_select_indices(
    scores: Sequence[float],
    *,
    ratio: float,
    tau: float,
    seed: int,
    min_size: int,
    multiple: int = 1,
) -> list[int]:
    """Select high-DDS and low-DDS samples using the paper's SDU split."""

    if len(scores) == 0:
        return []
    if not 0.0 <= tau <= 1.0:
        raise ValueError("DA-KD tau must be in [0, 1].")

    numeric_scores = [float(score) for score in scores]
    if not all(math.isfinite(score) for score in numeric_scores):
        raise ValueError("DA-KD DDS scores must be finite.")

    total = len(numeric_scores)
    target_size = selection_size(total, ratio=ratio, min_size=min_size, multiple=multiple)
    if target_size == total:
        return list(range(total))

    ordered = sorted(range(total), key=lambda index: (-numeric_scores[index], index))
    high_partition = ordered[:target_size]
    low_partition = ordered[target_size:]

    high_count = min(len(high_partition), int(round((1.0 - tau) * target_size)))
    low_count = target_size - high_count
    if low_count > len(low_partition):
        low_count = len(low_partition)
        high_count = target_size - low_count

    rng = random.Random(seed)
    selected = rng.sample(high_partition, high_count) + rng.sample(low_partition, low_count)
    rng.shuffle(selected)
    return selected


def per_sample_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute one teacher-forced CE value per sequence."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, sequence, vocab] and labels [batch, sequence].")
    if logits.shape[1] < 2:
        raise ValueError("At least two tokens are required for causal cross-entropy.")

    shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    valid_count = valid.sum(dim=-1)
    if (valid_count == 0).any():
        raise ValueError("Each DA-KD sample must contain at least one response token.")

    token_loss = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
        ignore_index=ignore_index,
    )
    if not torch.isfinite(token_loss).all():
        raise ValueError("DA-KD per-sample cross-entropy must be finite.")
    return token_loss.masked_fill(~valid, 0.0).sum(dim=-1) / valid_count.to(token_loss.dtype)
