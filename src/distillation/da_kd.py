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


def stratified_select_grouped_indices(
    scores: Sequence[float],
    *,
    group_size: int,
    ratio: float,
    tau: float,
    seed: int,
    min_size: int,
    multiple: int = 1,
) -> list[int]:
    """Run SDU selection on fixed groups and return every row in chosen groups.

    Prepared multitask data encodes a same-question selector contrast in each
    two-batch block. Group-level selection prevents DA-KD from retaining only
    the YES or only the NO member. A trailing partial group contains only
    majority-task rows and is omitted from the active subset.
    """

    if group_size <= 0:
        raise ValueError("DA-KD group size must be positive.")
    if multiple <= 0:
        raise ValueError("DA-KD selection multiple must be positive.")
    complete_group_count = len(scores) // group_size
    if complete_group_count == 0:
        raise ValueError("DA-KD dataset is too small for one complete contrast group.")

    grouped_scores = [
        sum(float(value) for value in scores[offset : offset + group_size]) / group_size
        for offset in range(0, complete_group_count * group_size, group_size)
    ]
    minimum_groups = math.ceil(min_size / group_size)
    group_multiple = multiple // math.gcd(group_size, multiple)
    selected_groups = stratified_select_indices(
        grouped_scores,
        ratio=ratio,
        tau=tau,
        seed=seed,
        min_size=minimum_groups,
        multiple=group_multiple,
    )
    return [
        index
        for group_index in selected_groups
        for index in range(group_index * group_size, (group_index + 1) * group_size)
    ]


def summarize_da_kd_selection(
    scores: Sequence[float],
    student_losses: Sequence[float],
    teacher_losses: Sequence[float],
    active_indices: Sequence[int],
) -> dict[str, float | int]:
    """Summarize DDS values and the realized high/low SDU mixture.

    The paper defines the high-DDS partition as the first ``rN`` examples
    after sorting.  ``active_indices`` also has size ``rN``, so it determines
    the partition boundary even when the low partition is too small to
    realize the requested ``tau`` mixture exactly.
    """

    total = len(scores)
    if total == 0:
        raise ValueError("DA-KD diagnostics require a non-empty dataset.")
    if len(student_losses) != total or len(teacher_losses) != total:
        raise ValueError("DA-KD score and loss arrays must have equal lengths.")

    selected = [int(index) for index in active_indices]
    if not selected:
        raise ValueError("DA-KD diagnostics require at least one active sample.")
    if len(set(selected)) != len(selected) or any(index < 0 or index >= total for index in selected):
        raise ValueError("DA-KD active indices must be unique valid dataset indices.")

    numeric_scores = torch.tensor([float(value) for value in scores], dtype=torch.float64)
    numeric_student = torch.tensor([float(value) for value in student_losses], dtype=torch.float64)
    numeric_teacher = torch.tensor([float(value) for value in teacher_losses], dtype=torch.float64)
    if not torch.isfinite(torch.cat((numeric_scores, numeric_student, numeric_teacher))).all():
        raise ValueError("DA-KD diagnostic values must be finite.")

    selected_tensor = torch.tensor(selected, dtype=torch.long)
    selected_scores = numeric_scores[selected_tensor]
    ordered = sorted(range(total), key=lambda index: (-float(numeric_scores[index]), index))
    high_partition = set(ordered[: len(selected)])
    selected_high = sum(index in high_partition for index in selected)
    selected_low = len(selected) - selected_high
    quantiles = torch.quantile(numeric_scores, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))

    return {
        "dds_min": numeric_scores.min().item(),
        "dds_p10": quantiles[0].item(),
        "dds_median": quantiles[1].item(),
        "dds_p90": quantiles[2].item(),
        "dds_max": numeric_scores.max().item(),
        "dds_mean": numeric_scores.mean().item(),
        "selected_dds_mean": selected_scores.mean().item(),
        "student_ce_mean": numeric_student.mean().item(),
        "teacher_ce_mean": numeric_teacher.mean().item(),
        "teacher_better_fraction": (numeric_student > numeric_teacher).to(torch.float64).mean().item(),
        "selected_high_count": selected_high,
        "selected_low_count": selected_low,
        "selected_low_fraction": selected_low / len(selected),
    }


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
