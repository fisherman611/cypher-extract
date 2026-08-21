from __future__ import annotations

import random

import numpy as np
import torch

from .distributed import get_rank


def seed_everything(seed: int, *, rank_offset: bool = True, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch and return the effective seed."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    effective_seed = seed + get_rank() if rank_offset else seed
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return effective_seed
