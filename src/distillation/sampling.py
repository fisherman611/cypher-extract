from __future__ import annotations

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import BatchSampler


def _distributed_batch_count(size: int, batch_size: int, num_replicas: int) -> int:
    full_batches, remainder = divmod(size, batch_size)
    if num_replicas == 1:
        return full_batches + int(remainder > 0)
    return (full_batches // num_replicas) * num_replicas


def _preserved_distributed_batches(
    indices: list[int],
    *,
    batch_size: int,
    num_replicas: int,
    seed: int,
    epoch: int,
) -> Iterator[list[int]]:
    """Shuffle whole prepared batches without shuffling their samples.

    The prepared dataset already encodes the intended generator/selector batch
    composition. Accelerate assigns each consecutive group of batches across
    ranks, so shuffling complete batches changes data order and rank ownership
    without destroying that composition.
    """

    full_batch_count, remainder = divmod(len(indices), batch_size)
    batches = [indices[start * batch_size : (start + 1) * batch_size] for start in range(full_batch_count)]
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(full_batch_count, generator=generator).tolist()
    if num_replicas > 1:
        # Each consecutive group of batches is sharded one-per-rank by
        # Accelerate. Drop fewer than ``num_replicas`` whole batches so every
        # rank has equal steps; never split or pad a prepared batch.
        order = order[: (full_batch_count // num_replicas) * num_replicas]
    for batch_index in order:
        yield batches[batch_index]
    if num_replicas == 1 and remainder:
        # Accelerate accepts a single-process incomplete batch only at the end.
        yield indices[full_batch_count * batch_size :]


class TemplateDistributedBatchSampler(BatchSampler):
    """Distribute prepared batches without destroying their task composition.

    With ``even_batches=False``, ranks have equal step counts and no examples
    are duplicated. Only whole batches and their rank ownership are shuffled.
    """

    def __init__(self, dataset: Sized, *, batch_size: int, num_replicas: int, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        super().__init__(range(len(dataset)), batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        yield from _preserved_distributed_batches(
            list(range(len(self.dataset))),
            batch_size=self.batch_size,
            num_replicas=self.num_replicas,
            seed=self.seed,
            epoch=self.epoch,
        )

    def __len__(self) -> int:
        return _distributed_batch_count(len(self.dataset), self.batch_size, self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DifficultyAwareDistributedBatchSampler(BatchSampler):
    """Distributed batch sampler whose active indices can change each epoch."""

    def __init__(self, dataset: Sized, *, batch_size: int, num_replicas: int, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        super().__init__(range(len(dataset)), batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self.active_indices: list[int] | None = None

    def set_active_indices(self, indices: list[int]) -> None:
        if not indices:
            raise ValueError("DA-KD cannot train with an empty active dataset.")
        if any(index < 0 or index >= len(self.dataset) for index in indices):
            raise IndexError("DA-KD active indices must refer to the training dataset.")
        self.active_indices = list(dict.fromkeys(int(index) for index in indices))

    def _ordered_indices(self) -> list[int]:
        pool = self.active_indices if self.active_indices is not None else list(range(len(self.dataset)))
        return sorted(pool)

    def __iter__(self) -> Iterator[list[int]]:
        yield from _preserved_distributed_batches(
            self._ordered_indices(),
            batch_size=self.batch_size,
            num_replicas=self.num_replicas,
            seed=self.seed,
            epoch=self.epoch,
        )

    def __len__(self) -> int:
        active_size = len(self.active_indices) if self.active_indices is not None else len(self.dataset)
        return _distributed_batch_count(active_size, self.batch_size, self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
