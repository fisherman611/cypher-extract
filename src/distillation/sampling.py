from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import BatchSampler, Sampler


class _TemplateDistributedSampler(Sampler[int]):
    """Global shuffled indices truncated like DistributedSampler(drop_last=True)."""

    def __init__(self, dataset: Sized, *, num_replicas: int, seed: int = 0) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self.total_size = (len(dataset) // num_replicas) * num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(indices[: self.total_size])

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class TemplateDistributedBatchSampler(BatchSampler):
    """Interleave per-rank batches from the template's distributed sampler.

    Accelerate normally pads incomplete distributed batches with duplicate
    examples. This sampler first reproduces PyTorch DistributedSampler with
    ``drop_last=True`` and then emits rank batches in process order. Prepared
    with ``even_batches=False``, each process receives exactly the same batches
    as the legacy template, including its incomplete final local batch.
    """

    def __init__(self, dataset: Sized, *, batch_size: int, num_replicas: int, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        sampler = _TemplateDistributedSampler(dataset, num_replicas=num_replicas, seed=seed)
        super().__init__(sampler, batch_size=batch_size, drop_last=False)
        self.num_replicas = num_replicas

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(self.sampler)
        rank_indices = [indices[rank :: self.num_replicas] for rank in range(self.num_replicas)]
        samples_per_rank = len(indices) // self.num_replicas
        for start in range(0, samples_per_rank, self.batch_size):
            end = start + self.batch_size
            for rank in range(self.num_replicas):
                yield rank_indices[rank][start:end]

    def __len__(self) -> int:
        samples_per_rank = len(self.sampler) // self.num_replicas
        return math.ceil(samples_per_rank / self.batch_size) * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)


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

    def _shuffled_indices(self) -> list[int]:
        pool = self.active_indices if self.active_indices is not None else list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(pool), generator=generator).tolist()
        indices = [pool[position] for position in permutation]
        total_size = (len(indices) // self.num_replicas) * self.num_replicas
        return indices[:total_size]

    def __iter__(self) -> Iterator[list[int]]:
        indices = self._shuffled_indices()
        rank_indices = [indices[rank :: self.num_replicas] for rank in range(self.num_replicas)]
        samples_per_rank = len(indices) // self.num_replicas
        for start in range(0, samples_per_rank, self.batch_size):
            end = start + self.batch_size
            for rank in range(self.num_replicas):
                yield rank_indices[rank][start:end]

    def __len__(self) -> int:
        active_size = len(self.active_indices) if self.active_indices is not None else len(self.dataset)
        samples_per_rank = active_size // self.num_replicas
        return math.ceil(samples_per_rank / self.batch_size) * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
