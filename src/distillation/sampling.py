from __future__ import annotations

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import BatchSampler, Sampler


class _EpochAwareSequentialSampler(Sampler[int]):
    """Expose epoch state where Accelerate expects to propagate it."""

    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


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
    """Shuffle two-batch contrast blocks without shuffling their samples.

    The prepared dataset already encodes the intended generator/selector batch
    composition and same-question selector pairs. At batch size two, a contrast
    pair spans two consecutive prepared batches. Shuffling complete two-batch
    blocks preserves both the local task mix and, with two replicas, places the
    YES/NO pair in the same distributed micro-step.
    """

    full_batch_count, remainder = divmod(len(indices), batch_size)
    batches = [indices[start * batch_size : (start + 1) * batch_size] for start in range(full_batch_count)]
    generator = torch.Generator().manual_seed(seed + epoch)
    complete_block_count, trailing_batch_count = divmod(full_batch_count, 2)
    blocks = [batches[start * 2 : (start + 1) * 2] for start in range(complete_block_count)]
    block_order = torch.randperm(complete_block_count, generator=generator).tolist()
    ordered_batches = [batch for block_index in block_order for batch in blocks[block_index]]
    if trailing_batch_count:
        # The prepared train layout puts any unpaired full batch after all
        # contrast blocks (it contains generator rows only), so leave it last.
        ordered_batches.append(batches[-1])
    if num_replicas > 1:
        # Each consecutive group of batches is sharded one-per-rank by
        # Accelerate. Drop fewer than ``num_replicas`` whole batches so every
        # rank has equal steps; never split or pad a prepared batch.
        ordered_batches = ordered_batches[: (full_batch_count // num_replicas) * num_replicas]
    yield from ordered_batches
    if num_replicas == 1 and remainder:
        # Accelerate accepts a single-process incomplete batch only at the end.
        yield indices[full_batch_count * batch_size :]


class TemplateDistributedBatchSampler(BatchSampler):
    """Distribute prepared contrast blocks without destroying their composition.

    With ``even_batches=False``, ranks have equal step counts and no examples
    are duplicated. Only complete two-batch blocks and their rank ownership are
    shuffled.
    """

    def __init__(self, dataset: Sized, *, batch_size: int, num_replicas: int, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        self._index_sampler = _EpochAwareSequentialSampler(dataset)
        super().__init__(self._index_sampler, batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed

    @property
    def epoch(self) -> int:
        return self._index_sampler.epoch

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
        self._index_sampler.set_epoch(epoch)
