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
    droppable_batch_start: int = 0,
    droppable_batch_count: int = 0,
) -> Iterator[list[int]]:
    """Shuffle two-batch prepared blocks without shuffling their samples.

    The prepared dataset already encodes the intended generator/selector batch
    composition and same-question selector pairs. At batch size two, a contrast
    pair spans two consecutive prepared batches; extra selector negatives are
    also arranged in two-batch blocks. Shuffling complete blocks preserves both
    the local task mix and, with two replicas, places each YES/NO pair in the
    same distributed micro-step.
    """

    full_batch_count, remainder = divmod(len(indices), batch_size)
    batches = [indices[start * batch_size : (start + 1) * batch_size] for start in range(full_batch_count)]
    drop_count = full_batch_count % num_replicas if num_replicas > 1 else 0
    if drop_count:
        droppable_stop = droppable_batch_start + droppable_batch_count
        if (
            droppable_batch_start < 0
            or droppable_stop > full_batch_count
            or droppable_batch_count < drop_count
        ):
            raise ValueError(
                "Distributed sampling needs enough explicitly droppable non-contrast batches "
                "to equalize rank lengths."
            )
        first_offset = (epoch * drop_count) % droppable_batch_count
        dropped = {
            droppable_batch_start + (first_offset + offset) % droppable_batch_count
            for offset in range(drop_count)
        }
        batches = [batch for index, batch in enumerate(batches) if index not in dropped]
        full_batch_count = len(batches)
    generator = torch.Generator().manual_seed(seed + epoch)
    complete_block_count, trailing_batch_count = divmod(full_batch_count, 2)
    blocks = [batches[start * 2 : (start + 1) * 2] for start in range(complete_block_count)]
    block_order = torch.randperm(complete_block_count, generator=generator).tolist()
    ordered_batches = [batch for block_index in block_order for batch in blocks[block_index]]
    if trailing_batch_count:
        # Keep the incomplete two-batch block last so it cannot shift the
        # distributed-rank alignment of any preceding contrast pair.
        ordered_batches.append(batches[-1])
    if num_replicas > 1 and full_batch_count % num_replicas:
        raise AssertionError("Distributed batch count was not equalized across ranks")
    yield from ordered_batches
    if num_replicas == 1 and remainder:
        # Accelerate accepts a single-process incomplete batch only at the end.
        yield indices[full_batch_count * batch_size :]


class TemplateDistributedBatchSampler(BatchSampler):
    """Distribute prepared two-batch blocks without destroying their composition.

    With ``even_batches=False``, ranks have equal step counts and no examples
    are duplicated. Only complete two-batch blocks and their rank ownership are
    shuffled.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        num_replicas: int,
        seed: int = 0,
        droppable_batch_start: int = 0,
        droppable_batch_count: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        self._index_sampler = _EpochAwareSequentialSampler(dataset)
        super().__init__(self._index_sampler, batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.droppable_batch_start = droppable_batch_start
        self.droppable_batch_count = droppable_batch_count

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
            droppable_batch_start=self.droppable_batch_start,
            droppable_batch_count=self.droppable_batch_count,
        )

    def __len__(self) -> int:
        return _distributed_batch_count(len(self.dataset), self.batch_size, self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self._index_sampler.set_epoch(epoch)
