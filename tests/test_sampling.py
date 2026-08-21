from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, DistributedSampler

from distillation.sampling import DifficultyAwareDistributedBatchSampler, TemplateDistributedBatchSampler


def _template_rank_batches(
    dataset: range,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    epoch: int,
) -> list[list[int]]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    sampler.set_epoch(epoch)
    return [list(batch) for batch in BatchSampler(sampler, batch_size=batch_size, drop_last=False)]


def test_distributed_batches_match_template_without_padding_duplicates() -> None:
    dataset = range(11)
    world_size = 2
    batch_size = 4
    sampler = TemplateDistributedBatchSampler(dataset, batch_size=batch_size, num_replicas=world_size)

    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        for rank in range(world_size):
            shard = BatchSamplerShard(
                sampler,
                num_processes=world_size,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
            actual = [list(batch) for batch in shard]
            expected = _template_rank_batches(
                dataset,
                rank=rank,
                world_size=world_size,
                batch_size=batch_size,
                epoch=epoch,
            )
            assert actual == expected
            assert [len(batch) for batch in actual] == [4, 1]


def test_single_process_keeps_incomplete_final_batch_like_template() -> None:
    dataset = range(10)
    sampler = TemplateDistributedBatchSampler(dataset, batch_size=4, num_replicas=1)
    actual = [list(batch) for batch in sampler]
    expected = _template_rank_batches(dataset, rank=0, world_size=1, batch_size=4, epoch=0)
    assert actual == expected
    assert [len(batch) for batch in actual] == [4, 4, 2]


def test_difficulty_aware_sampler_uses_active_indices() -> None:
    sampler = DifficultyAwareDistributedBatchSampler(range(10), batch_size=2, num_replicas=1, seed=0)
    sampler.set_active_indices([1, 3, 5, 7])
    batches = [list(batch) for batch in sampler]
    assert len(batches) == 2
    assert sorted(index for batch in batches for index in batch) == [1, 3, 5, 7]
