from accelerate.data_loader import BatchSamplerShard

from distillation.sampling import DifficultyAwareDistributedBatchSampler, TemplateDistributedBatchSampler


def test_distributed_batches_preserve_contiguous_groups_without_duplicates() -> None:
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
            assert [len(batch) for batch in actual] == [4]
            assert all(batch == list(range(batch[0], batch[0] + len(batch))) for batch in actual)
        batches = [list(batch) for batch in sampler]
        flattened = [index for batch in batches for index in batch]
        assert sorted(flattened) == list(range(8))
        assert len(flattened) == len(set(flattened))


def test_single_process_keeps_incomplete_final_batch_and_prepared_order() -> None:
    dataset = range(10)
    sampler = TemplateDistributedBatchSampler(dataset, batch_size=4, num_replicas=1)
    actual = [list(batch) for batch in sampler]
    assert [len(batch) for batch in actual] == [4, 4, 2]
    assert all(batch == list(range(batch[0], batch[0] + len(batch))) for batch in actual)
    assert sorted(index for batch in actual for index in batch) == list(dataset)


def test_distributed_sampler_preserves_generator_selector_pairs() -> None:
    sampler = TemplateDistributedBatchSampler(range(12), batch_size=2, num_replicas=2)

    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        for rank in range(2):
            shard = BatchSamplerShard(
                sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
            for batch in shard:
                assert len(batch) == 2
                assert batch[0] % 2 == 0
                assert batch[1] == batch[0] + 1


def test_sampler_shuffles_complete_two_batch_contrast_blocks() -> None:
    sampler = TemplateDistributedBatchSampler(range(32), batch_size=2, num_replicas=2, seed=17)

    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        batches = [list(batch) for batch in sampler]
        for offset in range(0, len(batches), 2):
            first, second = batches[offset : offset + 2]
            assert second[0] == first[0] + 2
            assert second[1] == first[1] + 2


def test_difficulty_aware_sampler_uses_active_indices() -> None:
    sampler = DifficultyAwareDistributedBatchSampler(range(10), batch_size=2, num_replicas=1, seed=0)
    sampler.set_active_indices([1, 3, 5, 7])
    batches = [list(batch) for batch in sampler]
    assert len(batches) == 2
    assert sorted(index for batch in batches for index in batch) == [1, 3, 5, 7]
    assert all(batch == sorted(batch) for batch in batches)
