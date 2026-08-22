from __future__ import annotations

import math
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from torch.utils.data import DataLoader, IterableDataset, Subset
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint, seed_worker

from .arguments import DistillationArguments
from .da_kd import per_sample_causal_cross_entropy, selection_ratio, selection_size, stratified_select_indices
from .distillm import AdaptiveRolloutScheduler, ReplayBuffer, RolloutSource, StudentRolloutGenerator
from .fdd import causal_response_mask, fdd_loss
from .losses import compute_distillation_loss, compute_hpd_loss
from .sampling import DifficultyAwareDistributedBatchSampler, TemplateDistributedBatchSampler
from .utils import all_gather_tensor, distributed_is_initialized, get_rank, get_world_size, print_rank

DISTILLM_STATE_NAME = "distillm_state"
DA_KD_STATE_NAME = "da_kd_state"


def _unwrap_model(model: torch.nn.Module, accelerator: Any) -> torch.nn.Module:
    try:
        return accelerator.unwrap_model(model)
    except Exception:
        return getattr(model, "module", model)


def _lm_head(model: torch.nn.Module, accelerator: Any) -> torch.nn.Module:
    unwrapped = _unwrap_model(model, accelerator)
    output_embeddings = unwrapped.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError(f"{type(unwrapped).__name__} does not expose output embeddings required by FDD.")
    return output_embeddings


class _DistillMStateCallback(TrainerCallback):
    def __init__(self, trainer: KDTrainer) -> None:
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        del control, kwargs
        checkpoint = Path(args.output_dir, f"checkpoint-{state.global_step}")
        self.trainer.save_distillm_state(checkpoint)


class _DistillMInitialEvalCallback(TrainerCallback):
    def __init__(self, trainer: KDTrainer) -> None:
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        self.trainer.initialize_distillm_baseline()


class _DAKDStateCallback(TrainerCallback):
    def __init__(self, trainer: KDTrainer) -> None:
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        del control, kwargs
        checkpoint = Path(args.output_dir, f"checkpoint-{state.global_step}")
        self.trainer.save_da_kd_state(checkpoint)


class _DAKDDataUpdateCallback(TrainerCallback):
    def __init__(self, trainer: KDTrainer) -> None:
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        del args, control

        # When resuming in the middle of an epoch, keep the exact subset that
        # was active when the checkpoint was written.  Re-scoring here would
        # use the partially-trained model and silently change the remainder of
        # the resumed epoch.
        resume_epoch = self.trainer._da_kd_resume_active_epoch
        if resume_epoch is not None:
            current_epoch = float(state.epoch or 0.0)
            self.trainer._da_kd_resume_active_epoch = None
            if int(current_epoch) == resume_epoch and current_epoch > resume_epoch:
                return

        if state.epoch is None:
            return
        epoch = int(state.epoch)
        if epoch <= 0:
            return
        # CallbackHandler exposes ``trainer.model`` (the unwrapped model),
        # while the training loop forwards ``model_wrapped`` to the actual
        # DDP/DeepSpeed/FSDP step.  DDS must use the latter as well.
        model = getattr(self.trainer, "model_wrapped", None)
        if model is None:
            model = kwargs.get("model", self.trainer.model)
        self.trainer.update_da_kd_dataset(model, epoch)


class KDTrainer(CustomSeq2SeqTrainer):
    """Shared LlamaFactory trainer for SFT and teacher-based KD baselines."""

    def __init__(
        self,
        *args,
        distillation_args: DistillationArguments,
        data_args: Any,
        generating_args: Any,
        **kwargs,
    ) -> None:
        self.distillation_args = distillation_args
        self.data_args = data_args
        self.generating_args = generating_args
        if self.distillation_args.uses_da_kd and not hasattr(CustomSeq2SeqTrainer, "_run_epoch"):
            raise RuntimeError("DA-KD requires Transformers >= 5.5.0 for dynamic epoch lengths.")
        super().__init__(*args, **kwargs)
        if self.distillation_args.uses_kd and self.ref_model is None:
            raise ValueError("Knowledge distillation requires LlamaFactory ref_model.")

        self.replay_buffer: ReplayBuffer | None = None
        self.rollout_scheduler: AdaptiveRolloutScheduler | None = None
        self.rollout_generator: StudentRolloutGenerator | None = None
        self._stored_hpd_metrics: defaultdict[str, list[float]] = defaultdict(list)
        self._da_kd_batch_sampler: DifficultyAwareDistributedBatchSampler | None = None
        self._da_kd_scoring_dataset = None
        self._da_kd_scoring_collator = None
        self._da_kd_active_indices: list[int] | None = None
        self._da_kd_active_epoch = 0
        self._da_kd_resume_active_epoch: int | None = None
        self._rollout_counts = {source.value: 0 for source in RolloutSource}

        if self.distillation_args.is_adaptive:
            rank_seed = int(self.args.seed) + int(self.args.process_index)
            self.replay_buffer = ReplayBuffer(self.distillation_args.capacity, seed=rank_seed)
            self.rollout_scheduler = AdaptiveRolloutScheduler(
                threshold=self.distillation_args.init_threshold,
                loss_eps=self.distillation_args.loss_eps,
                replay_ratio=self.distillation_args.replay_ratio,
                seed=rank_seed,
            )
            self.rollout_generator = StudentRolloutGenerator(
                tokenizer=self.processing_class,
                cutoff_len=self.data_args.cutoff_len,
                rollout_context_length=self.distillation_args.rollout_context_length,
                do_sample=self.generating_args.do_sample,
                top_p=self.distillation_args.gen_top_p,
                top_k=self.generating_args.top_k,
                temperature=self.generating_args.temperature,
                repetition_penalty=self.generating_args.repetition_penalty,
            )
            self.add_callback(_DistillMInitialEvalCallback(self))
            self.add_callback(_DistillMStateCallback(self))
        if self.distillation_args.uses_da_kd:
            self.add_callback(_DAKDStateCallback(self))
            self.add_callback(_DAKDDataUpdateCallback(self))

    def get_train_dataloader(self) -> DataLoader:
        """Build the template's DistributedSampler(drop_last=True) batches."""

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, IterableDataset):
            raise ValueError("Template-parity sampling requires a sized training dataset.")
        if self.args.dataloader_drop_last:
            raise ValueError("Set dataloader_drop_last=false; the template drops only at the distributed sampler.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, HFDataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="Training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="Training")

        if self.distillation_args.uses_da_kd:
            batch_sampler = DifficultyAwareDistributedBatchSampler(
                train_dataset,
                batch_size=self._train_batch_size,
                num_replicas=self.accelerator.num_processes,
                seed=0,
            )
            if self._da_kd_active_indices is not None:
                batch_sampler.set_active_indices(self._da_kd_active_indices)
            self._da_kd_batch_sampler = batch_sampler
            self._da_kd_scoring_dataset = train_dataset
            self._da_kd_scoring_collator = data_collator
        else:
            batch_sampler = TemplateDistributedBatchSampler(
                train_dataset,
                batch_size=self._train_batch_size,
                num_replicas=self.accelerator.num_processes,
                # PyTorch DistributedSampler defaults to seed zero in the template.
                seed=0,
            )
        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            multiprocessing_context="fork" if should_fork else None,
            worker_init_fn=partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        )

        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = False
        try:
            return self.accelerator.prepare_data_loader(dataloader)
        finally:
            self.accelerator.even_batches = previous_even_batches

    def _state_file(self, directory: str | os.PathLike[str]) -> Path:
        return Path(directory, f"{DISTILLM_STATE_NAME}_rank{self.args.process_index}.pt")

    def _da_kd_state_file(self, directory: str | os.PathLike[str]) -> Path:
        return Path(directory, f"{DA_KD_STATE_NAME}_rank{self.args.process_index}.pt")

    def save_distillm_state(self, directory: str | os.PathLike[str]) -> None:
        if self.rollout_scheduler is None or self.replay_buffer is None:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scheduler": self.rollout_scheduler.state_dict(),
                "replay_buffer": self.replay_buffer.state_dict(),
                "rollout_counts": self._rollout_counts,
            },
            self._state_file(directory),
        )

    def load_distillm_state(self, directory: str | os.PathLike[str]) -> bool:
        if self.rollout_scheduler is None or self.replay_buffer is None:
            return False
        state_file = self._state_file(directory)
        if not state_file.exists():
            return False
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        self.rollout_scheduler.load_state_dict(state["scheduler"])
        self.replay_buffer.load_state_dict(state["replay_buffer"])
        self._rollout_counts.update(state.get("rollout_counts", {}))
        return True

    def save_da_kd_state(self, directory: str | os.PathLike[str]) -> None:
        if not self.distillation_args.uses_da_kd:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "active_indices": self._da_kd_active_indices,
                "active_epoch": self._da_kd_active_epoch,
            },
            self._da_kd_state_file(directory),
        )

    def load_da_kd_state(self, directory: str | os.PathLike[str]) -> bool:
        if not self.distillation_args.uses_da_kd:
            return False
        state_file = self._da_kd_state_file(directory)
        if not state_file.exists():
            return False
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        active_indices = state.get("active_indices")
        self._da_kd_active_indices = None if active_indices is None else [int(index) for index in active_indices]
        self._da_kd_active_epoch = int(state.get("active_epoch", 0))
        self._da_kd_resume_active_epoch = self._da_kd_active_epoch
        return True

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        checkpoint_path: str | os.PathLike[str] | None = None
        if isinstance(resume_from_checkpoint, str | os.PathLike):
            checkpoint_path = resume_from_checkpoint
        elif resume_from_checkpoint is True:
            checkpoint_path = get_last_checkpoint(self.args.output_dir)

        if checkpoint_path is not None:
            self.load_distillm_state(checkpoint_path)
            self.load_da_kd_state(checkpoint_path)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def initialize_distillm_baseline(self) -> None:
        """Evaluate once after DeepSpeed setup and before the first training batch."""

        if self.rollout_scheduler is None or self.rollout_scheduler.previous_validation_loss is not None:
            return
        metrics = super().evaluate(metric_key_prefix="distillm_initial")
        try:
            initial_loss = float(metrics["distillm_initial_loss"])
        except KeyError as exc:
            raise RuntimeError("DistiLLM initial evaluation did not return a validation loss.") from exc
        self.rollout_scheduler.initialize(initial_loss)
        print_rank(f"DistiLLM initial validation loss={initial_loss:.6f}")

    def save_state(self) -> None:
        super().save_state()
        self.save_distillm_state(self.args.output_dir)
        self.save_da_kd_state(self.args.output_dir)

    def _training_progress(self) -> float:
        max_steps = max(int(getattr(self.state, "max_steps", 0)), 1)
        # The template numbers the first optimizer step as one.
        return min(max(float(self.state.global_step + 1) / max_steps, 0.0), 1.0)

    def _collate_rollouts(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self.data_collator(features)
        return self._prepare_inputs(batch)

    def _da_kd_minimum_size(self) -> int:
        return self._train_batch_size * max(int(self.accelerator.num_processes), 1)

    def _da_kd_active_size_for_epoch(self, epoch: int) -> int:
        if self._da_kd_scoring_dataset is None:
            raise RuntimeError("DA-KD dataset is not initialized.")
        total = len(self._da_kd_scoring_dataset)
        if epoch <= 0:
            return total
        ratio = selection_ratio(float(epoch), float(self.args.num_train_epochs), self.distillation_args.da_kd_schedule)
        return selection_size(
            total,
            ratio=ratio,
            min_size=self._da_kd_minimum_size(),
            multiple=max(int(self.accelerator.num_processes), 1),
        )

    def _da_kd_micro_batches_for_size(self, active_size: int) -> int:
        world_size = max(int(self.accelerator.num_processes), 1)
        samples_per_rank = (active_size // world_size)
        if samples_per_rank <= 0:
            return 0
        return math.ceil(samples_per_rank / self._train_batch_size)

    def _da_kd_update_steps_for_epoch(self, epoch: int) -> int:
        micro_batches = self._da_kd_micro_batches_for_size(self._da_kd_active_size_for_epoch(epoch))
        if micro_batches <= 0:
            raise ValueError("DA-KD needs at least one batch per rank in every epoch.")
        return max(math.ceil(micro_batches / self.args.gradient_accumulation_steps), 1)

    def set_initial_training_values(self, args, dataloader):
        """Make the scheduler and stopping condition account for DA-KD's shrinking epochs."""

        values = super().set_initial_training_values(args, dataloader)
        if not self.distillation_args.uses_da_kd or self._da_kd_scoring_dataset is None:
            return values

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        ) = values

        if args.max_steps < 0:
            total_epochs = float(args.num_train_epochs)
            full_epochs = int(math.floor(total_epochs))
            fractional_epoch = total_epochs - full_epochs
            dynamic_max_steps = sum(self._da_kd_update_steps_for_epoch(epoch) for epoch in range(full_epochs))
            if fractional_epoch > 0.0:
                partial_steps = self._da_kd_update_steps_for_epoch(full_epochs)
                dynamic_max_steps += max(math.ceil(fractional_epoch * partial_steps), 1)
            max_steps = max(dynamic_max_steps, 1)
            num_train_samples = max_steps * total_train_batch_size
        else:
            # Explicit max_steps must remain authoritative.  A shrinking
            # dataset may require more outer epochs to reach that step count.
            candidate_epochs = max(int(num_train_epochs), 1)
            smallest_steps = min(self._da_kd_update_steps_for_epoch(epoch) for epoch in range(candidate_epochs))
            num_train_epochs = max(num_train_epochs, math.ceil(args.max_steps / smallest_steps))

        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        )

    def _init_training_state(
        self,
        max_steps,
        num_update_steps_per_epoch,
        num_train_epochs,
        resume_from_checkpoint,
        trial,
    ):
        result = super()._init_training_state(
            max_steps,
            num_update_steps_per_epoch,
            num_train_epochs,
            resume_from_checkpoint,
            trial,
        )
        if (
            not self.distillation_args.uses_da_kd
            or resume_from_checkpoint is None
            or self._da_kd_scoring_dataset is None
            or int(self.state.global_step) <= 0
        ):
            return result

        # Trainer's default resume arithmetic assumes a constant number of
        # steps per epoch.  Reconstruct the epoch boundary from DA-KD's
        # deterministic schedule instead.
        global_step = int(self.state.global_step)
        completed_steps = 0
        for epoch in range(max(int(num_train_epochs), 1)):
            epoch_steps = self._da_kd_update_steps_for_epoch(epoch)
            if global_step < completed_steps + epoch_steps:
                micro_batches = (global_step - completed_steps) * self.args.gradient_accumulation_steps
                return epoch, micro_batches
            completed_steps += epoch_steps
        return result

    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        if not self.distillation_args.uses_da_kd:
            return super()._run_epoch(
                model,
                epoch,
                train_dataloader,
                steps_in_epoch,
                num_update_steps_per_epoch,
                trial,
                ignore_keys_for_eval,
                start_time,
                resume_from_checkpoint,
                epochs_trained,
                steps_trained_in_current_epoch,
            )

        full_steps_in_epoch = len(train_dataloader)
        if full_steps_in_epoch <= 0:
            raise ValueError("DA-KD produced no training batches; increase the dataset or batch size.")

        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        full_update_steps = max(math.ceil(full_steps_in_epoch / gradient_accumulation_steps), 1)
        skipped_steps = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            skipped_steps = steps_trained_in_current_epoch // gradient_accumulation_steps

        remaining_steps = max(int(self.state.max_steps) - int(self.state.global_step), 0)
        if remaining_steps == 0:
            self.control.should_training_stop = True
            return None

        dynamic_update_steps = min(full_update_steps, skipped_steps + remaining_steps)
        dynamic_steps_in_epoch = min(full_steps_in_epoch, dynamic_update_steps * gradient_accumulation_steps)
        if dynamic_update_steps <= skipped_steps or dynamic_steps_in_epoch <= steps_trained_in_current_epoch:
            self.control.should_training_stop = True
            return None

        result = super()._run_epoch(
            model,
            epoch,
            train_dataloader,
            dynamic_steps_in_epoch,
            dynamic_update_steps,
            trial,
            ignore_keys_for_eval,
            start_time,
            resume_from_checkpoint,
            epochs_trained,
            steps_trained_in_current_epoch,
        )
        if self.state.global_step >= self.state.max_steps:
            self.control.should_training_stop = True
        return result

    def _gather_da_kd_scores(
        self,
        local_indices: list[int],
        local_scores: list[float],
        total_size: int,
    ) -> list[float]:
        if len(local_indices) != len(local_scores):
            raise RuntimeError("DA-KD score/index lengths do not match.")
        if not distributed_is_initialized():
            scores = [0.0] * total_size
            for index, score in zip(local_indices, local_scores, strict=True):
                scores[index] = float(score)
            return scores

        device = self.accelerator.device
        world_size = get_world_size()
        local = torch.tensor(local_scores, dtype=torch.float32, device=device)
        lengths = all_gather_tensor(
            torch.tensor([local.numel()], dtype=torch.long, device=device),
            operation="stack",
        ).reshape(-1)
        max_length = max(int(lengths.max().item()), 1)
        padded = torch.zeros(max_length, dtype=torch.float32, device=device)
        padded[: local.numel()] = local
        gathered = all_gather_tensor(padded, operation="stack")

        scores = [0.0] * total_size
        for rank in range(world_size):
            rank_length = int(lengths[rank].item())
            for offset in range(rank_length):
                index = rank + offset * world_size
                if index < total_size:
                    scores[index] = float(gathered[rank, offset].item())
        return scores

    def update_da_kd_dataset(self, model: torch.nn.Module, epoch: int) -> None:
        """Score the full dataset and activate the next DiffUp/SDU subset."""

        if (
            self._da_kd_batch_sampler is None
            or self._da_kd_scoring_dataset is None
            or self._da_kd_scoring_collator is None
        ):
            return

        total_size = len(self._da_kd_scoring_dataset)
        world_size = get_world_size() if distributed_is_initialized() else max(int(self.accelerator.num_processes), 1)
        rank = get_rank() if distributed_is_initialized() else int(self.args.process_index)
        # Keep the number of scoring batches identical on every rank.  This
        # matters for DDP/ZeRO models whose forward pass may synchronize
        # buffers/partitions.  Padded indices are only used for scoring and
        # are ignored when scores are reconstructed below.
        samples_per_rank = math.ceil(total_size / world_size) if total_size else 0
        local_indices = [
            (rank + offset * world_size) % total_size
            for offset in range(samples_per_rank)
        ]
        scoring_dataset = (
            self._da_kd_scoring_dataset
            if world_size == 1
            else Subset(self._da_kd_scoring_dataset, local_indices)
        )
        scoring_loader = DataLoader(
            scoring_dataset,
            batch_size=self._train_batch_size,
            shuffle=False,
            collate_fn=self._da_kd_scoring_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )
        was_training = model.training
        ref_was_training = self.ref_model.training
        model.eval()
        self.ref_model.eval()
        local_scores: list[float] = []
        try:
            with torch.no_grad():
                for features in scoring_loader:
                    inputs = self._prepare_inputs(features)
                    labels = inputs.pop("labels")
                    student_outputs = model(**inputs, use_cache=False)
                    teacher_outputs = self.ref_model(**inputs, use_cache=False)
                    if student_outputs.logits.shape[-1] != teacher_outputs.logits.shape[-1]:
                        raise ValueError("DA-KD requires student and teacher models with the same vocabulary size.")
                    score_labels = labels
                    student_loss = per_sample_causal_cross_entropy(student_outputs.logits, score_labels)
                    teacher_loss = per_sample_causal_cross_entropy(teacher_outputs.logits, score_labels)
                    dds = student_loss / teacher_loss.clamp_min(torch.finfo(student_loss.dtype).eps)
                    if not torch.isfinite(dds).all():
                        raise ValueError("DA-KD produced a non-finite DDS score.")
                    local_scores.extend(dds.detach().float().cpu().tolist())
        finally:
            if was_training:
                model.train()
            if ref_was_training:
                self.ref_model.train()

        scores = self._gather_da_kd_scores(local_indices, local_scores, total_size)
        total_epochs = float(self.args.num_train_epochs)
        ratio = selection_ratio(epoch, total_epochs, self.distillation_args.da_kd_schedule)
        minimum_size = self._train_batch_size * self.accelerator.num_processes
        active_indices = stratified_select_indices(
            scores,
            ratio=ratio,
            tau=self.distillation_args.da_kd_tau,
            seed=int(self.args.seed) + epoch,
            min_size=minimum_size,
            multiple=max(int(self.accelerator.num_processes), 1),
        )
        self._da_kd_batch_sampler.set_active_indices(active_indices)
        self._da_kd_active_indices = list(active_indices)
        self._da_kd_active_epoch = epoch
        print_rank(
            f"DA-KD epoch={epoch + 1} ratio={ratio:.4f} "
            f"active={len(active_indices)}/{len(scores)} "
            f"dds_mean={sum(scores) / max(len(scores), 1):.4f}"
        )

    def _store_hpd_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._stored_hpd_metrics[key].append(float(value))

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._stored_hpd_metrics:
            logs = dict(logs)
            metric_values = []
            metric_keys = []
            for key, values in self._stored_hpd_metrics.items():
                metric_keys.append(key)
                metric_values.append(torch.tensor(values, dtype=torch.float32, device=self.accelerator.device).mean())
            reduced_values = self.accelerator.reduce(torch.stack(metric_values), "mean").tolist()
            logs.update(dict(zip(metric_keys, reduced_values, strict=True)))
            self._stored_hpd_metrics.clear()
        super().log(logs, *args, **kwargs)

    def _maybe_replace_with_student_data(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if (
            not self.distillation_args.is_adaptive
            or not model.training
            or self.rollout_scheduler is None
            or self.replay_buffer is None
            or self.rollout_generator is None
        ):
            return inputs

        batch_size = int(inputs["input_ids"].shape[0])
        source = self.rollout_scheduler.choose(
            progress=self._training_progress(),
            replay_size=len(self.replay_buffer),
            capacity=self.replay_buffer.capacity,
            batch_size=batch_size,
        )
        self._rollout_counts[source.value] += 1

        if source is RolloutSource.FRESH:
            generation_model = _unwrap_model(model, self.accelerator)
            features = self.rollout_generator.generate(generation_model, inputs)
            # A student can emit EOS immediately for every selected span.
            # Keep the original supervised batch in that case rather than
            # collating an empty/all-masked rollout batch.
            if not features:
                return inputs
            self.replay_buffer.extend(features)
            return self._collate_rollouts(features)
        if source is RolloutSource.REPLAY:
            return self._collate_rollouts(self.replay_buffer.sample(batch_size))
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        inputs = self._maybe_replace_with_student_data(model, inputs)
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("KDTrainer requires labels from the LlamaFactory SFT collator.")

        output_hidden_states = self.distillation_args.uses_fdd and model.training
        student_outputs = model(**inputs, output_hidden_states=output_hidden_states, use_cache=False)
        if student_outputs.loss is None:
            raise ValueError("The student model did not return an LM loss.")
        lm_loss = student_outputs.loss

        # SFT is deliberately teacher-free.  This also makes every existing
        # KD method with kd_ratio=0 a true SFT run: no teacher forward, no KD
        # loss dispatch, and no KD-specific data selection.
        if not self.distillation_args.uses_kd:
            return (lm_loss, student_outputs) if return_outputs else lm_loss

        # The reference evaluation loop measures student LM loss only. This
        # value drives DistiLLM's adaptive threshold and does not require a
        # teacher forward pass.
        if not model.training:
            return (lm_loss, student_outputs) if return_outputs else lm_loss

        teacher_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        with torch.no_grad():
            self.ref_model.eval()
            teacher_outputs = self.ref_model(
                **teacher_inputs,
                output_hidden_states=output_hidden_states,
                use_cache=False,
            )

        if self.distillation_args.base_method == "hpd":
            kd_loss, hpd_metrics = compute_hpd_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
                sample_in_fp32=self.distillation_args.hpd_sample_in_fp32,
            )
            self._store_hpd_metrics(hpd_metrics)
        else:
            kd_loss = compute_distillation_loss(
                self.distillation_args.base_method,
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
                skew_alpha=self.distillation_args.skew_alpha,
                amid_div_name=self.distillation_args.amid_div_name,
                amid_div_order=self.distillation_args.amid_div_order,
                amid_alpha=self.distillation_args.amid_alpha,
                amid_lam=self.distillation_args.amid_lam,
                bdl_lambda=self.distillation_args.bdl_lambda,
            )

        if self.distillation_args.uses_fdd:
            if student_outputs.hidden_states is None or teacher_outputs.hidden_states is None:
                raise ValueError("FDD requires output_hidden_states from student and teacher.")
            feature_mask = causal_response_mask(labels, inputs["attention_mask"])
            feature_loss = fdd_loss(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states,
                feature_mask,
                _lm_head(model, self.accelerator),
                _lm_head(self.ref_model, self.accelerator),
                self.distillation_args.student_layer_mapping,
                self.distillation_args.teacher_layer_mapping,
            )
            kd_component = 2.0 * (
                (1.0 - self.distillation_args.fdd_weight) * kd_loss + self.distillation_args.fdd_weight * feature_loss
            )
        else:
            kd_component = kd_loss

        loss = (1.0 - self.distillation_args.kd_ratio) * lm_loss + self.distillation_args.kd_ratio * kd_component
        return (loss, student_outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        original_padding_side = self.processing_class.padding_side
        if self.args.predict_with_generate:
            self.processing_class.padding_side = "left"
        try:
            metrics = super().evaluate(*args, **kwargs)
        finally:
            self.processing_class.padding_side = original_padding_side
        if self.rollout_scheduler is not None and "eval_loss" in metrics:
            changed = False
            if self.is_in_train:
                changed = self.rollout_scheduler.update(float(metrics["eval_loss"]))
            metrics["eval_distillm_threshold"] = self.rollout_scheduler.threshold
            metrics["eval_distillm_threshold_changed"] = float(changed)
            metrics.update({f"eval_rollout_{key}_steps": value for key, value in self._rollout_counts.items()})
            print_rank(
                f"DistiLLM threshold={self.rollout_scheduler.threshold:.2f} after eval_loss={metrics['eval_loss']:.6f}"
            )
        return metrics

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
        **gen_kwargs,
    ):
        """Preserve student LM loss while LlamaFactory generates metric predictions."""

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )

        prepared_inputs = self._prepare_inputs(inputs)
        labels = prepared_inputs["labels"]
        attention_mask = prepared_inputs.get("attention_mask")
        prompt_rows = []
        for index in range(labels.size(0)):
            if attention_mask is not None and attention_mask.ndim == 2:
                valid = attention_mask[index].bool()
                input_ids = prepared_inputs["input_ids"][index][valid]
                row_labels = labels[index][valid]
            else:
                input_ids = prepared_inputs["input_ids"][index]
                row_labels = labels[index]
            response_positions = torch.nonzero(row_labels.ne(-100), as_tuple=False).flatten()
            if response_positions.numel() == 0:
                raise ValueError("Generative evaluation requires at least one unmasked response token per row.")
            prompt_rows.append(input_ids[: int(response_positions[0])])

        prompt_length = max(row.numel() for row in prompt_rows)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        prompt_input_ids = prepared_inputs["input_ids"].new_full(
            (len(prompt_rows), prompt_length), pad_token_id
        )
        prompt_attention_mask = prepared_inputs["input_ids"].new_zeros(
            (len(prompt_rows), prompt_length)
        )
        for index, row in enumerate(prompt_rows):
            prompt_input_ids[index, -row.numel() :] = row
            prompt_attention_mask[index, -row.numel() :] = 1

        generation_inputs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
        }
        _, generated_tokens, _ = super().prediction_step(
            model,
            generation_inputs,
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, prepared_inputs).detach().mean()
        return loss, generated_tokens, labels
