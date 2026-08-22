from __future__ import annotations

from typing import Any

from .arguments import DistillationArguments
from .data import register_tool_dataset_converters
from .metrics import ComputeTaskMetrics
from .trainer import KDTrainer


def run_kd(
    model_args: Any,
    data_args: Any,
    training_args: Any,
    finetuning_args: Any,
    generating_args: Any,
    distillation_args: DistillationArguments,
) -> None:
    """Run the LlamaFactory SFT/KD workflow with the shared trainer."""

    from llamafactory.data import (
        SFTDataCollatorWith4DAttentionMask,
        get_dataset,
        get_template_and_fix_tokenizer,
    )
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.extras.misc import calculate_tps
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.train.trainer_utils import create_modelcard_and_push, create_ref_model

    # Tool-use datasets use LlamaFactory's native ShareGPT/OpenAI converters
    # and model-native chat templates (qwen3_nothink, llama3, ...).
    register_tool_dataset_converters()
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # Keep eval tokenization supervised even when generative metrics are
    # enabled. KDTrainer extracts prompt-only inputs for model.generate while
    # retaining the full sequence for the validation LM loss.
    generate_eval = training_args.predict_with_generate
    if generate_eval:
        training_args.predict_with_generate = False
    try:
        dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    finally:
        training_args.predict_with_generate = generate_eval
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    ref_model = None
    if distillation_args.uses_kd:
        ref_model = create_ref_model(model_args, finetuning_args)
        if ref_model is None:
            raise ValueError("Set ref_model to the teacher checkpoint for knowledge distillation.")

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeTaskMetrics(tokenizer)

    # Evaluation is deterministic and bounded. The longest gold response in
    # the prepared corpus is well below this budget.
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs.update(do_sample=False, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)
    extra_eos_ids = getattr(tokenizer, "additional_special_tokens_ids", []) or []
    eos_ids = [tokenizer.eos_token_id, *extra_eos_ids]
    valid_eos_ids = (token_id for token_id in eos_ids if token_id is not None and token_id >= 0)
    gen_kwargs["eos_token_id"] = list(dict.fromkeys(valid_eos_ids))

    trainer = KDTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        gen_kwargs=gen_kwargs,
        ref_model=ref_model,
        distillation_args=distillation_args,
        data_args=data_args,
        generating_args=generating_args,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        raise ValueError(
            "Tool-use training does not provide text-only prediction; evaluate tool calls with an executor."
        )

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
