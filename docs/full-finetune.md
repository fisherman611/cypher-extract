# Optional full fine-tuning

The default KD method folders keep LoRA student/teacher training (their SFT
baseline is already full). Full fine-tuning for every method is available as a
separate, non-normalized setting in:

- `configs/llama3_full_finetune`
- `configs/qwen2.5_full_finetune`
- `configs/qwen3_full_finetune`

Each folder contains the same 12 student methods as its default model-family
folder. These presets use `finetuning_type: full`, a learning rate of `2e-5`,
and write to:

```text
results/full_finetune/<model-family>/<method>
```

KD methods distill from the same LoRA teacher as the `lora` setting
(`results/lora/<model-family>/teacher_lora`), loaded through `ref_model`,
`ref_model_revision` and `ref_model_adapters`. Train that teacher first if it
does not exist yet. For example:

```bash
RUN_GPUS=0,1 bash scripts/train.sh \
  configs/distillation/teacher_lora_qwen3.yaml
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen3_full_finetune/fkl.yaml
```

To train all 12 Qwen3 methods in sequence:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen3_full_finetune.sh
```

`train_all.sh` reuses `results/lora/<model-family>/teacher_lora` when its
adapter already exists and trains it there otherwise. Set
`TEACHER_RESULTS_ROOT` to use a teacher from another results root. There is no
full-finetune teacher config; every setting distills from a LoRA teacher.

Student inference uses the existing method names with a different checkpoint
root:

```bash
bash scripts/infer_all_qwen3_full_finetune.sh
```

This wrapper runs all 12 student methods; the shared LoRA teacher is inferred
under its own LoRA setting. Pass `--methods fkl` to run only one method.
The Qwen 2.5 config directory omits `_coder` for consistency with the normalized
presets, while its checkpoint model-family remains `qwen2.5_coder`.
