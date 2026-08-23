from __future__ import annotations

import pytest
import torch

from distillation.da_kd import (
    per_sample_causal_cross_entropy,
    selection_ratio,
    selection_size,
    stratified_select_indices,
    summarize_da_kd_selection,
)
from distillation.losses import compute_distillation_loss, compute_hpd_loss, forward_kl, reverse_kl


@pytest.mark.parametrize(
    "method",
    [
        "fkl",
        "rkl",
        "sfkl",
        "srkl",
        "csd",
    ],
)
def test_core_losses_are_finite_and_mask_non_target_tokens(method: str) -> None:
    generator = torch.Generator().manual_seed(7)
    student = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 11, generator=generator)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5], [-100, 6, 7, -100, 8]],
        dtype=torch.long,
    )

    actual = compute_distillation_loss(method, student, teacher, labels, skew_alpha=0.1)
    actual.backward()
    assert torch.isfinite(actual)
    assert student.grad is not None
    # Row zero's logits at position zero predict labels[0, 1], which is masked.
    torch.testing.assert_close(student.grad[0, 0], torch.zeros_like(student.grad[0, 0]))


def test_csd_bf16_is_finite() -> None:
    generator = torch.Generator().manual_seed(17)
    student = torch.randn(2, 4, 9, generator=generator, dtype=torch.bfloat16)
    teacher = torch.randn(2, 4, 9, generator=generator, dtype=torch.bfloat16)
    labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, 4, 5]])
    actual = compute_distillation_loss("csd", student, teacher, labels)
    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual)


def test_forward_kl_retains_teacher_entropy_constant() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    labels = torch.tensor([[1]])
    loss = forward_kl(logits, logits.detach(), labels)
    assert loss.item() > 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_reverse_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    labels = torch.tensor([[1]])
    torch.testing.assert_close(reverse_kl(logits, logits, labels), torch.tensor(0.0))


def test_causal_alignment_uses_shifted_response_mask() -> None:
    student = torch.zeros(1, 4, 3)
    teacher = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    baseline = compute_distillation_loss("rkl", student, teacher, labels)

    changed_masked_position = student.clone()
    changed_masked_position[:, 0, 0] = 100.0
    masked = compute_distillation_loss("rkl", changed_masked_position, teacher, labels)
    torch.testing.assert_close(masked, baseline)

    changed_response_position = student.clone()
    changed_response_position[:, 1, 0] = 100.0
    unmasked = compute_distillation_loss("rkl", changed_response_position, teacher, labels)
    assert unmasked > baseline


def test_empty_response_mask_is_rejected() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.full((1, 3), -100)
    with pytest.raises(ValueError, match="no response tokens"):
        compute_distillation_loss("fkl", logits, logits, labels)


def test_hpd_loss_is_finite_and_returns_metrics() -> None:
    generator = torch.Generator().manual_seed(31)
    student = torch.randn(2, 6, 13, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 6, 13, generator=generator)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5, 6], [-100, 2, 7, -100, 8, 9]],
        dtype=torch.long,
    )

    torch.manual_seed(11)
    actual, metrics = compute_hpd_loss(student, teacher, labels)
    assert torch.isfinite(actual)
    assert {"hpd_loss", "ce_loss", "sampled_diff_pct", "k1_gt_raw_mean"}.issubset(metrics)
    actual.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_compute_distillation_loss_dispatches_to_hpd() -> None:
    generator = torch.Generator().manual_seed(37)
    student = torch.randn(1, 5, 9, generator=generator)
    teacher = torch.randn(1, 5, 9, generator=generator)
    labels = torch.tensor([[-100, -100, 1, 2, 3]])

    torch.manual_seed(13)
    direct, _ = compute_hpd_loss(student, teacher, labels)
    torch.manual_seed(13)
    dispatched = compute_distillation_loss("hpd", student, teacher, labels)
    torch.testing.assert_close(dispatched, direct)


def test_bdl_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, -1.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 1]])
    loss = compute_distillation_loss("bdl", logits, logits.detach(), labels, bdl_lambda=0.9)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    loss.backward()
    assert logits.grad is not None


def test_bdl_matches_bidirectional_mixture_definition() -> None:
    generator = torch.Generator().manual_seed(31)
    student = torch.randn(2, 4, 7, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 4, 7, generator=generator)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]])
    lam = 0.9

    actual = compute_distillation_loss("bdl", student, teacher, labels, bdl_lambda=lam)

    student_probs = torch.softmax(student[:, :-1].float(), dim=-1)
    teacher_probs = torch.softmax(teacher[:, :-1].float(), dim=-1)
    p_mix = (1.0 - lam) * teacher_probs + lam * student_probs
    q_mix = lam * teacher_probs + (1.0 - lam) * student_probs
    token_loss = (p_mix * (p_mix.log() - q_mix.log())).sum(dim=-1)
    response_mask = labels[:, 1:].ne(-100)
    expected = (token_loss * response_mask).sum() / response_mask.sum()

    torch.testing.assert_close(actual, expected)


def test_bdl_rejects_lambda_that_makes_the_loss_identically_zero() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[-100, 1]])
    with pytest.raises(ValueError, match="identical"):
        compute_distillation_loss("bdl", logits, logits, labels, bdl_lambda=0.5)


def test_da_kd_selection_ratio_and_stratified_sampling() -> None:
    assert selection_ratio(0, 10, "cosine") == 1.0
    assert selection_ratio(5, 10, "linear") == 0.5
    assert selection_ratio(5, 10, "cosine") == pytest.approx(0.5)
    assert selection_size(100, ratio=0.31, min_size=8, multiple=8) == 32

    scores = [0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0]
    selected = stratified_select_indices(scores, ratio=0.5, tau=0.1, seed=7, min_size=1)
    assert len(selected) == 5
    assert set(selected).issubset(range(len(scores)))
    assert len(set(selected)) == len(selected)
    assert any(index in {1, 2, 4, 6} for index in selected)

    selected_multiple = stratified_select_indices(
        scores,
        ratio=0.7,
        tau=0.1,
        seed=7,
        min_size=1,
        multiple=4,
    )
    assert len(selected_multiple) == 8


def test_da_kd_cosine_schedule_matches_the_paper_iteration_fraction() -> None:
    ratios = [selection_ratio(epoch, 10, "cosine") for epoch in range(10)]
    assert ratios[0] == 1.0
    assert sum(ratios) / len(ratios) == pytest.approx(0.55)


def test_da_kd_uses_all_low_samples_when_the_low_partition_is_too_small() -> None:
    scores = list(range(100))
    ratio = selection_ratio(1, 10, "cosine")
    selected = stratified_select_indices(scores, ratio=ratio, tau=0.1, seed=7, min_size=1)

    assert len(selected) == 98
    assert {0, 1}.issubset(selected)
    assert len(set(selected)) == len(selected)


def test_da_kd_selection_summary_reports_realized_mixture_and_ce() -> None:
    summary = summarize_da_kd_selection(
        scores=[4.0, 3.0, 2.0, 1.0],
        student_losses=[4.0, 6.0, 2.0, 1.0],
        teacher_losses=[2.0, 3.0, 4.0, 1.0],
        active_indices=[0, 2],
    )

    assert summary["selected_high_count"] == 1
    assert summary["selected_low_count"] == 1
    assert summary["selected_low_fraction"] == 0.5
    assert summary["dds_median"] == pytest.approx(2.5)
    assert summary["student_ce_mean"] == pytest.approx(3.25)
    assert summary["teacher_ce_mean"] == pytest.approx(2.5)
    assert summary["teacher_better_fraction"] == 0.5


def test_da_kd_rejects_invalid_scores_and_empty_responses() -> None:
    with pytest.raises(ValueError, match="finite"):
        stratified_select_indices([0.1, float("nan")], ratio=0.5, tau=0.1, seed=0, min_size=1)

    with pytest.raises(ValueError, match="response token"):
        per_sample_causal_cross_entropy(torch.zeros(1, 3, 4), torch.full((1, 3), -100))

    with pytest.raises(ValueError, match="schedule"):
        selection_ratio(0, 10, "invalid")


def test_per_sample_causal_cross_entropy_respects_ignore_index() -> None:
    logits = torch.zeros(2, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2], [-100, 0, 0, -100]])
    losses = per_sample_causal_cross_entropy(logits, labels)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(losses[0], torch.tensor(3.0).log())


@pytest.mark.parametrize(
    ("div_name", "div_order"),
    [(div_name, div_order) for div_name in ("fkl", "ab") for div_order in ("pr", "qr", "rp", "rq")],
)
def test_amid_loss_and_gradient_are_finite(div_name: str, div_order: str) -> None:
    generator = torch.Generator().manual_seed(23)
    student = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 11, generator=generator)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5], [-100, 6, 7, -100, 8]],
        dtype=torch.long,
    )

    actual = compute_distillation_loss(
        "amid",
        student,
        teacher,
        labels,
        amid_div_name=div_name,
        amid_div_order=div_order,
        amid_alpha=0.5,
        amid_lam=0.5,
    )
    actual.backward()
    assert torch.isfinite(actual)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
