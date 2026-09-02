from __future__ import annotations

import pytest
import torch

from distillation.losses import causal_lm_loss, compute_distillation_loss, compute_hpd_loss, forward_kl, reverse_kl


def test_causal_lm_loss_matches_huggingface_token_mean() -> None:
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 5, 7, generator=generator)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, -100, 6]])

    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(causal_lm_loss(logits, labels), expected)


def test_causal_lm_loss_upcasts_bf16_logits() -> None:
    logits = torch.randn(2, 4, 7, dtype=torch.bfloat16)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])
    loss = causal_lm_loss(logits, labels)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


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


@pytest.mark.parametrize("method", ["fkl", "rkl", "sfkl", "srkl", "csd", "bdl", "amid"])
def test_distillation_aligns_different_padded_vocab_sizes(method: str) -> None:
    generator = torch.Generator().manual_seed(41)
    student = torch.randn(1, 4, 7, generator=generator, requires_grad=True)
    teacher = torch.randn(1, 4, 9, generator=generator)
    labels = torch.tensor([[-100, -100, 3, 8]])

    actual = compute_distillation_loss(method, student, teacher, labels)
    actual.backward()

    assert torch.isfinite(actual)
    assert student.grad is not None
    # Token 8 exists only in the teacher's padded output vocabulary and is
    # excluded from the shared-vocabulary supervision mask.
    torch.testing.assert_close(student.grad[:, 2], torch.zeros_like(student.grad[:, 2]))


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
