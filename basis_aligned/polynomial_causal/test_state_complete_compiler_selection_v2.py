from __future__ import annotations

import pytest
import torch

import state_complete_compiler_selection_v2 as selection


def _affine(family: str, rank: int, recovery: float, copy: float = 0.0):
    return {
        "state": {"grammar": "affine", "interface": "state_complete_p",
                  "family": family, "rank": rank, "lambda": 0.001},
        "metrics": {"recovery": recovery, "copy_worsening": copy},
    }


def _native(family: str, k: int, recovery: float, copy: float = 0.0):
    return {
        "state": {"grammar": "native", "interface": "state_complete_p",
                  "family": family, "k": k},
        "metrics": {"recovery": recovery, "copy_worsening": copy},
    }


def test_teacher_kl_is_zero_for_identity_and_positive_for_changed_logits() -> None:
    generator = torch.Generator().manual_seed(5)
    teacher = torch.randn(3, 4, 7, generator=generator)
    valid = torch.ones(3, 4, dtype=torch.bool)
    assert abs(selection.token_weighted_teacher_kl(teacher, teacher, valid)) < 1e-7
    student = teacher.clone()
    student[..., 0] += 1.0
    assert selection.token_weighted_teacher_kl(teacher, student, valid) > 0.0


def test_direct_recovery_requires_positive_denominator() -> None:
    metrics = selection.direct_recovery(0.25, 1.0)
    assert metrics["remaining_kl_ratio"] == 0.25
    assert metrics["recovery"] == 0.75
    with pytest.raises(ValueError, match="positive"):
        selection.direct_recovery(0.1, 0.0)


def test_selection_excludes_A_and_copy_failures_then_takes_smallest_near_best() -> None:
    bank = {
        "A8": _affine("A_v1_like_z_only_affine_euclidean", 8, 0.99),
        "B8": _affine("B_state_complete_affine_euclidean", 8, 0.80),
        "B64": _affine("B_state_complete_affine_euclidean", 64, 0.90),
        "C8": _affine("C_state_complete_affine_causal", 8, 0.895),
        "D8": _native("D_state_complete_native_euclidean", 8, 0.91, copy=0.02),
        "E8": _native("E_state_complete_native_causal", 8, 0.895),
    }
    frozen = selection.freeze_validation_selection(bank, recovery_slack=0.99)
    assert frozen["selected"] == "C8"
    assert frozen["selected_family"] == "C_state_complete_affine_causal"
    assert frozen["family_representatives"]["A_v1_like_z_only_affine_euclidean"] == "A8"
    assert "D8" not in frozen["eligible"]


def test_selection_fails_closed_without_positive_B_to_E_candidate() -> None:
    bank = {
        "A8": _affine("A_v1_like_z_only_affine_euclidean", 8, 0.9),
        "B8": _affine("B_state_complete_affine_euclidean", 8, -0.1),
        "C8": _affine("C_state_complete_affine_causal", 8, -0.2),
        "D8": _native("D_state_complete_native_euclidean", 8, -0.1),
        "E8": _native("E_state_complete_native_causal", 8, -0.3),
    }
    with pytest.raises(RuntimeError, match="no B-E"):
        selection.freeze_validation_selection(bank)


def test_nonpositive_family_representative_uses_exact_argmax() -> None:
    bank = {
        "A8": _affine("A_v1_like_z_only_affine_euclidean", 8, -0.1),
        "A64": _affine("A_v1_like_z_only_affine_euclidean", 64, -0.2),
        "B8": _affine("B_state_complete_affine_euclidean", 8, 0.2),
        "C8": _affine("C_state_complete_affine_causal", 8, 0.2),
        "D8": _native("D_state_complete_native_euclidean", 8, 0.2),
        "E8": _native("E_state_complete_native_causal", 8, 0.2),
    }
    frozen = selection.freeze_validation_selection(bank)
    assert frozen["family_representatives"][
        "A_v1_like_z_only_affine_euclidean"
    ] == "A8"


def test_shuffle_control_freezes_when_every_recovery_is_nonpositive() -> None:
    bank = {
        "A8": _affine("A_v1_like_z_only_affine_euclidean", 8, 0.9),
        "B8": _affine("B_state_complete_affine_euclidean", 8, -0.1),
        "B64": _affine("B_state_complete_affine_euclidean", 64, -0.2),
        "C8": _affine("C_state_complete_affine_causal", 8, -0.1),
        "D8": _native("D_state_complete_native_euclidean", 8, -0.3),
        "E8": _native("E_state_complete_native_causal", 8, -0.4),
    }
    frozen = selection.freeze_control_selection(bank)
    assert frozen["selected"] == "B8"
    assert frozen["best_signed_constrained_recovery"] == -0.1
    assert frozen["positive_recovery_required"] is False
