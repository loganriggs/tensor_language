"""Focused CPU tests for rung-522 corrected protocol statistics."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

import pytest
import torch


PATH = Path(__file__).with_name("attention8_selective_shared_projector_rung522_protocol.py")
SPEC = importlib.util.spec_from_file_location("rung522_protocol_stats", PATH)
PROTOCOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROTOCOL
SPEC.loader.exec_module(PROTOCOL)


def test_complete_four_bit_null_preserves_strata_and_attains_maximum_movement():
    size = 240
    codes = tuple((index % 15) + 1 for index in range(size))
    classes = tuple(0 if index < 120 else 2 for index in range(size))
    positions = tuple(17 if index < 120 else 81 for index in range(size))
    deciles = tuple(3 if index < 120 else 7 for index in range(size))
    parents = (15,) * size
    keyword = dict(
        token_classes=classes,
        token_positions=positions,
        ce_deciles=deciles,
        parent_slice_codes=parents,
        position_ids=range(1_000, 1_000 + size),
        seed=52_300,
    )
    result = PROTOCOL.permute_four_bit_memberships(codes, **keyword)
    assert result == PROTOCOL.permute_four_bit_memberships(codes, **keyword)
    assert result.attains_maximum_possible_movement
    assert result.moved_nonzero_count == result.maximum_possible_moved_nonzero_count
    for start, stop in ((0, 120), (120, 240)):
        assert Counter(result.permuted_codes[start:stop]) == Counter(codes[start:stop])
    assert all(
        code & ~parent == 0
        for code, parent in zip(result.permuted_codes, parents, strict=True)
    )


def test_membership_null_reports_constraint_forced_inert_stratum():
    inert = PROTOCOL.permute_four_bit_memberships(
        (0b1011,) * 20,
        token_classes=(0,) * 20,
        token_positions=(0,) * 20,
        ce_deciles=(0,) * 20,
        parent_slice_codes=(0b1111,) * 20,
        seed=52_300,
    )
    assert inert.moved_nonzero_count == 0
    assert inert.maximum_possible_moved_nonzero_count == 0
    assert inert.attains_maximum_possible_movement
    with pytest.raises(ValueError, match="outside its circuit parent"):
        PROTOCOL.permute_four_bit_memberships(
            [0b0010],
            token_classes=[0],
            token_positions=[0],
            ce_deciles=[0],
            parent_slice_codes=[0b0001],
            seed=52_300,
        )


def _registered_frames():
    basis = torch.eye(8, dtype=torch.float64)
    real = {
        seed: {"omit202": basis[:, :2], "omit211": basis[:, :2], "omit221": basis[:, :2]}
        for seed in PROTOCOL.REGISTERED_REAL_SEEDS
    }
    null = {
        seed: {
            "omit202": basis[:, :2],
            "omit211": basis[:, 2:4],
            "omit221": basis[:, 4:6],
        }
        for seed in PROTOCOL.REGISTERED_NULL_SEEDS
    }
    return real, null


def test_three_fold_stability_matches_each_seed_before_real_null_comparison():
    real, null = _registered_frames()
    result = PROTOCOL.summarize_matched_three_fold_stability(real, null)
    assert len(result.real) == 5
    assert len(result.null) == 16
    assert all(seed.minimum_overlap == pytest.approx(1) for seed in result.real)
    assert all(seed.minimum_overlap == pytest.approx(0) for seed in result.null)
    assert result.null_q95_higher == 0
    assert result.real_strict_exceed_count == 5
    assert result.passes_four_of_five


def test_stability_fails_on_wrong_seed_or_fold_identity_or_frame_shape():
    real, null = _registered_frames()
    with pytest.raises(ValueError, match="real seeds 52200"):
        PROTOCOL.summarize_matched_three_fold_stability(
            {seed + 1: frames for seed, frames in real.items()}, null
        )
    broken_folds = dict(null)
    broken_folds[52_300] = {"x": torch.eye(8)[:, :2], "y": torch.eye(8)[:, 2:4]}
    with pytest.raises(ValueError, match="three matched folds"):
        PROTOCOL.summarize_matched_three_fold_stability(real, broken_folds)
    broken_shape = dict(null)
    broken_shape[52_300] = {
        key: torch.eye(10, dtype=torch.float64)[:, :2]
        for key in ("omit202", "omit211", "omit221")
    }
    with pytest.raises(ValueError, match="same registered shape"):
        PROTOCOL.summarize_matched_three_fold_stability(real, broken_shape)


def test_bounded_selectivity_fourfold_margin_and_joint_statistic_are_literal():
    value = PROTOCOL.selectivity_from_rms(4, 1)
    assert value.concentration == 4
    assert value.bounded_selectivity == pytest.approx(3 / (5 + 1e-12))
    assert value.fourfold_margin == 0
    joint = PROTOCOL.bounded_joint_statistic([0.8, 0.7, 0.9], [0.4, 0.5, 0.6])
    assert joint.minimum_heldout_selectivity == pytest.approx(0.7)
    assert joint.minimum_heldout_aligned_recovery == pytest.approx(0.4)
    assert joint.product == pytest.approx(0.28)


def _constant_rows(member: float, control: float, count: int = 12):
    return PROTOCOL.RowPairSquares.from_sequences(
        [member * member] * count,
        [1] * count,
        [control * control] * count,
        [1] * count,
        pair_ids=range(count),
    )


def test_bootstrap_indices_follow_exact_sha_little_endian_counter_rule():
    cell_id = "validation:omit202:D0:forward:seed52200"
    indices = PROTOCOL.deterministic_row_bootstrap_indices(3, cell_id=cell_id, draws=2)
    expected = []
    for replicate in range(2):
        row = []
        for draw in range(3):
            digest = hashlib.sha256(
                f"a8-r522-row-bootstrap-v1:{cell_id}:{replicate}:{draw}".encode()
            ).digest()
            row.append(int.from_bytes(digest[:8], "little", signed=False) % 3)
        expected.append(row)
    assert indices.tolist() == expected
    assert "seed" not in inspect.signature(PROTOCOL.deterministic_row_bootstrap).parameters


def test_row_bootstrap_is_cell_keyed_reproducible_and_uses_two_thousand_by_default():
    primary = _constant_rows(5, 1)
    comparison = _constant_rows(3, 1)
    cell_id = "test:omit211:D1:reverse:seed52203"
    first = PROTOCOL.deterministic_row_bootstrap(
        primary, cell_id=cell_id, comparison=comparison
    )
    repeat = PROTOCOL.deterministic_row_bootstrap(
        primary, cell_id=cell_id, comparison=comparison
    )
    assert first == repeat
    assert first.draws == 2_000
    assert first.cell_id == cell_id
    assert first.fourfold_margin_lower95_higher == pytest.approx(1)
    assert first.bounded_selectivity_improvement_lower95_higher == pytest.approx(
        4 / 6 - 2 / 4, abs=1e-12
    )


def test_row_bootstrap_changes_with_full_cell_id_and_uses_pooled_token_rms():
    stats = PROTOCOL.RowPairSquares.from_sequences(
        [1, 100],
        [1, 100],
        [1, 4],
        [1, 100],
        pair_ids=[70, 80],
    )
    one = PROTOCOL.deterministic_row_bootstrap(stats, cell_id="cell:a", draws=80)
    repeat = PROTOCOL.deterministic_row_bootstrap(stats, cell_id="cell:a", draws=80)
    two = PROTOCOL.deterministic_row_bootstrap(stats, cell_id="cell:b", draws=80)
    assert one.sha256 == repeat.sha256
    assert one.fourfold_margin_samples == repeat.fourfold_margin_samples
    assert one.fourfold_margin_samples != two.fourfold_margin_samples
    assert one.point.member_rms == pytest.approx((101 / 101) ** 0.5)
    assert one.point.control_rms == pytest.approx((5 / 101) ** 0.5)


def test_paired_bootstrap_requires_same_ordered_row_identities():
    primary = _constant_rows(5, 1, 4)
    reordered = PROTOCOL.RowPairSquares.from_sequences(
        [9] * 4, [1] * 4, [1] * 4, [1] * 4, pair_ids=[1, 0, 2, 3]
    )
    with pytest.raises(ValueError, match="ordered paired-row identities"):
        PROTOCOL.deterministic_row_bootstrap(
            primary, cell_id="comparison:test", draws=10, comparison=reordered
        )


def test_higher_quantile_uses_higher_interpolation_at_both_tail_probabilities():
    assert PROTOCOL.higher_quantile(range(16), 0.95) == 15
    assert PROTOCOL.higher_quantile(range(32), 0.95) == 30
    assert PROTOCOL.higher_quantile(range(2_000), 0.05) == 100


def test_exact_five_pair_sign_flip_null_has_all_32_values_and_a_strict_gate():
    result = PROTOCOL.exact_five_pair_sign_flip_null([1, 1, 1, 1, 1])
    assert len(result.null_means) == 32
    assert result.observed_mean == 1
    assert result.null_q95_higher == pytest.approx(0.6)
    assert result.strictly_exceeds_q95
    tied = PROTOCOL.exact_five_pair_sign_flip_null([0, 0, 0, 0, 0])
    assert tied.observed_mean == tied.null_q95_higher == 0
    assert not tied.strictly_exceeds_q95
    with pytest.raises(ValueError, match="exactly five"):
        PROTOCOL.exact_five_pair_sign_flip_null([1, 2, 3, 4])


def test_protocol_rejects_cuda_inputs_without_initializing_cuda():
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="nonempty full cell"):
        PROTOCOL.deterministic_row_bootstrap_indices(3, cell_id="", draws=2)
