from __future__ import annotations

import pytest
import torch

import finite_horizon_tangent_bundle as bundle


def _responses(
    bases: list[torch.Tensor], *, probes: int = 4, scales: list[float] | None = None,
) -> torch.Tensor:
    scales = scales or [1.0] * len(bases)
    rows = []
    for basis, scale in zip(bases, scales, strict=True):
        coefficients = torch.eye(basis.shape[0], dtype=torch.float64)[
            torch.arange(probes) % basis.shape[0]
        ]
        rows.append(scale * coefficients @ basis)
    return torch.cat(rows)


def test_shared_dictionary_passes_and_is_invariant_to_context_scale() -> None:
    basis = torch.eye(8, dtype=torch.float64)[:2]
    primary = _responses([basis] * 5, scales=[1, 2, 3, 4, 5])
    replication = _responses([basis] * 4, scales=[9, 1, 7, 2])
    report = bundle.analyze_context_normalized_bundle(
        primary, replication, probes_per_context=4,
    )
    assert report["passes"] is True
    assert report["primary_selected_rank"] <= 2
    assert report["replication_capture_by_primary_dictionary"] == pytest.approx(1.0)
    assert report["raw_responses_returned"] is False
    assert report["density_matrices_returned"] is False
    assert report["projectors_returned"] is False


def test_rotating_local_codes_have_large_pooled_rank_and_fail() -> None:
    identity = torch.eye(8, dtype=torch.float64)
    bases = [identity[start:start + 2] for start in (0, 2, 4, 6)]
    primary = _responses(bases)
    replication = _responses(list(reversed(bases)))
    report = bundle.analyze_context_normalized_bundle(
        primary, replication, probes_per_context=4,
    )
    assert report["passes"] is False
    assert report["primary_selected_rank"] > report["rank_limit"]
    assert report["bundle_rotation_gap"] > 0
    assert report["local_rank_summary"]["primary_maximum"] <= 2


def test_primary_dictionary_is_evaluated_directionally_on_replication() -> None:
    identity = torch.eye(8, dtype=torch.float64)
    primary = _responses([identity[:2]] * 4)
    replication = _responses([identity[2:4]] * 4)
    report = bundle.analyze_context_normalized_bundle(
        primary, replication, probes_per_context=4,
    )
    assert report["primary_capture"] == pytest.approx(1.0)
    assert report["replication_capture_by_primary_dictionary"] == pytest.approx(0.0)
    assert report["gates"]["replication_energy_capture"] is False


def test_orthogonal_coordinate_gauge_preserves_scalar_diagnostics() -> None:
    torch.manual_seed(7)
    primary = torch.randn(20, 8, dtype=torch.float64)
    replication = torch.randn(16, 8, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    left = bundle.analyze_context_normalized_bundle(
        primary, replication, probes_per_context=4,
    )
    right = bundle.analyze_context_normalized_bundle(
        primary @ q, replication @ q, probes_per_context=4,
    )
    for key in (
        "primary_capture", "replication_capture_by_primary_dictionary",
        "absolute_capture_gap", "normalized_barycenter_frobenius_distance",
    ):
        assert left[key] == pytest.approx(right[key], abs=1e-12)
    assert left["primary_selected_rank"] == right["primary_selected_rank"]
    assert left["replication_selected_rank"] == right["replication_selected_rank"]


@pytest.mark.parametrize("bad", [
    torch.ones(3, 4, dtype=torch.float64),
    torch.zeros(4, 4, dtype=torch.float64),
    torch.full((4, 4), float("nan"), dtype=torch.float64),
])
def test_malformed_or_zero_contexts_fail_closed(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        bundle.analyze_context_normalized_bundle(
            bad, torch.ones(4, 4, dtype=torch.float64), probes_per_context=2,
        )


def test_split_width_and_constants_are_validated() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        bundle.analyze_context_normalized_bundle(
            torch.ones(4, 3), torch.ones(4, 4), probes_per_context=2,
        )
    with pytest.raises(ValueError, match="constants"):
        bundle.analyze_context_normalized_bundle(
            torch.ones(4, 3), torch.ones(4, 3), probes_per_context=2,
            maximum_fractional_rank=1.0,
        )


def test_site_registry_requires_complete_explicit_blocks() -> None:
    block = torch.eye(4, dtype=torch.float64).repeat(2, 1)
    report = bundle.analyze_site_bundles(
        {(3, 0): block, (3, 1): block},
        {(3, 0): block, (3, 1): block},
        target_site=3, source_sites=(0, 1), probes_per_context=4,
    )
    assert set(report) == {"0", "1"}
    with pytest.raises(ValueError, match="every and only"):
        bundle.analyze_site_bundles(
            {(3, 0): block}, {(3, 0): block},
            target_site=3, source_sites=(0, 1), probes_per_context=4,
        )


def test_repeated_probe_audit_detects_stable_rotating_physical_bundle() -> None:
    directions = torch.eye(8, dtype=torch.float64)
    identity = torch.eye(8, dtype=torch.float64)
    bases = [identity[start:start + 2] for start in (0, 2, 4, 6)]
    first = _responses(bases, probes=4)
    second = _responses(bases, probes=4, scales=[2, 3, 4, 5])
    report = bundle.analyze_repeated_probe_physical_bundle(
        first, second, directions, probes_per_half=4, fixed_ranks=(2,),
        local_rank_limit=2, bootstrap_repetitions=200,
    )
    assert report["status"] == "stable_context_varying_response_bundle"
    assert report["stable_local_low_rank_fraction"] == 1.0
    assert report["response_bundle_gate"] is True
    assert report["fixed_rank_physical_projectors"]["2"][
        "same_context_mean_distance"
    ] == pytest.approx(0.0, abs=1e-12)
    assert report["physical_frames_returned"] is False


def test_repeated_probe_audit_does_not_call_one_shared_frame_a_bundle() -> None:
    directions = torch.eye(8, dtype=torch.float64)
    basis = torch.eye(8, dtype=torch.float64)[:2]
    first = _responses([basis] * 4, probes=4)
    second = _responses([basis] * 4, probes=4, scales=[2, 3, 4, 5])
    report = bundle.analyze_repeated_probe_physical_bundle(
        first, second, directions, probes_per_half=4, fixed_ranks=(2,),
        local_rank_limit=2, bootstrap_repetitions=200,
    )
    assert report["status"] == "no_admitted_local_bundle"
    assert report["response_bundle_gate"] is False


def test_repeated_probe_audit_validates_physical_map_and_context_count() -> None:
    responses = torch.eye(4, dtype=torch.float64).repeat(2, 1)
    with pytest.raises(ValueError, match="at least 3|constants"):
        bundle.analyze_repeated_probe_physical_bundle(
            responses[:4], responses[:4], torch.eye(4, dtype=torch.float64),
            probes_per_half=2, fixed_ranks=(1,), bootstrap_repetitions=100,
        )
    collapsed = torch.zeros(4, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="full row rank"):
        bundle.analyze_repeated_probe_physical_bundle(
            responses, responses, collapsed, probes_per_half=2,
            fixed_ranks=(1,), bootstrap_repetitions=100,
        )


def test_physical_frames_are_invariant_to_nonorthogonal_direction_reparameterization() -> None:
    torch.manual_seed(19)
    directions = torch.randn(4, 9, dtype=torch.float64)
    physical_covectors = torch.randn(7, 9, dtype=torch.float64)
    responses = physical_covectors @ directions.T
    change = torch.tensor([
        [2.0, 0.7, 0.0, 0.0],
        [0.0, 0.5, -0.4, 0.0],
        [0.3, 0.0, 1.5, 0.2],
        [0.0, 0.0, 0.0, 0.8],
    ], dtype=torch.float64)
    changed_directions = change @ directions
    changed_responses = responses @ change.T
    for rank in (1, 2, 3):
        original = bundle._physical_frame(responses, directions, rank)
        changed = bundle._physical_frame(
            changed_responses, changed_directions, rank,
        )
        assert bundle._frame_distance(original, changed) == pytest.approx(
            0.0, abs=2e-7,
        )


def test_unstable_fixed_rank_tails_cannot_promote_a_low_rank_bundle() -> None:
    identity = torch.eye(12, dtype=torch.float64)
    first_contexts = []
    second_contexts = []
    for start in (0, 2, 4, 6):
        first = torch.zeros(6, 12, dtype=torch.float64)
        second = torch.zeros(6, 12, dtype=torch.float64)
        first[0, start] = second[0, start] = 10.0
        first[1, start + 1] = second[1, start + 1] = 10.0
        first[2, 8] = first[3, 9] = 0.1
        second[2, 10] = second[3, 11] = 0.1
        first_contexts.append(first)
        second_contexts.append(second)
    report = bundle.analyze_repeated_probe_physical_bundle(
        torch.cat(first_contexts), torch.cat(second_contexts), identity,
        probes_per_half=6, fixed_ranks=(2, 4), local_rank_limit=4,
        bootstrap_repetitions=200,
    )
    assert report["bundle_promotion_fixed_rank"] == 4
    assert report["stable_local_low_rank_fraction"] == 0.0
    assert report["response_bundle_gate"] is False
    assert report["status"] == "no_admitted_local_bundle"


def test_fixed_rank_above_support_is_not_an_identified_projector() -> None:
    basis = torch.eye(8, dtype=torch.float64)[:2]
    first = _responses([basis] * 4, probes=4)
    second = _responses([basis] * 4, probes=4, scales=[2, 3, 4, 5])
    report = bundle.analyze_repeated_probe_physical_bundle(
        first, second, torch.eye(8, dtype=torch.float64),
        probes_per_half=4, fixed_ranks=(2, 4), local_rank_limit=4,
        bootstrap_repetitions=200,
    )
    assert report["fixed_rank_physical_projectors"]["4"][
        "evaluable_contexts"
    ] == 0
    assert report["fixed_rank_physical_projectors"]["4"][
        "cross_minus_same_bootstrap_lcb_95"
    ] is None
    assert report["stable_local_low_rank_fraction"] == 0.0
    assert report["response_bundle_gate"] is False


def test_excluded_contexts_cannot_supply_stable_subset_bundle_signal() -> None:
    identity = torch.eye(10, dtype=torch.float64)
    stable = identity[:2].repeat(2, 1)
    contexts = [stable.clone() for _ in range(12)]
    for start in (2, 3, 4, 5):
        contexts.append(identity[start:start + 4])
    responses = torch.cat(contexts)
    report = bundle.analyze_repeated_probe_physical_bundle(
        responses, responses.clone(), identity, probes_per_half=4,
        fixed_ranks=(2,), local_rank_limit=2, bootstrap_repetitions=200,
    )
    assert report["stable_local_low_rank_fraction"] == 0.75
    assert report["fixed_rank_physical_projectors"]["2"][
        "cross_minus_same_mean"
    ] > 0
    assert report["promotion_stable_fraction"] == 0.75
    assert report["response_bundle_gate"] is False
    assert report["status"] == "no_admitted_local_bundle"


def test_fixed_promotion_cohort_cannot_borrow_signal_from_diagnostics() -> None:
    identity = torch.eye(10, dtype=torch.float64)
    stable = identity[:2].repeat(2, 1)
    contexts = [stable.clone() for _ in range(12)]
    for start in (2, 3, 4, 5):
        contexts.append(identity[start:start + 4])
    responses = torch.cat(contexts)
    report = bundle.analyze_repeated_probe_physical_bundle(
        responses, responses.clone(), identity, probes_per_half=4,
        fixed_ranks=(2,), local_rank_limit=2, bootstrap_repetitions=200,
        promotion_contexts=tuple(range(12)),
    )
    assert report["promotion_stable_fraction"] == 1.0
    assert report["fixed_promotion_cohort_contrast"][
        "cross_minus_same_mean"
    ] == pytest.approx(0.0, abs=1e-12)
    assert report["response_bundle_gate"] is False
    ledger = report["per_context_scalar_ledger"]
    assert len(ledger) == 16
    assert all(row["in_fixed_promotion_cohort"] == (index < 12)
               for index, row in enumerate(ledger))
    assert all(set(row) == {
        "context_index", "in_fixed_promotion_cohort", "first", "second",
        "same_context_physical_projector_distance", "probe_limited_high_rank",
        "stable_local_low_rank",
    } for row in ledger)
