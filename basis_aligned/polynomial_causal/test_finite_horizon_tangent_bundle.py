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
