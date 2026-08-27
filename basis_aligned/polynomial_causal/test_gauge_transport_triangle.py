import pytest
import torch

import gauge_transport_triangle as TRIANGLE


def test_suffix_mask_is_causal_and_row_specific():
    positions = torch.tensor([0, 2, 4], dtype=torch.long)
    mask = TRIANGLE.suffix_mask(positions, length=5)
    assert mask.tolist() == [
        [True, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, False, True],
    ]


def test_scale_selection_requires_band_and_is_geometric():
    failed = TRIANGLE.choose_scale([
        {"amplitude": 1.0, "median_suffix_kl": 0.001},
        {"amplitude": 2.0, "median_suffix_kl": 0.5},
    ])
    assert not failed["passed"]
    passed = TRIANGLE.choose_scale([
        {"amplitude": 1.0, "median_suffix_kl": 0.02},
        {"amplitude": 2.0, "median_suffix_kl": 0.05},
        {"amplitude": 3.0, "median_suffix_kl": 0.18},
    ])
    assert passed["passed"]
    assert passed["selected"]["amplitude"] == 2.0


def test_sparse_delta_changes_only_declared_positions():
    basis = torch.eye(5)[:, :2]
    coordinates = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    positions = torch.tensor([1, 3], dtype=torch.long)
    delta = TRIANGLE.sparse_physical_delta(coordinates, basis, positions, length=4)
    assert torch.equal(delta[0, 1], torch.tensor([2.0, 3.0, 0.0, 0.0, 0.0]))
    assert torch.equal(delta[1, 3], torch.tensor([4.0, 5.0, 0.0, 0.0, 0.0]))
    assert int(torch.count_nonzero(delta[:, [0, 2]])) == 0


def test_output_accumulator_uses_ratio_of_sums_and_raw_centering():
    baseline = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, -2.0]]])
    early = baseline + torch.tensor([[[0.4, -0.2, 0.1], [-0.3, 0.2, 0.5]]])
    accumulator = TRIANGLE.empty_arm_accumulator()
    TRIANGLE.accumulate_output_arm(
        accumulator,
        baseline, baseline, early, early, early + 9.0, early + 9.0,
        torch.tensor([0], dtype=torch.long),
    )
    result = TRIANGLE.finish_output_arm(accumulator)
    assert result["e_out"] == pytest.approx(0.0, abs=1e-10)
    assert result["centered_raw_logit_relative_rmse"] < 1e-5


def test_screen_gates_fail_independently():
    passing = {
        "full_oracle": {"e_out": 0.0, "centered_raw_logit_relative_rmse": 0.0},
        "projected_oracle": {"e_out": 0.2},
        "direct": {"e_out": 0.2, "coordinate_response_r2": 0.8},
        "chain": {"e_out": 0.25, "coordinate_response_r2": 0.7},
    }
    assert all(TRIANGLE.screen_decisions(passing).values())
    broken = {name: dict(value) for name, value in passing.items()}
    broken["projected_oracle"]["e_out"] = 0.3
    decisions = TRIANGLE.screen_decisions(broken)
    assert not decisions["projected_u14_sufficient"]
    assert decisions["direct_response_transport"]


def test_registered_specs_are_disjoint_and_receipt_addressed():
    assert len({TRIANGLE.BASIS_SPEC, TRIANGLE.FIT_SPEC, TRIANGLE.EVAL_SPEC}) == 3
    assert TRIANGLE.BASIS_SPEC == (96, 80)
    assert TRIANGLE.FIT_SPEC == (96, 1200)
    assert TRIANGLE.EVAL_SPEC == (192, 11000)
    assert (
        TRIANGLE.CALIBRATION_ROWS
        + TRIANGLE.MAP_FIT_ROWS
        + TRIANGLE.MAP_VALIDATION_ROWS
    ) == TRIANGLE.FIT_SPEC[0]


def test_document_provenance_is_mandatory_and_cross_split_docs_fail():
    with pytest.raises(RuntimeError, match="document_provenance"):
        TRIANGLE.require_document_disjoint_receipt({})

    sets = {}
    for n, skip in (TRIANGLE.BASIS_SPEC, TRIANGLE.FIT_SPEC, TRIANGLE.EVAL_SPEC):
        sets[f"n{n}_skip{skip}"] = [
            {"document_id": f"{skip}-{index}", "chunk_id": 0}
            for index in range(n)
        ]
    receipt = {"document_provenance": {"schema_version": 1, "sets": sets}}
    assert len(TRIANGLE.require_document_disjoint_receipt(receipt)) == 3
    sets["n96_skip1200"][0]["document_id"] = sets["n96_skip80"][0]["document_id"]
    with pytest.raises(RuntimeError, match="crosses triangle splits"):
        TRIANGLE.require_document_disjoint_receipt(receipt)
