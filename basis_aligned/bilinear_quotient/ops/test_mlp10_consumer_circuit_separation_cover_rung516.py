import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("mlp10_consumer_circuit_separation_cover_rung516.py")
SPEC = importlib.util.spec_from_file_location("r516", SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_planted_suite_recovers_every_exact_witness_set():
    result = R.planted_suite()
    assert result["all_exact_set_recoveries_and_bars_hold"]
    assert len(result["cases"]) == 8


def test_no_selected_coordinates_witness_nothing():
    table = {
        "beta": torch.ones(3),
        "left0": torch.ones(3, 32), "right0": torch.ones(3, 32),
        "left1": torch.ones(3, 32), "right1": torch.ones(3, 32),
        "task_compatible": torch.ones(3, dtype=torch.bool),
    }
    assert not R.split_mask(table, [], 0).any()


def test_exact_proportional_vectors_do_not_split():
    generator = torch.Generator().manual_seed(1)
    right = torch.randn(7, 32, generator=generator, dtype=torch.float64)
    beta = torch.linspace(.5, 2.0, 7, dtype=torch.float64)
    table = {
        "beta": beta,
        "left0": beta[:, None] * right, "right0": right,
        "left1": beta[:, None] * right, "right1": right,
        "task_compatible": torch.ones(7, dtype=torch.bool),
    }
    assert not R.split_mask(table, list(range(32)), 0).any()
    assert not R.split_mask(table, list(range(32)), 1).any()


def test_wrong_coordinate_has_no_signal_in_planted_case():
    case = R.planted_case(51680)
    assert set(case["planted"]) == set(case["recovered"])
    assert case["holds"]


def test_cover_sizes_and_seeds_are_frozen():
    assert R.COVER_SIZES == (1, 2, 4, 8, 16, 32)
    assert R.CONTROL_SEEDS == tuple(range(51600, 51616))
    assert R.PLANTED_SEEDS == tuple(range(51680, 51688))


def test_terminal_zero_pair_route_is_hash_pinned():
    result, bundle = R.validate_route()
    assert result["pred_a_exact_live_identifiable_finite_downstream_instrument"] is True
    assert result["analysis"]["discovery_summary"]["candidate_count"] == 0
    assert set(bundle["collections"]) == {"discovery"}
