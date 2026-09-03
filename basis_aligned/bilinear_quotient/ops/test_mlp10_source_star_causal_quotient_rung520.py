import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp10_source_star_causal_quotient_rung520.py")
SPEC = importlib.util.spec_from_file_location("r520", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class Linear:
    def __init__(self, weight):
        self.weight = weight


class ToyMLP:
    def __init__(self):
        generator = torch.Generator().manual_seed(520)
        self.Down = Linear(torch.randn(7, 11, generator=generator))


def toy_factors():
    generator = torch.Generator().manual_seed(521)
    return {
        "left": torch.randn(2, 3, 22, 11, generator=generator),
        "right": torch.randn(2, 3, 22, 11, generator=generator),
    }


def test_every_source_star_has_one_self_and_twenty_one_cross_terms():
    assert len(R.STAR_INDICES) == 22
    for source, indices in enumerate(R.STAR_INDICES):
        assert len(indices) == len(set(indices)) == 22
        pairs = [R.r507.SOURCE_PAIRS[index] for index in indices]
        assert sum(left == right == source for left, right in pairs) == 1
        assert all(source in pair for pair in pairs)


def test_star_sum_equals_independent_two_branch_source_removal():
    factors = toy_factors()
    for source in range(22):
        torch.testing.assert_close(
            R._star_hidden(factors, source),
            R._independent_star_hidden(factors, source),
            rtol=1e-5, atol=2e-5)


def test_star_output_is_down_projection_of_exact_star_sum():
    factors = toy_factors()
    mlp = ToyMLP()
    for source in (0, 7, 21):
        expected = R.r507._linear(
            R._independent_star_hidden(factors, source), mlp.Down.weight.float())
        torch.testing.assert_close(R._star_output(mlp, factors, source), expected,
                                   rtol=2e-5, atol=5e-5)


def test_node_indexing_crosses_actions_and_sources_exactly():
    assert len(R.NODE_NAMES) == 88
    assert R.node_parts(0) == (0, 0)
    assert R.node_parts(21) == (0, 21)
    assert R.node_parts(22) == (1, 0)
    assert R.node_parts(87) == (3, 21)
    assert R.NODE_NAMES[22] == "P::E"


def test_exact_mask_deduplication_keeps_first_identity():
    masks = {
        "a": {"member": torch.tensor([1, 0]), "slice_control": torch.tensor([0, 1])},
        "b": {"member": torch.tensor([1, 0]), "slice_control": torch.tensor([0, 1])},
        "c": {"member": torch.tensor([0, 1]), "slice_control": torch.tensor([1, 0])},
    }
    tags, identity = R.deduplicate_circuit_tags(masks, ("a", "b", "c"))
    assert tags == ("a", "c")
    assert identity["duplicates"] == {"b": "a"}


def _planted_matrices():
    generator = torch.Generator().manual_seed(522)
    circuit = torch.zeros(88, 32, dtype=torch.float64)
    task = torch.zeros(88, 4, dtype=torch.float64)
    circuit[3] = .003 * torch.randn(32, generator=generator)
    task[3] = .003 * torch.randn(4, generator=generator)
    circuit[44] = 1.5 * circuit[3]
    task[44] = 1.5 * task[3]
    return {window: {"circuit": circuit.clone(), "task": task.clone()}
            for window in ("half0", "half1", "pooled")}


def test_all_pairs_detector_recovers_planted_cross_source_relation():
    candidates, summary = R.discover_pairs(_planted_matrices())
    assert summary["unordered_pairs_tested"] == 3828
    assert summary["candidate_count"] == 1
    assert candidates[0]["left_node"] == 3
    assert candidates[0]["right_node"] == 44
    assert candidates[0]["cross_source"]


def test_fixed_controls_destroy_planted_circuit_identity():
    assert R.permutation_control_counts(_planted_matrices()) == [0] * 16


def test_confirmation_uses_discovery_scale_without_refitting():
    matrices = _planted_matrices()
    candidates, _ = R.discover_pairs(matrices)
    candidates[0]["beta_left_from_right"] *= 3
    confirmed, checks = R.confirmation_pairs(matrices, candidates)
    assert confirmed == []
    assert not next(iter(checks.values()))["holds"]


def test_support_requires_every_task_and_circuit_cell_in_both_halves():
    collection = {
        "bounds": (0, 4, 2),
        "task_counts": torch.ones(4, len(R.r507.TASK_CELLS)),
        "circuit_counts": torch.ones(2, 2, 3),
    }
    assert R._support(collection)["holds"]
    collection["circuit_counts"][1, 0, 2] = 0
    assert not R._support(collection)["holds"]


def test_cycle_consistency_rejects_incomplete_or_inconsistent_groups():
    edges = [
        {"left_node": 0, "right_node": 1, "beta_left_from_right": 2.0},
        {"left_node": 0, "right_node": 2, "beta_left_from_right": 4.0},
        {"left_node": 1, "right_node": 2, "beta_left_from_right": 2.0},
    ]
    groups = R.quotient_groups(edges)
    assert len(groups) == 1 and groups[0]["nodes"] == [0, 1, 2]
    edges[-1]["beta_left_from_right"] = .5
    assert R.quotient_groups(edges) == []


def test_eight_planted_tables_recover_and_controls_destroy():
    suite = R.planted_suite()
    assert suite["holds"]
    assert len(suite["cases"]) == 8

