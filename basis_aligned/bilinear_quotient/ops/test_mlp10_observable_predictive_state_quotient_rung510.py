import torch

import mlp10_observable_predictive_state_quotient_rung510 as rung


def _planted_matrices(scale=-2.0):
    generator = torch.Generator(device="cpu").manual_seed(510)
    circuit = torch.randn(rung.N_NODES, 32, generator=generator, dtype=torch.float64) * .002
    task = torch.randn(rung.N_NODES, 4, generator=generator, dtype=torch.float64) * .002
    circuit[1] = scale * circuit[0]
    task[1] = scale * task[0]
    return {
        window: {"circuit": circuit.clone(), "task": task.clone()}
        for window in ("half0", "half1", "pooled")}


def test_node_vocabulary_and_registered_price():
    assert rung.N_NODES == 1012
    assert rung.N_NODES * (rung.N_NODES - 1) // 2 == 511566
    assert rung.node_parts(0) == (0, 0)
    assert rung.node_parts(rung.N_TERMS + 1) == (1, 1)
    assert 2 * 63116 + 372 + 124 * 16 == 128588


def test_all_pairs_detector_recovers_signed_scaled_equivalence():
    pairs, summary = rung.discover_pairs(_planted_matrices())
    planted = [pair for pair in pairs
               if pair["left_node"] == 0 and pair["right_node"] == 1]
    assert len(planted) == 1
    assert abs(planted[0]["beta_left_from_right"] + .5) < 1e-10
    assert planted[0]["holds"] is True
    assert summary["unordered_pairs_tested"] == 511566


def test_detector_does_not_rank_or_truncate_overlarge_relation():
    matrices = _planted_matrices()
    for node in range(1, 7):
        matrices["half0"]["circuit"][node] = matrices["half0"]["circuit"][0]
        matrices["half1"]["circuit"][node] = matrices["half1"]["circuit"][0]
        matrices["pooled"]["circuit"][node] = matrices["pooled"]["circuit"][0]
        matrices["half0"]["task"][node] = matrices["half0"]["task"][0]
        matrices["half1"]["task"][node] = matrices["half1"]["task"][0]
        matrices["pooled"]["task"][node] = matrices["pooled"]["task"][0]
    pairs, summary = rung.discover_pairs(matrices)
    clique = [pair for pair in pairs if pair["left_node"] < 7 and pair["right_node"] < 7]
    assert len(clique) == 21
    assert summary["candidate_count"] >= 21
    assert summary["small_relation"] is False


def test_confirmation_uses_frozen_beta_instead_of_refitting():
    discovery_pairs, _ = rung.discover_pairs(_planted_matrices())
    planted = next(pair for pair in discovery_pairs
                   if pair["left_node"] == 0 and pair["right_node"] == 1)
    confirmation = _planted_matrices(scale=-1.0)
    passing, checks = rung.confirmation_pairs(confirmation, [planted])
    key = f"{planted['left_name']} <-> {planted['right_name']}"
    assert passing == []
    assert checks[key]["holds"] is False


def test_quotient_groups_require_complete_scale_consistent_graph():
    names = rung.NODE_NAMES
    pair01 = {"left_node": 0, "right_node": 1,
              "left_name": names[0], "right_name": names[1],
              "beta_left_from_right": .5}
    pair02 = {"left_node": 0, "right_node": 2,
              "left_name": names[0], "right_name": names[2],
              "beta_left_from_right": .25}
    assert rung.quotient_groups([pair01, pair02]) == []
    pair12 = {"left_node": 1, "right_node": 2,
              "left_name": names[1], "right_name": names[2],
              "beta_left_from_right": .5}
    groups = rung.quotient_groups([pair01, pair02, pair12])
    assert len(groups) == 1
    assert groups[0]["nodes"] == [0, 1, 2]
    assert groups[0]["cycle_consistent"] is True


def test_substituted_response_uses_target_background_intact_baseline():
    cells = len(rung.r509.parent.TASK_CELLS)
    tags = 3
    docs = 4
    exact = {
        "arms": ("intact",) + tuple(rung.r509.parent.PAIR_NAMES),
        "task": torch.zeros(4, 254, docs, cells, dtype=torch.float64),
        "circuit_sums": torch.zeros(4, 254, 2, 2, tags, dtype=torch.float64),
    }
    substitutions = {
        "bounds": (0, 4, 2),
        "directions": [{"target": 0}],
        "task": torch.ones(1, docs, cells, dtype=torch.float64),
        "task_counts": torch.ones(docs, cells, dtype=torch.float64),
        "circuit_sums": torch.ones(1, 2, 2, tags, dtype=torch.float64),
        "circuit_counts": torch.ones(2, 2, tags, dtype=torch.float64),
    }
    response = rung._substituted_response(substitutions, exact, 0, "pooled")
    torch.testing.assert_close(response["task"], torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(response["circuit"], torch.zeros(tags, dtype=torch.float64))


def test_frozen_rung509_failure_route_and_circuit_partition_validate():
    _rows, _tasks, _circuits, _scales, discovery, confirmation, metadata = \
        rung.validate_inputs()
    assert len(discovery) == 32
    assert len(confirmation) == 30
    assert metadata["rung509_model_outcome_absent"] is True
