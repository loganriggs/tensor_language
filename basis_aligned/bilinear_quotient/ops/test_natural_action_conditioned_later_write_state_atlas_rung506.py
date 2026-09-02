import sys
from pathlib import Path

import torch


OPS = Path(__file__).resolve().parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import natural_action_conditioned_later_write_state_atlas_rung506 as rung


def _synthetic_singletons():
    patch_sets = ((),) + tuple((site,) for site in rung.SITES)
    data = rung._empty_collection((0, 4, 2), ("r.0", "r.2", "r.4"), patch_sets, True)
    data["task_counts"].fill_(1)
    data["circuit_counts"].fill_(1)
    fingerprint = torch.tensor([0.002, -0.001, 0.003], dtype=torch.float64)
    task = torch.tensor([0.004, 0.003, 0.005, 0.002], dtype=torch.float64)
    for source_index, _source in enumerate(rung.SOURCES):
        for site in ("m8", "a9"):
            arm = rung._arm_index(data, site)
            data["source_circuit_sums"][source_index, arm, :, 0] = fingerprint
            data["source_task"][source_index, arm, :, rung.CELLS.index("all_positive")] = 0.003
            data["source_task"][source_index, arm, :, rung.CELLS.index("off_target")] = 0.0002
            for cell_index, value in zip(rung.TASK_CONTEXT_INDICES, task):
                data["source_task"][source_index, arm, :, cell_index] = value
    return data


def _synthetic_additive_pairs(singletons):
    name = rung.edge_name("m8", "a9")
    pairs = rung._empty_collection((0, 4, 2), singletons["tags"], (("m8", "a9"),), False)
    pairs["task_counts"].copy_(singletons["task_counts"])
    pairs["circuit_counts"].copy_(singletons["circuit_counts"])
    for source_index, source in enumerate(rung.SOURCES):
        pair_index = rung._arm_index(pairs, name)
        left_index = rung._arm_index(singletons, "m8")
        right_index = rung._arm_index(singletons, "a9")
        pairs["source_circuit_sums"][source_index, pair_index] = (
            singletons["source_circuit_sums"][source_index, left_index]
            + singletons["source_circuit_sums"][source_index, right_index]
        )
        pairs["source_task"][source_index, pair_index] = (
            singletons["source_task"][source_index, left_index]
            + singletons["source_task"][source_index, right_index]
        )
    return name, pairs


def test_edge_names_are_ordered_and_round_trip():
    name = rung.edge_name("m8", "a9")
    assert name == "m8+a9"
    assert rung.parse_edge(name) == ("m8", "a9")


def test_discovery_retains_all_and_only_qualifying_edges():
    data = _synthetic_singletons()
    eligible, edges, checks = rung.discover_edges(data)
    assert eligible == ["m8", "a9"]
    assert edges == ["m8+a9"]
    assert checks["pairs"]["m8+a9"]["holds"] is True


def test_additive_composition_rule_is_recovered_exactly():
    singletons = _synthetic_singletons()
    name, pairs = _synthetic_additive_pairs(singletons)
    rule = rung.fit_composition_rule(singletons, pairs, name)
    assert rule["identified"] is True
    assert rule["kind"] == "additive"
    assert rule["interaction_over_joint"] < 1e-12


def test_confirmation_validation_and_selective_composition_pass_exact_fixture():
    singletons = _synthetic_singletons()
    name, pairs = _synthetic_additive_pairs(singletons)
    confirmed, confirmation = rung.confirm_edges(singletons, [name])
    validated, validation = rung.validate_edges(singletons, [name])
    rule = rung.fit_composition_rule(singletons, pairs, name)
    composition = rung.score_composition(singletons, pairs, name, rule)
    assert confirmed == [name] and confirmation[name]["holds"] is True
    assert validated == [name] and validation[name]["holds"] is True
    assert composition["holds"] is True


def test_calibration_uses_signed_native_recovery_and_document_cosine():
    data = _synthetic_singletons()
    all_index = rung.CELLS.index("all_positive")
    off_index = rung.CELLS.index("off_target")
    data["base_task"][:, all_index] = 1.0
    intact = rung._arm_index(data, "intact")
    for source_index, source in enumerate(rung.SOURCES):
        data["source_task"][source_index, intact, :, all_index] = 0.0 if source == "N" else -0.1
        data["source_task"][source_index, intact, :, off_index] = 0.0 if source == "N" else 0.001
    reports = rung.calibration(data)
    assert rung.calibration_holds(reports) is True
    assert reports["pooled"]["P"]["recovery_vs_native"] == 1.1


def test_one_scalar_prediction_uses_the_frozen_beta():
    left = torch.tensor([1.0, 2.0])
    right = torch.tensor([3.0, 4.0])
    predicted = rung.predict_joint(
        {"kind": "one_scalar_interaction", "beta": 0.5}, left, right)
    assert torch.equal(predicted, torch.tensor([6.0, 9.0]))


def test_registered_maximum_price_arithmetic():
    assert 20729 + 496 * rung.MAX_EDGES + 500 * rung.MAX_EDGES == 28697
