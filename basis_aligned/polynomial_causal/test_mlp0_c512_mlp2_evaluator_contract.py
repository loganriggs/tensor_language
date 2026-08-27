import copy
import json

import pytest
import torch

import mlp0_c512_mlp2_evaluator_contract as contract


def prior_receipt():
    path = contract.__file__.replace(
        "polynomial_causal/mlp0_c512_mlp2_evaluator_contract.py",
        "bilinear_quotient/mlp0_c512_mlp1_interchange_v1_rows_receipt.json",
    )
    return json.load(open(path))


def test_unit_identity_rebuilds_registered_document_waves():
    identity = contract.build_unit_identity(prior_receipt())
    assert len(identity["ordered_ids"]) == 384
    assert identity["wave_labels"] == ["A"] * 192 + ["B"] * 192
    assert len(identity["row_to_unit"]) == 1170
    occupancy = torch.bincount(torch.tensor(identity["row_to_unit"]), minlength=384)
    assert set(occupancy.tolist()).issubset({2, 4, 6})
    assert set(contract.unit_identity_hashes(identity)) == {
        "ordered_ids_sha256", "row_to_unit_sha256", "occupancy_sha256",
        "wave_labels_sha256",
    }


def test_document_cannot_cross_waves():
    receipt = copy.deepcopy(prior_receipt())
    records = receipt["document_provenance"]["sets"]["eval"]
    duplicate = next(record for record in records if record["source_document_ordinal"] == 0)
    duplicate["wave"] = "B"
    with pytest.raises(RuntimeError, match="crosses replication waves"):
        contract.build_unit_identity(receipt)


def test_expected_call_contract_is_phase_and_site_complete():
    result = contract.expected_call_contract(1170)
    # ceil(1170 / 4) = 293 evaluation batches.
    assert result["exact_call_counts"] == {
        "candidate_original_down_calls": 0,
        "poison_canary_calls": 1,
        "c512_proxy_calls": 1172,
    }
    assert result["n_eval_windows"] == 1170
    phases = result["exact_phase_site_call_counts"]
    assert phases["mlp1_teacher_capture"] == {"1": 1172}
    assert phases["mlp2_teacher_capture"] == {"2": 1172}
    assert phases["parent_replay_mlp_sites"]["1"] == 1172
    assert phases["parent_replay_mlp_sites"]["2"] == 586
    assert set(phases["crossed_suffix_replay"].values()) == {2344}


def test_contrast_orientations_are_literal():
    logits = {arm: torch.randn(1, 2, 3) for arm in contract.CONTRAST_ORIENTATIONS.values()
              for arm in arm if arm != "additive_CO_plus_OC_minus_OO"}
    # The comprehension above obtains the exact raw eight-arm set.
    output = contract.contrast_logits(logits)
    assert output["observational"] == (logits["OO"], logits["CC"])
    assert output["write_on_candidate_state"] == (logits["CO"], logits["CC"])
    assert output["interaction"][0] is logits["CC"]


def test_coverage_uses_wave_labels_not_row_boundary():
    identity = contract.build_unit_identity(prior_receipt())
    unit_ids = torch.tensor([0, 192, 1, 193])
    valid = torch.tensor([[1, 1], [0, 0], [1, 0], [1, 1]], dtype=torch.bool)
    result = contract.coverage_by_wave(valid, unit_ids, identity["wave_labels"])
    assert result == pytest.approx({"wave_A": .75, "wave_B": .5, "pooled": .625})
    with pytest.raises(ValueError, match="out-of-range"):
        contract.coverage_by_wave(valid, torch.tensor([0, 192, 1, 384]),
                                  identity["wave_labels"])


def test_control_receipt_and_derangement_invariants():
    recipient = torch.tensor([0, 1, 192, 193])
    donor = torch.tensor([1, 0, 193, 192])
    groups = torch.tensor([3, 3, 19, 19])
    permutation = torch.tensor([1, 0, 3, 2])
    checks = contract.verify_derangement(
        permutation, recipient, donor, groups, groups.clone()
    )
    assert all(checks.values())
    digest = contract.control_realization_sha256(
        permutation, recipient, donor, groups
    )
    assert len(digest) == 64
    assert len(contract.control_contract_sha256()) == 64
    broken = contract.verify_derangement(
        torch.tensor([0, 0, 2, 3]), recipient, recipient, groups, groups
    )
    assert not broken["derangement_bijection"]
    assert not broken["donor_arrays_indexed_by_permutation"]
    assert not broken["derangement_no_same_document"]


def test_arm_carried_state_provenance_is_frozen():
    assert contract.ARM_CARRIED_PATH == {
        "OO": "O", "OC": "O", "O0": "O", "ON": "O",
        "CC": "C", "CO": "C", "C0": "C", "CS": "C",
    }
    interfaces = {
        path: {"v1": torch.tensor([index]), "x0": torch.tensor([index + 2])}
        for index, path in enumerate(("O", "C"))
    }
    for arm, path in contract.ARM_CARRIED_PATH.items():
        v1, x0 = contract.carried_inputs_for_arm(arm, interfaces)
        assert v1 is interfaces[path]["v1"]
        assert x0 is interfaces[path]["x0"]


def test_derangement_rejects_float_permutation_before_coercion():
    values = torch.arange(4)
    with pytest.raises(ValueError, match="integer dtype"):
        contract.verify_derangement(
            torch.tensor([1.9, 0.1, 3.0, 2.0]), values, values,
            values, values,
        )
