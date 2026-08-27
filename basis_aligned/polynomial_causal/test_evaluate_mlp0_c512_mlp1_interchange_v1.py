import json

import pytest
import torch

import evaluate_mlp0_c512_mlp1_interchange_v1 as evaluator


def test_expected_call_counts_are_mechanically_frozen():
    assert evaluator.expected_call_counts(960, {"fineweb": 1170, "code": 192}) == {
        "candidate_original_down_calls": 0,
        "poison_canary_calls": 1,
        "mlp1_teacher_calls": 2968,
        "c512_proxy_calls": 1364,
    }


def test_wave_coverage_uses_source_unit_identity_not_row_boundary():
    # Wave A has noncontiguous rows and more rows than wave B.
    unit_ids = torch.tensor([0, 2, 0, 3, 1])
    valid = torch.tensor([[1, 1], [0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.bool)
    coverage = evaluator.coverage_by_unit_partition(valid, unit_ids, split_unit=2)
    assert coverage == pytest.approx({"wave_A": 4 / 6, "wave_B": 2 / 4, "pooled": 6 / 10})


def test_registered_unit_identity_has_exact_occupancy_and_hashes():
    receipt = json.loads(evaluator.ROW_RECEIPT.read_text())
    artifact = torch.load(evaluator.CODE_REGISTER, map_location="cpu", weights_only=False)
    identity = evaluator.build_unit_identity(receipt, artifact["manifest"])
    assert len(identity["fineweb"]["row_to_unit"]) == 1170
    assert len(identity["code"]["row_to_unit"]) == 192
    fine_occupancy = torch.bincount(torch.tensor(identity["fineweb"]["row_to_unit"]), minlength=384)
    code_occupancy = torch.bincount(torch.tensor(identity["code"]["row_to_unit"]), minlength=48)
    assert set(fine_occupancy.tolist()).issubset({2, 4, 6})
    assert torch.equal(code_occupancy, torch.full((48,), 4))
    hashes = evaluator.unit_identity_hashes(identity)
    assert set(hashes) == {"fineweb", "code"}
    assert all(len(value) == 64 for domain in hashes.values() for value in domain.values())


def test_contrast_orientation_and_signed_ce_effect():
    logits = {
        "OO": torch.tensor([[[0.0, 0.0]]]),
        "OC": torch.tensor([[[1.0, 0.0]]]),
        "CO": torch.tensor([[[0.0, 1.0]]]),
        "CC": torch.tensor([[[2.0, 1.0]]]),
        "shuffle": torch.tensor([[[0.0, 2.0]]]),
        "native_write": torch.tensor([[[2.0, 0.0]]]),
    }
    contrasts = evaluator.contrast_logits(logits)
    assert contrasts["write_on_O"] == (logits["OO"], logits["OC"])
    assert contrasts["write_on_C"] == (logits["CO"], logits["CC"])
    assert contrasts["upstream_state"] == (logits["OO"], logits["CO"])
    target = torch.zeros((1, 1), dtype=torch.long)
    effect = evaluator.pair_effects(logits["OO"], logits["OC"], target, logit_scale=1.0)
    assert float(effect["ce_abs"]) < 0  # candidate improves CE; sign must not be discarded
