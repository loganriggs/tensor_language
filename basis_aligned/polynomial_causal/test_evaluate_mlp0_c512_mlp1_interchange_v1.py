import json
import subprocess
import sys

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


def test_runner_style_import_resolves_repository_model_module():
    script = evaluator.PC / "evaluate_mlp0_c512_mlp1_interchange_v1.py"
    command = (
        "import runpy; "
        f"ns=runpy.run_path({str(script)!r}, run_name='authority_preflight_import'); "
        "import jacclust.tt_model; "
        "assert ns['ROOT'].is_dir()"
    )
    subprocess.run([sys.executable, "-c", command], cwd=evaluator.BQ, check=True)


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


def test_fineweb_derangement_never_crosses_replication_wave():
    unit_ids = torch.tensor([0, 1, 192, 193])
    cells = torch.zeros((4, 2), dtype=torch.long)
    groups = evaluator.derangement_groups("fineweb", cells, unit_ids)
    flat_units = unit_ids[:, None].expand_as(cells).reshape(-1)
    permutation = evaluator.document_derangement(flat_units, groups.reshape(-1))
    recipient_wave = flat_units >= 192
    donor_wave = flat_units[permutation] >= 192
    assert torch.equal(recipient_wave, donor_wave)
    assert not bool((flat_units == flat_units[permutation]).any())


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
    assert contrasts["interaction"][0] is logits["CC"]
    expected_additive = evaluator.additive_interaction_prediction(logits)
    torch.testing.assert_close(contrasts["interaction"][1], expected_additive)
    target = torch.zeros((1, 1), dtype=torch.long)
    effect = evaluator.pair_effects(logits["OO"], logits["OC"], target, logit_scale=1.0)
    assert float(effect["ce_abs"]) < 0  # candidate improves CE; sign must not be discarded
