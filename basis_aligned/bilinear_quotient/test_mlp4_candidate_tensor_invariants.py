import json

import torch

from . import mlp4_bilinear_residual_codec as native_codec
from . import mlp4_candidate_tensor_invariants as audit


def test_frozen_audit_binds_roster_hashes_and_rank_invariants():
    result = json.loads(audit.OUTPUT.read_text())
    inventory = json.loads(audit.INVENTORY.read_text())
    assert result["candidate_bytes_sha256"] == audit.sha(audit.BYTES)
    assert result["inventory_sha256"] == audit.sha(audit.INVENTORY)
    expected = []
    for pair in inventory["native_random_actual_bit_pairings"]:
        expected.extend((pair["native_candidate_id"], pair["random_candidate_id"]))
    assert [row["candidate_id"] for row in result["rows"]] == expected
    assert len(result["pair_comparisons"]) == 5
    for row in result["rows"]:
        assert row["rank"] == row["components"]
        assert 1 <= row["stable_rank"] <= row["rank"]
        assert row["stable_rank"] <= row["entropy_rank"] <= row["rank"]
        assert 0 < row["largest_mode_energy_fraction"] <= \
            row["top8_mode_energy_fraction"] <= \
            row["top32_mode_energy_fraction"] <= 1+1e-12
        assert row["serialized_bits"] > 0 and row["stable_rank_per_mbit"] > 0


def test_smallest_production_spectrum_recomputes_from_frozen_bytes():
    result = json.loads(audit.OUTPUT.read_text())
    artifact = torch.load(audit.BYTES, map_location="cpu", weights_only=False)
    expected = next(row for row in result["rows"]
                    if row["candidate_id"] == "native_k32")
    decoded = native_codec.decode(artifact["encoded"]["native_k32"])
    actual = audit.invariants.output_mode_spectrum(
        decoded["A"], decoded["B"], decoded["C"])
    assert actual["rank"] == expected["rank"]
    assert abs(actual["stable_rank"]-expected["stable_rank"]) < 1e-10
    assert abs(actual["entropy_rank"]-expected["entropy_rank"]) < 1e-10
