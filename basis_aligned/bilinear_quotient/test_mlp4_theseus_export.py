import json

from . import mlp4_theseus_export as export


def test_unscored_export_preserves_price_claims_and_closed_lanes(tmp_path):
    result = export.build(tmp_path/"absent.json")
    assert len(result["candidates"]) == 18
    assert result["coverage"] == {lane: 0 for lane in export.LANES}
    by_family = {}
    for row in result["candidates"]:
        by_family.setdefault(row["family"], []).append(row)
        assert row["declared_inputs"] == ["blocks.4.mlp.rmsnorm_input"]
        assert not row["frontier_eligible"]
        assert all(lane["score"] is None for lane in row["operational_lanes"].values())
    assert all(row["price"]["eligible_for_unconditional_mdl"]
               for row in by_family["linear"])
    assert all(not row["price"]["eligible_for_unconditional_mdl"]
               for family in ("native_product", "seeded_random_product")
               for row in by_family[family])
    assert all(row["structural_tensor"]["status"] == "not_applicable_affine"
               for row in by_family["linear"])
    audited = [row for family in ("native_product", "seeded_random_product")
               for row in by_family[family]
               if row["structural_tensor"]["status"] ==
               "measured_factorization_invariant"]
    assert len(audited) == 10
    assert len(result["tensor_pair_comparisons"]) == 5
    assert all(pair["energy_majorization"]["relation"] ==
               "incomparable_crossing"
               for pair in result["tensor_pair_comparisons"])
    assert all(row["structural_tensor"]["not_behavioral_evidence"]
               and row["structural_tensor"]["not_description_length"]
               for row in audited)
    alternatives = [alternative for row in by_family["native_product"]
                    for alternative in row["alternative_serializations"]]
    assert len(alternatives) == 5
    assert all(not alternative["behavioral_score_inherited"]
               and not alternative["frontier_eligible"]
               and alternative["relative_coefficient_tensor_frobenius_error"] < 1e-3
               for alternative in alternatives)


def test_complete_fake_results_open_only_heldout(tmp_path):
    inventory = json.loads(export.INVENTORY.read_text())
    protocol = json.loads(export.PROTOCOL.read_text())
    points = [{"candidate_id": row["candidate_id"],
               "program_hash": row["canonical_bytes_hash"],
               "ce": 2.0, "delta_ce": .1, "fidelity": .75}
              for row in inventory["candidates"]]
    path = tmp_path/"complete.json"
    path.write_text(json.dumps({"partial": False,
                                "protocol_id": protocol["protocol_id"],
                                "points": points}))
    result = export.build(path)
    assert result["coverage"]["held_out"] == 18
    assert all(result["coverage"][lane] == 0 for lane in export.LANES[1:])
    assert all(not row["frontier_eligible"] for row in result["candidates"])


def test_partial_or_hash_mismatched_results_are_rejected(tmp_path):
    path = tmp_path/"bad.json"
    path.write_text(json.dumps({"partial": True}))
    try:
        export.build(path)
    except ValueError as error:
        assert "partial" in str(error)
    else:
        raise AssertionError("partial results were accepted")
