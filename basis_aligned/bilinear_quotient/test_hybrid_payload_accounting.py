import json

from . import hybrid_payload_accounting as hybrid


def test_hybrid_partition_is_scope_complete_and_nonoverlapping():
    result = hybrid.build()
    assert result["scope"]["checkpoint_element_count"] == 545902902
    assert result["scope"]["covered_element_count"] == 545902902
    assert result["scope"]["uncovered_element_count"] == 0
    assert result["scope"]["double_counted_element_count"] == 0
    by_id = {row["charge_id"]: row for row in result["charges"]}
    assert len(by_id) == len(result["charges"]) == 8
    assert by_id["candidate_attention_qk139"]["covered_checkpoint_elements"] \
        == 139*4*128*1152
    assert by_id["retained_attention_qk23"]["covered_checkpoint_elements"] \
        == 23*4*128*1152


def test_hybrid_total_and_claim_boundaries():
    result = hybrid.build()
    accounting = result["accounting"]
    assert accounting["hybrid_learned_constant_payload_bits"] == 12592891853
    assert accounting["identity_checkpoint_tensor_payload_bits"] == 16541356896
    assert accounting["payload_bits_removed"] == 3948465043
    assert 0.76 < accounting["hybrid_to_identity_payload_ratio"] < 0.77
    claims = result["claims"]
    assert claims["scope_comparable_to_identity_tensor_payload"]
    assert claims["complete_learned_constant_payload_accounting"]
    for key in ("identity_replay", "joint_operational_fidelity_certified",
                "decoder_graph_schema_charged", "complete_program_bound",
                "quotient_price", "minimal_description_length"):
        assert not claims[key]


def test_checked_in_artifact_matches_generator():
    assert json.loads(hybrid.OUTPUT.read_text()) == hybrid.build()
