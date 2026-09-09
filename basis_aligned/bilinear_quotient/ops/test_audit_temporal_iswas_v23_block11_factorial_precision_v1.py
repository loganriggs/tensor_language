import copy
import json

import audit_temporal_iswas_v23_block11_factorial_precision_v1 as audit


def result():
    return json.loads(audit.RESULT.read_text())


def test_real_invalid_factorial_receives_only_scoped_successor_license():
    payload = audit.audit_payload(result())
    assert all(payload["predictions"].values())
    assert payload["original_terminal"] == "invalid_instrument"
    assert payload["original_factorial_relabelled"] is False
    assert payload["downstream_reader_license"] is True
    assert payload["terminal"] == "licensed_downstream_reader_only"


def test_final_hidden_failure_is_not_erased():
    payload = audit.audit_payload(result())
    assert payload["joint_restore"]["final_hidden_max_abs"] == .03125
    assert payload["joint_restore"]["final_hidden_max_abs"] > payload["bars"]["interface_restore"]


def test_block_output_or_logit_failure_revokes_license():
    for field in ("block11_output_max_abs", "logits_max_abs"):
        changed = copy.deepcopy(result())
        changed["joint_restore"][field] = .001
        payload = audit.audit_payload(changed)
        assert not payload["predictions"][
            "pred_c_block_output_logits_and_behavior_close"]
        assert not payload["downstream_reader_license"]
