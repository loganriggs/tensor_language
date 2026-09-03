from __future__ import annotations

import rung528_leave_one_action_out_consensus_diagnosis as diagnosis


def test_diagnosis_is_descriptive_and_uses_all_leave_one_out_targets():
    report = diagnosis.run()
    assert report["claim_level"].endswith("not_physical_state_evidence")
    assert set(report["leave_one_action_out_consensus"]) == {"N", "P", "Z7", "Z8"}
    assert report["leave_one_action_out_consensus"]["Z7"]["D0"]["relative_residual"] < .35
    assert report["leave_one_action_out_consensus"]["Z7"]["D1"]["relative_residual"] < .35
    assert report["physical_consensus_insertion_opened"] is False
