#!/usr/bin/env python3
# BQLANE: cpu

import run_circuit_fast_screen_task14_select_cross_noun as runner


def test_protocol_is_a_targeted_cross_noun_select_profile() -> None:
    protocol = runner.PROTOCOL
    assert protocol.phase == "SELECT"
    assert protocol.partition == "HELD_OUT"
    assert "cross_noun" in protocol.validation_scope
    assert protocol.expected_authority_sha256 == \
        "9d5151f9e297788c0c8799cc60cc4c9bf1e6196e10df93793fb53094566091ae"
    assert protocol.candidate.compile_plan()["price"]["forward_calls"] == 8


def test_predictions_are_exposed_to_managed_preflight() -> None:
    assert runner.REGISTERED_PREDICTIONS == (
        "pred_a_native_capability",
        "pred_b_attention11_cross_syntax",
        "pred_c_head11_3_cross_syntax",
    )
