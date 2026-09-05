#!/usr/bin/env python3
# BQLANE: cpu

from datetime import datetime, timedelta, timezone

import pytest

import circuit_fast_screen_candidate_task14_select_cross_syntax as candidate
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_task14_cross_syntax as shared
import run_circuit_fast_screen_task14_select_cross_syntax as select_run


class PassingBackend:
    def __init__(self) -> None:
        self.native_calls = 0
        self.patched_calls = 0

    def native(self, batch, *, capture):
        self.native_calls += 1
        captured = ({
            (row_id, site_id): (site_id, row_id)
            for row_id in batch.row_ids for site_id in candidate.SITE_IDS
        } if capture else {})
        return producer.BatchOutput(tuple((3.0, 1.0) for _ in batch.row_ids), captured)

    def patched(self, batch, *, site, donor_cache):
        self.patched_calls += 1
        assert all((row_id, site.site_id) in donor_cache for row_id in batch.row_ids)
        return producer.BatchOutput(tuple((0.6, 1.0) for _ in batch.row_ids), {})


def test_select_protocol_scores_held_out_rows_without_gpu() -> None:
    backend = PassingBackend()
    now = datetime(2026, 9, 5, 5, 0, tzinfo=timezone.utc)
    wall = iter((now, now + timedelta(seconds=1)))
    monotonic = iter((10.0, 11.0))
    result = shared.run_science(
        protocol=select_run.PROTOCOL, backend=backend,
        wall_clock=lambda: next(wall), monotonic_clock=lambda: next(monotonic),
    )
    assert backend.native_calls == 4
    assert backend.patched_calls == 4
    assert result["terminal"] == "screen"
    assert result["phase"] == "SELECT"
    assert result["partition"] == "HELD_OUT"
    assert result["validation_scope"] == candidate.VALIDATION_SCOPE
    assert result["active_price"] == result["maximum_price"]
    assert result["checkpoint"] == {
        "weights_sha256": select_run.PROTOCOL.checkpoint_sha256,
        "config_sha256": select_run.PROTOCOL.config_sha256,
        "verified_before_model_load": False,
    }
    assert result["predictions"] == {
        "pred_a_native_capability": True,
        "pred_b_attention11_cross_syntax": True,
        "pred_c_head11_3_cross_syntax": True,
    }
    assert all(item["overall_mean_recovery"] == pytest.approx(0.6)
               for item in result["site_results"])


def test_protocol_refuses_candidate_phase_mismatch() -> None:
    changed = shared.TargetedCrossSyntaxProtocol(
        **{**select_run.PROTOCOL.__dict__, "phase": "TEST"}
    )
    with pytest.raises(shared.CrossSyntaxRunError, match="candidate PHASE"):
        shared.run_science(protocol=changed, backend=PassingBackend())
