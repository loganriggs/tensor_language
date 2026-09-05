#!/usr/bin/env python3
# BQLANE: cpu

from datetime import datetime, timedelta, timezone
import torch

import circuit_fast_screen_candidate_task14_test_cross_syntax as candidate
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_task14_test_transfer_removal as runner


class PassingBackend:
    def native(self, batch, *, capture):
        tag = 1.0 if batch.side == "base" else 2.0
        cache = {(row_id, runner.HEAD_SITE_ID): torch.full((128,), tag)
                 for row_id in batch.row_ids}
        return producer.BatchOutput(tuple((2.0, 0.0) for _ in batch.row_ids), cache)

    def patched(self, batch, *, site, donor_cache):
        pairs = []
        for row_id in batch.row_ids:
            tag = float(donor_cache[(row_id, site.site_id)][0])
            pairs.append({0.0: (1.0, 0.0), 1.0: (2.0, 0.0), 2.0: (0.0, 0.0)}[tag])
        return producer.BatchOutput(tuple(pairs), {})


def test_fake_TEST_transfer_removal_and_replay_pass() -> None:
    now = datetime(2026, 9, 5, 5, 45, tzinfo=timezone.utc)
    wall = iter((now, now + timedelta(seconds=1))); mono = iter((10.0, 11.0))
    result = runner.run_science(backend=PassingBackend(), wall_clock=lambda: next(wall),
                                monotonic_clock=lambda: next(mono))
    assert result["terminal"] == "screen"
    assert all(result["predictions"].values())
    assert result["active_price"] == result["maximum_price"]
    assert all(cell["mean_recovery"] == 0.5 for cell in result["transfer"]["cells"])
    assert all(cell["median_normalized_damage"] == 0.5 for cell in result["removal"]["cells"])
    assert result["replay_max_abs_logit_error"] == 0.0


def test_dry_run_is_model_free(capsys) -> None:
    runner.main(["--dry-run"])
    assert candidate.compile_plan()["compiled_sha256"] in capsys.readouterr().out


def test_protocol_can_run_precompiled_ood_candidate() -> None:
    import circuit_fast_screen_candidate_task14_ood_cross_syntax as ood
    protocol = runner.RunProtocol(
        candidate=ood, request_id="fake", experiment_id="fake",
        result_relative=runner.Path("fake.json"), prior_art_sha256="a" * 64,
        result_schema="fake_ood", novelty="fake", limits="fake",
    )
    now = datetime(2026, 9, 5, 5, 50, tzinfo=timezone.utc)
    wall = iter((now, now + timedelta(seconds=1))); mono = iter((20.0, 21.0))
    result = runner.run_science(protocol=protocol, backend=PassingBackend(),
                                wall_clock=lambda: next(wall), monotonic_clock=lambda: next(mono))
    assert result["terminal"] == "screen"
    assert result["phase"] == "OOD"
    assert result["plan_sha256"] == ood.compile_plan()["compiled_sha256"]
