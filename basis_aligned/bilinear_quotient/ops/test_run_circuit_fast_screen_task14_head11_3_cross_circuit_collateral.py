#!/usr/bin/env python3
# BQLANE: cpu

from datetime import datetime, timedelta, timezone

import torch

import circuit_fast_screen_candidate_task14_head11_3_cross_circuit_collateral as candidate
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_task14_head11_3_cross_circuit_collateral as runner


class FakeBackend:
    def __init__(self, damage):
        self.damage = damage

    def native(self, batch, *, capture):
        captured = {(row_id, candidate.SITE_ID): torch.ones(128) for row_id in batch.row_ids}
        return producer.BatchOutput(tuple((2.0, 0.0) for _ in batch.row_ids), captured)

    def patched(self, batch, *, site, donor_cache):
        pairs = []
        for row_id in batch.row_ids:
            replacement = donor_cache[(row_id, site.site_id)]
            pairs.append((2.0 - self.damage, 0.0) if float(replacement.norm()) == 0 else (2.0, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def _run(damage):
    now = datetime(2026, 9, 5, 5, 35, tzinfo=timezone.utc)
    wall = iter((now, now + timedelta(seconds=1)))
    mono = iter((10.0, 11.0))
    return runner.run_science(backend=FakeBackend(damage),
                              wall_clock=lambda: next(wall), monotonic_clock=lambda: next(mono))


def test_small_collateral_passes() -> None:
    result = _run(0.1)
    assert result["terminal"] == "screen"
    assert all(result["predictions"].values())
    assert result["active_price"] == result["maximum_price"]
    assert result["minimum_native_head_norm"] > 0


def test_large_collateral_is_honest_null() -> None:
    result = _run(1.0)
    assert result["terminal"] == "null"
    assert result["reason"] == "head11_3_removal_has_cross_circuit_collateral"


def test_dry_run_is_model_free(capsys) -> None:
    runner.main(["--dry-run"])
    assert candidate.compile_plan()["compiled_sha256"] in capsys.readouterr().out
