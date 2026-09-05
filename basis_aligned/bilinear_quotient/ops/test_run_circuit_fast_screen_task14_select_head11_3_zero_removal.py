#!/usr/bin/env python3
# BQLANE: cpu

from datetime import datetime, timedelta, timezone

import torch

import circuit_fast_screen_candidate_task14_select_head11_3_zero_removal as candidate
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_task14_select_head11_3_zero_removal as runner


class PassingBackend:
    def __init__(self, rows):
        self.family = {row["row_id"]: row["family"] for row in rows}
        self.native_calls = self.patched_calls = 0

    def native(self, batch, *, capture):
        self.native_calls += 1
        captured = {}
        for row_id in batch.row_ids:
            captured[(row_id, candidate.HEAD_SITE_ID)] = torch.ones(128)
            captured[(row_id, candidate.ATTENTION_SITE_ID)] = torch.ones(1152)
        return producer.BatchOutput(tuple((2.0, 0.0) for _ in batch.row_ids), captured)

    def patched(self, batch, *, site, donor_cache):
        self.patched_calls += 1
        pairs = []
        for row_id in batch.row_ids:
            replacement = donor_cache[(row_id, site.site_id)]
            is_zero = float(replacement.abs().max()) == 0.0
            if not is_zero:
                pairs.append((2.0, 0.0))
            elif self.family[row_id] in {"P", "C"}:
                pairs.append((2.0, 0.0))
            elif site.site_id == candidate.HEAD_SITE_ID:
                pairs.append((1.0, 0.0))
            else:
                pairs.append((0.0, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def test_fake_execution_passes_literal_removal_without_gpu() -> None:
    rows = candidate.build_rows()
    backend = PassingBackend(rows)
    now = datetime(2026, 9, 5, 5, 20, tzinfo=timezone.utc)
    wall = iter((now, now + timedelta(seconds=2)))
    mono = iter((10.0, 12.0))
    result = runner.run_science(
        backend=backend, wall_clock=lambda: next(wall),
        monotonic_clock=lambda: next(mono),
    )
    assert result["terminal"] == "screen"
    assert all(result["predictions"].values())
    assert backend.native_calls == 4
    assert backend.patched_calls == 12
    assert result["active_price"] == result["maximum_price"]
    assert result["replay_max_abs_logit_error"] == 0.0
    assert result["control_mean_absolute_head_damage"] == {"P": 0.0, "C": 0.0}
    targets = [cell for cell in result["cells"] if cell["family"] in {"A1", "A2"}]
    assert all(cell["mean_head_damage"] == 0.5 for cell in targets)
    assert all(cell["mean_attention_damage"] == 1.0 for cell in targets)


def test_dry_run_is_model_free_and_exact(capsys) -> None:
    runner.main(["--dry-run"])
    output = capsys.readouterr().out
    assert candidate.compile_plan()["compiled_sha256"] in output
