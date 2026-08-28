#!/usr/bin/env python3
"""Static verifier for the preregistered MLP4 z4 validation boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "mlp4_z4_validation_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate(path=PATH):
    p = json.loads(path.read_text())
    assert p["status"] == "preregistered_gpu_launch_requires_claim_and_driver_check"
    for group in ("pinned_artifacts", "pinned_sources"):
        for filename, expected in p[group].items():
            assert sha(HERE/filename) == expected, filename
    assert len(p["candidate_order"]) == 18 and len(set(p["candidate_order"])) == 18
    assert p["controls"] == ["retained_live_mlp4", "fit_mean_mlp4_output"]
    assert p["metrics"]["all_candidates_evaluated_once"]
    assert p["metrics"]["no_early_stopping"] and p["metrics"]["no_validation_refit_or_selection"]
    assert not p["data"]["fit_rows_may_be_opened"]
    assert not p["data"]["combined_rows_may_be_opened"]
    assert not p["data"]["ood_may_be_opened"]
    assert p["resources"]["batch_size"] == 4
    assert p["resources"]["hard_abort_peak_gib"] <= 10
    assert p["resources"]["hard_abort_temperature_c"] <= 82
    runner = (HERE/"mlp4_z4_validation.py").read_text()
    assert p["data"]["file"] in runner
    for forbidden in ("mlp4_frontier_fit_rows.pt", "mlp4_frontier_rows.pt",
                      "mlp4_z4_fit_artifact.pt"):
        assert forbidden not in runner
    assert "block.mlp(z)" in runner
    assert 'layer == 4 and mode == "program"' in runner
    assert "execute(program, z)" in runner
    assert "resource_guard(protocol)" in runner and "nvidia-smi" in runner
    return p


if __name__ == "__main__":
    p = load_and_validate()
    print(f"validated {p['protocol_id']}: {p['status']}")
