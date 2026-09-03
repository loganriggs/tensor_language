"""CPU tests for rung-522 scheduler-aligned spike analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    path = OPS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load("attention8_selective_shared_projector_rung522_spike_analysis")


def test_spikes_are_joined_to_exact_maximizing_target_row_and_map():
    history = [1.0] * 200
    history[3] = 101.0
    history[7] = 250.0
    record = SimpleNamespace(
        spec=SimpleNamespace(family="real_leave_one_out", frame_id="frame"),
        fit_record_payload={
            "loss_history": history,
            "maximizing_targets": ["target-a"] * 200,
        },
        fit_scheduler_payload={
            "roles": [
                {
                    "target": "target-a", "kind": "member",
                    "permutation": [11],
                },
                {
                    "target": "target-a", "kind": "control",
                    "permutation": [12],
                },
            ]
        },
    )
    result = ANALYSIS.analyze_spikes({"frame": record}, threshold=100.0)
    assert result["spike_count"] == 2
    assert result["fits_with_a_spike"] == 1
    assert result["first_spike_update_median"] == 3
    assert result["by_donor_map"]["3"]["spikes"] == 2
    assert result["by_maximizing_target"]["target-a"]["spikes"] == 2
    repeated = result["repeated_target_map_member_row_patterns"]
    assert repeated == [{
        "target": "target-a",
        "donor_map_index": 3,
        "member_rows": [11],
        "control_rows": [12],
        "visits": 50,
        "spikes": 2,
        "spike_percent": 4.0,
        "fits_with_pattern": 1,
        "fits_with_a_spike": 1,
    }]
    assert result["patterns_spiking_in_multiple_fits"] == 0
    assert result["patterns_spiking_in_every_fit_that_saw_them"] == 0
