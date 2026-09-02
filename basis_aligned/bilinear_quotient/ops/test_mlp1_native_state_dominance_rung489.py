#!/usr/bin/env python3
"""CPU-only classification checks for rung489."""

from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mlp1_native_state_dominance_rung489 as rung


def _bundle(kind):
    generator = torch.Generator().manual_seed(489)
    own = torch.randn(3, 500, 3, generator=generator, dtype=torch.float64)
    benefits = torch.zeros(3, 500, 3, len(rung.MODES), dtype=torch.float64)
    for target, branch in enumerate(rung.BRANCHES):
        own_index = rung.MODES.index(f"mid_{branch}")
        benefits[target, ..., own_index] = own[target]
        benefits[target, ..., rung.MODES.index("curvature")] = .1 * own[target]
        for mode in ("native", "mid_T", "mid_C", "mid_I"):
            benefits[target, ..., rung.MODES.index(mode)] = own[target]
    if kind == "specific":
        noise = torch.randn(3, 500, 3, generator=generator, dtype=torch.float64)
        for target, desired in ((0, "mid_I"), (2, "mid_T")):
            benefits[target, ..., rung.MODES.index("native")] = (
                .6 * own[target] + .8 * noise[target])
            benefits[target, ..., rung.MODES.index("mid_C")] = (
                .6 * own[target] - .8 * noise[target])
            benefits[target, ..., rung.MODES.index(desired)] = own[target]
    absent = torch.zeros(3, 500, 3, dtype=torch.float64)
    arms = absent[..., None] - benefits
    write_cosines = torch.zeros(2, 3, len(rung.WRITE_MODES),
                                1 + len(rung.POSITION_SHIFTS), dtype=torch.float64)
    write_cosines[..., 0] = 1.0
    return {
        "arms": arms,
        "absent": absent,
        "native": torch.zeros(500, 3),
        "write_cosines": write_cosines,
        "instrument": {},
    }


def test_common_native():
    report = rung.analyze_phase(_bundle("common"))
    assert report["half_classifications"] == ["common_native", "common_native"]
    assert report["pred_b_common_native_reader"] is True
    assert report["pred_c_T_I_specific_midpoint"] is False
    assert report["pred_d_stable_nonnull_classification"] is True


def test_specific():
    report = rung.analyze_phase(_bundle("specific"))
    assert report["half_classifications"] == ["ti_specific", "ti_specific"]
    assert report["pred_c_T_I_specific_midpoint"] is True
    assert report["pred_d_stable_nonnull_classification"] is True


def test_frozen_class_rejects_switch():
    report = rung.analyze_phase(_bundle("common"), frozen_class="ti_specific")
    assert report["frozen_classification_holds"] is False


if __name__ == "__main__":
    test_common_native()
    test_specific()
    test_frozen_class_rejects_switch()
    print("rung489 classification tests passed")
