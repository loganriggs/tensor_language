#!/usr/bin/env python3
"""CPU-only gates for rung 505."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import torch


OPS = Path(__file__).resolve().parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import equality_score_gauged_downstream_program_rung505 as r505


def test_fixed_vocabulary_and_price():
    assert r505.SITES == ("m8", "m9", "m12", "a14", "m17")
    assert r505.TASK_MASK == 7
    assert r505.SUPPRESSOR_MASK == 24
    assert r505.ALL_MASK == 31
    assert len(r505.SUBSETS) == 32
    assert r505.EXPECTED_FORWARDS == 17875
    assert r505.EXPECTED_CAPTURES == 2375
    assert r505.EXPECTED_PATCHES == 52000


def test_subset_enumeration_is_complete_and_ordered():
    assert r505.subset_sites(0) == ()
    assert r505.subset_sites(r505.TASK_MASK) == ("m8", "m9", "m12")
    assert r505.subset_sites(r505.SUPPRESSOR_MASK) == ("a14", "m17")
    assert r505.subset_sites(r505.ALL_MASK) == r505.SITES
    assert sum(len(r505.subset_sites(mask)) for mask in r505.SUBSETS) == 80


def test_signed_scale_changes_only_score_orientation():
    scales = {
        "L5H5->L8H4": {"score_ratio": 2.0, "payload_ratio": 3.0},
        "L7H3->L8H4": {"score_ratio": 5.0, "payload_ratio": 7.0},
        "L8H3->L8H4": {"score_ratio": 11.0, "payload_ratio": 13.0},
    }
    assert r505.signed_scales(scales, "N") is None
    assert r505.signed_scales(scales, "P") == {"score_ratio": 2.0, "payload_ratio": 3.0}
    assert r505.signed_scales(scales, "Z7") == {"score_ratio": -5.0, "payload_ratio": 7.0}
    assert r505.signed_scales(scales, "Z8") == {"score_ratio": -11.0, "payload_ratio": 13.0}
    assert r505.signed_scales(scales, "W7") == {"score_ratio": 5.0, "payload_ratio": 7.0}


def test_metrics_and_sign_definitions():
    vector = [-1.0, 2.0, 3.0, -4.0]
    assert r505.sign_pattern(vector)
    assert not r505.sign_pattern([1.0, 2.0, 3.0, -4.0])
    assert r505.all_negative([-1.0, -2.0, -3.0, -4.0])
    report = r505.metrics(vector, [2 * value for value in vector])
    assert abs(report["cosine"] - 1.0) < 1e-12
    assert abs(report["right_projection_on_left"] - 2.0) < 1e-12
    assert abs(report["norm_ratio"] - 2.0) < 1e-12


def test_effect_report_uses_token_weighted_cross_entropy():
    base = torch.tensor([[2.0, 0, 0, 0, 0, 0], [8.0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    other = torch.tensor([[1.0, 0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    counts = torch.tensor([[1.0, 1, 1, 1, 1, 1], [4.0, 1, 1, 1, 1, 1]], dtype=torch.float64)
    report = r505.effect_report(base, other, counts, 0, 2)
    assert abs(report["all_positive"]["effect_nat"] - 1.0) < 1e-12
    assert report["all_positive"]["tokens"] == 5


def test_mobius_interaction_vanishes_for_additive_subset_values():
    data = r505._empty_collection()
    data["counts"].fill_(1.0)
    # Give every source a nonconstant positive full effect and five additive removal costs.
    row_scale = torch.linspace(0.8, 1.2, r505.DOCUMENTS, dtype=torch.float64)
    context = torch.tensor([0.2, -0.03, 0.06, 0.04, -0.02, 0.0], dtype=torch.float64)
    for si in range(len(r505.SOURCES)):
        full = (1 + .1 * si) * row_scale[:, None] * context[None, :]
        for mask in r505.SUBSETS:
            removal = sum((bit + 1) * .001 for bit in range(5) if mask & (1 << bit))
            effect = full - removal
            data["losses"][si, mask] = -effect
        data["direct"][si] = -(full - .01)
    for wi in range(len(r505.WRONG_SIGNS)):
        for mi, mask in enumerate(r505.WRONG_MASKS):
            data["wrong"][wi, mi] = -(
                row_scale[:, None] * context[None, :] - .001 * mask.bit_count())
    window = r505._window(data, 0, r505.DOCUMENTS)
    for source in r505.SOURCES:
        for mask in r505.SUBSETS:
            if mask.bit_count() >= 2:
                assert max(abs(value) for value in window["mobius_dividends"][source][mask].values()) < 1e-12


def test_true_dry_run_opens_no_model_or_outcomes():
    environment = dict(os.environ)
    environment["BQLIB_DRYRUN"] = "1"
    completed = subprocess.run(
        [sys.executable, str(OPS / "equality_score_gauged_downstream_program_rung505.py")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "dry_run_passed"
    assert receipt["model_loaded"] is False
    assert receipt["subset_outcomes_opened"] is False
    assert receipt["expected_forwards"] == 17875
