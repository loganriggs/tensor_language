#!/usr/bin/env python3
"""Focused CPU tests for the native-order Task14 current/cache v2 repair."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "source_term_sum_max_absolute_error": value,
        "raw_effective_value_max_absolute_error": value,
        "projected_effective_value_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
        "complete_head_vector_max_absolute_error": value,
    }


def _evidence():
    effects = {
        "native_value": 0.0,
        "opposite_current_only": .8,
        "opposite_cached_only": .1,
        "opposite_both": 1.0,
        "lexical_current_only": .05,
        "lexical_cached_only": .05,
        "lexical_both": .05,
        "complete_opposite_head": 1.1,
    }
    output = []
    for row in run.build_rows():
        sign = 1.0 if row["direction_id"] == "singular_to_plural" else -1.0
        for condition, effect in effects.items():
            output.append({
                "row_id": row["row_id"],
                "cell_id": f"{row['direction_id']}__{row['template_id']}",
                "condition": condition,
                "target_margin_improvement": effect,
                "target_CE_improvement": effect,
                "fixed_are_minus_is_change": sign * effect,
            })
    return output


def test_v2_candidate_specific_license_holdout_and_unchanged_price():
    plan = run.compile_plan()
    assert plan["candidate_id"].endswith("factorial_v2")
    assert plan["numerical_repair_only"] is True
    assert plan["split"] == "LICENSED_HOLDOUT" and plan["row_count"] == 16
    assert plan["conditions"] == list(run.v1.CONDITIONS)
    assert plan["bars"] == run.v1.BARS
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 352,
                             "backwards": 0, "parameter_updates": 0}


def test_dry_run_validates_repair_and_license_without_model_access():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_wrong_repair_hash_and_license_fail_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.CurrentCachedFactorialV2Error, match="repair receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "LICENSE_SHA256", "0" * 64)
    with pytest.raises(run.licensing.CapabilityLicenseError, match="license hash changed"):
        run.validate_preflight()


def test_raw_branches_are_mixed_before_one_projection():
    torch = pytest.importorskip("torch")

    class Linear:
        def __init__(self, weight):
            self.weight = weight

    class Attention:
        pass

    attention = Attention()
    attention.c_v = Linear(torch.eye(18))
    attention.c_proj = Linear(torch.arange(18 * 18, dtype=torch.float32).reshape(18, 18) / 1000)
    attention.lamb = torch.tensor(.3)
    state = torch.arange(2 * 4 * 18, dtype=torch.float32).reshape(2, 4, 18) / 100
    bus = torch.flip(state.reshape(2, 4, 9, 2), dims=(1,))
    current, cached, effective, projection = run._raw_value_branches(
        state, bus, attention, torch, torch.nn.functional)
    assert current.shape == cached.shape == effective.shape == (2, 4, 2)
    assert torch.equal(effective, current + cached)
    assert torch.equal(projection, attention.c_proj.weight[:, 6:8])

    class CountingF:
        calls = 0

        @classmethod
        def linear(cls, value, weight):
            cls.calls += 1
            return torch.nn.functional.linear(value, weight)

    observed = run._project_once(current[:, 2], cached[:, 2], projection, CountingF)
    expected = torch.nn.functional.linear(effective[:, 2], projection)
    assert CountingF.calls == 1
    assert torch.equal(observed, expected)


def test_compile_invokes_projection_once_per_mixed_value_arm():
    torch = pytest.importorskip("torch")
    rows = run.build_rows(); n, head_width, out_width = len(rows), 3, 5

    def side(offset):
        p = torch.ones(n, 9) + offset
        current = torch.ones(n, 9, head_width) * (1 + offset)
        cached = torch.ones(n, 9, head_width) * (2 + offset)
        projection = torch.arange(out_width * head_width, dtype=torch.float32).reshape(
            out_width, head_width) / 10
        u = torch.nn.functional.linear(current + cached, projection)
        return {"p": p, "u": u, "current_pre": current, "cached_pre": cached,
                "effective_pre": current + cached,
                "head": torch.einsum("bk,bkd->bd", p, u)}, projection

    recipient, projection = side(0)
    opposite, _ = side(1)
    lexical, _ = side(2)
    tokens = torch.tensor([row["endpoints"]["recipient"]["ids"] for row in rows])

    class CountingF:
        calls = 0

        @classmethod
        def linear(cls, value, weight):
            cls.calls += 1
            return torch.nn.functional.linear(value, weight)

    patch = run._compile(tokens, recipient, opposite, lexical, projection, rows, torch, CountingF)
    assert CountingF.calls == 7
    assert len(patch["specs"]) == 128
    assert int(patch["native_reinstall_mask"].sum()) == 16


def test_raw_and_projected_exactness_checks_each_can_fail():
    assert run.score(_evidence(), _exact())["predictions"]["pred_a_instrument_live"]
    for name in ("raw_effective_value_max_absolute_error",
                 "projected_effective_value_max_absolute_error"):
        exactness = _exact()
        exactness[name] = 1.0
        assert not run.score(_evidence(), exactness)["predictions"]["pred_a_instrument_live"]


def test_missing_v2_exactness_metric_fails_closed():
    exactness = _exact()
    del exactness["raw_effective_value_max_absolute_error"]
    with pytest.raises(run.CurrentCachedFactorialV2Error, match="exactness evidence is incomplete"):
        run.score(_evidence(), exactness)
