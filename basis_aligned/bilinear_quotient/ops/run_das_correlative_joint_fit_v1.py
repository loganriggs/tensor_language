#!/usr/bin/env python3
# BQGATE: rank stays at 1; this is a different fit, not a rank increase.
"""Fit ONE direction on BOTH correlative pairs at once, and see whether it serves both.

The single-behaviour fit left the shared-feature question inconclusive by design: a direction
fitted on `both`/`neither` transferred to `either`/`neither` at 0.377/0.457, above the 0.15 that
would have meant separate lexical associations and below the 0.50 that would have meant a shared
feature.

Two readings survive that, and they differ in something testable. If the pairs share a feature and
the single-behaviour fit merely landed off-axis, then a direction fitted on the UNION should serve
both well. If they are genuinely distinct directions that happen to overlap, a joint fit is a
compromise and should serve neither well.

RANK STAYS AT 1. This is a different fit, not permission to raise rank.

REGISTERED BEFORE THE RUN:
  pred_c shared feature      -> BOTH behaviours recover >= 0.80 from the joint direction
  pred_e distinct directions -> at least one recovers <= 0.50
  Between those is INCONCLUSIVE and will be reported as such. The single-behaviour fit reached
  0.980 on its own held-out rows, so 0.80 is a real bar and not a formality.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_correlative_pair as pair_bn      # both / neither
import circuit_fast_screen_candidate_correlative_state as pair_en     # either / neither
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_correlative_joint_fit_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
SHARED, DISTINCT = 0.80, 0.50
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20, 1024


def _plan():
    return {"candidate_id": "correlative.joint_fit_das_v1", "site": SITE, "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _a1(mod):
    return [r for r in mod.build_rows() if r["family"] == "A1"]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    bn, en = _a1(pair_bn), _a1(pair_en)
    ok, worst = das.verify_head(backend, bn[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    half_bn, half_en = len(bn) // 2, len(en) // 2
    fit_rows = bn[:half_bn] + en[:half_en]          # the union, fitted once
    held = {"both_neither_heldout": bn[half_bn:], "either_neither_heldout": en[half_en:]}

    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    report = {}
    for label, rows in held.items():
        base, donor, _ = das.capture_site(backend, rows, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
        report[label] = {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    lo = min(v["mean_recovery"] for v in report.values())
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_joint_fit_converged": bool(lo > 0.0),
        "pred_c_shared_feature": bool(lo >= SHARED),
        "pred_d_both_positive_transfer": bool(lo > 0.15),
        "pred_e_distinct_directions": bool(lo <= DISTINCT),
    }
    reading = ("shared_feature" if lo >= SHARED
               else "distinct_directions" if lo <= DISTINCT
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_joint_fit_result_v1",
              "candidate_id": "correlative.joint_fit_das_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "union of both/neither and either/neither A1 rows",
              "registered_thresholds": {"shared": SHARED, "distinct": DISTINCT},
              "single_behaviour_reference": {"own_heldout": 0.980, "cross_pair": 0.377},
              "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report}, indent=2))


if __name__ == "__main__":
    main()
