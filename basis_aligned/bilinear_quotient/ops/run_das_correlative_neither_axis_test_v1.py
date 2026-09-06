#!/usr/bin/env python3
# BQGATE: rank stays at 1; thresholds registered before the run.
"""Is the shared correlative direction a correlative-state feature, or a `neither` axis?

`das_correlative_joint_fit_v1` found ONE rank-1 direction serving both `both`/`neither` (0.975)
and `either`/`neither` (0.963). But both pairs put `neither` -> ` nor` on their donor side, so the
direction may encode `neither` rather than correlative state.

This refits that joint direction identically (same union, same seed, same rank) and evaluates it,
untouched, on `correlative_pair.both_vs_either` — a pair with NO negative member, screened
selective at resid:17.

REGISTERED BEFORE THE RUN:
  RANK = 1, unchanged. This is a transfer test, not a new fit on the probed behaviour.
  pred_c correlative_state reading -> transfer >= 0.50
  pred_e neither_axis reading      -> transfer <= 0.15
  Between is INCONCLUSIVE and reported as such.

  Note the asymmetry of prior expectation: a rank-1 direction separating {both, either} from
  {neither} places `both` and `either` on the SAME side, so the `neither`-axis reading predicts
  near-zero transfer here. A high number would be the surprising outcome.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_both_either as probed          # both / either, no neither
import circuit_fast_screen_candidate_correlative_pair as bn         # both / neither
import circuit_fast_screen_candidate_correlative_state as en        # either / neither
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_correlative_neither_axis_test_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
STATE, NEITHER = 0.50, 0.15
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 24, 1280


def _plan():
    return {"candidate_id": "correlative.neither_axis_test_v1", "site": SITE, "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _a1(m):
    return [r for r in m.build_rows() if r["family"] == "A1"]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    a, b = _a1(bn), _a1(en)
    ok, worst = das.verify_head(backend, a[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    # identical joint fit to das_correlative_joint_fit_v1
    fit_rows = a[: len(a) // 2] + b[: len(b) // 2]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    report = {}
    for label, rows in (("fitted_both_neither_heldout", a[len(a) // 2:]),
                        ("fitted_either_neither_heldout", b[len(b) // 2:]),
                        ("probed_both_either_A1", _a1(probed)),
                        ("probed_both_either_A2", [r for r in probed.build_rows() if r["family"] == "A2"])):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
        report[label] = {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    transfer = report["probed_both_either_A1"]["mean_recovery"]
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_joint_fit_replicates": bool(
            min(report["fitted_both_neither_heldout"]["mean_recovery"],
                report["fitted_either_neither_heldout"]["mean_recovery"]) >= 0.80),
        "pred_c_correlative_state_feature": bool(transfer >= STATE),
        "pred_d_cross_construction_on_probed": bool(
            report["probed_both_either_A2"]["mean_recovery"] >= STATE),
        "pred_e_neither_axis": bool(transfer <= NEITHER),
    }
    reading = ("correlative_state_feature" if transfer >= STATE
               else "neither_axis" if transfer <= NEITHER
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_confound_test_result_v1",
              "candidate_id": "correlative.neither_axis_test_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "union of both/neither and either/neither",
              "probed_on": "correlative_pair.both_vs_either (no negative member)",
              "registered_thresholds": {"correlative_state": STATE, "neither_axis": NEITHER},
              "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report}, indent=2))


if __name__ == "__main__":
    main()
