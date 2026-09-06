#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Is the polarity/neither distinctness SYMMETRIC, or an artefact of the fitting slot?

`das_polarity_licensing_rank1_v1` found the correlative `neither` direction transfers to polarity
licensing at only 0.132 (registered bar 0.20), reading `neither_axis_is_narrower_than_negation`,
with abs cosine 0.165 between the two rank-1 directions.

There is an alternative explanation I do not get to dismiss for free. The correlative direction
was fitted at a COORDINATION SLOT on connective tokens; polarity is read at a verb four tokens
downstream of its licensor. A direction fitted at one slot may fail to transfer to a different
slot for reasons that have nothing to do with which feature it encodes. Under that reading the
low transfer is about position, not about negation.

The discriminating test is the reverse direction, and it is cheap. Fit rank-1 on polarity
licensing (identical fit, same seed, same rank) and apply it UNTOUCHED to the two correlative
pairs. Two distinct features predict near-zero BOTH ways. A fitting-slot artefact predicts an
ASYMMETRY -- one direction of transfer working much better than the other.

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_c symmetric_distinct_features -> BOTH correlative pairs recover <= 0.20 from the polarity
                                        direction (matching the 0.132 seen the other way)
  pred_d asymmetric_fitting_slot     -> EITHER pair recovers >= 0.50
  Between is INCONCLUSIVE and reported as such.

  Prior expectation: symmetric near-zero, because the measured cosine of 0.165 is a property of
  the two directions themselves and does not depend on which one was applied to which rows. But
  cosine is not recovery -- a small component along a high-gain direction can still move an
  answer -- so the reverse transfer is worth measuring rather than inferring.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_polarity_licensing as pol
import circuit_fast_screen_candidate_correlative_pair as bn         # both / neither
import circuit_fast_screen_candidate_correlative_state as en        # either / neither
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_polarity_reverse_transfer_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
SYMMETRIC, ASYMMETRIC = 0.20, 0.50
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 32, 2048


def _plan():
    return {"candidate_id": "polarity_licensing.reverse_transfer_v1", "site": SITE, "rank": RANK,
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
    a1 = _a1(pol)
    ok, worst = das.verify_head(backend, a1[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    # identical polarity fit to das_polarity_licensing_rank1_v1
    half = len(a1) // 2
    fit_rows = a1[:half]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q_pol = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    report = {}
    for label, rows in (("polarity_a1_heldout_reference", a1[half:]),
                        ("correlative_both_neither", _a1(bn)),
                        ("correlative_either_neither", _a1(en))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q_pol,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
        report[label] = {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    worst_pair = max(report["correlative_both_neither"]["mean_recovery"],
                     report["correlative_either_neither"]["mean_recovery"])
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_polarity_fit_replicates": bool(
            report["polarity_a1_heldout_reference"]["mean_recovery"] >= 0.50),
        "pred_c_symmetric_distinct_features": bool(worst_pair <= SYMMETRIC),
        "pred_d_asymmetric_fitting_slot": bool(worst_pair >= ASYMMETRIC),
    }
    reading = ("asymmetric_fitting_slot_explanation_survives" if worst_pair >= ASYMMETRIC
               else "symmetric_distinct_features" if worst_pair <= SYMMETRIC
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_confound_test_result_v1",
              "candidate_id": "polarity_licensing.reverse_transfer_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "polarity_licensing.never_vs_often A1 first half",
              "probed_on": "correlative both/neither and either/neither",
              "forward_direction_recovery": 0.132,
              "registered_thresholds": {"symmetric": SYMMETRIC, "asymmetric": ASYMMETRIC},
              "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
