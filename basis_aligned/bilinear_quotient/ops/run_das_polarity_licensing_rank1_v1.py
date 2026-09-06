#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Which subspace at resid:18 carries polarity licensing -- and is it this lane's `neither` axis?

`polarity_licensing_never_vs_often_v1` screened selective at resid:18 (recovery 1.000, P 0.059,
C 0.133), with no single head or MLP carrying it: the strongest component was attn:07 at 0.180
while the residual stream climbed gradually to 1.000. Interchange therefore localizes this to a
whole residual site and nothing finer, which is exactly the case DAS exists for.

Two questions, one run.

  1. STANDARD DAS PASS. Fit rank-1 on the A1 negative/positive contrast and require all four
     hypotheses of the screen to survive at the subspace level -- held-out A1, the A2 report
     construction, the answer-preserving agent rewrite P, and the unrelated control C.

  2. THE TRANSFER TEST THIS BEHAVIOUR WAS AUTHORED FOR. `das_correlative_neither_axis_test_v1`
     resolved the shared correlative direction as a `neither` axis (it transferred to a pair with
     no negative member at 0.036 / 0.020). That left open whether `neither` is a narrow lexical
     feature or one end of a GENERAL negation axis. Polarity licensing is the first negation
     behaviour in the corpus that is not a connective contrast, so it can finally be asked: does
     the jointly-fitted correlative direction, untouched, carry polarity here?

REGISTERED BEFORE THE RUN (both branches, so neither is retrofitted):
  RANK = 1 for the fit AND the transfer. A null is not permission to raise rank.
  pred_c shared_negation_axis   -> transfer >= 0.50
  pred_f neither_specific       -> transfer <= 0.20
  Between is INCONCLUSIVE and is reported as such rather than resolved by refitting.

  Prior expectation is genuinely open here, unlike the neither-axis test. `never` and `neither`
  are both negation, so a shared axis is plausible; but the correlative direction was fitted at a
  coordination slot on connective tokens, and polarity is read four tokens downstream of its
  licensor, so no transfer is equally plausible. Worth stating that I do not have a confident
  prediction, rather than manufacturing one after the fact.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_polarity_licensing as probed
import circuit_fast_screen_candidate_correlative_pair as bn         # both / neither
import circuit_fast_screen_candidate_correlative_state as en        # either / neither
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_polarity_licensing_rank1_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, SPECIFIC = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 2560


def _plan():
    return {"candidate_id": "polarity_licensing.das_rank1_v1", "site": SITE, "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _fam(m, family):
    return [r for r in m.build_rows() if r["family"] == family]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    a1 = _fam(probed, "A1")
    ok, worst = das.verify_head(backend, a1[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    half = len(a1) // 2
    fit_rows, held = a1[:half], a1[half:]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    scale = das.target_scale(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows])

    report = {}
    for label, rows in (("a1_heldout", held), ("a2_report_frame", _fam(probed, "A2"))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
        report[label] = {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    # P and C share an answer across their two sides, so a recovery ratio has a degenerate
    # denominator; the kernel's same-answer disturbance measure is the comparable one.
    for label, rows in (("p_agent_rewrite", _fam(probed, "P")), ("c_control", _fam(probed, "C"))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        effect, n = das.subspace_same_answer_effect(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows], scale)
        report[label] = {"same_answer_effect": effect, "rows": n}

    # --- the transfer test: the correlative joint direction, refitted identically, applied here
    ca, cb = _fam(bn, "A1"), _fam(en, "A1")
    joint_rows = ca[: len(ca) // 2] + cb[: len(cb) // 2]
    jb, jd, _ = das.capture_site(backend, joint_rows, SITE)
    q_corr = das.fit_subspace(
        backend, jb, jd,
        [r["donor_answer_id"] for r in joint_rows], [r["donor_foil_id"] for r in joint_rows],
        rank=RANK, steps=STEPS)
    base_t, donor_t, _ = das.capture_site(backend, a1, SITE)
    transfer, transfer_abs, tn = das.subspace_recovery(
        backend, base_t, donor_t, q_corr,
        [r["donor_answer_id"] for r in a1], [r["donor_foil_id"] for r in a1])

    cosine = float(abs((q[:, 0] * q_corr[:, 0]).sum()))

    native = min(report["a1_heldout"]["mean_recovery"], report["a2_report_frame"]["mean_recovery"])
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_rank1_carries_both_constructions": bool(native >= CARRY),
        "pred_c_shared_negation_axis": bool(transfer >= SHARED),
        "pred_d_subspace_spares_p_and_c": bool(
            report["p_agent_rewrite"]["same_answer_effect"] <= SPARE
            and report["c_control"]["same_answer_effect"] <= SPARE),
        "pred_e_selective_at_subspace_level": bool(
            native >= CARRY
            and report["p_agent_rewrite"]["same_answer_effect"] <= SPARE
            and report["c_control"]["same_answer_effect"] <= SPARE),
        "pred_f_neither_specific": bool(transfer <= SPECIFIC),
    }
    reading = ("shared_negation_axis" if transfer >= SHARED
               else "neither_axis_is_narrower_than_negation" if transfer <= SPECIFIC
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_subspace_result_v1",
              "candidate_id": "polarity_licensing.das_rank1_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "polarity_licensing.never_vs_often A1 first half",
              "transfer_source": "union of both/neither and either/neither (joint correlative fit)",
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared_negation_axis": SHARED,
                                        "neither_specific": SPECIFIC},
              "reading": reading,
              "target_scale": scale,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "transfer": {"mean_recovery": transfer, "mean_absolute_recovery": transfer_abs,
                           "rows": tn, "abs_cosine_with_native_direction": cosine},
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report,
                      "transfer": result["transfer"], "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
