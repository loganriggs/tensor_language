#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Is the shared direction NUMBER, or just AUXILIARY SELECTION at that position?

Another control on my own claim, and the gate is inverted: here a LOW number is what supports the
claim I already hold, so the branch that would cost me is the high one.

Four number behaviours share a rank-1 direction at an auxiliary directly after the subject NP, and
`das_number_token_control_v1` excluded a token axis by varying answer vocabulary (0.691 / 0.778,
cosine 0.651). But all four perform the same STRUCTURAL operation -- choosing between two auxiliary
forms at that position. A direction encoding "which auxiliary goes here" would reproduce every
transfer observed without being number at all, and nothing run so far separates those readings.

`temporal_auxiliary.will_vs_had` supplies the missing configuration. Its subject is SINGULAR in
every row, so number is constant and uninformative; a fronted temporal adverb drives the auxiliary.
Same read position, vocabulary disjoint from both number families. It screened selective at
resid:18 (recovery 1.000, P 0.072, C 0.099).

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_e number_direction_is_specific -> BOTH transfers <= 0.20. The direction does not carry a
                                         non-number variable at the same slot, and the
                                         four-behaviour number claim stands AS STATED.
  pred_d direction_is_auxiliary_selection_not_number -> EITHER transfer >= 0.50. The direction
                                         encodes auxiliary choice at this position, and the claim
                                         must be RESTATED in those terms -- it would not be a
                                         number feature at all.
  Between is INCONCLUSIVE.

  Stated prior: I expect specificity, but I expected the coordination separation to hold too and
  it did not. Registering the costly branch in the words I would have to use.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_lexical_number_pp as coord
import circuit_fast_screen_candidate_temporal_auxiliary as poss
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_number_specificity_control_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, DISTINCT = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 48, 3072


def _plan():
    return {"candidate_id": "number.specificity_control_v1", "site": SITE,
            "rank": RANK, "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _fam(m, family):
    return [r for r in m.build_rows() if r["family"] == family]


def _fit(backend, rows):
    base, donor, _ = das.capture_site(backend, rows, SITE)
    return das.fit_subspace(
        backend, base, donor,
        [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows],
        rank=RANK, steps=STEPS)


def _apply(backend, rows, q):
    base, donor, _ = das.capture_site(backend, rows, SITE)
    mean, absmean, n = das.subspace_recovery(
        backend, base, donor, q,
        [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
    return {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    c1, p1 = _fam(coord, "A1"), _fam(poss, "A1")
    ok, worst = das.verify_head(backend, c1[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    ch, ph = len(c1) // 2, len(p1) // 2
    q_wasw = _fit(backend, c1[:ch])
    q_temp = _fit(backend, p1[:ph])

    report = {
        "was_were_a1_heldout": _apply(backend, c1[ch:], q_wasw),
        "was_were_a2_report_frame": _apply(backend, _fam(coord, "A2"), q_wasw),
        "temporal_a1_heldout": _apply(backend, p1[ph:], q_temp),
        "number_direction_on_temporal": _apply(backend, p1, q_wasw),
        "temporal_direction_on_number": _apply(backend, c1, q_temp),
    }

    # P and C share an answer across their sides; use the kernel's same-answer disturbance measure.
    base_f, donor_f, _ = das.capture_site(backend, c1[:ch], SITE)
    scale = das.target_scale(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in c1[:ch]], [r["donor_foil_id"] for r in c1[:ch]])
    for label, rows in (("p_head_noun_rewrite", _fam(coord, "P")), ("c_control", _fam(coord, "C"))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        effect, n = das.subspace_same_answer_effect(
            backend, base, donor, q_wasw,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows], scale)
        report[label] = {"same_answer_effect": effect, "rows": n}

    cosine = float(abs((q_wasw[:, 0] * q_temp[:, 0]).sum()))
    t1 = report["number_direction_on_temporal"]["mean_recovery"]
    t2 = report["temporal_direction_on_number"]["mean_recovery"]
    best = max(t1, t2)
    native = min(report["was_were_a1_heldout"]["mean_recovery"],
                 report["was_were_a2_report_frame"]["mean_recovery"])
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_rank1_carries_both_constructions": bool(native >= CARRY),
        "pred_c_subspace_spares_p_and_c": bool(
            report["p_head_noun_rewrite"]["same_answer_effect"] <= SPARE
            and report["c_control"]["same_answer_effect"] <= SPARE),
        "pred_d_shared_number_direction": bool(best >= SHARED),
        "pred_e_distinct_number_directions": bool(best <= DISTINCT),
    }
    reading = ("direction_is_auxiliary_selection_not_number" if best >= SHARED
               else "number_direction_is_specific" if best <= DISTINCT
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_subspace_result_v1",
              "candidate_id": "number.specificity_control_v1",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared": SHARED, "distinct": DISTINCT},
              "registered_specificity_control": (
                  "The subject is SINGULAR in every temporal row, so number is constant and "
                  "carries no information there; a fronted temporal adverb drives the auxiliary "
                  "instead. Read position is identical to the number family and the vocabulary "
                  "is disjoint from both number vocabularies. A LOW transfer is therefore the "
                  "outcome that SUPPORTS the existing number claim, which inverts the gate "
                  "relative to every earlier transfer test in this lane."),
              "reading": reading, "target_scale": scale,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report, "abs_cosine_between_directions": cosine,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report, "abs_cosine": cosine,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
