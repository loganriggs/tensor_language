#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Does OVERRIDING number join the shared number direction, or is it carried separately?

Three routes to number now exist in this corpus, all answering at the SAME auxiliary slot:

    lexical_number.pp_intervener      COPIES number from a plural token
    coordination_agreement.and_vs_or  COMPOSES number from two singulars
    quantifier_number.each_vs_all     OVERRIDES the number an invariantly plural noun carries

The first two share a rank-1 direction at resid:18 (transfer 1.053). This asks whether the third
joins them. Overriding is the case where a separate direction is most plausible: the model must
SUPPRESS a cue that is present rather than build or propagate one.

Because all three answer at the same read position, this transfer is not confounded by read slot
-- the failure mode `das_slot_confound_calibration_v1` measured, where the same variable at
different slots transferred at 0.067 / 0.086 against 1.053 at a matched slot. That calibration is
why this comparison is worth running and the cross-slot ones were not.

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_d shared_number_direction    -> EITHER transfer >= 0.50; override joins the shared direction
                                       and number looks like one feature reached three ways.
  pred_e distinct_number_directions -> BOTH transfers <= 0.20; overriding is carried separately,
                                       which would be the FIRST interpretable feature separation
                                       this lane has produced, since it is not cross-slot.
  Between is INCONCLUSIVE.

  Stated prior: I lean shared, because the two routes already tested came back shared and the
  answer vocabulary is identical here. A separation would be the more informative outcome and I
  am registering it as the branch that would genuinely surprise me.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_coordination_agreement as coord
import circuit_fast_screen_candidate_quantifier_number as poss
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_quantifier_override_transfer_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, DISTINCT = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 48, 3072


def _plan():
    return {"candidate_id": "quantifier_number.override_transfer_v1", "site": SITE,
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
    q_coord = _fit(backend, c1[:ch])
    q_quant = _fit(backend, p1[:ph])

    report = {
        "coordination_a1_heldout": _apply(backend, c1[ch:], q_coord),
        "coordination_a2_report_frame": _apply(backend, _fam(coord, "A2"), q_coord),
        "quantifier_a1_heldout": _apply(backend, p1[ph:], q_quant),
        "coordination_direction_on_quantifier": _apply(backend, p1, q_coord),
        "quantifier_direction_on_coordination": _apply(backend, c1, q_quant),
    }

    # P and C share an answer across their sides; use the kernel's same-answer disturbance measure.
    base_f, donor_f, _ = das.capture_site(backend, c1[:ch], SITE)
    scale = das.target_scale(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in c1[:ch]], [r["donor_foil_id"] for r in c1[:ch]])
    for label, rows in (("p_first_noun_rewrite", _fam(coord, "P")), ("c_control", _fam(coord, "C"))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        effect, n = das.subspace_same_answer_effect(
            backend, base, donor, q_coord,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows], scale)
        report[label] = {"same_answer_effect": effect, "rows": n}

    cosine = float(abs((q_coord[:, 0] * q_quant[:, 0]).sum()))
    t1 = report["coordination_direction_on_quantifier"]["mean_recovery"]
    t2 = report["quantifier_direction_on_coordination"]["mean_recovery"]
    best = max(t1, t2)
    native = min(report["coordination_a1_heldout"]["mean_recovery"],
                 report["coordination_a2_report_frame"]["mean_recovery"])
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_rank1_carries_both_constructions": bool(native >= CARRY),
        "pred_c_subspace_spares_p_and_c": bool(
            report["p_first_noun_rewrite"]["same_answer_effect"] <= SPARE
            and report["c_control"]["same_answer_effect"] <= SPARE),
        "pred_d_shared_number_direction": bool(best >= SHARED),
        "pred_e_distinct_number_directions": bool(best <= DISTINCT),
    }
    reading = ("shared_number_direction" if best >= SHARED
               else "distinct_number_directions" if best <= DISTINCT
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_subspace_result_v1",
              "candidate_id": "quantifier_number.override_transfer_v1",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared": SHARED, "distinct": DISTINCT},
              "registered_matched_slot": (
                  "quantifier_number.each_vs_all answers at the SAME auxiliary in the SAME "
                  "position as coordination_agreement.and_vs_or, so this transfer is not "
                  "confounded by read position -- the failure mode measured in "
                  "das_slot_confound_calibration_v1."),
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
