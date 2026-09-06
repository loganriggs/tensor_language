#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""CALIBRATION: how much does a READ-SLOT difference alone depress DAS transfer?

Three of this lane's cross-behaviour conclusions depend on the answer, so this measures it
directly instead of patching each result separately.

`das_matched_slot_number_transfer_v1` retracted an earlier reading by showing that two number
behaviours sharing a read slot transfer at 1.053, while `das_coordination_number_transfer_v1` had
measured 0.018 / 0.030 for number behaviours at DIFFERENT slots. That is strong circumstantial
evidence for a read-position confound, but the two comparisons also differed in the variable
(composed versus copied number), so position was never isolated.

This isolates it. Both behaviours below read the SAME variable -- head noun number:

    lexical_number.pp_intervener        "The leaders near the maple" -> " were"    AUXILIARY slot
    possessive_number.adjacent_antecedent "The leaders checked"      -> " their"   PRONOUN slot

Same variable, disjoint vocabularies, different read position. Nothing else differs. This is the
known-good / known-bad instrument check of standing lesson 6, applied to the transfer measure
itself rather than to a behaviour.

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_d position_is_not_the_confound -> EITHER transfer >= 0.50. Then a slot difference does NOT
                                         destroy transfer, my retraction's premise is wrong, and
                                         the composed-versus-copied separation returns as a real
                                         finding -- I would be retracting the retraction.
  pred_e position_confound_confirmed  -> BOTH transfers <= 0.20. Then read slot alone destroys
                                         transfer between behaviours reading the SAME variable,
                                         and EVERY cross-slot transfer this lane has run --
                                         negation 0.165, verb frames 0.172 -- is uninterpretable
                                         as evidence about features.
  Between is INCONCLUSIVE.

  Stated prior: I expect the confound confirmed, because that is what the retraction assumed. I am
  aware that makes this a test of my own current position, which is the reason to register both
  branches explicitly and to name what each would cost me.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_lexical_number_pp as coord
import circuit_fast_screen_candidate_possessive_adjacent as poss
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_slot_confound_calibration_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, DISTINCT = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 48, 3072


def _plan():
    return {"candidate_id": "number.slot_confound_calibration_v1", "site": SITE,
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
    q_aux = _fit(backend, c1[:ch])
    q_pron = _fit(backend, p1[:ph])

    report = {
        "aux_slot_a1_heldout": _apply(backend, c1[ch:], q_aux),
        "aux_slot_a2_report_frame": _apply(backend, _fam(coord, "A2"), q_aux),
        "pronoun_slot_a1_heldout": _apply(backend, p1[ph:], q_pron),
        "aux_direction_on_pronoun_slot": _apply(backend, p1, q_aux),
        "pronoun_direction_on_aux_slot": _apply(backend, c1, q_pron),
    }

    # P and C share an answer across their sides; use the kernel's same-answer disturbance measure.
    base_f, donor_f, _ = das.capture_site(backend, c1[:ch], SITE)
    scale = das.target_scale(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in c1[:ch]], [r["donor_foil_id"] for r in c1[:ch]])
    for label, rows in (("p_head_noun_rewrite", _fam(coord, "P")), ("c_control", _fam(coord, "C"))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        effect, n = das.subspace_same_answer_effect(
            backend, base, donor, q_aux,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows], scale)
        report[label] = {"same_answer_effect": effect, "rows": n}

    cosine = float(abs((q_aux[:, 0] * q_pron[:, 0]).sum()))
    t1 = report["aux_direction_on_pronoun_slot"]["mean_recovery"]
    t2 = report["pronoun_direction_on_aux_slot"]["mean_recovery"]
    best = max(t1, t2)
    native = min(report["aux_slot_a1_heldout"]["mean_recovery"],
                 report["aux_slot_a2_report_frame"]["mean_recovery"])
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_rank1_carries_both_constructions": bool(native >= CARRY),
        "pred_c_subspace_spares_p_and_c": bool(
            report["p_head_noun_rewrite"]["same_answer_effect"] <= SPARE
            and report["c_control"]["same_answer_effect"] <= SPARE),
        "pred_d_shared_number_direction": bool(best >= SHARED),
        "pred_e_distinct_number_directions": bool(best <= DISTINCT),
    }
    reading = ("shared_number_direction" if best >= SHARED
               else "distinct_number_directions" if best <= DISTINCT
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_subspace_result_v1",
              "candidate_id": "number.slot_confound_calibration_v1",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared": SHARED, "distinct": DISTINCT},
              "registered_calibration": (
                  "Both behaviours read the SAME variable -- head noun number (leaders vs "
                  "leader). Only the READ SLOT differs: an auxiliary after a prepositional "
                  "phrase versus a possessive pronoun after a verb. Vocabularies are disjoint. "
                  "Any gap between this transfer and the matched-slot 1.053 is therefore "
                  "attributable to read position alone."),
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
