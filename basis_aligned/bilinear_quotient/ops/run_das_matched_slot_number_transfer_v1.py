#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Does CONSTRUCTED number share a direction with COPIED number, at a MATCHED read slot?

`das_coordination_number_transfer_v1` read `distinct_number_directions`: constructed number
(`and`/`or` across two singular nouns) and antecedent number (possessive) came back
near-orthogonal at resid:18, cosine 0.072, transfer 0.018 / 0.030 both ways. Disjoint vocabularies
ruled out a shared-token artefact. They did NOT rule out a READ-POSITION artefact: coordination is
read at an auxiliary after the second noun, possessive at a pronoun slot, and differing slots
depress transfer in both directions exactly as distinct features would. Symmetry cannot separate
those explanations, so that result was reported with the confound still open.

`lexical_number.pp_intervener` was authored to close it. Its number lives ON A TOKEN (`leaders`),
three tokens back, and its answer is the SAME auxiliary in the SAME position as coordination. It
screened selective at resid:18 (recovery 1.000, P 0.069, C 0.111). Transferring between the two
holds read position FIXED and varies only whether the number was copied or composed.

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_d shared_number_direction    -> EITHER transfer >= 0.50, meaning the earlier near-zero was
                                       a read-position artefact and my previous result is
                                       RETRACTED, not nuanced.
  pred_e distinct_number_directions -> BOTH transfers <= 0.20, and the earlier reading stands at
                                       a matched slot.
  Between is INCONCLUSIVE.

  Stated prior: I expect SHARED here, against this corpus's base rate. The three previous
  separations each paired behaviours reading genuinely DIFFERENT variables; this pairs ONE
  variable reached two ways at one slot. Distinct would be the more surprising outcome.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_coordination_agreement as coord
import circuit_fast_screen_candidate_lexical_number_pp as poss
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_matched_slot_number_transfer_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, DISTINCT = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 48, 3072


def _plan():
    return {"candidate_id": "coordination_agreement.matched_slot_transfer_v1", "site": SITE,
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
    q_lex = _fit(backend, p1[:ph])

    report = {
        "coordination_a1_heldout": _apply(backend, c1[ch:], q_coord),
        "coordination_a2_report_frame": _apply(backend, _fam(coord, "A2"), q_coord),
        "lexical_a1_heldout": _apply(backend, p1[ph:], q_lex),
        "coordination_direction_on_lexical": _apply(backend, p1, q_coord),
        "lexical_direction_on_coordination": _apply(backend, c1, q_lex),
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

    cosine = float(abs((q_coord[:, 0] * q_lex[:, 0]).sum()))
    t1 = report["coordination_direction_on_lexical"]["mean_recovery"]
    t2 = report["lexical_direction_on_coordination"]["mean_recovery"]
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
              "candidate_id": "coordination_agreement.matched_slot_transfer_v1",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared": SHARED, "distinct": DISTINCT},
              "registered_matched_slot": (
                  "lexical_number.pp_intervener answers at the SAME auxiliary in the "
                  "SAME position as coordination_agreement.and_vs_or, so this transfer "
                  "holds read position fixed and varies only whether the number was "
                  "copied from a token or composed by a connective."),
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
