#!/usr/bin/env python3
# BQGATE: rank fixed at 1 in advance; transfer thresholds registered before the run.
"""DAS at resid:18 for `possessive_number`, with four matched siblings as built-in transfer tests.

This is the best-controlled target in the corpus. Five configurations of the same behaviour screen
`selective_causal_site`, differing only in what sits between the antecedent and the pronoun:

    adjacent        distance 1, nothing intervening          <- the direction is fitted here
    medial          distance 4, one prepositional phrase
    long_simple     distance 7, two stacked prepositional phrases
    inanimate_arg   distance 5, an inanimate direct object
    verb_final      distance 6, a VP with its own object, verb-final

If `resid:18` carries a genuine NUMBER feature, one direction fitted on the adjacent design should
carry it in all four others, which differ in intervening material and in distance from 1 to 7. If
instead the fit latches onto something specific to the short frame, transfer collapses.

REGISTERED BEFORE THE RUN:
  RANK = 1. A null is not permission to raise it.
  pred_c number_feature      -> the MINIMUM transfer across all four siblings >= 0.50
  pred_f design_specific     -> that minimum <= 0.15
  Between is INCONCLUSIVE and reported as such.
  pred_d selectivity          -> P and C same-answer effects <= 0.2 on the fitted behaviour

The two configurations that FAIL natively (animate attractor, particle-final) are deliberately
excluded: their base-to-donor separation is unreliable, so a recovery ratio computed on them
would be noise dressed as evidence.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer

import circuit_fast_screen_candidate_possessive_adjacent as adjacent
import circuit_fast_screen_candidate_possessive_medial as medial
import circuit_fast_screen_candidate_possessive_long_simple as long_simple
import circuit_fast_screen_candidate_possessive_argument as inanimate_arg
import circuit_fast_screen_candidate_possessive_verbfinal as verb_final

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_possessive_number_resid18_rank1_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
FEATURE, SPECIFIC = 0.50, 0.15
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 2048

SIBLINGS = (("medial_dist4_pp", medial), ("long_simple_dist7_two_pp", long_simple),
            ("inanimate_argument_dist5", inanimate_arg), ("verb_final_dist6", verb_final))


def _plan():
    return {"candidate_id": "possessive_number.das_resid18_rank1_v1", "site": SITE, "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _fam(mod, name):
    return [r for r in mod.build_rows() if r["family"] == name]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    a1 = _fam(adjacent, "A1")
    ok, worst = das.verify_head(backend, a1[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    fit_rows, held_rows = a1[: len(a1) // 2], a1[len(a1) // 2:]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    def recovery(rows):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
        return {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    report = {"fitted_adjacent_heldout": recovery(held_rows),
              "fitted_adjacent_A2": recovery(_fam(adjacent, "A2"))}
    for label, mod in SIBLINGS:
        report[label] = recovery(_fam(mod, "A1"))

    base_h, donor_h, _ = das.capture_site(backend, held_rows, SITE)
    scale = das.target_scale(backend, base_h, donor_h,
                             [r["donor_answer_id"] for r in held_rows],
                             [r["donor_foil_id"] for r in held_rows])
    for name in ("P", "C"):
        rows = _fam(adjacent, name)
        base, donor, _ = das.capture_site(backend, rows, SITE)
        effect, n = das.subspace_same_answer_effect(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows], scale)
        report[name] = {"same_answer_effect": effect, "mean_absolute_recovery": effect, "rows": n}

    sibling_min = min(report[label]["mean_recovery"] for label, _ in SIBLINGS)
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_heldout_transfer": bool(report["fitted_adjacent_heldout"]["mean_recovery"] >= 0.5),
        "pred_c_number_feature": bool(sibling_min >= FEATURE),
        "pred_d_subspace_selective": bool(report["P"]["same_answer_effect"] <= 0.2
                                          and report["C"]["same_answer_effect"] <= 0.2),
        "pred_e_cross_construction": bool(report["fitted_adjacent_A2"]["mean_recovery"] >= 0.5),
        "pred_f_design_specific": bool(sibling_min <= SPECIFIC),
    }
    reading = ("number_feature_survives_intervening_material" if sibling_min >= FEATURE
               else "design_specific_direction" if sibling_min <= SPECIFIC
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_sibling_transfer_result_v1",
              "candidate_id": "possessive_number.das_resid18_rank1_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "possessive_number.adjacent_antecedent A1 (distance 1)",
              "registered_thresholds": {"number_feature": FEATURE, "design_specific": SPECIFIC},
              "sibling_minimum": sibling_min, "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "sibling_minimum": sibling_min,
                      "families": report}, indent=2))


if __name__ == "__main__":
    main()
