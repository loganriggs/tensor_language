#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Is the shared number direction a NUMBER feature, or a was/were token axis?

This is a control on my own strongest result, not a new claim.

Three number behaviours share one rank-1 direction at a matched auxiliary slot -- copy
(`pp_intervener`), compose (`coordination_agreement`), override (`quantifier_number`) -- with
transfers of 1.053 and 0.600. But all three answer with the IDENTICAL pair ` was` / ` were`. A
rank-1 direction that merely separates those two tokens would reproduce every one of those numbers
without any shared NUMBER feature existing.

The comparison that could have caught this was never available. The one disjoint-vocabulary number
behaviour, possessive `their`/`his`, ALSO differs in read slot, and `das_slot_confound_calibration_v1`
measured that a slot difference alone collapses transfer from 1.053 to 0.067. So it could not
discriminate a token axis from a position artefact.

`perfect_number.have_vs_has` was authored to fill that gap: same variable (head noun number), same
read position (an auxiliary directly after the subject NP), DISJOINT answer tokens. It screened
selective at resid:18 (recovery 1.000, P 0.057, C 0.113).

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_d shared_number_direction    -> EITHER transfer >= 0.50. The direction survives a complete
                                       change of answer vocabulary, and the shared-number reading
                                       stands as a claim about number.
  pred_e distinct_number_directions -> BOTH transfers <= 0.20. The earlier 1.053 and 0.600 were a
                                       ` was`/` were` token axis, and the shared-number claim --
                                       the strongest result this lane holds -- is RETRACTED.
  Between is INCONCLUSIVE.

  Stated prior: genuinely uncertain. The three-way agreement across copy, compose and override is
  hard to get from a pure token axis, since those routes differ in everything except the number
  they end up asserting. But I have already had one confident cross-behaviour reading overturned
  this session, so I am registering the retraction branch in the same words I would have to use.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_lexical_number_pp as coord
import circuit_fast_screen_candidate_perfect_number as poss
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_number_token_control_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, DISTINCT = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 48, 3072


def _plan():
    return {"candidate_id": "number.token_control_v1", "site": SITE,
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
    q_have = _fit(backend, p1[:ph])

    report = {
        "was_were_a1_heldout": _apply(backend, c1[ch:], q_wasw),
        "was_were_a2_report_frame": _apply(backend, _fam(coord, "A2"), q_wasw),
        "have_has_a1_heldout": _apply(backend, p1[ph:], q_have),
        "was_were_direction_on_have_has": _apply(backend, p1, q_wasw),
        "have_has_direction_on_was_were": _apply(backend, c1, q_have),
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

    cosine = float(abs((q_wasw[:, 0] * q_have[:, 0]).sum()))
    t1 = report["was_were_direction_on_have_has"]["mean_recovery"]
    t2 = report["have_has_direction_on_was_were"]["mean_recovery"]
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
    reading = ("shared_number_direction" if best >= SHARED
               else "distinct_number_directions" if best <= DISTINCT
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_subspace_result_v1",
              "candidate_id": "number.token_control_v1",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared": SHARED, "distinct": DISTINCT},
              "registered_token_control": (
                  "Both behaviours read head noun number at an auxiliary in the SAME position, "
                  "so read slot is held fixed. Their answer vocabularies are DISJOINT "
                  "(was/were against have/has), so a direction that merely separates two answer "
                  "tokens cannot transfer between them. Matched slot with disjoint vocabulary is "
                  "the only configuration in this corpus where transfer tests the FEATURE."),
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
