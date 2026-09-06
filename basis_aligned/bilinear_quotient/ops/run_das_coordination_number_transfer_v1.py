#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds registered before the run.
"""Does CONSTRUCTED subject number share a direction with ANTECEDENT number?

`coordination_agreement.and_vs_or` screened selective at resid:17 (recovery 1.002, P 0.082,
C 0.165) and is also fully carried at resid:18 (1.000). Its variable is unusual for this corpus:
BOTH coordinated nouns are singular in every row, so no token carries the subject's number. The
connective builds it. That makes it the first behaviour here whose causal variable is composed
rather than copied from a token.

The natural question is whether that composed number is the SAME feature the model uses for
number it can read off a token.

A REGISTERED SUBSTITUTION, DISCLOSED RATHER THAN MADE SILENTLY.
The receipt for `coordination_agreement` registered this transfer against
`subject_verb.number_agreement`. That module is unusable: it pins an immutable task dossier whose
hash has since drifted, and it belongs to the other lane so I am not editing it. The other
were/was behaviour, `existential_agreement.were_vs_was`, is a NULL with no selective site, and the
DAS precondition is a terminal selective_causal_site receipt. I have therefore substituted
`possessive_number.adjacent_antecedent`, which is selective at resid:18 and whose variable is
number carried from an antecedent.

The substitution is in one way BETTER than the registered counterpart: the two vocabularies are
disjoint (' were'/' was' against ' their'/' his'), so the shared-token confound that forced an
asymmetric gate in `das_verb_frame_transfer_v2` does not exist here. A transfer either way cannot
be manufactured by a token axis the two behaviours share, because they share none.

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_d shared_number_direction    -> EITHER transfer >= 0.50
  pred_e distinct_number_directions -> BOTH transfers <= 0.20
  Between is INCONCLUSIVE.

  Stated prior: two domains have now come back as separate directions (negation cosine 0.165,
  verb frames 0.172), so the base rate in this corpus favours DISTINCT. But those were pairs of
  behaviours reading different variables in one domain; this is one variable -- number -- reached
  by two different routes, which is the case where sharing is most plausible. I genuinely do not
  know, and that is the reason to run it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_coordination_agreement as coord
import circuit_fast_screen_candidate_possessive_adjacent as poss
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_coordination_number_transfer_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
CARRY, SPARE = 0.50, 0.20
SHARED, DISTINCT = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 48, 3072


def _plan():
    return {"candidate_id": "coordination_agreement.das_rank1_transfer_v1", "site": SITE,
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
    q_poss = _fit(backend, p1[:ph])

    report = {
        "coordination_a1_heldout": _apply(backend, c1[ch:], q_coord),
        "coordination_a2_report_frame": _apply(backend, _fam(coord, "A2"), q_coord),
        "possessive_a1_heldout": _apply(backend, p1[ph:], q_poss),
        "coordination_direction_on_possessive": _apply(backend, p1, q_coord),
        "possessive_direction_on_coordination": _apply(backend, c1, q_poss),
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

    cosine = float(abs((q_coord[:, 0] * q_poss[:, 0]).sum()))
    t1 = report["coordination_direction_on_possessive"]["mean_recovery"]
    t2 = report["possessive_direction_on_coordination"]["mean_recovery"]
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
              "candidate_id": "coordination_agreement.das_rank1_transfer_v1",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"carry": CARRY, "spare": SPARE,
                                        "shared": SHARED, "distinct": DISTINCT},
              "registered_substitution": (
                  "Receipt registered subject_verb.number_agreement; that module pins an immutable "
                  "dossier whose hash has drifted and belongs to the other lane, and the other "
                  "were/was behaviour existential_agreement.were_vs_was is a null with no selective "
                  "site. Substituted possessive_number.adjacent_antecedent, which is selective at "
                  "resid:18 and carries antecedent number. Vocabularies are DISJOINT, so no "
                  "shared-token confound exists here."),
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
