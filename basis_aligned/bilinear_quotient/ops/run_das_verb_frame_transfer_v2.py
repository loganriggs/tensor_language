#!/usr/bin/env python3
# BQGATE: rank fixed at 1 before the run; thresholds and the confound gate registered before the run.
"""Is verb subcategorization ONE direction at resid:17, or several frame-specific ones?

Two behaviours in this corpus read "which frame does this verb open", and both screened selective
at the SAME site, resid:17:

    finiteness_selection.to_vs_that_v3   "refused tersely" -> ' to'  / "insisted tersely" -> ' that'
    verb_preposition.relied_vs_objected  "relied for years" -> ' on' / "objected for years" -> ' to'

The negation domain turned out to hold TWO near-orthogonal directions rather than one shared
feature (`das_polarity_reverse_transfer_v1`: 0.132 forward, 0.051 / 0.020 back). This asks the
same question of verb frames, bidirectionally, at rank 1.

A CONFOUND I AM REGISTERING RATHER THAN DISCOVERING AFTERWARD.
The two behaviours SHARE the token ' to'. It is finiteness's BASE answer and verb-preposition's
DONOR answer. A rank-1 direction that merely encodes "' to' versus not" could therefore produce
transfer between them with no shared verb-frame feature at all. This makes the two outcomes
asymmetric in what they license, so the gate is asymmetric too:

  LOW transfer both ways  -> frame_specific_directions is SAFE to read. A shared-token artefact
                             would INFLATE transfer, not deflate it, so a null survives the
                             confound. This is the same asymmetry that made the neither-axis
                             null readable.
  HIGH transfer           -> NOT readable as a shared verb-frame direction. It is reported as
                             INCONCLUSIVE_pending_token_control and requires a third verb-frame
                             behaviour whose vocabulary is disjoint from both. I am registering
                             that now so a high number cannot be claimed as the interesting result.

WHY THIS IS v2, AND WHY THE SITE MOVED.
v1 registered resid:17, the site both screens selected, and DIED at head verification with a max
absolute difference of 7.11. That failure was correct and mine. `circuit_das_subspace` reconstructs
logits from a captured site with `30*tanh(lm_head(rms_norm(x))/30)` and nothing else, which is only
valid at resid:18 -- the FINAL residual site, where the map to logits really is just the head. At
resid:17 a whole transformer block still has to run, so the reconstruction is meaningless. The
guard caught an invalid instrument before it produced a single number.

The site therefore moves to resid:18 for an INSTRUMENT reason, not because it gives a better
answer, and both behaviours are independently selective there: finiteness recovery 1.000 (P 0.066,
C 0.107) and verb-preposition recovery 1.000 (P 0.058, C 0.124). The question -- one direction or
two -- does not depend on which of the two carrying sites it is asked at. Everything else below,
including the shared-token confound gate, is unchanged from the v1 registration.

REGISTERED BEFORE THE RUN:
  RANK = 1 both ways. A null is not permission to raise rank.
  pred_c frame_specific_directions -> BOTH transfers <= 0.20
  pred_d shared_direction_candidate -> EITHER transfer >= 0.50 (reads INCONCLUSIVE, not a result)
  Between is INCONCLUSIVE.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_finiteness as fin
import circuit_fast_screen_candidate_verb_preposition as vp
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_verb_frame_transfer_v2_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
SPECIFIC, SHARED = 0.20, 0.50
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 2560


def _plan():
    return {"candidate_id": "verb_frame.transfer_v2", "site": SITE, "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _a1(m):
    return [r for r in m.build_rows() if r["family"] == "A1"]


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
    f_rows, v_rows = _a1(fin), _a1(vp)
    ok, worst = das.verify_head(backend, v_rows[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    fh, vh = len(f_rows) // 2, len(v_rows) // 2
    q_fin = _fit(backend, f_rows[:fh])
    q_vp = _fit(backend, v_rows[:vh])

    report = {
        "finiteness_heldout_reference": _apply(backend, f_rows[fh:], q_fin),
        "verb_preposition_heldout_reference": _apply(backend, v_rows[vh:], q_vp),
        "finiteness_direction_on_verb_preposition": _apply(backend, v_rows, q_fin),
        "verb_preposition_direction_on_finiteness": _apply(backend, f_rows, q_vp),
    }
    cosine = float(abs((q_fin[:, 0] * q_vp[:, 0]).sum()))

    t1 = report["finiteness_direction_on_verb_preposition"]["mean_recovery"]
    t2 = report["verb_preposition_direction_on_finiteness"]["mean_recovery"]
    worst_transfer = max(t1, t2)
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_both_fits_replicate": bool(
            min(report["finiteness_heldout_reference"]["mean_recovery"],
                report["verb_preposition_heldout_reference"]["mean_recovery"]) >= 0.50),
        "pred_c_frame_specific_directions": bool(worst_transfer <= SPECIFIC),
        "pred_d_shared_direction_candidate": bool(worst_transfer >= SHARED),
    }
    reading = ("INCONCLUSIVE_pending_token_control" if worst_transfer >= SHARED
               else "frame_specific_directions" if worst_transfer <= SPECIFIC
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_confound_test_result_v1",
              "candidate_id": "verb_frame.transfer_v2",
              "site": SITE, "rank": RANK,
              "registered_thresholds": {"frame_specific": SPECIFIC, "shared": SHARED},
              "registered_confound": (
                  "The two behaviours share the token ' to' (finiteness BASE answer, "
                  "verb-preposition DONOR answer). A shared-token direction would INFLATE "
                  "transfer, so a LOW transfer survives the confound and a HIGH transfer does "
                  "not; a high number is reported INCONCLUSIVE_pending_token_control and needs "
                  "a third verb-frame behaviour with disjoint vocabulary."),
              "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "abs_cosine_between_directions": cosine,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report,
                      "abs_cosine": cosine, "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
