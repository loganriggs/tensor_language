#!/usr/bin/env python3
# BQGATE: rank stays at 1; thresholds registered before the run.
"""Does the number direction still move the answer where the model FAILS?

`das_possessive_number_resid18_rank1_v1` found a rank-1 direction at resid:18 carrying antecedent
number across four matched configurations (minimum transfer 0.694). Two sibling configurations
FAIL natively — an animate number-mismatched attractor, and a particle-final prediction site —
and both were excluded from that run because their base-to-donor separation is unreliable, so a
recovery RATIO on them would be noise dressed as evidence.

The question they raise is still well posed, and answerable with a measure that does not divide by
a degenerate denominator: **does patching along the direction still MOVE the answer on those rows,
by as much as it does where the model succeeds?**

    effect = |margin_patched - margin_base| / scale

with `scale` the passing configurations' median native separation — the same quantity the kernel
uses for same-answer families. That is defined whether or not the model gets the row right.

Two readings, and they say different things about where the failure lives:

  * the direction still moves the answer as much as it does in passing frames -> the number
    representation at resid:18 survives the attractor, and the native failure is elsewhere: the
    model has the information and does not use it.
  * the direction's effect collapses too -> the attractor degrades the representation at resid:18
    itself.

REGISTERED BEFORE THE RUN:
  RANK = 1, unchanged; this is a probe with the same fit, not a new one.
  pred_c representation_intact     -> failing-config effect >= 0.50 x the passing reference
  pred_e representation_disrupted  -> <= 0.20 x the passing reference
  Between is INCONCLUSIVE and reported as such.
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
import circuit_fast_screen_candidate_possessive_attractor as attractor      # fails natively
import circuit_fast_screen_candidate_possessive_number as particle_final    # fails natively

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_possessive_attractor_probe_v1_result.json"
SITE, RANK, STEPS = "resid:18", 1, 300
INTACT, DISRUPTED = 0.50, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 32, 1536


def _plan():
    return {"candidate_id": "possessive_number.das_attractor_probe_v1", "site": SITE, "rank": RANK,
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
    a1 = _a1(adjacent)
    ok, worst = das.verify_head(backend, a1[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    fit_rows, held_rows = a1[: len(a1) // 2], a1[len(a1) // 2:]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    base_h, donor_h, _ = das.capture_site(backend, held_rows, SITE)
    scale = das.target_scale(backend, base_h, donor_h,
                             [r["donor_answer_id"] for r in held_rows],
                             [r["donor_foil_id"] for r in held_rows])

    def effect(rows):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        value, n = das.subspace_same_answer_effect(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows], scale)
        return {"absolute_effect": value, "rows": n}

    report = {
        "passing_adjacent_heldout": effect(held_rows),
        "passing_medial": effect(_a1(medial)),
        "failing_animate_attractor": effect(_a1(attractor)),
        "failing_particle_final": effect(_a1(particle_final)),
        "target_scale": scale,
    }
    reference = (report["passing_adjacent_heldout"]["absolute_effect"]
                 + report["passing_medial"]["absolute_effect"]) / 2.0
    ratios = {k: report[k]["absolute_effect"] / reference
              for k in ("failing_animate_attractor", "failing_particle_final")}
    report["passing_reference"] = reference
    report["ratio_to_passing_reference"] = ratios
    worst_ratio = min(ratios.values())

    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_reference_is_material": bool(reference > 0.2),
        "pred_c_representation_intact": bool(worst_ratio >= INTACT),
        "pred_d_both_failing_configs_agree": bool(
            abs(ratios["failing_animate_attractor"] - ratios["failing_particle_final"]) < 0.25),
        "pred_e_representation_disrupted": bool(worst_ratio <= DISRUPTED),
    }
    reading = ("representation_intact_failure_is_elsewhere" if worst_ratio >= INTACT
               else "representation_disrupted_at_site" if worst_ratio <= DISRUPTED
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_failing_config_probe_v1",
              "candidate_id": "possessive_number.das_attractor_probe_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "possessive_number.adjacent_antecedent A1",
              "registered_thresholds": {"intact": INTACT, "disrupted": DISRUPTED},
              "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "ratios": ratios, "families": report}, indent=2))


if __name__ == "__main__":
    main()
