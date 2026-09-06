#!/usr/bin/env python3
# BQGATE: rank fixed at 1 in advance; cross-behaviour transfer thresholds registered before the run.
"""Does ONE direction carry both correlative pairs, or are they two lexical associations?

`das_correlative_pair_resid18_rank1_v2` found a single direction at resid:18 that carries the
`both`/`neither` correlative: 0.980 held-out, 0.821 across constructions, P 0.053, C 0.001.

That leaves the question the result was built to raise. `correlative_state.either_vs_neither` is a
different pair with a different second element and a disjoint answer vocabulary (` or`/` nor`
against ` and`/` nor`). If the SAME direction carries it, the model has a correlative-state
feature. If it does not, it has two separately-learned lexical associations that happen to screen
alike.

The test fits the direction on `both`/`neither` only, then evaluates it, untouched, on
`either`/`neither` rows — no refitting, no per-behaviour tuning.

REGISTERED BEFORE THE RUN:
  RANK = 1, as before. A null is not permission to raise it.
  pred_c cross-behaviour transfer >= 0.50  -> one shared correlative-state feature
  pred_e separate-associations reading     -> transfer <= 0.15
  Between 0.15 and 0.50 is INCONCLUSIVE and will be reported as such rather than rounded to
  whichever reading is more interesting.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_correlative_pair as fitted      # both / neither
import circuit_fast_screen_candidate_correlative_state as probed     # either / neither
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_correlative_shared_subspace_v1_result.json"
SITE = "resid:18"
RANK = 1
STEPS = 300
SHARED_THRESHOLD = 0.50
SEPARATE_THRESHOLD = 0.15
MODEL_FORWARDS_MAX = 16
EXAMPLE_EVALUATIONS_MAX = 768


def _plan() -> dict:
    return {"candidate_id": "correlative.shared_subspace_das_v1", "site": SITE, "rank": RANK,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": RANK * 1152,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _fam(rows, name):
    return [r for r in rows if r["family"] == name]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True))
        return

    backend = producer.Bilin18TorchBackend.load("cuda")
    fit_rows_all = _fam(fitted.build_rows(), "A1")
    probe_rows = probed.build_rows()

    ok, worst = das.verify_head(backend, fit_rows_all[:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED ({worst:.6f})")

    fit_rows = fit_rows_all[: len(fit_rows_all) // 2]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS)

    report = {}
    # within-behaviour sanity, then the cross-behaviour question
    for label, rows in (("fitted_A1_heldout", fit_rows_all[len(fit_rows_all) // 2:]),
                        ("probed_A1", _fam(probe_rows, "A1")),
                        ("probed_A2", _fam(probe_rows, "A2"))):
        base, donor, _ = das.capture_site(backend, rows, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rows], [r["donor_foil_id"] for r in rows])
        report[label] = {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    transfer = report["probed_A1"]["mean_recovery"]
    predictions = {
        "pred_a_head_reproduces_producer": bool(ok),
        "pred_b_fitted_behaviour_replicates": bool(report["fitted_A1_heldout"]["mean_recovery"] >= 0.5),
        "pred_c_shared_correlative_feature": bool(transfer >= SHARED_THRESHOLD),
        "pred_d_cross_construction_on_probed": bool(report["probed_A2"]["mean_recovery"] >= SHARED_THRESHOLD),
        "pred_e_separate_lexical_associations": bool(transfer <= SEPARATE_THRESHOLD),
    }
    reading = ("shared_feature" if transfer >= SHARED_THRESHOLD
               else "separate_associations" if transfer <= SEPARATE_THRESHOLD
               else "INCONCLUSIVE_between_registered_thresholds")

    result = {**predictions, "predictions": predictions,
              "schema": "circuit_das_shared_subspace_result_v1",
              "candidate_id": "correlative.shared_subspace_das_v1",
              "site": SITE, "rank": RANK,
              "fitted_on": "correlative_pair.both_vs_neither (and/nor)",
              "probed_on": "correlative_state.either_vs_neither (or/nor)",
              "registered_thresholds": {"shared": SHARED_THRESHOLD, "separate": SEPARATE_THRESHOLD},
              "reading": reading,
              "head_verification": {"passed": ok, "max_abs_difference": worst},
              "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "families": report, "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
