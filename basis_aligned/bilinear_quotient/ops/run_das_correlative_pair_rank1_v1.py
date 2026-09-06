#!/usr/bin/env python3
# BQGATE: rank is fixed at 1 in advance; a null is not permission to raise it.
"""DAS at resid:18 for `correlative_pair.both_vs_neither`, rank fixed at 1.

Registered before running (ops/README.md, "DAS follow-up on localized circuits"):

  RANK = 1, chosen in advance. A null at rank 1 is NOT permission to raise the rank.
  PREDICTION: if the open correlative is carried by a low-dimensional feature at resid:18, a
  single learned direction recovers a substantial fraction of the whole-site effect on HELD-OUT
  rows, and it does so for A2 as well as A1 while leaving P and C near zero. If a single
  direction recovers A1 but not A2, it is a construction-specific direction rather than a
  carrier. If P or C moves with it, it is not selective and the result is withdrawn.

Target chosen because it is the best-conditioned circuit in the corpus: P 0.024 and C 0.053 at
the selected site, the lowest of both, so there is least competing structure to confuse an
alignment.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_candidate_correlative_pair as candidate
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/das_correlative_pair_resid18_rank1_v1_result.json"
SITE = "resid:18"
RANK = 1
STEPS = 300


# Six capture passes (base+donor for A1-fit, A1-heldout, A2, P, C) over 128 rows; the fit itself
# runs only the final head, so it costs no transformer forward.
MODEL_FORWARDS_MAX = 12
EXAMPLE_EVALUATIONS_MAX = 512


def _plan(rows) -> dict:
    return {
        "candidate_id": "correlative_pair.both_vs_neither.das_resid18_rank1_v1",
        "site": SITE, "rank": RANK, "steps": STEPS,
        "row_count": len(rows),
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,          # backward passes touch the final head only, not the model
        "model_updates": 0,
        "fit_parameters": RANK * 1152,  # the learned basis, discarded after evaluation
        "gpu_accessed": False,
        "model_loaded": False,
        "execution_policy": "managed_queue_only",
    }


def main() -> None:
    rows = candidate.build_rows()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True))
        return
    backend = producer.Bilin18TorchBackend.load("cuda")
    families = {f: [r for r in rows if r["family"] == f] for f in ("A1", "A2", "P", "C")}

    ok, worst = das.verify_head(backend, families["A1"][:8], SITE)
    if not ok:
        raise SystemExit(f"HEAD VERIFICATION FAILED (max abs diff {worst:.5f}) - nothing below is trustworthy")

    a1 = families["A1"]
    fit_rows, held_rows = a1[: len(a1) // 2], a1[len(a1) // 2 :]
    base_f, donor_f, _ = das.capture_site(backend, fit_rows, SITE)
    q = das.fit_subspace(
        backend, base_f, donor_f,
        [r["donor_answer_id"] for r in fit_rows], [r["donor_foil_id"] for r in fit_rows],
        rank=RANK, steps=STEPS,
    )

    report = {}
    for name, rs in (("A1_heldout", held_rows), ("A2", families["A2"]),
                     ("P", families["P"]), ("C", families["C"])):
        base, donor, _ = das.capture_site(backend, rs, SITE)
        mean, absmean, n = das.subspace_recovery(
            backend, base, donor, q,
            [r["donor_answer_id"] for r in rs], [r["donor_foil_id"] for r in rs])
        report[name] = {"mean_recovery": mean, "mean_absolute_recovery": absmean, "rows": n}

    # Registered predictions, evaluated from the numbers above. Thresholds fixed in advance.
    a1 = report["A1_heldout"]["mean_absolute_recovery"]
    a2 = report["A2"]["mean_absolute_recovery"]
    pv = report["P"]["mean_absolute_recovery"]
    cv = report["C"]["mean_absolute_recovery"]
    predictions = {
        # a: the differentiable head is the model's own head
        "pred_a_head_reproduces_producer": bool(ok),
        # b: one direction carries the variable on rows the fit never saw
        "pred_b_heldout_a1_transfer": bool(a1 >= 0.5),
        # c: it is a carrier, not a construction-specific direction
        "pred_c_cross_construction_transfer": bool(a2 >= 0.5),
        # d: it is selective - the same subspace leaves the invariance edit and the control alone
        "pred_d_subspace_selective": bool(pv <= 0.2 and cv <= 0.2),
    }

    result = {
        **predictions,
        "predictions": predictions,
        "schema": "circuit_das_subspace_result_v1",
        "candidate_id": "correlative_pair.both_vs_neither.das_resid18_rank1_v1",
        "site": SITE, "rank": RANK, "steps": STEPS,
        "head_verification": {"passed": ok, "max_abs_difference": worst},
        "fit_rows": len(fit_rows), "families": report,
        "whole_site_reference": {"A1_recovery": 1.000, "P": 0.024, "C": 0.053},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["families"], indent=2))
    print("head verification max abs diff:", worst)


if __name__ == "__main__":
    main()
