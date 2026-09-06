#!/usr/bin/env python3
# BQGATE: frozen predictions; source direction, targets, floor and bars fixed before the run.
"""v11: how far does attn:11:head:03's number axis reach?

v10 (`unit_hub_head_axes_v10_result.json`) found the only shared direction among the hub heads:
11:03's block diff-in-means direction fit on lexical_number (were/was) serves perfect_number
(have/has) at 0.93-1.01 (|cos| 0.96) and drives coordination_agreement (and/or) and
quantifier_number (each/all) one-way. That is already not a token axis (four different answer
pairs). The open question is whether it is the model's ONE number variable: does the same
direction carry number where it is read off an ANTECEDENT for a possessive (their/his), and in the
existential frame (there were/was), and does it carry each behaviour's A2 (fresh construction)?
Source direction: fit on lexical_number's even A1 rows (fixed). Everything else is evaluation.

  targets (odd A1 rows and all A2 rows of each)
    lexical_number.pp_intervener (held-out self), perfect_number.have_vs_has,
    coordination_agreement.and_vs_or, quantifier_number.each_vs_all,
    existential_agreement.were_vs_was (terminal NULL at the module level -- 11:03 may not carry it),
    possessive_number.adjacent_antecedent (v3/v9 greedy sets never picked 11:03)
  per target (rows whose donor beats the base only; dropped rows counted): exact single-head 11:03 effect; the source direction's fraction and complement
  fraction (block-live); the target's OWN 11:03 direction (fit on its even rows) and |cos| to the
  source. A target counts only where the exact single-head effect >= 0.10 on those rows.

REGISTERED BEFORE THE RUN
    pred_a_possessive_reads_1103      possessive adjacent: exact single-head 11:03 effect on odd A1
                                      rows >= 0.10. Worked example: 0.14 -> True; 0.04 -> False.
    pred_b_axis_serves_possessive     pred_a holds AND the lexical_number direction gives >= 0.50 of
                                      that exact effect with complement <= 0.30. Worked example:
                                      exact 0.14, direction 0.09 (0.64), complement 0.02 -> True.
    pred_c_axis_serves_existential    existential counts (exact >= 0.10) AND the lexical direction
                                      gives >= 0.50 with complement <= 0.30. Worked example: exact
                                      0.05 -> False (does not count); exact 0.2, direction 0.15 -> True.
    pred_d_possessive_cos             |cos|(lexical direction, possessive's own 11:03 direction) >= 0.70.
                                      Worked example: 0.75 -> True; 0.3 -> False.
    pred_e_axis_serves_every_a2       on the A2 rows of every target that counts there, the lexical
                                      direction gives >= 0.50 with complement <= 0.30. Worked
                                      example: 5 counting targets, all >= 0.50 -> True; one at 0.41 -> False.

    Priors. a unsure (possessive's sets sit at layers 3-10, but 11:03 was never in the top-12 pool
    reported); b, d conditional on a, leaning True if the head carries it at all. c: existential was
    a module-level null, I expect it not to count (False by the floor). e expected True (v9: the
    number behaviours' A2 fractions were 0.97-1.15).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_lexical_number_pp as m_lexical
import circuit_fast_screen_candidate_perfect_number as m_perfect
import circuit_fast_screen_candidate_coordination_agreement as m_coord
import circuit_fast_screen_candidate_quantifier_number as m_quantifier
import circuit_fast_screen_candidate_existential as m_existential
import circuit_fast_screen_candidate_possessive_adjacent as m_possessive

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_number_axis_reach_v11_result.json"
HEAD = ["attn:11:head:03"]
TARGETS = {"lexical_number": m_lexical, "perfect_number": m_perfect, "coordination_agreement": m_coord,
           "quantifier_number": m_quantifier, "existential": m_existential, "possessive_adjacent": m_possessive}
FLOOR, SERVE, COMP_BAR, COS_BAR = 0.10, 0.50, 0.30, 0.70
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6400


def _plan():
    return {"candidate_id": "corpus.unit_number_axis_reach_v11", "source": "lexical_number even A1",
            "head": HEAD, "targets": list(TARGETS),
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _eval(backend, prep, q_src):
    b = g.block_direction_battery(backend, prep, HEAD, q_src)
    b["counts"] = b["exact_set"] >= FLOOR
    b["served"] = bool(b["counts"] and b["subspace_fraction"] is not None
                       and b["subspace_fraction"] >= SERVE and abs(b["complement_fraction"]) <= COMP_BAR)
    return b


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    src_fit = g.prepare(backend, g.rows_of(m_lexical, "A1")[0::2])
    q_src = g.block_diff_in_means(backend, src_fit, HEAD)
    key = next(iter(q_src))
    report = {}
    for name, module in TARGETS.items():
        a1 = g.rows_of(module, "A1")
        # valid_only: existential is a module-level null with donor-side capability failures on
        # some rows; the kernel refuses a non-positive donor denominator (v5 crashed the same way)
        fit, held = g.prepare(backend, a1[0::2], valid_only=True), g.prepare(backend, a1[1::2], valid_only=True)
        a2 = g.prepare(backend, g.rows_of(module, "A2"), valid_only=True)
        q_own = g.block_diff_in_means(backend, fit, HEAD)
        report[name] = {"rows_dropped": {"fit": fit.dropped, "heldout": held.dropped, "a2": a2.dropped},
                        "a1_heldout": _eval(backend, held, q_src), "a2": _eval(backend, a2, q_src),
                        "cos_to_source": float((q_own[key][:, 0] @ q_src[key][:, 0]).abs()),
                        "own_direction_a1_heldout": g.block_direction_battery(backend, held, HEAD, q_own)}
        r = report[name]
        print(name, json.dumps({"exact": round(r["a1_heldout"]["exact_set"], 3),
                                "src_frac": round(r["a1_heldout"]["subspace_fraction"] or 0, 3),
                                "comp": round(r["a1_heldout"]["complement_fraction"] or 0, 3),
                                "a2_exact": round(r["a2"]["exact_set"], 3),
                                "a2_src_frac": round(r["a2"]["subspace_fraction"] or 0, 3),
                                "cos": round(r["cos_to_source"], 3),
                                "own_frac": round(r["own_direction_a1_heldout"]["subspace_fraction"] or 0, 3)}),
              flush=True)

    poss, exist = report["possessive_adjacent"], report["existential"]
    a2_counting = {n: r["a2"] for n, r in report.items() if r["a2"]["counts"]}
    predictions = {
        'pred_a_possessive_reads_1103': poss["a1_heldout"]["counts"],
        'pred_b_axis_serves_possessive': poss["a1_heldout"]["served"],
        'pred_c_axis_serves_existential': exist["a1_heldout"]["served"],
        'pred_d_possessive_cos': poss["cos_to_source"] >= COS_BAR,
        'pred_e_axis_serves_every_a2': bool(a2_counting) and all(b["served"] for b in a2_counting.values()),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_number_axis_reach_result_v1",
              "candidate_id": "corpus.unit_number_axis_reach_v11", "semantics": "block_live",
              "head": HEAD, "source": "lexical_number.pp_intervener even A1 rows",
              "bars": {"floor": FLOOR, "serve": SERVE, "complement": COMP_BAR, "cos": COS_BAR},
              "a2_counting_targets": sorted(a2_counting), "targets": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
