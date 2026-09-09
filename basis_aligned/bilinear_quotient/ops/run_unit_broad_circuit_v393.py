#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v393: NINE DISCARDED CELLS, RE-ASKED. Does a pooled direction cover the shape variants that missed a battery row?

WHAT THIS IS CASHING IN. v379 showed that two shapes which missed rows 2 and 4 of their own battery are covered when
pooled with shapes that passed (1.043 each), and v391 showed the same holds when they are held OUT of the fit
entirely (0.939 and 1.027, against 0.823 for a battery-PASSING held-out shape -- the ordering inverted). A scan of
every battery receipt on disk then found 186 cells that missed at least one row, 134 that missed only rows 2 and/or 4,
and -- the number that actually matters -- ELEVEN of those that are SHAPE VARIANTS of an existing mapping, which is
the only class a joint fit has any reason to carry. Two of the eleven are already settled by v391. This rung takes the
remaining nine, one group per (cue -> token) mapping, each group being that mapping's frame-A cell plus the variant
that missed:
    woke/calmed + fk (causal frame, row 4)            raved/delved + fb (row 4)
    cheered/knuckled + fc (row 2)                     scoffed/agonised + fc (row 4)
    faded/livened + fc (row 2)                        depended/complained + fb (row 4)
    hunkered/caved + fc (row 4)                       embarked/plunged + fc (row 2)
Eight groups; the ninth of the nine is fb_up_down, which is already inside the eleven-shape pool of v387 and needs
nothing. Every one of these variants cleared its CPU capability floor when authored, so the model can do the task in
that frame; what failed was a per-shape fit on sixteen rows.
WHAT A PASS BUYS, STATED HONESTLY: breadth on entries that ALREADY count, not new circuits. Eight groups covered
would raise eight entries from breadth 1 or 2 to breadth 2 or 3 and would retire nine receipts currently filed as
misses. It does not move the count, and I am registering that here so a good result cannot be read as growth.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_missers_covered   in at least k_groups = 6 of the 8 groups, the variant that MISSED a battery row reaches
                           joint extraction >= joint_min = 0.80. This is the claim. Worked example: six variants at
                           0.9-1.0 and two at 0.6 gives TRUE at exactly the bar; five and three gives FALSE.
                                                                                                        prior 65%
  pred_b_both_shapes_covered  in at least 6 of the 8, BOTH shapes are covered -- the pooled direction must not buy
                           the variant by giving up the frame-A cell that already counts.               prior 70%
  pred_c_control_clean     in at least 6 of the 8, own-C damage UB975 <= 0.01 on both shapes.           prior 60%
  pred_d_no_error          every group returns a receipt; an error fails this predicate rather than being dropped.
                                                                                                        prior 90%
  pred_e_some_misser_resists  at least ONE variant is NOT covered. Registered because ten rungs have now found
                           nothing below 0.80 and a rung that cannot fail is not worth the GPU; if every one of nine
                           discarded cells is covered, the row-2/row-4 miss on a shape variant carries no information
                           at all and I should say so in the ledger rather than keep filing them as misses.
                                                                                                        prior 55%
COUNTING. Nothing here is counted.
Smoke: V393_SMOKE=<out.json> (CPU, V393_SMOKE_ROWS=4 per split).
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_broad_circuit_v393_result.json"
# each group is ONE cue -> token mapping: its frame-A cell plus the shape variant that MISSED a battery row
GROUPS = {
    "woke_calmed":        ("verb_particle_up_down", "verb_particle_fk_up_down"),
    "cheered_knuckled":   ("verb_particle_up_down_b", "verb_particle_fc_up_down_b"),
    "faded_livened":      ("verb_particle_away_up_b", "verb_particle_fc_away_up_b"),
    "hunkered_caved":     ("verb_particle_down_in", "verb_particle_fc_down_in"),
    "raved_delved":       ("verb_preposition_about_into", "verb_preposition_fb_about_into"),
    "scoffed_agonised":   ("verb_preposition_at_over", "verb_preposition_fc_at_over"),
    "depended_complained":("verb_preposition_on_about", "verb_preposition_fb_on_about"),
    "embarked_plunged":   ("verb_preposition_on_into", "verb_preposition_fc_on_into"),
}
MISSER = {g: cells[1] for g, cells in GROUPS.items()}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "k_groups": 6}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v393", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and all("error" not in g for g in G.values())
    covered = [k for k, g in G.items() if "error" not in g
               and (g["per_shape"].get(MISSER[k], {}).get("joint_extraction") or 0) >= B["joint_min"]]
    both = [k for k, g in G.items() if "error" not in g
            and all(s.get("joint_extraction") is not None and s["joint_extraction"] >= B["joint_min"]
                    for s in g["per_shape"].values())]
    units_ok = [k for k, g in G.items() if "error" not in g
                and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"]
                        for s in g["per_shape"].values())]
    clean = [k for k, g in G.items() if "error" not in g
             and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"] for s in g["per_shape"].values())]
    return {"pred_a_missers_covered": bool(ok and len(covered) >= B["k_groups"]),
            "pred_b_both_shapes_covered": bool(ok and len(both) >= B["k_groups"]),
            "pred_c_control_clean": bool(ok and len(clean) >= B["k_groups"]),
            "pred_d_no_error": bool(ok),
            "pred_e_some_misser_resists": bool(ok and len(covered) < len(G))}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V393_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V393_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]
    V = dict(valid_only=True)

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float()
                                for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0)
                for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)

    groups = {}
    for gname, cells in GROUPS.items():
        try:
            pooled_fit, pooled_c_fit, per_shape_rows = [], [], {}
            for cell in cells:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                pooled_fit += cut(fit_half(a1))
                pooled_c_fit += cut(fit_half(cc))
            P_fit = g.prepare(backend, pooled_fit, **V)
            P_cfit = g.prepare(backend, pooled_c_fit)
            singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=TARGET,
                                                     min_gain=MIN_GAIN, max_units=max_units)
            units = list(greedy["chosen"])
            mu_joint = mu_of(P_fit, units)
            q_joint, hist = g.fit_block_subspace_constrained(
                backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
                controls=(P_cfit,), control_weight=LAM, mu=mu_joint)
            per_shape = {}
            for cell in cells:
                held_rows, c_rows = per_shape_rows[cell]
                p_held = g.prepare(backend, held_rows, **V)
                p_c = g.prepare(backend, c_rows)
                e_exact = ext(p_held, units)
                cdmg = dmg(p_c, units, q_joint, mu_joint)
                per_shape[cell] = {
                    "units_recovery": e_exact,
                    "joint_extraction": round(ext(p_held, units, q=q_joint) / e_exact, 3) if abs(e_exact) > 1e-6 else None,
                    "A1_damage": dmg(p_held, units, q_joint, mu_joint),
                    "C_damage": cdmg, "C_ub975": cdmg["ce_ub975"], "n_held": len(p_held.base_batch.row_ids)}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v393",
              "candidate_id": "corpus.unit_broad_circuit_v393", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                         "lam": LAM, "steps": steps, "lr": LR, "complement_weight": CW},
              "groups": groups, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "groups": {k: (v.get("error") or {c: s["joint_extraction"] for c, s in v["per_shape"].items()})
                                 for k, v in groups.items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
