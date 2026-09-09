#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v433: the same contrast INSIDE one construction -- removing the confound in my own result from an hour ago.

THE RESULT THIS TESTS IS MINE AND IT HAS A CONFOUND. v429 pooled five determiner mappings that all encode ONE variable
(grammatical number, realised on different nouns) and one rank-1 direction covered them at 0.965 to 1.011, mean 0.992.
v431 pooled five particle mappings that encode DIFFERENT variables and got 0.795 to 0.885, mean 0.829, with the first
sub-0.80 cell in fifteen pooled rungs. I reported the 0.164 gap as evidence that pooled coverage degrades with
variable diversity. But those two pools differ in TWO ways at once: variable-sameness and construction. A determiner
pool against a particle pool cannot separate them.
This rung runs both pools inside verb_particle, three cells each, same frame, same position, same P control, same
recipe, same rank:
    same_variable   up_down (woke/calmed), up_down_b (cheered/knuckled), up_down_c (teamed/wound)
                    -- three DIFFERENT cue pairs all setting the SAME variable, ` up` versus ` down`
    diff_variable   up_down (woke/calmed), out_down (found/settled), away_up (shied/sidled)
                    -- three cue pairs setting THREE different variables
up_down appears in both pools deliberately: it is the shared anchor, so any difference between the two groups cannot
come from which cells happen to be strong. Three cells per pool rather than five keeps the fits matched in size.
WHAT EACH OUTCOME MEANS. If same_variable covers all three and diff_variable does not, the v429/v431 contrast survives
the confound and the degradation is about variables. If both pools cover, the contrast was about the constructions and
my 19:11 ledger row overstates it -- which I would correct in the same terms I used to write it.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_same_variable_covers  every cell in the same-variable pool reaches joint extraction >= joint_min = 0.80.
                               v371 and v375 already merged these three under a pooled fit, so this is the arm I
                               expect to hold and it is the weaker half of the claim.                    prior 75%
  pred_b_diff_variable_resists at least one cell in the diff-variable pool falls BELOW 0.80. v431 found exactly one
                               of five below the bar, so at three cells this is genuinely uncertain.     prior 45%
  pred_c_gap                   the same-variable pool's MEAN joint extraction exceeds the diff-variable pool's by at
                               least gap = 0.10. This is the confound-free version of the 0.164 I measured across
                               constructions. Worked example: means of 0.99 and 0.84 give 0.15, TRUE; 0.97 and 0.92
                               give 0.05, FALSE, and the earlier gap was carried by the construction rather than the
                               variables.                                                                prior 50%
  pred_d_units_transfer        every cell in both pools reaches exact-set recovery >= unit_min = 0.80 under its
                               pool's greedy set, so a coverage difference cannot be blamed on the units.
                                                                                                        prior 55%
  pred_e_control_clean         own-C damage UB975 <= 0.01 on every cell in both pools. Both v429 and v431 FAILED this
                               (0.0193 and 0.0144) where per-cell directions run 0.001-0.008, so I expect it to fail
                               again and am registering it to keep that pattern on the record rather than to pass it.
                                                                                                        prior 25%
COUNTING. Nothing here is counted; all four distinct cells already count. What is at stake is whether a claim I made
an hour ago survives its own control.
Smoke: V433_SMOKE=<out.json> (CPU, V433_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v433_result.json"
GROUPS = {
    "same_variable": ("verb_particle_up_down", "verb_particle_up_down_b", "verb_particle_up_down_c"),
    "diff_variable": ("verb_particle_up_down", "verb_particle_out_down", "verb_particle_away_up"),
}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gap": 0.10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v433", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and all("error" not in g for g in G.values()) and set(G) == {"same_variable", "diff_variable"}

    def vals(k):
        g = G.get(k, {})
        if "error" in g or not g:
            return []
        return [s["joint_extraction"] for s in g["per_shape"].values() if s.get("joint_extraction") is not None]

    same, diff = vals("same_variable"), vals("diff_variable")
    a = ok and bool(same) and all(v >= B["joint_min"] for v in same)
    b = ok and bool(diff) and any(v < B["joint_min"] for v in diff)
    c = ok and bool(same) and bool(diff) and (sum(same) / len(same) - sum(diff) / len(diff)) >= B["gap"]
    d = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_same_variable_covers": bool(a), "pred_b_diff_variable_resists": bool(b),
            "pred_c_gap": bool(c), "pred_d_units_transfer": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V433_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V433_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v433",
              "candidate_id": "corpus.unit_broad_circuit_v433", "bars": BARS,
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
