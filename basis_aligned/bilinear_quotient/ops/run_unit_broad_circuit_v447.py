#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v447: TOKENS or CELLS? Seven behaviours and four behaviours spanning the SAME six readout tokens.

THE CONFOUND IN MY OWN RESULT, NAMED IN ITS OWN LEDGER ROW. v445 found that two five-cell preposition pools differ by
0.139 depending on how many distinct readout tokens they span -- five tokens gave mean 0.899 min 0.821 and covered,
ten tokens gave 0.760 and 0.731 and did not, with sites available in both. I wrote there that token count and cell
count are "not fully independent", and that is the weakness this rung removes. Every size series I have run varied
BOTH at once: adding a cell usually adds tokens.
So: two pools of DIFFERENT cell counts spanning the SAME number of tokens, found by exhaustive search over the 24
separable frame-A preposition members.
    seven_cells_six_tokens  about_against, about_into, from_about, into_with, on_about, on_into, with_against
                            -- SEVEN behaviours, SIX distinct tokens (about, against, into, from, with, on)
    four_cells_six_tokens   about_against, about_into, at_over, at_to
                            -- FOUR behaviours, SIX distinct tokens (about, against, into, at, over, to)
An arbitrary seven-cell preposition pool scored mean 0.718, min 0.576 at v441. If the token account is right, this
seven-cell pool -- same size, same construction, same budget, but six tokens instead of eleven -- should score far
better than that and about the same as the four-cell pool. If cell count is what matters, it should score like the
arbitrary seven and unlike the four.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_seven_cells_cover  the SEVEN-cell pool's MINIMUM joint extraction reaches joint_min = 0.80. No seven-cell
                            pool has done this: verb_particle managed 0.777 and the arbitrary preposition seven gave
                            0.576. If a seven-cell pool covers when its tokens are few, cell count is not the limit.
                                                                                                        prior 45%
  pred_b_beats_mixed_seven  the seven-cell pool's MEAN exceeds the arbitrary seven-cell pool's 0.718 by at least
                            gain = 0.05. Worked example: 0.85 gives 0.132, TRUE; 0.74 gives 0.022, FALSE and the
                            token account does not survive its own confound.                             prior 55%
  pred_c_sizes_match        the seven-cell and four-cell MEANS agree within tol_pool = 0.05. This is the strong form:
                            with tokens held equal, nearly doubling the number of behaviours should not matter. It
                            can fail while pred_b passes, which would mean tokens explain PART of the effect and cell
                            count the rest -- a third outcome worth distinguishing rather than collapsing.
                                                                                                        prior 40%
  pred_d_units_available    both pools reach minimum pooled-unit recovery >= unit_min = 0.80, so any difference is the
                            direction and not the selection.                                            prior 80%
  pred_e_control_clean      own-C damage UB975 <= 0.01 on every cell in both pools. It failed in v445's covering pool
                            at 0.0128.                                                                   prior 45%
COUNTING. Nothing here is counted; all nine cells already count separately. What is at stake is whether the account I
put on the board an hour ago -- that a shared direction fails on distinct readout tokens rather than on the number of
behaviours -- survives the one confound I flagged in it myself.
Smoke: V447_SMOKE=<out.json> (CPU, V447_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v447_result.json"
SEVEN = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_from_about",
         "verb_preposition_into_with", "verb_preposition_on_about", "verb_preposition_on_into",
         "verb_preposition_with_against")      # SEVEN cells, SIX distinct readout tokens
FOUR = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_at_over",
        "verb_preposition_at_to")              # FOUR cells, SIX distinct readout tokens
GROUPS = {"seven_cells_six_tokens": SEVEN, "four_cells_six_tokens": FOUR}
RELAXED = "ALL"
PRIOR = {"prep_7_mixed": (0.718, 0.576), "prep_5_ten_tokens": (0.760, 0.731), "prep_5_five_tokens": (0.899, 0.821)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v447", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and all("error" not in g for g in G.values()) and set(G) == set(GROUPS)

    def stats(k):
        g = G.get(k, {})
        if not g or "error" in g:
            return None, None, None
        v = [s["joint_extraction"] for s in g["per_shape"].values() if s.get("joint_extraction") is not None]
        u = [s["units_recovery"] for s in g["per_shape"].values() if s.get("units_recovery") is not None]
        return (sum(v) / len(v) if v else None, min(v) if v else None, min(u) if u else None)

    m7, n7, u7 = stats("seven_cells_six_tokens")
    m4, n4, u4 = stats("four_cells_six_tokens")
    a = ok and n7 is not None and n7 >= B["joint_min"]
    b = ok and m7 is not None and (m7 - PRIOR["prep_7_mixed"][0]) >= B["gain"]
    c = ok and m7 is not None and m4 is not None and abs(m7 - m4) <= B["tol_pool"]
    d = ok and all(u is not None and u >= B["unit_min"] for u in (u7, u4))
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_seven_cells_cover": bool(a), "pred_b_beats_mixed_seven": bool(b), "pred_c_sizes_match": bool(c),
            "pred_d_units_available": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V447_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V447_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            tgt, mg = (0.97, 0.001)          # every pool in this rung uses the relaxed budget
            singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=tgt,
                                                     min_gain=mg, max_units=max_units)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v447",
              "candidate_id": "corpus.unit_broad_circuit_v447", "bars": BARS,
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
