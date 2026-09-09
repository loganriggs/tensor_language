#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v451: are the two factors ADDITIVE? The token effect measured again at a different cell count.

WHAT IS ESTABLISHED. Two factors move pooled coverage, each measured with the other held fixed and the sites
available:
    TOKENS at five cells   five tokens 0.899 / ten tokens 0.760      gap 0.139   (v445)
    CELLS at six tokens    four 1.004, five 0.913, six 0.786, seven 0.692        (v449, 5/5)
The token effect has been measured at exactly ONE cell count. If the two factors are additive, the same token
manipulation at FOUR cells should produce a similar gap; if they interact, the gap should change with the number of
behaviours -- and an interaction would mean neither number can be quoted without the other, which matters for any
merge rule that tries to use them.
Two four-cell preposition pools, found by exhaustive search over the 24 separable frame-A members, sharing one member
so a difference cannot come from one pool holding a strong cell:
    cells4_tok4  about_against, about_into, into_with, with_against   -- FOUR distinct tokens (each used twice)
    cells4_tok8  about_against, at_over, by_from, for_toward          -- EIGHT distinct tokens (each used once)
v449's four-cell six-token pool sat at mean 1.004, between these two token counts, so it provides a mid-point already
measured at this cell count. Rank stays at 1; budget stays relaxed.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_low_tokens_cover   the four-token pool's MINIMUM joint extraction reaches joint_min = 0.80.   prior 85%
  pred_b_high_tokens_cover  the eight-token pool's MINIMUM also reaches 0.80. Registered as a SEPARATE clause from
                            pred_a rather than as its negation, because at four cells the token effect may shift the
                            mean without pushing any cell under the bar -- and that is a real outcome, not a
                            half-failure.                                                                prior 55%
  pred_c_token_gap_present  the four-token pool's MEAN exceeds the eight-token pool's by at least gain = 0.05, i.e.
                            the token effect reproduces at all outside the five-cell case where it was found.
                            Worked example: 1.00 against 0.90 gives 0.10, TRUE.                          prior 65%
  pred_d_units_available    both pools reach minimum pooled-unit recovery >= unit_min = 0.80.            prior 85%
  pred_e_gap_matches_five_cell  the gap measured here agrees with the five-cell gap of 0.139 within tol_pool = 0.05.
                            This is the additivity claim stated as a number. Worked example: a gap of 0.11 differs by
                            0.029, TRUE and the factors add; a gap of 0.02 differs by 0.119, FALSE and they interact,
                            in which case neither factor can be quoted alone.                            prior 45%
COUNTING. Nothing here is counted; all seven cells already count separately. What is at stake is whether the two
constraints the corpus now has can be stated independently or only jointly.
Smoke: V451_SMOKE=<out.json> (CPU, V451_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v451_result.json"
# FOUR cells throughout; only the number of distinct readout tokens changes
LO4 = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_into_with",
       "verb_preposition_with_against")            # 4 cells, FOUR distinct tokens
HI4 = ("verb_preposition_about_against", "verb_preposition_at_over", "verb_preposition_by_from",
       "verb_preposition_for_toward")              # 4 cells, EIGHT distinct tokens
GROUPS = {"cells4_tok4": LO4, "cells4_tok8": HI4}
RELAXED = "ALL"
PRIOR = {"cells5_tok5": 0.899, "cells5_tok10": 0.760, "cells4_tok6": 1.004}   # gap at five cells was 0.139
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v451", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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

    ml, nl, ul = stats("cells4_tok4")
    mh, nh, uh = stats("cells4_tok8")
    gap5 = PRIOR["cells5_tok5"] - PRIOR["cells5_tok10"]
    a = ok and nl is not None and nl >= B["joint_min"]
    b = ok and nh is not None and nh >= B["joint_min"]
    c = ok and ml is not None and mh is not None and (ml - mh) >= B["gain"]
    d = ok and all(u is not None and u >= B["unit_min"] for u in (ul, uh))
    e = ok and ml is not None and mh is not None and abs((ml - mh) - gap5) <= B["tol_pool"]
    return {"pred_a_low_tokens_cover": bool(a), "pred_b_high_tokens_cover": bool(b),
            "pred_c_token_gap_present": bool(c), "pred_d_units_available": bool(d),
            "pred_e_gap_matches_five_cell": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V451_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V451_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v451",
              "candidate_id": "corpus.unit_broad_circuit_v451", "bars": BARS,
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
