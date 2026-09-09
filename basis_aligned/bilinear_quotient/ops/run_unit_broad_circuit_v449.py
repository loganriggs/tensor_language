#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v449: the pure CELL-COUNT curve, with distinct readout tokens held at six throughout.

WHY THE ENDPOINTS ARE NOT ENOUGH. v445 showed token count matters at fixed cell count (five tokens 0.899, ten tokens
0.760, gap 0.139) and v447 showed cell count matters at fixed token count (four cells 0.980, seven cells 0.692, gap
0.288), which refuted the exclusive clause I had attached to v445 and left both factors standing. What v447 gives is
two endpoints. A merge rule needs the SHAPE: whether coverage holds flat and then falls, or declines from the start,
decides whether "up to five members" is a safe region or the middle of a slope.
Four NESTED pools, every one spanning EXACTLY six distinct readout tokens -- about, against, from, into, on, with --
so the token count is constant while the number of behaviours goes four, five, six, seven:
    cells4_tok6  about_against, about_into, from_about, into_with
    cells5_tok6  + on_about
    cells6_tok6  + on_into
    cells7_tok6  + with_against        (this is v447's seven-cell pool, mean 0.692, min 0.506)
Nested, so a difference cannot come from which cells were drawn, and the seven-cell endpoint is a direct reproduction
check against v447. Note that cells4_tok6 here is NOT v447's four-cell pool -- that one used at_over and at_to for its
last two tokens -- so the four-cell number is an independent second measurement at six tokens rather than a repeat.
Rank stays at 1; budget stays relaxed.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_covers        the four-cell pool's MINIMUM joint extraction reaches joint_min = 0.80. v447's other
                            four-cell six-token pool gave 0.937, so a failure here would mean the four-cell result
                            depended on which cells were chosen and the whole curve needs re-reading. prior 80%
  pred_b_seven_resists      the seven-cell pool's MINIMUM falls below 0.80, reproducing v447's 0.506. This is the
                            instrument check at the other end.                                          prior 85%
  pred_c_monotone_within_tol  the means do not INCREASE as cells are added, allowing tol_pool = 0.05 of slack at each
                            step: mean(4) >= mean(5) - 0.05 >= mean(6) - 0.10 >= mean(7) - 0.15 pairwise. Registered
                            because the standard-budget series at v435 looked monotone and the relaxed one was flat
                            from five to six, so "monotone" has already misled me once tonight and this states it
                            with slack rather than as a strict order.                                    prior 60%
  pred_d_units_available    every pool reaches minimum pooled-unit recovery >= unit_min = 0.80, so the curve is the
                            direction and not the selection.                                            prior 85%
  pred_e_endpoint_gap       mean(4) - mean(7) >= gain = 0.05, i.e. the cell-count effect reproduces at all in this
                            nested series. Worked example: 0.98 against 0.69 gives 0.29, TRUE.          prior 85%
COUNTING. Nothing here is counted; all seven cells already count separately. What is at stake is the SHAPE of the
constraint that every merge rule in the corpus depends on.
Smoke: V449_SMOKE=<out.json> (CPU, V449_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v449_result.json"
# nested pools all spanning EXACTLY SIX distinct readout tokens: about, against, from, into, on, with
P7 = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_from_about",
      "verb_preposition_into_with", "verb_preposition_on_about", "verb_preposition_on_into",
      "verb_preposition_with_against")
GROUPS = {"cells4_tok6": P7[:4], "cells5_tok6": P7[:5], "cells6_tok6": P7[:6], "cells7_tok6": P7}
RELAXED = "ALL"
PRIOR = {"cells7_tok6": (0.692, 0.506), "cells4_tok6_other": (0.980, 0.937)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v449", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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

    S = {k: stats(k) for k in GROUPS}
    means = {k: S[k][0] for k in GROUPS}
    mins = {k: S[k][1] for k in GROUPS}
    order = ["cells4_tok6", "cells5_tok6", "cells6_tok6", "cells7_tok6"]
    have = ok and all(means[k] is not None for k in order)
    a = ok and mins["cells4_tok6"] is not None and mins["cells4_tok6"] >= B["joint_min"]
    b = ok and mins["cells7_tok6"] is not None and mins["cells7_tok6"] < B["joint_min"]
    c = have and all(means[order[i]] - means[order[i + 1]] >= -B["tol_pool"] for i in range(3))
    d = ok and all(S[k][2] is not None and S[k][2] >= B["unit_min"] for k in GROUPS)
    e = have and (means["cells4_tok6"] - means["cells7_tok6"]) >= B["gain"]
    return {"pred_a_four_covers": bool(a), "pred_b_seven_resists": bool(b), "pred_c_monotone_within_tol": bool(c),
            "pred_d_units_available": bool(d), "pred_e_endpoint_gap": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V449_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V449_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v449",
              "candidate_id": "corpus.unit_broad_circuit_v449", "bars": BARS,
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
