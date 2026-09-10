#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v453: is the pooled-coverage decline DISTINCTNESS, or is it the rank-1 direction running out of room?

WHAT IS ESTABLISHED. Pooled coverage falls as a pool grows -- 1.004 / 0.913 / 0.786 / 0.692 at four, five, six and
seven distinct preposition mappings at six fixed tokens (v449) -- and the token spread costs more the more
behaviours there are (v451: 0.052 at four cells against 0.139 at five). Every one of those pools mixed DISTINCT
mappings, so the decline has two readings that no receipt so far separates:
    DISTINCTNESS   the direction covers less because the behaviours really are different objects, which is what the
                   count rule wants pooling to detect;
    CAPACITY       one rank-1 direction over a growing set simply runs out of room, in which case a failure to pool
                   at five or more members says nothing about identity and R2's pooled merges are uninterpretable
                   above whatever size the capacity limit sits at.
THE CONTROL THIS NEEDS IS A KNOWN-IDENTICAL POOL AT THE SAME SIZE (lesson 6: check the instrument against a
known-good case). Frame copies of ONE mapping are that case: same cue pair, same readout pair, different sentence
frame -- R7 already treats groups of two to four as one object, and nothing has tested five. Four such pools exist
in the corpus, so this is a replication and not a single comparison:
    same_about_for   about_for  + fb fc fd fe
    same_at_to       at_to      + fb fd fj fm
    same_by_from     by_from    + fc fd fj fm
    same_from_for    from_for   + fb fd fj fm
    dist5_tok6       v449's five DISTINCT mappings, re-run here so the comparison is made by ONE instrument on ONE
                     day rather than against a number carried across receipts.
Rank stays at 1; budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_same_mapping_pools_cover   all four same-mapping pools reach MINIMUM joint extraction joint_min = 0.80.
                                    Capable of failing: the six- and seven-member distinct pools sat at 0.786 and
                                    0.692, so 0.80 is a bar this instrument has missed before.        prior 70%
  pred_b_control_reproduces         the distinct-mapping pool re-measured here agrees with v449's 0.913 within
                                    tol_pool = 0.05. This is the instrument check; if it fails, the comparison below
                                    is void whichever way it comes out. Worked example: 0.93 differs by 0.017, TRUE.
                                                                                                      prior 80%
  pred_c_gap                        the MEAN over the four same-mapping pool means exceeds the distinct pool mean by
                                    at least gain = 0.05 -- larger than the tolerance the instrument reproduces
                                    itself to, so a gap smaller than that is not distinguishable from noise.
                                    Worked example: 0.99 against 0.913 is 0.077, TRUE.                prior 60%
  pred_d_units_available            every group reaches minimum pooled-unit recovery >= unit_min = 0.80.  prior 85%
  pred_e_every_same_pool_beats_it   EACH of the four same-mapping pools individually exceeds the distinct pool mean,
                                    registered apart from pred_c because an average over four can be carried by one
                                    strong pool while two sit at the distinct level.                  prior 55%
SCOPE. Whatever comes out is a statement about five-member pools of ONE construction (verb_preposition) at this rank
and budget. It can say that mapping-sharing pools cover better at five, or that they do not; it CANNOT say that
frame is irrelevant, because frame is what varies inside every same-mapping pool here.
Smoke: V453_SMOKE=<out.json> (CPU, V453_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v453_result.json"
# FIVE members in every group; only whether the members share a MAPPING changes
VP = "verb_preposition_"
SAME = {"same_about_for": ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for"),
        "same_at_to": ("at_to", "fb_at_to", "fd_at_to", "fj_at_to", "fm_at_to"),
        "same_by_from": ("by_from", "fc_by_from", "fd_by_from", "fj_by_from", "fm_by_from"),
        "same_from_for": ("from_for", "fb_from_for", "fd_from_for", "fj_from_for", "fm_from_for")}
DIST5 = ("about_against", "about_into", "from_about", "into_with", "on_about")   # v449's five distinct mappings
GROUPS = {k: tuple(VP + c for c in v) for k, v in SAME.items()}
GROUPS["dist5_tok6"] = tuple(VP + c for c in DIST5)
SAME_KEYS = tuple(SAME)
RELAXED = "ALL"
PRIOR = {"dist5_tok6": 0.913}                 # v449, five distinct mappings at six fixed tokens
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v453", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    same = [S[k] for k in SAME_KEYS]
    md, nd, ud = S["dist5_tok6"]
    have = ok and md is not None and all(x[0] is not None for x in same)
    a = ok and all(x[1] is not None and x[1] >= B["joint_min"] for x in same)
    b = have and abs(md - PRIOR["dist5_tok6"]) <= B["tol_pool"]
    c = have and (sum(x[0] for x in same) / len(same) - md) >= B["gain"]
    d = ok and all(S[k][2] is not None and S[k][2] >= B["unit_min"] for k in GROUPS)
    e = have and all(x[0] > md for x in same)
    return {"pred_a_same_mapping_pools_cover": bool(a), "pred_b_control_reproduces": bool(b),
            "pred_c_gap": bool(c), "pred_d_units_available": bool(d),
            "pred_e_every_same_pool_beats_it": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V453_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V453_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v453",
              "candidate_id": "corpus.unit_broad_circuit_v453", "bars": BARS,
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
