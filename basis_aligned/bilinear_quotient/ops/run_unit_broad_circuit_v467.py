#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v467: the pooling result in a SECOND construction, with the known-identical control carried to TEN.

WHAT IS ESTABLISHED, ALL OF IT IN ONE CONSTRUCTION. Pooled coverage falls as a pool of DISTINCT mappings grows
(1.004 / 0.913 / 0.786 / 0.692 at four to seven) while pools of frame copies of ONE mapping cover at 0.996-1.023
at five and 1.015 at seven -- so the decline is distinctness and not a rank-1 direction running out of room
(v453, v455, each with the distinct pools reproducing v449 to three decimals). Every one of those numbers comes
from verb_preposition. Tonight already showed what happens when an account measured on one part of the corpus is
generalised: the surface-form binding held on about/for and turned out to be one mapping in four (v459 -> v465).
The pooled result is what R2 and R7 -- and therefore the count of 139 -- actually rest on, so it is worth more than
the form account was, and it deserves the same treatment.
verb_particle is the natural second construction and it is a STRONGER test than the first: up/down has FOURTEEN
frame copies in the corpus, so the known-identical control can be carried to TEN, past anything the preposition
cells could reach, while the construction supplies seven mappings with genuinely different particle pairs.
    same7 / same10   up_down + fb fc fd fe fg fh (+ fj fk fm)   one mapping, seven then ten sentence frames
    dist5 / dist7    up_down, out_down, out_up, down_in, away_up (+ on_away, on_out) -- distinct particle pairs
Rank stays at 1; budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_same7_covers      the seven-frame same-mapping pool reaches MINIMUM joint extraction joint_min = 0.80,
                           as the preposition seven-frame pool did at min 0.965.                      prior 75%
  pred_b_same10_covers     the TEN-frame same-mapping pool also reaches 0.80. This is the new size and the
                           capacity question at a scale nothing has tested; a failure here would put a ceiling
                           between seven and ten and would cap the merge rule at whatever size it sits.  prior 60%
  pred_c_distinct_declines the seven-mapping DISTINCT pool's mean is at least gain = 0.05 below the seven-frame
                           same-mapping pool's mean. Worked example: 0.75 against 1.00 is 0.25 below, TRUE.
                                                                                                      prior 75%
  pred_d_units_available   every group reaches minimum pooled-unit recovery >= unit_min = 0.80.        prior 80%
  pred_e_decline_within_construction the FIVE-mapping distinct pool's mean exceeds the SEVEN-mapping distinct
                           pool's by at least gain, i.e. the decline with pool size reproduces INSIDE this
                           construction rather than being imported from the preposition curve. No bar here cites a
                           preposition number: both sides are measured in this run.                    prior 70%
HOW IT READS: a, c and e together say the pooled criterion behaves the same way in a second construction and R2/R7
are not preposition artefacts. b alone extends the capacity control. If c or e fails while a passes, the criterion
does NOT transfer and every merge outside verb_preposition needs its own control before it is trusted.
SCOPE. Two constructions at this rank and budget; nothing here says anything about the other constructions in the
corpus, and the honest instruction if it passes is still per-construction rather than universal.
Smoke: V467_SMOKE=<out.json> (CPU, V467_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v467_result.json"
# second construction; same-mapping control carried to ten frames
VP = "verb_particle_"
SAME10 = ("up_down", "fb_up_down", "fc_up_down", "fd_up_down", "fe_up_down", "fg_up_down",
          "fh_up_down", "fj_up_down", "fk_up_down", "fm_up_down")
DIST7 = ("up_down", "out_down", "out_up", "down_in", "away_up", "on_away", "on_out")
GROUPS = {"same7": tuple(VP + c for c in SAME10[:7]), "same10": tuple(VP + c for c in SAME10),
          "dist5": tuple(VP + c for c in DIST7[:5]), "dist7": tuple(VP + c for c in DIST7)}
RELAXED = "ALL"
PRIOR = {}                                    # every number this rung compares is measured in this run
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v467", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    m7, n7, _ = S["same7"]
    m10, n10, _ = S["same10"]
    md5, _, _ = S["dist5"]
    md7, _, _ = S["dist7"]
    have = ok and all(x is not None for x in (m7, m10, md5, md7))
    a = ok and n7 is not None and n7 >= B["joint_min"]
    b = ok and n10 is not None and n10 >= B["joint_min"]
    c = have and (m7 - md7) >= B["gain"]
    d = ok and all(S[k][2] is not None and S[k][2] >= B["unit_min"] for k in GROUPS)
    e = have and (md5 - md7) >= B["gain"]
    return {"pred_a_same7_covers": bool(a), "pred_b_same10_covers": bool(b),
            "pred_c_distinct_declines": bool(c), "pred_d_units_available": bool(d),
            "pred_e_decline_within_construction": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V467_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V467_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v467",
              "candidate_id": "corpus.unit_broad_circuit_v467", "bars": BARS,
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
