#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v455: the known-identical control carried to SEVEN, where the distinct pools actually fail.

WHAT v453 SETTLED AND WHAT IT DID NOT. At FIVE members a rank-1 direction covers a known-identical pool (four
replicates, means 0.996-1.023) while five distinct mappings reach only 0.913 -- so at five the pooled decline is
DISTINCTNESS, not the direction running out of room. But the distinct pools do not fail at five; they fail at six
and seven (0.786, 0.692, v449), and v453's control stopped at five because the corpus held no larger same-mapping
group. CAPACITY THEREFORE REMAINS A LIVE READING EXACTLY WHERE THE COUNT RULE IS DOING REAL WORK.
This rung extends the control by two frames authored for it, both chosen to differ from frames A-E in KIND rather
than in wording -- A-E are all declarative matrix-plus-complement shapes ending in an adverbial:
    frame G  `Did the pilot really care, then,`              interrogative, bare verb under do-support
    frame H  `Because of the ledger, the pilot cared, quietly,`   main clause behind a fronted adjunct
Both were capability-checked on CPU before authoring this runner: A1 32/32 rows kept, 0 dropped, on each.
    same6 / same7   about_for + fb fc fd fe (+ fg) (+ fh)   one mapping, six then seven sentence frames
    dist6 / dist7   v449's six and seven DISTINCT mappings, re-measured here so both sides come from one
                    instrument on one day (v453's re-measurement of the five-member pool reproduced 0.913 exactly,
                    which is why this check is cheap and worth repeating).
Rank stays at 1; budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_same7_covers        the seven-member same-mapping pool reaches MINIMUM joint extraction
                             joint_min = 0.80. Capable of failing on this instrument: the seven distinct
                             mappings sat at 0.692.                                                   prior 65%
  pred_b_controls_reproduce  BOTH distinct pools agree with v449 (0.786 at six, 0.692 at seven) within
                             tol_pool = 0.05. The instrument check; if it fails the comparison is void whichever
                             way the rest comes out. Worked example: 0.80 against 0.786 differs by 0.014, TRUE.
                                                                                                      prior 80%
  pred_c_gap7                the seven-member same-mapping MEAN exceeds the seven-member distinct MEAN by at least
                             gain = 0.05. Worked example: 0.98 against 0.692 is 0.288, TRUE.          prior 65%
  pred_d_units_available     every group reaches minimum pooled-unit recovery >= unit_min = 0.80.     prior 85%
  pred_e_same_holds_from_five  the seven-member same-mapping mean stays within tol_pool of this mapping's FIVE-member
                             mean of 1.023 (v453) -- i.e. adding two members, and two frames of a different clause
                             type, does not degrade a mapping-sharing pool. Worked example: 0.99 differs by 0.033,
                             TRUE; 0.90 differs by 0.123, FALSE and size costs something even when the mapping is
                             held fixed.                                                              prior 55%
SCOPE. One construction (verb_preposition), one mapping on the identical side, this rank and budget. A pass says a
rank-1 direction still has room at seven when the mapping is shared, which would make the distinct pools' failure at
six and seven readable as distinctness. A failure says size alone costs coverage by seven, and then no pooled verdict
above five can be read as identity -- which would cap the merge rule rather than the count.
Smoke: V455_SMOKE=<out.json> (CPU, V455_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v455_result.json"
# six and seven members, same mapping against distinct mappings
VP = "verb_preposition_"
SAME7 = ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for",
         "fg_about_for", "fh_about_for")
DIST7 = ("about_against", "about_into", "from_about", "into_with", "on_about", "on_into", "with_against")
GROUPS = {"same6": tuple(VP + c for c in SAME7[:6]), "same7": tuple(VP + c for c in SAME7),
          "dist6": tuple(VP + c for c in DIST7[:6]), "dist7": tuple(VP + c for c in DIST7)}
RELAXED = "ALL"
PRIOR = {"dist6": 0.786, "dist7": 0.692, "same5_about_for": 1.023}   # v449 (distinct), v453 (same mapping at five)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v455", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    md6, _, _ = S["dist6"]
    md7, _, _ = S["dist7"]
    have = ok and all(x is not None for x in (m7, md6, md7))
    a = ok and n7 is not None and n7 >= B["joint_min"]
    b = have and abs(md6 - PRIOR["dist6"]) <= B["tol_pool"] and abs(md7 - PRIOR["dist7"]) <= B["tol_pool"]
    c = have and (m7 - md7) >= B["gain"]
    d = ok and all(S[k][2] is not None and S[k][2] >= B["unit_min"] for k in GROUPS)
    e = have and abs(m7 - PRIOR["same5_about_for"]) <= B["tol_pool"]
    return {"pred_a_same7_covers": bool(a), "pred_b_controls_reproduce": bool(b),
            "pred_c_gap7": bool(c), "pred_d_units_available": bool(d),
            "pred_e_same_holds_from_five": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V455_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V455_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v455",
              "candidate_id": "corpus.unit_broad_circuit_v455", "bars": BARS,
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
