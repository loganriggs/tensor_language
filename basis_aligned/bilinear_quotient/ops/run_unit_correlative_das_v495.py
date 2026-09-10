#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v495: the causal counterpart of v493, on a NAMED DAS target, nested so two effects are attributable apart.

WHAT v493 SETTLED BEHAVIOURALLY. Every cell in the correlative family contains exactly ONE correlative word, so
"nearest correlative" and "still-open correlative" are the same token on 100% of rows. Interposing a DISCHARGED
`either ... or` nearer than the live `both`/`neither`, the model did NOT collapse -- 32/32 rows kept, and the
correlative-free control in the same frame likewise -- so the MODEL tracks the open correlative. That says nothing
about what a fitted direction carries, which is this rung.
correlative_pair.both_vs_neither is the FIRST named target of the standing DAS follow-up, so the fit is on the
canonical cell rather than on anything authored tonight, and RANK IS FIXED AT 1 AND REGISTERED HERE IN ADVANCE per
the protocol; a null is not permission to raise it. All three cells carry the SAME mapping -- `both` -> ` and`,
`neither` -> ` nor` -- so a failure to transfer cannot be explained by directions being cue-pair keyed, the defect
that voided v487.
THE DESIGN IS NESTED, WHICH IS THE POINT. The distractor cell differs from the canonical one in TWO ways at once: a
much longer frame AND a discharged correlative inside it. A flat comparison could not say which mattered, so the
plain long frame is evaluated too and pred_a carries it:
    fit_canonical  correlative_pair (`The pilot praised both the guide`, cue at -2)
                   held out: correlative_open_neu  -- long frame, NO correlative interposed
                             correlative_open_rec  -- same long frame WITH a discharged either ... or
    fit_distractor correlative_open_rec; evaluate the canonical cell held out
    control        correlative_or_and, a different mapping in the same family, on ABSOLUTE recovery
Relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_reaches_long_frame   the canonical cell's own held-out rows AND the plain long-frame cell both reach
                              MINIMUM joint extraction joint_min = 0.80. This establishes that the direction
                              survives the frame change on its own; without it pred_b is unreadable.  prior 65%
  pred_b_survives_discharged  the canonical-fitted direction also reaches 0.80 on the cell whose interposition
                              contains a DISCHARGED correlative. Given pred_a, this isolates the discharged
                              correlative from the frame length.                                      prior 55%
  pred_c_reverse_transfer     the direction fitted on the distractor cell reaches 0.80 back on the canonical one.
                                                                                                      prior 55%
  pred_d_units_available      every evaluated cell except the control reports units_recovery >= unit_min = 0.80,
                              so a transfer failure is the DIRECTION and not units that miss the shape. prior 70%
  pred_e_different_mapping    the control's ABSOLUTE recovery stays at or below cross_max = 0.30. Registered as a
                              CONTROL, not a finding: the cue-pair baseline predicts it.              prior 85%
HOW IT READS, one outcome per branch. a TRUE and b TRUE: the direction reads the open correlative and a discharged
one costs it nothing -- the behavioural result of v493 has a causal counterpart, on the DAS target. a TRUE and
b FALSE: the long frame is fine but the DISCHARGED CORRELATIVE specifically breaks the direction, which would be the
sharpest outcome available here -- the site would be confusable by a competing correlative even though the model's
OUTPUT is not, and that gap between behaviour and mechanism is worth more than either half alone. a FALSE: the frame
change alone breaks it, b is unreadable whatever it shows, and I will not interpret it.
SCOPE. One mapping, rank 1, this budget. It is a statement about the canonical cell's direction, not about the
family's other four cells.
Smoke: V495_SMOKE=<out.json> (CPU, V495_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_correlative_das_v495_result.json"
# nested: canonical cell -> plain long frame -> long frame with a DISCHARGED correlative
CANON, PLAIN, DISTR = "correlative_pair", "correlative_open_neu", "correlative_open_rec"
CTRL = "correlative_or_and"
MAPS = {"fit_canonical": {"fit": (CANON,), "held": (PLAIN, DISTR)},
        "fit_distractor": {"fit": (DISTR,), "held": (CANON,)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) + (CTRL,) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v493_behavioural": "model kept 32/32 rows with the discharged correlative interposed"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_correlative_das_v495", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery")
    canon_own = je("fit_canonical", CANON)
    plain_v = je("fit_canonical", PLAIN)
    distr_v = je("fit_canonical", DISTR)
    rev_v = je("fit_distractor", CANON)
    have = ok and all(x is not None for x in (canon_own, plain_v, distr_v, rev_v))
    a = have and min(canon_own, plain_v) >= B["joint_min"]
    b = have and distr_v >= B["joint_min"]
    c = have and rev_v >= B["joint_min"]
    d = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["held"]))
    cross = [G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery") for k in GROUPS]
    e = ok and all(x is not None and abs(x) <= B["cross_max"] for x in cross)
    return {"pred_a_reaches_long_frame": bool(a), "pred_b_survives_discharged": bool(b),
            "pred_c_reverse_transfer": bool(c), "pred_d_units_available": bool(d),
            "pred_e_different_mapping": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V495_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V495_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                if cell in cells:                       # only this mapping's own frames enter the fit
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
                backend, P_fit, units, rank=RANK, steps=steps, lr=LR, seed=0, complement_weight=CW,
                controls=(P_cfit,), control_weight=LAM, mu=mu_joint)
            per_shape = {}
            for cell in EVAL[gname]:
                held_rows, c_rows = per_shape_rows[cell]
                p_held = g.prepare(backend, held_rows, **V)
                p_c = g.prepare(backend, c_rows)
                e_exact = ext(p_held, units)
                cdmg = dmg(p_c, units, q_joint, mu_joint)
                per_shape[cell] = {
                    "units_recovery": e_exact,
                    "cross_abs_recovery": round(ext(p_held, units, q=q_joint), 3),
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
    result = {"predictions": predictions, "schema": "unit_correlative_das_v495",
              "candidate_id": "corpus.unit_correlative_das_v495", "bars": BARS,
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
