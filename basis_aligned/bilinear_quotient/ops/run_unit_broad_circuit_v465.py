#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v465: how common is form-binding? Two more mappings through the identical design.

WHERE THIS STANDS. On about/for a rank-1 direction fitted on -ed frames reaches 0.931 on a wh-interrogative that
keeps the -ed form and only 0.658 on a declarative that switches to the bare verb (v459); on at/to, run through the
SAME design with the cells derived by substitution, the bare-form cell reaches 1.262 and nothing is form-bound
(v463). One of those two is the exception and no argument decides which -- but two more mappings do, and each costs
about 45 seconds of GPU.
    by/from    swore vs withdrew    fit on by_from + fc fd fj fm
    from/for   prevented vs blamed  fit on from_for + fb fd fj fm
Both mappings take an IRREGULAR verb pair, so their bare forms differ from the -ed forms by more than a suffix
(swear/swore, withdraw/withdrew, and prevent/prevented, blame/blamed on the regular side of from/for). If the
surface form is what binds a direction, an irregular pair is the easier case for it to show up in, not the harder.
Each mapping gets its own fit, its own two new cells and the OTHER mapping in this rung as its cross-mapping
control on absolute recovery. All four new cells were capability-checked on CPU before this runner was written
(A1 32/32 rows kept, 0 dropped, each). Rank stays at 1; budget stays relaxed; nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution    held-out rows of the fitted frames reach MINIMUM joint extraction joint_min = 0.80 in
                            BOTH mappings. If this fails nothing else is readable.                    prior 85%
  pred_b_by_from_form_fails by/from's bare-form cell lands BELOW 0.80, as about/for's did.           prior 50%
  pred_c_from_for_form_fails from/for's bare-form cell lands BELOW 0.80.                              prior 50%
  pred_d_type_cells_transfer BOTH wh-interrogative cells, which keep the -ed form, reach 0.80 -- the clause-type
                            half held in both mappings so far (0.931 and 0.905).                      prior 75%
  pred_e_units_available    every evaluated cell except the controls reports units_recovery >= unit_min = 0.80.
                                                                                                      prior 80%
HOW THE COUNT READS, fixed before the run: b and c BOTH TRUE makes it three of four mappings form-bound and at/to
the exception. BOTH FALSE makes it one of four and about/for the exception -- and then the v461 repair, which was
built on about/for, describes a special case rather than a method. ONE OF EACH leaves it two and two, form-binding
is mapping-specific with no majority, and the per-mapping check stands as the only safe instruction either way.
Whatever the count, it does not explain WHY mappings differ; that needs a factor varied deliberately, and this rung
does not vary one.
SCOPE. Four mappings in ONE construction at this rank and budget.
Smoke: V465_SMOKE=<out.json> (CPU, V465_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v465_result.json"
# two mappings, each with its own fit, its own two new cells, and the other as its cross-mapping control
VP = "verb_preposition_"
MAPS = {
    "by_from": {"fit": ("by_from", "fc_by_from", "fd_by_from", "fj_by_from", "fm_by_from"),
                "form": "fi_by_from", "type": "fk_by_from", "control": "from_for"},
    "from_for": {"fit": ("from_for", "fb_from_for", "fd_from_for", "fj_from_for", "fm_from_for"),
                 "form": "fi_from_for", "type": "fk_from_for", "control": "by_from"},
}
GROUPS = {k: tuple(VP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(VP + v[x] for x in ("form", "type", "control")) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"about_for_form": 0.658, "at_to_form": 1.262}          # v459 form-bound, v463 not
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v465", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VP + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VP + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(VP + MAPS[k]["control"], {}).get("cross_abs_recovery")
    fit_v = [je(k, c) for k in GROUPS for c in MAPS[k]["fit"]]
    form_v = {k: je(k, MAPS[k]["form"]) for k in GROUPS}
    type_v = {k: je(k, MAPS[k]["type"]) for k in GROUPS}
    have = ok and all(x is not None for x in fit_v + list(form_v.values()) + list(type_v.values()))
    a = have and min(fit_v) >= B["joint_min"]
    b = have and form_v["by_from"] < B["joint_min"]
    c = have and form_v["from_for"] < B["joint_min"]
    d = have and all(x >= B["joint_min"] for x in type_v.values())
    e = ok and all(ur(k, c) is not None and ur(k, c) >= B["unit_min"]
                   for k in GROUPS for c in tuple(MAPS[k]["fit"]) + (MAPS[k]["form"], MAPS[k]["type"]))
    return {"pred_a_in_distribution": bool(a), "pred_b_by_from_form_fails": bool(b),
            "pred_c_from_for_form_fails": bool(c), "pred_d_type_cells_transfer": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V465_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V465_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
                backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v465",
              "candidate_id": "corpus.unit_broad_circuit_v465", "bars": BARS,
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
