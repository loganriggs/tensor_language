#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v471: does the READOUT-ENVIRONMENT binding replicate, or is it about/for again?

WHAT v469 FOUND, ON ONE MAPPING. Two cells with no punctuation anywhere after the cue -- `Word spread that the pilot
cared a lot` and `Near the ledger the pilot cared again` -- held out of a fit made entirely of comma-adverbial
frames, reached 0.662 and 0.622 against 1.023 for the fitted frames, with units_recovery 0.844 and 1.005. The sites
reach the new environment; the direction does not. Two different shared tails give nearly the same number, so it is
not one word.
WHY THIS RUNG IS MANDATORY RATHER THAN OPTIONAL. Earlier tonight the surface-form binding was measured on about/for
(0.658), written up with a repair recipe attached, and then turned out to be ONE MAPPING IN FOUR (0.818 / 0.975 /
1.262 on the other three). v469 is about/for again, and it has a much larger consequence attached to it -- 193 of the
corpus's 480 cells end with the identical suffix `, of course,` -- so it gets the replication BEFORE the consequence
is stated, not after.
Three mappings, each fitted on its own five comma-adverbial frames, each evaluated on its own two comma-free cells
held out of the fit, each using another mapping in this rung as its cross-mapping control on absolute recovery. All
six new cells were capability-checked on CPU before this runner was written (A1 32/32 rows kept, 0 dropped, each).
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution  held-out rows of the fitted comma-adverbial frames reach MINIMUM joint extraction
                          joint_min = 0.80 in ALL THREE mappings. If this fails nothing else is readable. prior 85%
  pred_b_at_to_env_fails  BOTH comma-free cells of at/to land BELOW 0.80, as about/for's did.          prior 50%
  pred_c_by_from_env_fails BOTH comma-free cells of by/from land below 0.80.                            prior 50%
  pred_d_from_for_env_fails BOTH comma-free cells of from/for land below 0.80.                          prior 50%
  pred_e_units_available  every evaluated cell except the controls reports units_recovery >= unit_min = 0.80, so a
                          failure is attributable to the DIRECTION and not to units that miss the shape. prior 80%
HOW THE COUNT READS, fixed before the run. b, c and d ALL TRUE makes it four mappings of four: the readout
environment binds every mapping tested, and the corpus's concentration on one suffix is a limitation inherited by
every circuit counted -- which would be the most consequential finding of the night and would oblige a stimulus
redesign rather than a note. ALL FALSE makes it one of four and about/for the exception again, in which case the
suffix concentration is cosmetic and v469 says something about that cue pair only. A MIXED count is the honest
middle: the environment binds some mappings, the proportion is the result, and no single number may be quoted for
the corpus. Note that a mapping counts as bound only if BOTH its cells fail -- one of two is reported as such and
named as a split rather than folded into either side.
SCOPE. Four mappings in one construction at this rank and budget. Even four of four would not license the claim for
verb_particle or the smaller families; those need their own rung, which is cheap now that the cells derive by
substitution.
Smoke: V471_SMOKE=<out.json> (CPU, V471_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v471_result.json"
# three mappings; each fitted on comma-adverbial frames, each evaluated on its own two comma-free cells
VP = "verb_preposition_"
MAPS = {
    "at_to": {"fit": ("at_to", "fb_at_to", "fd_at_to", "fj_at_to", "fm_at_to"),
              "env": ("fn_at_to", "fq_at_to"), "control": "from_for"},
    "by_from": {"fit": ("by_from", "fc_by_from", "fd_by_from", "fj_by_from", "fm_by_from"),
                "env": ("fn_by_from", "fq_by_from"), "control": "at_to"},
    "from_for": {"fit": ("from_for", "fb_from_for", "fd_from_for", "fj_from_for", "fm_from_for"),
                 "env": ("fn_from_for", "fq_from_for"), "control": "by_from"},
}
GROUPS = {k: tuple(VP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(VP + c for c in v["env"]) + (VP + v["control"],) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"about_for_env": (0.662, 0.622), "about_for_fitted": 1.023}      # v469
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v471", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    env_v = {k: [je(k, c) for c in MAPS[k]["env"]] for k in GROUPS}
    have = ok and all(x is not None for x in fit_v) and all(x is not None for v in env_v.values() for x in v)
    a = have and min(fit_v) >= B["joint_min"]
    bound = {k: all(x < B["joint_min"] for x in env_v[k]) for k in GROUPS} if have else {}
    b = have and bound["at_to"]
    c = have and bound["by_from"]
    d = have and bound["from_for"]
    e = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["env"]))
    return {"pred_a_in_distribution": bool(a), "pred_b_at_to_env_fails": bool(b),
            "pred_c_by_from_env_fails": bool(c), "pred_d_from_for_env_fails": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V471_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V471_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v471",
              "candidate_id": "corpus.unit_broad_circuit_v471", "bars": BARS,
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
