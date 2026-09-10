#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v499: find a disjoint control the head set actually REACHES, and turn v497's bound into an estimate.

THE REQUIREMENT THIS ANSWERS, REGISTERED BY v497 AGAINST ITSELF. On the DAS target correlative_pair.both_vs_neither,
the partially related control (correlative_or_and, which shares the token ` and` with the mapping's base answer) sat
at 0.259 absolute while the disjoint control sat at 0.004 -- a leak of 0.255. But the disjoint control's
units_recovery was 0.622, below the floor, so its near-zero number partly reflected sites the fitted head set does
not reach rather than genuine inertness. A CONTROL WHOSE SITES ARE UNREACHABLE CANNOT BE INERT FOR AN INFORMATIVE
REASON, so 0.255 was only an upper bound.
The fix is not to guess a better control but to SCREEN A PANEL of them under the identical deterministic fit and
report units_recovery for each, then re-test the leak against whichever ones the head set reaches. Six candidates,
every one with an answer vocabulary disjoint from ` and`/` nor`, spanning four constructions:
    verb_preposition_at_to, verb_preposition_about_for, noun_preposition_interest,
    adjective_preposition_of_at, verb_particle_up_down, verb_preposition_in_to
All are answer-changing behaviours, so each has a ceiling near 1.0 at a site that carries it -- lesson 4's question
has a real answer for every member, which is what makes an inert reading meaningful instead of saturated.
correlative_or_and is evaluated alongside as the reference point. Rank stays at 1. Nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_reproduces_fit        the canonical cell's own held-out rows land within tol_pool = 0.05 of v497's 1.001.
                               The fit is deterministic; this is the instrument check.                prior 90%
  pred_b_some_control_reachable at least ONE of the six disjoint candidates reports units_recovery >= unit_min = 0.80.
                               Capable of failing: the one candidate measured so far came in at 0.622, and if all
                               six land under the floor then this head set reaches nothing outside its own family
                               and the clean comparison is not available at all -- which would itself be the
                               finding.                                                               prior 70%
  pred_c_reachable_are_inert   EVERY disjoint candidate that clears the units floor has absolute recovery at or
                               below cross_max = 0.30.                                                prior 80%
  pred_d_related_leak_holds    the related control's absolute recovery exceeds the LARGEST among the reachable
                               disjoint candidates by at least gain = 0.05. Worked example: 0.26 against a best
                               reachable 0.05 is 0.21, TRUE and v497's upper bound becomes an estimate; 0.26
                               against 0.24 is 0.02, FALSE and the leak was never about the shared TOKEN -- it is
                               about being another behaviour read at the same site, and my account of v497 was
                               wrong.                                                                 prior 65%
  pred_e_related_units_ok      the related control itself reports units_recovery >= 0.80, so its leak is
                               attributable to the direction rather than to reachability. It was 0.916, so this is
                               a guard rather than a question, and its prior says so.                 prior 90%
HOW IT READS, one outcome per branch. b TRUE, c TRUE, d TRUE: there is a usable disjoint control for this family,
v497's 0.255 becomes a measured leak rather than a bound, and the correlative rungs get a control that is inert for
an informative reason. d FALSE: the shared-token account is wrong and the 0.259 reflects behaviour-at-the-same-site,
which changes what every correlative control has been measuring. b FALSE: this head set reaches nothing outside its
own family, no clean disjoint control exists for it, and every correlative control number stays an upper bound --
in which case I report that limit rather than papering over it.
SCOPE. One fit of one mapping. This rung selects and measures controls; it makes no claim about the circuit.
Smoke: V499_SMOKE=<out.json> (CPU, V499_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_control_panel_v499_result.json"
# one deterministic fit; a PANEL of disjoint candidate controls screened for reachability, then for inertness
OWN = "correlative_pair"
RELATED = "correlative_or_and"               # shares ' and' with the base answer -- the reference point
DISJOINT_PANEL = ("verb_preposition_at_to", "verb_preposition_about_for", "noun_preposition_interest",
                  "adjective_preposition_of_at", "verb_particle_up_down", "verb_preposition_in_to")
SIBS = (RELATED,) + DISJOINT_PANEL
MAPS = {"fit_one": {"fit": (OWN,), "held": SIBS, "control": DISJOINT_PANEL[0]}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1
PRIOR = {"v497_canonical_own": 1.001, "v497_related": 0.259, "v497_disjoint_units": 0.622}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_control_panel_v499", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
        return G.get(k, {}).get("per_shape", {}).get(MAPS[k]["control"], {}).get("cross_abs_recovery")
    ps0 = G.get("fit_one", {}).get("per_shape", {})
    own_v = ps0.get(OWN, {}).get("joint_extraction")
    rel_abs = ps0.get(RELATED, {}).get("cross_abs_recovery")
    rel_u = ps0.get(RELATED, {}).get("units_recovery")
    panel = [(c, ps0.get(c, {}).get("cross_abs_recovery"), ps0.get(c, {}).get("units_recovery"))
             for c in DISJOINT_PANEL]
    have = ok and own_v is not None and rel_abs is not None and all(x is not None and u is not None
                                                                   for _, x, u in panel)
    reachable = [(c, x) for c, x, u in panel if have and u >= B["unit_min"]] if have else []
    a = have and abs(own_v - PRIOR["v497_canonical_own"]) <= B["tol_pool"]
    b = bool(reachable)
    c = b and all(abs(x) <= B["cross_max"] for _, x in reachable)
    d = b and rel_abs is not None and (abs(rel_abs) - max(abs(x) for _, x in reachable)) >= B["gain"]
    e = have and rel_u is not None and rel_u >= B["unit_min"]
    return {"pred_a_reproduces_fit": bool(a), "pred_b_some_control_reachable": bool(b),
            "pred_c_reachable_are_inert": bool(c), "pred_d_related_leak_holds": bool(d),
            "pred_e_related_units_ok": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V499_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V499_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_control_panel_v499",
              "candidate_id": "corpus.unit_control_panel_v499", "bars": BARS,
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
