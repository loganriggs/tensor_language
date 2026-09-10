#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v485: 115 of 119 cells share one subject slot. Is the direction an agent-bigram feature?

THE CHALLENGE. An adversarial audit reported that 115 of the 119 verb_preposition cells put a SINGLE-TOKEN,
SINGULAR, DEFINITE, ANIMATE, HUMAN OCCUPATIONAL noun from one 32-item table IMMEDIATELY before the cue, and that no
P row and no A2 row anywhere in the family perturbs a token between the subject and the readout. So the interchange
difference could be carried by an `the {occupation} + verb` bigram or an animacy-conditioned feature rather than a
verb-lexeme feature, and NOTHING IN THE CURRENT DESIGN COULD DETECT IT -- the whole window from subject to readout is
token-frozen across A1, A2 and P in every cell.
The new cell breaks three of those constants at once: `the delays across the region` is plural, inanimate, and three
tokens from the cue, with ` region` rather than an occupation in the pre-cue slot. THAT COMPOUND IS DELIBERATE AND IS
REGISTERED AS A FIRST PASS: if the incumbent direction still carries it, all three are ruled out together at one
screen's cost; if it does not, the failure cannot be attributed to any one of them and follow-ups splitting them are
required. I will not read a failure as being about animacy, or number, or adjacency. ` across` is not in this cell's
answer vocabulary, so the internal preposition cannot prime either answer.
Two fits, each on ONE cell. That is a validated instrument here rather than an assumption: v481 fitted at_to alone
and reached its four frame siblings at 0.914-1.090, so a one-cell fit is not inherently weak and an asymmetry in this
rung cannot be blamed on fit size the way v479's was.
    fit_normal  in_to (`the {agent} resulted/amounted`); evaluate the broken-subject cell held out
    fit_broken  sub_in_to (`the delays across the region resulted/amounted`); evaluate in_to held out
    control     about_for, a different mapping entirely, on ABSOLUTE recovery
Capability screened on CPU before this runner was written: the new cell keeps 32/32 rows with margins +/-5.1 and
+/-4.6, among the strongest in the corpus, so a transfer failure would not be a weak-behaviour artefact.
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_both_in_distribution  held-out rows of BOTH fitted cells reach MINIMUM joint extraction joint_min = 0.80.
                               If this fails nothing else is readable.                                prior 85%
  pred_b_normal_to_broken      the direction fitted on the ordinary cell reaches 0.80 on the broken-subject cell.
                                                                                                      prior 55%
  pred_c_broken_to_normal      the direction fitted on the broken-subject cell reaches 0.80 on the ordinary cell.
                                                                                                      prior 55%
  pred_d_units_available       every evaluated cell except the controls reports units_recovery >= unit_min = 0.80,
                               so a transfer failure is the DIRECTION and not units that miss the shape. prior 75%
  pred_e_different_mapping     the control's ABSOLUTE recovery stays at or below cross_max = 0.30.     prior 85%
HOW IT READS, one outcome per branch. b TRUE and c TRUE: the subject slot is not load-bearing -- animacy, number and
adjacency are ruled out TOGETHER, and the family's directions are not agent-bigram features. b FALSE and c FALSE: the
two cells are different objects and the family's 115 same-subject cells have been measuring something that does not
survive a changed subject; the compound cannot say which of the three constants did it, and the splitting follow-ups
become the next rungs. b TRUE with c FALSE: the ordinary direction is the broader object, and since v481 already
showed a one-cell fit transfers between ordinary cells, that asymmetry is content rather than fit size -- but which
constant it turns on is still unattributed. c TRUE with b FALSE: the same, reversed, and more surprising, since the
broken cell would then be the broader object.
SCOPE. One mapping at rank 1 and this budget. A pass licenses dropping three confounds for this mapping, not for the
family; the audit's point was about all 115 cells and this rung tests one of them.
Smoke: V485_SMOKE=<out.json> (CPU, V485_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_subject_slot_v485_result.json"
# two fits, one cell each: the family's frozen subject slot, and a cell that breaks three of its constants
VPP = "verb_preposition_"
NORMAL, BROKEN = "in_to", "sub_in_to"
CTRL = VPP + "about_for"
MAPS = {"fit_normal": {"fit": (NORMAL,), "held": (BROKEN,)},
        "fit_broken": {"fit": (BROKEN,), "held": (NORMAL,)}}
GROUPS = {k: tuple(VPP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(VPP + c for c in v["held"]) + (CTRL,) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v481_one_cell_to_sibs": (0.914, 1.090)}    # a one-cell fit is not inherently weak
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_subject_slot_v485", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPP + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPP + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery")
    n_own, b_own = je("fit_normal", NORMAL), je("fit_broken", BROKEN)
    n_to_b, b_to_n = je("fit_normal", BROKEN), je("fit_broken", NORMAL)
    have = ok and all(x is not None for x in (n_own, b_own, n_to_b, b_to_n))
    a = have and min(n_own, b_own) >= B["joint_min"]
    b = have and n_to_b >= B["joint_min"]
    c = have and b_to_n >= B["joint_min"]
    d = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["held"]))
    cross = [G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery") for k in GROUPS]
    e = ok and all(x is not None and abs(x) <= B["cross_max"] for x in cross)
    return {"pred_a_both_in_distribution": bool(a), "pred_b_normal_to_broken": bool(b),
            "pred_c_broken_to_normal": bool(c), "pred_d_units_available": bool(d),
            "pred_e_different_mapping": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V485_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V485_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_subject_slot_v485",
              "candidate_id": "corpus.unit_subject_slot_v485", "bars": BARS,
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
