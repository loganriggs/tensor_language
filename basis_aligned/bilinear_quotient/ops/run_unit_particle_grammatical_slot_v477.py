#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v477: verb_particle predicts a particle where English does not license one. Does it matter?

THE CHALLENGE. An adversarial audit reported that in all 40 verb_particle cells the target continuation is
`VERB , <adverbial> , PARTICLE` -- and a particle cannot follow a comma-bracketed parenthetical (*`the pilot woke,
of course, up`). If that is right, every measurement in the family is taken in a position where no particle
selection is syntactically licensed, so only the verb-particle collocation prior can act there. THIS REACHES BEYOND
THE FAMILY: v467 used verb_particle as the SECOND CONSTRUCTION validating the pooled distinctness rule that R2, R7
and the count of 139 rest on, and that validation is worth exactly as much as the family is.
The new cell keeps the mapping, the frame and the lexicon and changes only the slot -- ` back` as the shared matched
suffix, so the particle lands where `wake back up` and `calm back down` are ordinary English. Its CPU screen already
says something: 32/32 rows kept with margins +/-5.1, against +/-2 to +/-4 for the comma frames it is being compared
with. That is capability, not causality, and it is reported here rather than argued from.
TWO FITS, so the question is asked in both directions and neither answer can be assumed:
    fit_comma        fit on up_down + its six frame copies (all comma-slot); evaluate the GRAMMATICAL cell held out
    fit_grammatical  fit on the grammatical cell alone; evaluate all seven COMMA cells held out
    control          down_in, a different particle mapping, on ABSOLUTE recovery, in both groups
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_comma_in_distribution   held-out rows of the fitted comma cells reach MINIMUM joint extraction
                                 joint_min = 0.80. If this fails nothing else is readable.            prior 85%
  pred_b_comma_to_grammatical    the comma-fitted direction reaches 0.80 on the grammatical cell.     prior 50%
  pred_c_grammatical_in_distribution the grammatical-fitted direction reaches 0.80 on its own held-out rows -- the
                                 other half of the instrument check, and the one that says a failure of pred_d is
                                 about transfer rather than about an unreachable cell.                prior 85%
  pred_d_grammatical_to_comma    the grammatical-fitted direction reaches 0.80 on the MINIMUM of the seven comma
                                 cells.                                                               prior 50%
  pred_e_units_available         every evaluated cell except the controls reports units_recovery >= unit_min = 0.80,
                                 so a transfer failure is the DIRECTION and not units that miss the shape. prior 75%
HOW IT READS, fixed in advance. b TRUE and d TRUE: the two slots share a direction, the off-distribution worry does
not change what this family measures, and v467 stands as written. EITHER FALSE: the comma cells and the grammatical
cell are different objects, the family's 40 cells measure something that does not extend to where particles are
actually licensed, and v467's second-construction validation of the pooling rule inherits that -- which would mean
the rule is confirmed on one construction plus one artefact, not on two constructions. Asymmetry (one direction
transfers and the other does not) is a real outcome and is reported as such rather than folded into either side.
SCOPE. One mapping of the family at this rank and budget. A failure would not say the other 39 cells are worthless,
only that this one has been measured somewhere the grammar does not license and the burden moves.
Smoke: V477_SMOKE=<out.json> (CPU, V477_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_particle_grammatical_slot_v477_result.json"
# two fits of ONE mapping: the comma slot the family uses, and the slot English actually licenses
VPT = "verb_particle_"
COMMA = ("up_down", "fb_up_down", "fc_up_down", "fd_up_down", "fe_up_down", "fg_up_down", "fh_up_down")
GRAM = "gm_up_down"
CTRL = "down_in"
MAPS = {"fit_comma": {"fit": COMMA, "held": (GRAM,), "control": CTRL},
        "fit_grammatical": {"fit": (GRAM,), "held": COMMA, "control": CTRL}}
GROUPS = {k: tuple(VPT + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(VPT + c for c in v["held"]) + (VPT + v["control"],) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v467_same7": 1.005, "v467_same10": 1.005}      # comma-slot pools, this same mapping
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_particle_grammatical_slot_v477", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPT + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPT + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(VPT + MAPS[k]["control"], {}).get("cross_abs_recovery")
    comma_fit = [je("fit_comma", c) for c in COMMA]
    gram_from_comma = je("fit_comma", GRAM)
    gram_own = je("fit_grammatical", GRAM)
    comma_from_gram = [je("fit_grammatical", c) for c in COMMA]
    have = ok and all(x is not None for x in comma_fit + comma_from_gram + [gram_from_comma, gram_own])
    a = have and min(comma_fit) >= B["joint_min"]
    b = have and gram_from_comma >= B["joint_min"]
    c = have and gram_own >= B["joint_min"]
    d = have and min(comma_from_gram) >= B["joint_min"]
    e = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["held"]))
    return {"pred_a_comma_in_distribution": bool(a), "pred_b_comma_to_grammatical": bool(b),
            "pred_c_grammatical_in_distribution": bool(c), "pred_d_grammatical_to_comma": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V477_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V477_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_particle_grammatical_slot_v477",
              "candidate_id": "corpus.unit_particle_grammatical_slot_v477", "bars": BARS,
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
