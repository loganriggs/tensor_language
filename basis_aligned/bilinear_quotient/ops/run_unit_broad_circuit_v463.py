#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v463: does the surface-form binding replicate on a SECOND mapping, or is it about/for's quirk?

WHAT IS ESTABLISHED, ON ONE MAPPING. Fitted on five declarative frames whose cue is the -ed form, a rank-1
direction reaches 0.931 on a wh-interrogative that keeps the -ed form and only 0.658 on a declarative that switches
to the bare verb under do-support; the split across all eight cells is clean, with units_recovery 0.898-1.012
throughout, so the sites carry every shape and only the direction is bound (v459). Adding one bare-form cell to the
fit then lifts the other, still held out, 0.575 -> 0.839 at no cost to the fitted frames (v461).
EVERY ONE OF THOSE NUMBERS COMES FROM about/for (cared vs voted). A binding that holds for one cue pair and no other
is that behaviour's quirk, not a fact about these circuits, and the breadth recipe drawn from it -- one cell per
surface form -- would be worthless anywhere else. This rung repeats the v459 design on at/to (aimed vs appealed),
with the two new cells derived from fi and fk by substitution so the SHAPES are identical and only the mapping
differs. Both were capability-checked on CPU before this runner was written (A1 32/32 rows kept, 0 dropped, each).
    FIT      at_to + fb fd fj fm     five frames, all -ed cues, exactly as the about/for fit was
    fi_at_to `The pilot did really aim, then,`      BARE form, clause declarative -> the FORM cell
    fk_at_to `Who noticed the pilot aimed, then,`   wh-interrogative, -ed kept    -> the TYPE cell
    CONTROL  about_for, the mapping all the earlier numbers came from, scored on ABSOLUTE recovery
Rank stays at 1; budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution   held-out rows of the fitted at/to frames reach MINIMUM joint extraction
                           joint_min = 0.80. If this fails nothing else is readable.                  prior 90%
  pred_b_bare_form_fails   the bare-form cell lands BELOW 0.80, as it did on about/for at 0.658. prior 65%
  pred_c_ed_form_transfers the wh-interrogative cell, which keeps the -ed form, reaches 0.80, as it did on
                           about/for at 0.931.                                                        prior 70%
  pred_d_different_mapping about/for's ABSOLUTE recovery under this direction stays at or below cross_max = 0.30,
                           the same control that came in at 0.016 with the mappings swapped.          prior 80%
  pred_e_units_available   every evaluated cell except the control reports units_recovery >= unit_min = 0.80, so a
                           failure is attributable to the direction and not to units that do not reach the shape.
                                                                                                      prior 80%
HOW THE OUTCOMES READ, fixed before the run: b TRUE with c TRUE replicates the surface-form binding on a second
mapping and makes it a statement about these circuits rather than one cue pair. b FALSE with c TRUE means at/to's
direction is NOT form-bound and the account is about/for-specific -- the recipe would then need a per-mapping check
before use. b TRUE with c FALSE means at/to's direction fails on BOTH new shapes, which is a narrower direction
rather than a form-bound one, and would need the same separation v459 ran before anything is named.
SCOPE. Two mappings in one construction at this rank and budget. A replication here does not make it a fact about
other constructions, and the next question would be a second construction, not a third preposition pair.
Smoke: V463_SMOKE=<out.json> (CPU, V463_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v463_result.json"
# the v459 design repeated on a second mapping; shapes identical, only the cue pair differs
VP = "verb_preposition_"
FIT_CELLS = tuple(VP + c for c in ("at_to", "fb_at_to", "fd_at_to", "fj_at_to", "fm_at_to"))
FORM_CELL, TYPE_CELL = VP + "fi_at_to", VP + "fk_at_to"
NEW_TYPE = (FORM_CELL, TYPE_CELL)
CONTROL = VP + "about_for"                   # the mapping every earlier number came from
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_AE": FIT_CELLS}
RELAXED = "ALL"
PRIOR = {"about_for_form": 0.658, "about_for_type": 0.931}      # v459, same shapes, other mapping
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v463", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    g0 = G.get("fit_AE", {})
    ok = bool(g0) and "error" not in g0 and set(g0.get("per_shape", {})) == set(EVAL_CELLS)
    ps = g0.get("per_shape", {}) if ok else {}
    je = lambda c: ps.get(c, {}).get("joint_extraction")
    ur = lambda c: ps.get(c, {}).get("units_recovery")
    fit_v = [je(c) for c in FIT_CELLS]
    form_v, type_v = je(FORM_CELL), je(TYPE_CELL)
    have = ok and all(x is not None for x in fit_v + [form_v, type_v])
    a = have and min(fit_v) >= B["joint_min"]
    b = have and form_v < B["joint_min"]
    c = have and type_v >= B["joint_min"]
    cross = ps.get(CONTROL, {}).get("cross_abs_recovery")
    d = ok and cross is not None and abs(cross) <= B["cross_max"]
    e = ok and all(ur(x) is not None and ur(x) >= B["unit_min"] for x in FIT_CELLS + NEW_TYPE)
    return {"pred_a_in_distribution": bool(a), "pred_b_bare_form_fails": bool(b),
            "pred_c_ed_form_transfers": bool(c), "pred_d_different_mapping": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V463_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V463_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in EVAL_CELLS:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                if cell in cells:                       # ONLY frames A-E enter the fit
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
            for cell in EVAL_CELLS:
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v463",
              "candidate_id": "corpus.unit_broad_circuit_v463", "bars": BARS,
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
