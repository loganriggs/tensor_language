#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v375: THE MERGE AUDIT. Ten counted frame-copy groups, each asked whether one direction covers all of its shapes.

WHY THIS RUNS AGAINST MY OWN COUNT. v371 and v373 both found that a SINGLE greedy unit set and a SINGLE constrained
rank-1 direction, fitted on rows pooled across sentence shapes, reach the specialists' level on every shape:
0.993/0.994/1.000/1.008 against 0.991/0.998/0.998/0.998 in the particle class over four shapes, and 1.001/0.988/1.025
against 0.998 in the preposition class over three. Pooled-unit recovery was 0.806 to 0.957 and own-C damage 0.0018 to
0.0096. In both, pred_e_generality_costs came out FALSE: generality cost nothing.
Those same shapes are pairwise SEPARABLE at rank 1, which is the criterion under which I counted each of them as its
own circuit today. Both facts hold, and together they say the criterion is too weak: rank-1 separability shows that a
SPECIALISED direction exists which spares the other shapes, not that the mechanism differs. Where a joint direction
covers every shape at specialist level, the shapes are one circuit and the corpus has been counting views of it.
FOURTEEN frame copies are currently counted. Two of their groups are already tested (v371, v373). This rung tests the
other TEN, one joint fit per group, so the merge is decided per group from a receipt rather than by extrapolating two
results across twelve:
    particle_out_down   out_down + fb + fc          prep_from_for     from_for + fb
    particle_out_up     out_up + fc                 prep_into_with    into_with + fb
    prep_at_to          at_to + fb                  prep_by_from      by_from + fc
    prep_from_about     from_about + fb             prep_for_at       for_at + fc
    prep_of_into        of_into + fc                prep_through_with through_with + fc
Roughly forty seconds per group.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_groups_merge      in at least k_groups = 8 of the 10 groups, the joint direction reaches extraction
                           >= joint_min = 0.80 of the exact-set effect on EVERY shape in that group. This is the
                           merge verdict. Worked example: a group whose two shapes score 0.95 and 0.88 counts toward
                           the eight; a group scoring 0.95 and 0.62 does not, however good its average.  prior 70%
  pred_b_units_transfer    in at least 8 of the 10, the pooled greedy unit set reaches exact-set recovery
                           >= unit_min = 0.80 on every shape. Separates "different components" from "same components,
                           different direction".                                                        prior 75%
  pred_c_control_clean     in at least 8 of the 10, the joint direction's own-C damage UB975 stays <= 0.01 on every
                           shape, so a merge is never bought with a direction that damages the lexical control.
                                                                                                        prior 60%
  pred_d_no_error          every group returns a receipt. An error is recorded per group and counts as a failure of
                           this predicate rather than being silently dropped.                            prior 90%
  pred_e_some_group_resists  at least ONE of the ten groups does NOT merge. Registered deliberately in the direction
                           that would complicate my life: if every group merges, the frame axis produced no distinct
                           circuits at all and the count loses fourteen; if some resist, those are the shape pairs
                           where the mechanism genuinely differs and they are worth naming.              prior 45%
COUNTING. Nothing here is counted. Every group that merges REMOVES one or two entries from the corpus count, and I
will apply the result to ops/circuit_count.py from this receipt -- listing the merged groups explicitly in the tool so
the deduction is auditable -- rather than adjusting a number by hand.
Smoke: V375_SMOKE=<out.json> (CPU, V375_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v375_result.json"
GROUPS = {
    "particle_out_down": ("verb_particle_out_down", "verb_particle_fb_out_down", "verb_particle_fc_out_down"),
    "particle_out_up": ("verb_particle_out_up", "verb_particle_fc_out_up"),
    "prep_at_to": ("verb_preposition_at_to", "verb_preposition_fb_at_to"),
    "prep_from_about": ("verb_preposition_from_about", "verb_preposition_fb_from_about"),
    "prep_from_for": ("verb_preposition_from_for", "verb_preposition_fb_from_for"),
    "prep_into_with": ("verb_preposition_into_with", "verb_preposition_fb_into_with"),
    "prep_by_from": ("verb_preposition_by_from", "verb_preposition_fc_by_from"),
    "prep_for_at": ("verb_preposition_for_at", "verb_preposition_fc_for_at"),
    "prep_of_into": ("verb_preposition_of_into", "verb_preposition_fc_of_into"),
    "prep_through_with": ("verb_preposition_through_with", "verb_preposition_fc_through_with"),
}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "k_groups": 8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v375", "shapes": len(SHAPES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and all("error" not in g for g in G.values())
    covers = [k for k, g in G.items() if "error" not in g
              and all(s.get("joint_extraction") is not None and s["joint_extraction"] >= B["joint_min"]
                      for s in g["per_shape"].values())]
    units_ok = [k for k, g in G.items() if "error" not in g
                and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"]
                        for s in g["per_shape"].values())]
    clean = [k for k, g in G.items() if "error" not in g
             and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                     for s in g["per_shape"].values())]
    a = ok and len(covers) >= B["k_groups"]
    b = ok and len(units_ok) >= B["k_groups"]
    c = ok and len(clean) >= B["k_groups"]
    d = ok
    e = ok and len(covers) < len(G)
    return {"pred_a_groups_merge": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_no_error": bool(d), "pred_e_some_group_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V375_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V375_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=TARGET,
                                                     min_gain=MIN_GAIN, max_units=max_units)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v375",
              "candidate_id": "corpus.unit_broad_circuit_v375", "bars": BARS,
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
