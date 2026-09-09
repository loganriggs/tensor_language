#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v387: THE CIRCUIT IS A RECENCY READER -- does the same direction carry the case that proves it?

WHAT TWO CPU SCREENS JUST ESTABLISHED, at no GPU cost. Every cell in this corpus has contained exactly ONE
particle-selecting verb, so "the model tracks the matrix verb" and "the model tracks the nearest cue" have never made
different predictions. `The {agent} woke the cadets who calmed, of course,` separates them: the matrix verb is `woke`
(-> ` up`) and the nearest verb is `calmed` (-> ` down`). Screened both ways on identical sentences:
    keyed to the MATRIX verb    (verb_particle_fu_up_down)  A1 32 of 32 rows DROPPED, A2 32 of 32 dropped
    keyed to the NEAREST verb   (verb_particle_fv_down_up)  A1 -1.883/+1.705 drop 0, A2 -2.435/+2.365 drop 0
The donor never once beat the base on the matrix verb's axis, and beat it on every row on the nearest verb's axis. The
model follows the NEAREST particle-selecting verb, not the syntactic subject's verb. That reframes v383's breadth of
ten: all ten shapes put the matrix verb adjacent to the readout, so none of them distinguished the two accounts, and
the generality I measured is generality over sentence SHAPE at a fixed structural relation -- not over the relation.
THE CAUSAL QUESTION THIS RUNG ASKS. The screens are behavioural. Adding the competing-cue shape as an ELEVENTH member
of the pool asks whether the SAME unit set and the SAME rank-1 direction that carry the other ten also carry the case
where a distractor of the opposite class sits between the cue and the readout. If yes, the object I have been counting
is one recency reader and its description should say so. If the competing-cue shape resists while the ten hold, the
distractor case recruits different machinery and I have finally located an edge -- after seven rungs in which nothing
resisted.
    A_locative B_agreement C_report D_denial J_temporal M_question P_longparen R_relclause_subject
    S_conjoined_subject T_clause_between   (the ten from v383, joint extraction 0.984-1.010)
    U_competing_cue                        (`The {agent} woke the cadets who calmed, of course,`)

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_ten_still_covered   the joint direction still reaches >= joint_min = 0.80 on each of the ORIGINAL ten shapes
                           with the competing-cue shape added to the pool. If adding one hard shape degrades the ten,
                           the pooled fit is being pulled apart and the receipt should show it.          prior 75%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape
                           INCLUDING the competing-cue one.                                              prior 50%
  pred_c_control_clean     own-C damage UB975 <= 0.01 on every shape.                                    prior 50%
  pred_d_competing_cue_covered  the COMPETING-CUE shape specifically reaches >= 0.80 under the joint direction. This
                           is the edge question. Worked example: 0.91 gives TRUE and the entry is one recency reader
                           across eleven shapes; 0.45 gives FALSE and the distractor case is separate machinery,
                           which is the first located limit in seven rungs.                              prior 45%
  pred_e_some_shape_resists  at least one of the eleven falls below 0.80. Registered at a higher prior than pred_d
                           because I expect the distractor to be the shape that finally resists, and because six
                           previous rungs of universal coverage came from stimuli that were too easy.    prior 55%
COUNTING. Nothing here is counted. The competing-cue cell has no four-row battery and is not a candidate entry; it is
a probe of what the existing entry computes.
Smoke: V387_SMOKE=<out.json> (CPU, V387_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v387_result.json"
SHAPES = (("A_locative", "verb_particle_up_down", 0.991),
          ("B_agreement", "verb_particle_fb_up_down", 0.998),
          ("C_report", "verb_particle_fc_up_down", 0.998),
          ("D_denial", "verb_particle_fd_up_down", 0.998),
          ("J_temporal", "verb_particle_fj_up_down", None),
          ("M_question", "verb_particle_fm_up_down", None),
          ("P_longparen", "verb_particle_fp_up_down", None),
          ("R_relclause_subject", "verb_particle_fr_up_down", None),
          ("S_conjoined_subject", "verb_particle_fs_up_down", None),
          ("T_clause_between", "verb_particle_ft_up_down", None),
          ("U_competing_cue", "verb_particle_fv_down_up", None))
COMPETING = "U_competing_cue"
STRESS = ("R_relclause_subject", "S_conjoined_subject", "T_clause_between")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v387", "shapes": len(SHAPES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    per = R.get("per_shape", {})
    ok = bool(per) and all("error" not in s for s in per.values())
    jx = {k: s.get("joint_extraction") for k, s in per.items()}
    ten = {k: v for k, v in jx.items() if k != COMPETING}
    a = ok and all(v is not None and v >= B["joint_min"] for v in ten.values())
    b = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"] for s in per.values())
    c = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"] for s in per.values())
    d = ok and jx.get(COMPETING) is not None and jx[COMPETING] >= B["joint_min"]
    e = ok and any(v is not None and v < B["joint_min"] for v in jx.values())
    return {"pred_a_ten_still_covered": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_competing_cue_covered": bool(d), "pred_e_some_shape_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V387_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V387_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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

    pooled_fit, pooled_c_fit, per_shape_rows = [], [], {}
    for tag, name, _spec in SHAPES:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
        a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
        per_shape_rows[tag] = (cut(held_half(a1)), cut(held_half(cc)))
        pooled_fit += cut(fit_half(a1))
        pooled_c_fit += cut(fit_half(cc))

    P_fit = g.prepare(backend, pooled_fit, **V)
    P_cfit = g.prepare(backend, pooled_c_fit)
    singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
    units = list(greedy["chosen"])
    mu_joint = mu_of(P_fit, units)
    q_joint, hist = g.fit_block_subspace_constrained(
        backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
        controls=(P_cfit,), control_weight=LAM, mu=mu_joint)

    per_shape = {}
    for tag, name, spec in SHAPES:
        held_rows, c_rows = per_shape_rows[tag]
        try:
            p_held = g.prepare(backend, held_rows, **V)
            p_c = g.prepare(backend, c_rows)
            e_exact = ext(p_held, units)
            e_joint = ext(p_held, units, q=q_joint)
            per_shape[tag] = {
                "cell": name,
                "units_recovery": e_exact,
                "joint_extraction": round(e_joint / e_exact, 3) if abs(e_exact) > 1e-6 else None,
                "specialist_extraction": spec,
                "A1_damage": dmg(p_held, units, q_joint, mu_joint),
                "C_damage": dmg(p_c, units, q_joint, mu_joint),
                "n_held": len(p_held.base_batch.row_ids),
            }
            per_shape[tag]["C_ub975"] = per_shape[tag]["C_damage"]["ce_ub975"]
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            per_shape[tag] = {"cell": name, "error": f"{type(err).__name__}: {err}"}

    R = {"per_shape": per_shape}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v387",
              "candidate_id": "corpus.unit_broad_circuit_v387", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                         "lam": LAM, "steps": steps, "lr": LR, "complement_weight": CW},
              "joint_units": units, "n_joint_units": len(units),
              "pooled_fit_rows": len(P_fit.base_batch.row_ids), "pooled_control_rows": len(P_cfit.base_batch.row_ids),
              "final_loss": (round(float(hist[-1][-1]), 5) if isinstance(hist[-1], (list, tuple)) else round(float(hist[-1]), 5)) if hist else None,
              "per_shape": per_shape, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "n_joint_units": len(units),
                      "per_shape": {k: {kk: v[kk] for kk in ("joint_extraction", "specialist_extraction", "units_recovery")
                                        if kk in v} for k, v in per_shape.items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
