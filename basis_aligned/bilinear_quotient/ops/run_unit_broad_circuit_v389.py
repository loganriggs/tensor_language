#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v389: LEAVE-ONE-OUT TRANSFER. Does the direction reach a shape it was never fitted on?

WHY THIS IS A STRONGER CLAIM THAN THE POOLED ONES. Eight rungs have now fitted one direction on rows POOLED from
several shapes and scored it per shape, and every shape has been covered -- eleven of them at 0.982 to 1.009 in v387,
including a competing-cue sentence where a distractor of the opposite class sits between the cue and the readout. But
in every one of those, the scored shape contributed rows to the fit. "Covered" has meant "the pooled objective could
satisfy this shape too", which is weaker than what the word suggests.
This rung removes the shape from the fit entirely. For each of three held-out shapes, the greedy unit set AND the
rank-1 direction are computed on the OTHER TEN shapes only, and then evaluated on the held-out shape's rows. Nothing
about the held-out shape touches the units, the direction, the controls or the mean.
    U_competing_cue   `The {agent} woke the cadets who calmed, of course,`  the distractor case: the only shape whose
                      correct answer depends on the model reading the NEAREST cue rather than the matrix verb, which
                      two CPU screens established it does (32 of 32 rows dropped when keyed to the matrix verb, 0 of
                      32 when keyed to the nearest).
    T_clause_between  `Near the {object} the {agent} {verb}, once the lanterns had been put out,`  a full subordinate
                      clause, with its own verb, between cue and readout.
    M_question        `Did anyone notice the {agent} {verb}, really,`  interrogative mood.
Three held-out shapes rather than eleven because each costs its own greedy selection and its own fit; these three are
the ones whose failure would mean something distinct (relation, span, mood).

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_all_transfer      every held-out shape reaches transfer extraction >= joint_min = 0.80 of its own exact-set
                           effect, using units and a direction fitted without it.                        prior 55%
  pred_b_units_transfer    the unit set chosen on the other ten reaches exact-set recovery >= unit_min = 0.80 on the
                           held-out shape. This is the component half of transfer and can fail while the direction
                           half holds -- v379 saw exactly that at 0.763 units with 1.043 extraction. prior 55%
  pred_c_control_clean     own-C damage UB975 <= 0.01 on each held-out shape.                            prior 55%
  pred_d_competing_cue_transfers  the COMPETING-CUE shape specifically transfers at >= 0.80. Worked example: 0.92
                           means a direction fitted only on sentences where the governing verb is adjacent to the
                           readout still carries the case where it is NOT, which would say the circuit is keyed to
                           the nearest cue rather than to adjacency of the subject's verb; 0.41 means the pooled
                           coverage at v387 came from the competing-cue rows being IN the fit, and the honest
                           reading of v387 changes.                                                      prior 50%
  pred_e_some_holdout_resists  at least one of the three falls below 0.80. Registered because eight rungs of
                           universal coverage were all in-sample, and out-of-sample is where a limit should first
                           appear; if this fails as well, the generality is real rather than an artefact of pooling.
                                                                                                          prior 50%
COUNTING. Nothing here is counted. The output is a transfer figure for an entry that already counts once.
Smoke: V389_SMOKE=<out.json> (CPU, V389_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v389_result.json"
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
HELDOUT = ("U_competing_cue", "T_clause_between", "M_question")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v389", "shapes": len(SHAPES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    per = R.get("held_out", {})
    ok = bool(per) and all("error" not in s for s in per.values())
    tx = {k: s.get("transfer_extraction") for k, s in per.items()}
    a = ok and all(v is not None and v >= B["joint_min"] for v in tx.values())
    b = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"] for s in per.values())
    c = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"] for s in per.values())
    d = ok and tx.get("U_competing_cue") is not None and tx["U_competing_cue"] >= B["joint_min"]
    e = ok and any(v is not None and v < B["joint_min"] for v in tx.values())
    return {"pred_a_all_transfer": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_competing_cue_transfers": bool(d), "pred_e_some_holdout_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V389_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V389_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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

    held_out = {}
    for target in HELDOUT:
        try:
            pooled_fit, pooled_c_fit = [], []
            held_rows = c_rows = None
            for tag, name, _spec in SHAPES:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                if tag == target:
                    held_rows, c_rows = cut(held_half(a1)), cut(held_half(cc))
                    continue                      # the target contributes NOTHING to the fit
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
            p_held = g.prepare(backend, held_rows, **V)
            p_c = g.prepare(backend, c_rows)
            e_exact = ext(p_held, units)
            cdmg = dmg(p_c, units, q_joint, mu_joint)
            held_out[target] = {"n_units": len(units), "units_recovery": e_exact,
                                "transfer_extraction": round(ext(p_held, units, q=q_joint) / e_exact, 3) if abs(e_exact) > 1e-6 else None,
                                "A1_damage": dmg(p_held, units, q_joint, mu_joint),
                                "C_damage": cdmg, "C_ub975": cdmg["ce_ub975"],
                                "fit_shapes": [t for t, _, _ in SHAPES if t != target]}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            held_out[target] = {"error": f"{type(err).__name__}: {err}"}
        print(f"[{target}] {round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"held_out": held_out}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v389",
              "candidate_id": "corpus.unit_broad_circuit_v389", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                         "lam": LAM, "steps": steps, "lr": LR, "complement_weight": CW},
              "held_out": held_out, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "held_out": {k: (v.get("error") or {"transfer": v["transfer_extraction"], "units": v["units_recovery"]})
                                   for k, v in held_out.items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
