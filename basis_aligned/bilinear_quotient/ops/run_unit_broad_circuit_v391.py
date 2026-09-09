#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v391: does OUT-OF-SAMPLE transfer hold in the preposition class -- and does it reach the two shapes that failed
their own battery?

WHY. v389 established real transfer for the particle entry: units and a rank-1 direction fitted on ten shapes, with
the target contributing nothing to the fit, reached 0.951, 0.996 and 0.960 on three held-out shapes, the last of them
the competing-cue sentence the fit had never seen. Everything in that result is ONE mapping pair in ONE readout class,
and the preposition entries have breadth figures with no transfer figures at all. This rung repeats the design for
cared -> ` about` / voted -> ` for` over its five shapes, and it holds out the two that are the most informative:
    D_denial   `Nobody doubted the {agent} {verb}, honestly,`      MISSED rows 2 and 4 of its own battery at v361
    E_spread   `Word spread that the {agent} {verb}, evidently,`   MISSED rows 2 and 4 of its own battery at v361
    C_report   `It turned out the {agent} {verb}, apparently,`     passed its battery; held out as the comparison
v379 showed both failures are COVERED when their rows are in the pooled fit (1.043 each, above the three shapes that
passed). That could still have been the pool absorbing them. Holding them OUT asks whether a direction fitted only on
shapes that passed their batteries reaches shapes that did not -- which is the difference between "pooling rescues a
weak fit" and "the mechanism was there all along and the standalone battery could not see it on sixteen rows".

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_all_transfer      every held-out shape reaches transfer extraction >= joint_min = 0.80 of its own exact-set
                           effect, from units and a direction fitted without it.                        prior 55%
  pred_b_units_transfer    the unit set chosen on the other four reaches exact-set recovery >= unit_min = 0.80 on the
                           held-out shape. v379 measured 0.763 for the denial shape when its rows WERE in the fit, so
                           this is the predicate most likely to fail and the receipt separates the component half of
                           transfer from the direction half.                                             prior 40%
  pred_c_control_clean     own-C damage UB975 <= 0.01 on each held-out shape.                            prior 55%
  pred_d_battery_failures_transfer  BOTH battery-failing shapes transfer at >= 0.80 from a fit that contains only
                           battery-passing shapes. Worked example: 0.88 and 0.91 gives TRUE and a row-2/row-4 miss on
                           a capable cell is a small-sample fitting failure rather than an absent mechanism, which
                           changes how a class of nulls in this corpus should be read; 0.88 and 0.42 gives FALSE and
                           at least one of those misses is real.                                         prior 45%
  pred_e_some_holdout_resists  at least one of the three falls below 0.80. Nine rungs have found nothing that
                           resists, so this is registered as the outcome that would finally locate a limit, at a
                           prior above pred_a because the two held-out shapes here failed a bar of their own.
                                                                                                          prior 55%
COUNTING. Nothing here is counted; this entry already counts once.
Smoke: V391_SMOKE=<out.json> (CPU, V391_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v391_result.json"
SHAPES = (("A_locative", "verb_preposition_about_for", 0.998),
          ("B_agreement", "verb_preposition_fb_about_for", 0.998),
          ("C_report", "verb_preposition_fc_about_for", 0.998),
          ("D_denial", "verb_preposition_fd_about_for", None),
          ("E_spread", "verb_preposition_fe_about_for", None))
HELDOUT = ("D_denial", "E_spread", "C_report")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v391", "shapes": len(SHAPES),
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
    d = ok and all(tx.get(k) is not None and tx[k] >= B["joint_min"] for k in ("D_denial", "E_spread"))
    e = ok and any(v is not None and v < B["joint_min"] for v in tx.values())
    return {"pred_a_all_transfer": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_battery_failures_transfer": bool(d), "pred_e_some_holdout_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V391_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V391_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v391",
              "candidate_id": "corpus.unit_broad_circuit_v391", "bars": BARS,
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
