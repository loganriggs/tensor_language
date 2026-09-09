#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v403: does the NEW construction class transfer OUT OF SAMPLE, not just pool?

WHERE THIS SITS. v397 gave the corpus its first non-verb-cue circuit -- cue a function word (several / each), readout a
content noun's NUMBER -- and it passed all four rows first try. v401 pooled it across four sentence shapes on one
8-unit set and one direction at 0.974 to 0.996 with own-C 0.0007. Both of those scored shapes whose rows were IN the
fit, which is the weaker claim; v389 and v391 showed for the verb-cue circuits that the strong version -- fit without
the target, then evaluate on it -- also holds (0.951 to 0.996, and 0.939/1.027 for two shapes that had failed their
own battery). This rung asks the strong version for the new class.
For each of two held-out shapes the greedy unit set AND the rank-1 direction are computed on the other THREE shapes
only; the target contributes nothing to the units, the direction, the controls or the mean.
    D_denial   `Nobody doubted the {agent} inspected several damaged`   held out
    C_report   `It turned out the {agent} inspected several damaged`    held out
    (A_inspect and B_agreement remain in the fit; A is the shape that actually counts.)
Three fitting shapes is a thinner pool than the ten v389 used, so this is a harder test of transfer, not an easier
one, and I would rather report that than pad the pool to make the number look better.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_all_transfer      both held-out shapes reach transfer extraction >= joint_min = 0.80 of their own exact-set
                           effect, from units and a direction fitted without them.                       prior 60%
  pred_b_units_transfer    the unit set chosen on the other three reaches exact-set recovery >= unit_min = 0.80 on
                           the held-out shape. In-sample this class ran 0.898 to 0.914, the tightest spread of any
                           entry, which is why the bar is worth setting here.                            prior 65%
  pred_c_control_clean     own-C damage UB975 <= 0.01 on each held-out shape. In-sample it was 0.0007.   prior 70%
  pred_d_units_reach_085   the stricter version of pred_b: every held-out shape's unit recovery is at least 0.85, not
                           merely 0.80. Registered because 0.80 is the corpus bar and this class was tighter than any
                           other in sample, so 0.80 would be nearly free here while 0.85 can fail. Worked example:
                           recoveries of 0.91 and 0.87 give TRUE; 0.91 and 0.82 give FALSE while still passing
                           pred_b, which is exactly the distinction I want on the record.                prior 50%
  pred_e_some_holdout_resists  at least one held-out shape falls below 0.80. Thirteen rungs have found nothing below
                           that bar; this is the fourteenth chance and the first for a non-verb-cue class out of
                           sample.                                                                        prior 45%
COUNTING. Nothing here is counted; determiner_number_crates is still awaiting a separability rung, which needs a
sibling mapping to be separable against. That sibling is the next authoring job and this rung does not substitute for
it.
Smoke: V403_SMOKE=<out.json> (CPU, V403_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v403_result.json"
SHAPES = (("A_inspect", "determiner_number_crates", 0.898),
          ("B_agreement", "determiner_number_fb_crates", None),
          ("C_report", "determiner_number_fc_crates", None),
          ("D_denial", "determiner_number_fd_crates", None))
HELDOUT = ("D_denial", "C_report")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v403", "shapes": len(SHAPES),
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
    d = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= 0.85 for s in per.values())
    e = ok and any(v is not None and v < B["joint_min"] for v in tx.values())
    return {"pred_a_all_transfer": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_units_reach_085": bool(d), "pred_e_some_holdout_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V403_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V403_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v403",
              "candidate_id": "corpus.unit_broad_circuit_v403", "bars": BARS,
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
