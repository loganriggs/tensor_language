#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets and directions frozen from the v197 receipt; arms and bars fixed before the run.
"""v199: why does a rank-1 removal damage MORE than full-rank mean-ablation of the same set on the possessive sets?

v197 (amended battery, all 24): on the four possessive constructions the A1 diff-in-means direction's CE damage is 1.33-2.27x the
full-rank mean-ablation of the whole set (adjacent 0.52 vs 0.23, long_simple 0.58 vs 0.41, medial 0.60 vs 0.42, number 0.64 vs
0.48) and the A2-own share is 1.53-1.73 -- over the row-5 band's upper bound, while non-possessive sets sit at 0.9-1.3. Magnitudes
above are printed from the v197 receipt before this file was written. In block-live removal (v51.removal, q = per-block basis)
each unit is `live + q q^T (mu - live)`: the axis is clamped to the pooled mean, the complement stays LIVE and responds to the
upstream removals. With q = None every unit is set to mu entirely, so downstream complements are clamped. The two readings:
  (induced) the downstream units' live complements RESPOND to the upstream axis removal and that response carries the extra
            damage (memory item 15 in the patching direction: rank-1 > exact because exact clamps the induced response);
  (restore) the pooled mean's complement, when written at a unit, RESTORES part of the answer (the excess is an upstream effect
            and does not need downstream units live).
Arms per set (A1 held rows, mu and direction from the FIT rows, exactly as v197): R1 = rank-1 at every block (v197's dim A1);
F = full rank at every block (v197's a1_full_ceiling); split the set's blocks by layer into the earliest half U and the rest D;
H_uD = rank-1 at U, full rank at D (blocks absent from the q dict are patched at full rank -- g.forward_units contract);
H_Ud = full rank at U, rank-1 at D; R1_up / F_up = removal on the U units ONLY (D untouched, live) at rank 1 / full rank.
Sets: possessive_adjacent, possessive_long_simple, possessive_medial, possessive_number (over band); possessive_argument,
possessive_verbfinal (in band, 1.22 / 1.13); correlative_both_either (1.33 A1 share but 12 units), finiteness_selection (0.98)
as non-possessive controls. 8 sets x 6 removals x 32 held rows: ~1 min GPU.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_instrument   -> R1 and F reproduce v197's dim A1 CE and a1_full_ceiling CE within 0.03 on all 8 (same rows, mu, direction:
                         deterministic, disclosed as not an independent outcome).
  pred_b_induced      -> on the 4 over-band possessives H_uD (downstream clamped) falls to within [0.7, 1.3] x F, i.e. the excess
                         needs the downstream complements live.                                              prior 55%
  pred_c_not_restore  -> on the 4 over-band possessives R1_up / F_up <= 1.2, i.e. rank-1 vs full at the upstream half alone does
                         not produce the excess (refutes 'restore').                                          prior 55%
  pred_d_downstream_carries -> on the 4 over-band possessives (H_Ud - F) / (R1 - F) >= 0.5: keeping the downstream half at rank 1
                         keeps at least half of the excess.                                                   prior 50%
  pred_e_controls     -> on the 4 control sets |H_uD / F - 1| <= 0.3 AND R1 / F within [0.85, 1.4] (the in-band sets stay in band
                         under the same arms).                                                                prior 65%
Smoke: V199_SMOKE=<out.json> (CPU, V199_SMOKE_ROWS=4, V199_SMOKE_NAMES=possessive_adjacent).
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
import run_unit_tier3_batch_amended_all_v197 as v197

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_row5_ceiling_induced_response_v199_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
OVER = ("possessive_adjacent", "possessive_long_simple", "possessive_medial", "possessive_number")
CONTROLS = ("possessive_argument", "possessive_verbfinal", "correlative_both_either", "finiteness_selection")
BARS = {"instr_tol": 0.03, "induced_band": [0.7, 1.3], "restore_max": 1.2, "downstream_min": 0.5, "control_tol": 0.3, "control_r1_band": [0.85, 1.4]}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_row5_ceiling_induced_response_v199", "sets": len(OVER) + len(CONTROLS), "arms": 6,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "arms" in r and "error" not in r}
    ok = bool(good)
    ce = lambda n, arm: good[n]["arms"][arm]["ce_damage"]
    ratio = lambda x, y: (x / y) if abs(y) > 1e-6 else float("inf")
    a = ok and len(good) == 8 and all(abs(ce(n, "R1") - good[n]["v197_dim_a1"]) <= B["instr_tol"] and abs(ce(n, "F") - good[n]["v197_full"]) <= B["instr_tol"] for n in good)
    b = ok and all(n in good and B["induced_band"][0] <= ratio(ce(n, "H_uD"), ce(n, "F")) <= B["induced_band"][1] for n in OVER)
    c = ok and all(n in good and ratio(ce(n, "R1_up"), ce(n, "F_up")) <= B["restore_max"] for n in OVER)
    d = ok and all(n in good and ratio(ce(n, "H_Ud") - ce(n, "F"), ce(n, "R1") - ce(n, "F")) >= B["downstream_min"] for n in OVER)
    e = ok and all(n in good and abs(ratio(ce(n, "H_uD"), ce(n, "F")) - 1) <= B["control_tol"]
                   and B["control_r1_band"][0] <= ratio(ce(n, "R1"), ce(n, "F")) <= B["control_r1_band"][1] for n in CONTROLS)
    return {"pred_a_instrument": bool(a), "pred_b_induced": bool(b), "pred_c_not_restore": bool(c),
            "pred_d_downstream_carries": bool(d), "pred_e_controls": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V199_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V199_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    which = [n for n in OVER + CONTROLS if not smoke or n in os.environ.get("V199_SMOKE_NAMES", "possessive_adjacent").split(",")]
    prior = json.loads(V197.read_text())["behaviours"]
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage")}

    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            if "error" in prior.get(n, {"error": "missing"}):
                raise RuntimeError(f"v197 has no clean receipt for {n}")
            units = list(prior[n]["units"])
            m = importlib.import_module(f"circuit_fast_screen_candidate_{v197.NAMES[n]}")
            a1 = g.rows_of(m, "A1")
            fit, held = g.prepare(backend, cut(fit_half(a1))), g.prepare(backend, cut(held_half(a1)))
            mu = mu_of(fit, units)
            q1 = g.block_diff_in_means(backend, fit, units)
            blocks = list(g.blocks_of(units).keys())          # ordered as given; sort by layer for the split
            by_layer = sorted(blocks, key=lambda k: (k[0], k[1]))
            n_up = (len(by_layer) + 1) // 2
            U_blocks, D_blocks = by_layer[:n_up], by_layer[n_up:]
            U_units = [u for u in units if g.block_key(u) in U_blocks]
            arms = {"R1": dmg(held, units, q1, mu), "F": dmg(held, units, None, mu),
                    "H_uD": dmg(held, units, {k: q1[k] for k in U_blocks}, mu),
                    "H_Ud": dmg(held, units, {k: q1[k] for k in D_blocks}, mu) if D_blocks else None,
                    "R1_up": dmg(held, U_units, {k: q1[k] for k in U_blocks}, mu), "F_up": dmg(held, U_units, None, mu)}
            if arms["H_Ud"] is None:
                arms["H_Ud"] = dict(arms["R1"])   # a single-block set has no downstream half: H_Ud == R1 by construction (disclosed)
            r = lambda x, y: round(x / y, 3) if abs(y) > 1e-6 else None
            R[n] = {"units": units, "n_units": len(units), "blocks_by_layer": [f"{k[0]}:{k[1]}" for k in by_layer], "n_upstream_blocks": n_up,
                    "arms": arms, "ratios": {"R1_over_F": r(arms["R1"]["ce_damage"], arms["F"]["ce_damage"]),
                                             "HuD_over_F": r(arms["H_uD"]["ce_damage"], arms["F"]["ce_damage"]),
                                             "R1up_over_Fup": r(arms["R1_up"]["ce_damage"], arms["F_up"]["ce_damage"]),
                                             "downstream_share_of_excess": r(arms["H_Ud"]["ce_damage"] - arms["F"]["ce_damage"], arms["R1"]["ce_damage"] - arms["F"]["ce_damage"])},
                    "v197_dim_a1": prior[n]["arms"]["dim"]["A1"]["ce_damage"], "v197_full": prior[n]["a1_full_ceiling"]["ce_damage"],
                    "n_rows": {"fit": len(fit.rows), "held": len(held.rows)}, "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one set must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "arms")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_row5_ceiling_induced_response_v199", "candidate_id": "corpus.unit_row5_ceiling_induced_response_v199",
              "bars": BARS, "over_band": list(OVER), "controls": list(CONTROLS), "sets": R,
              "protocol": {"rows": "A1 within-direction: mu/direction from FIT rows[0::4]+rows[1::4], removal on HELD rows[2::4]+rows[3::4]",
                           "split": "blocks sorted by layer; upstream = earliest ceil(n/2) blocks", "units_source": "v197"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "ratios": {n: R[n].get("ratios") for n in R}}, indent=2))


if __name__ == "__main__":
    main()
