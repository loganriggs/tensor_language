#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, upstream halves and directions fixed by the v197/v199 receipts; arms and bars fixed before the run.
"""v200: the rank-1 removal excess lives at the UPSTREAM half -- is it the live complement, and is it low-rank?

v199 (printed from its receipt before this file was written): on the four over-band possessive sets, removing the diff-in-means
axis at the upstream half of the blocks alone (downstream live) damages 0.18-0.24 CE, while clamping the same upstream units
ENTIRELY to the pooled mean damages -0.04 / 0.09 / 0.13 / 0.04 (adjacent / long_simple / medial / number): R1_up / F_up =
-6.3, 2.5, 1.9, 4.4. Clamping the downstream half did not remove the excess on adjacent (H_uD / F 1.69), so the 'induced
downstream response' reading of v197's row-5 overshoot is refuted there and the excess sits at the upstream blocks: the
unit with its axis clamped and its complement LIVE damages, the unit clamped to a constant does not. Two readings:
  (dose)      the damage is carried by the live complement continuously: interpolating the complement from live to mu,
              complement = mu + alpha (live - mu) with the axis clamped, gives damage decreasing in alpha -> 0.
  (low-rank)  a few more directions of the complement carry the excess: a rank-r removal (axis + top principal components of
              the FIT residual) falls to the full-rank value at small r.
Instrument: q = [sqrt(alpha) q1, sqrt(1-alpha) I] per block gives Q Q^T = alpha q1 q1^T + (1-alpha) I, i.e. the axis fully clamped
and the complement moved (1-alpha) of the way to mu (g.forward_units applies (d @ q) @ q^T without orthonormalising); alpha = 1 is
v199's R1_up, alpha = 0 is F_up. Rank ladder: Q_r = orthonormal [q1, top r-1 PCs of the FIT-row residual (live - mu) after q1],
r in {1, 2, 4, 8, 16, D}; r = D is F_up. Random control: a unit-norm random axis (seed 0) per block instead of q1 at rank 1.
Sets, units, upstream halves exactly as v199 (8 sets); A1 held rows; mu / q1 / PCs from FIT rows. ~15 s GPU.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_instrument -> alpha = 1 reproduces v199 R1_up within 0.02, alpha = 0 and r = D reproduce v199 F_up within 0.02, on all 8
                       (deterministic, disclosed as not an independent outcome).
  pred_b_dose       -> on the 4 over-band possessives damage is non-increasing along alpha = 1, .75, .5, .25, 0 (each step
                       >= -0.02) and damage(alpha=.5) - F_up >= 0.5 (damage(alpha=1) - F_up): half the complement carries
                       at least half the excess.                                                          prior 60%
  pred_c_not_low_rank -> on the 4 over-band possessives damage at r = 4 is >= damage(r=1) - 0.05: three extra principal
                       directions do not remove the excess.                                                prior 50%
  pred_d_random     -> a random rank-1 axis removal at the upstream half damages |CE| <= 0.05 on all 8 (the axis is what
                       matters, not the rank-1 operation).                                                prior 70%
  pred_e_control    -> finiteness_selection: |damage(alpha=1) - damage(alpha=0)| <= 0.05 (v199: 0.375 vs 0.384) and its
                       rank ladder stays within 0.05 of F_up at every r.                                   prior 80%
Smoke: V200_SMOKE=<out.json> (CPU, V200_SMOKE_ROWS=4, V200_SMOKE_NAMES=possessive_adjacent).
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
import run_unit_row5_ceiling_induced_response_v199 as v199

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_removal_complement_dose_v200_result.json"
V199 = ROOT / "circuits/followups/unit_row5_ceiling_induced_response_v199_result.json"
OVER, CONTROLS = v199.OVER, v199.CONTROLS
ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
RANKS = (1, 2, 4, 8, 16)
BARS = {"instr_tol": 0.02, "step_tol": -0.02, "half_dose_min": 0.5, "rank4_tol": 0.05, "random_max": 0.05, "control_tol": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_removal_complement_dose_v200", "sets": 8, "arms": len(ALPHAS) + len(RANKS) + 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "dose" in r and "error" not in r}
    ok = bool(good)
    dose = lambda n, a: good[n]["dose"][str(a)]["ce_damage"]
    rank = lambda n, r: good[n]["rank"][str(r)]["ce_damage"]
    a = ok and len(good) == 8 and all(abs(dose(n, 1.0) - good[n]["v199_R1_up"]) <= B["instr_tol"] and abs(dose(n, 0.0) - good[n]["v199_F_up"]) <= B["instr_tol"]
                                      and abs(rank(n, "D") - good[n]["v199_F_up"]) <= B["instr_tol"] for n in good)
    def dose_ok(n):
        seq = [dose(n, x) for x in ALPHAS]
        steps = all(seq[i + 1] - seq[i] <= -B["step_tol"] for i in range(len(seq) - 1))
        excess = seq[0] - seq[-1]
        return steps and (dose(n, 0.5) - seq[-1]) >= B["half_dose_min"] * excess
    b = ok and all(n in good and dose_ok(n) for n in OVER)
    c = ok and all(n in good and rank(n, 4) >= rank(n, 1) - B["rank4_tol"] for n in OVER)
    d = ok and len(good) == 8 and all(abs(good[n]["random"]["ce_damage"]) <= B["random_max"] for n in good)
    f = "finiteness_selection"
    e = ok and f in good and abs(dose(f, 1.0) - dose(f, 0.0)) <= B["control_tol"] and all(abs(rank(f, r) - good[f]["v199_F_up"]) <= B["control_tol"] for r in RANKS)
    return {"pred_a_instrument": bool(a), "pred_b_dose": bool(b), "pred_c_not_low_rank": bool(c), "pred_d_random": bool(d), "pred_e_control": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V200_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    dev = backend.device
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V200_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    which = [n for n in OVER + CONTROLS if not smoke or n in os.environ.get("V200_SMOKE_NAMES", "possessive_adjacent").split(",")]
    prior = json.loads(V199.read_text())["sets"]
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]

    def unit_vals(p, u):
        return torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).to(dev)

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage")}

    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            if "error" in prior.get(n, {"error": "missing"}):
                raise RuntimeError(f"v199 has no clean receipt for {n}")
            units = list(prior[n]["units"])
            n_up = prior[n]["n_upstream_blocks"]
            m = importlib.import_module(f"circuit_fast_screen_candidate_{v197.NAMES[n]}")
            a1 = g.rows_of(m, "A1")
            fit, held = g.prepare(backend, cut(fit_half(a1))), g.prepare(backend, cut(held_half(a1)))
            mu = {u: unit_vals(fit, u).mean(0) for u in units}
            q1 = g.block_diff_in_means(backend, fit, units)
            by_layer = sorted(g.blocks_of(units).keys(), key=lambda k: (k[0], k[1]))
            U_blocks = by_layer[:n_up]
            U_units = [u for u in units if g.block_key(u) in U_blocks]
            blk_units = {k: [u for u in U_units if g.block_key(u) == k] for k in U_blocks}
            # per-block FIT residuals (live - mu) over both sides, in the block's unit order; PCs of the part orthogonal to q1
            pcs, D = {}, {}
            for k in U_blocks:
                X = torch.cat([unit_vals(fit, u) - mu[u] for u in blk_units[k]], dim=1)     # (n_fit*2, D_blk)
                qk = q1[k].to(dev).float()                                                  # (D_blk, 1)
                X = X - (X @ qk) @ qk.T
                _, _, Vt = torch.linalg.svd(X, full_matrices=False)
                pcs[k], D[k] = Vt.T, X.shape[1]                                             # (D_blk, n_fit*2)
            gen = torch.Generator(device="cpu").manual_seed(0)
            def q_dose(alpha):
                return {k: torch.cat([q1[k].to(dev).float() * alpha ** 0.5, torch.eye(D[k], device=dev) * (1 - alpha) ** 0.5], dim=1) for k in U_blocks}
            def q_rank(r):
                out = {}
                for k in U_blocks:
                    if r == "D":
                        out[k] = torch.eye(D[k], device=dev)
                    else:
                        Q, _ = torch.linalg.qr(torch.cat([q1[k].to(dev).float(), pcs[k][:, :r - 1]], dim=1))
                        out[k] = Q
                return out
            def q_random():
                out = {}
                for k in U_blocks:
                    v = torch.randn(D[k], 1, generator=gen)
                    out[k] = (v / v.norm()).to(dev)
                return out
            dose = {str(a): dmg(held, U_units, q_dose(a), mu) for a in ALPHAS}
            rank = {str(r): dmg(held, U_units, q_rank(r), mu) for r in RANKS + ("D",)}
            rnd = dmg(held, U_units, q_random(), mu)
            R[n] = {"units": units, "upstream_units": U_units, "upstream_blocks": [f"{k[0]}:{k[1]}" for k in U_blocks], "block_dims": {f"{k[0]}:{k[1]}": D[k] for k in U_blocks},
                    "dose": dose, "rank": rank, "random": rnd,
                    "dose_curve": [dose[str(a)]["ce_damage"] for a in ALPHAS], "rank_curve": [rank[str(r)]["ce_damage"] for r in RANKS + ("D",)],
                    "v199_R1_up": prior[n]["arms"]["R1_up"]["ce_damage"], "v199_F_up": prior[n]["arms"]["F_up"]["ce_damage"],
                    "n_rows": {"fit": len(fit.rows), "held": len(held.rows)}, "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one set must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k in ("dose_curve", "rank_curve", "random", "v199_R1_up", "v199_F_up", "error", "seconds")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_removal_complement_dose_v200", "candidate_id": "corpus.unit_removal_complement_dose_v200",
              "bars": BARS, "alphas": list(ALPHAS), "ranks": list(RANKS) + ["D"], "over_band": list(OVER), "controls": list(CONTROLS), "sets": R,
              "protocol": {"rows": "A1 within-direction: mu/q1/PCs from FIT rows[0::4]+rows[1::4], removal on HELD rows[2::4]+rows[3::4]",
                           "dose": "Q = [sqrt(a) q1, sqrt(1-a) I] -> axis clamped, complement = mu + a (live - mu)",
                           "rank": "Q_r = qr([q1, top r-1 PCs of FIT residual orthogonal to q1]); r = D is identity", "units_source": "v199 upstream halves"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "dose": {n: R[n].get("dose_curve") for n in R}, "rank": {n: R[n].get("rank_curve") for n in R}}, indent=2))


if __name__ == "__main__":
    main()
