#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v97 final), recipe, controls and two-sided bars fixed before the run.
"""v99: modal_remoteness full-specificity direction WITHOUT the P family in the controls (P is not a removal control).

v98 put own P EVEN rows in the control set. The P family alternates by answer like A1, so P EVEN is all would-rows and
P ODD all will-rows; the control taught the direction to be inert on would contexts, and the result was a direction that
costs 1.02 nat on the will side and 0.19 on the would side of ODD A1 -- and 1.02 on P ODD (will rows). Under mean-ablation a
same-answer P row is just another A1 row of its side: P tests that interchange between same-answer rows is inert, it is
not a removal control (design miss #12; the six greedy sets used C siblings, never P). This is the standard recipe
(rank 1 per block, fit on EVEN A1, complement 1.0, own C EVEN + six v80 A1 EVEN controls at 30 each, 120 steps, lr 0.05,
seed 0), evaluated on ODD rows with two-sided collateral bars; P ODD removal is reported against the A1 will side.

REGISTERED BEFORE THE RUN (ODD rows; removal = mean-ablation CE damage in nat; extraction = rank-1 / exact-set recovery)
    pred_a_extraction   ODD A1 extraction fraction >= 0.80 with paired-bootstrap LB >= 0.60. Worked: 0.86 (LB 0.74) True; 0.76 False.
    pred_b_removal      ODD A1 removal >= 0.40 with LB > 0. Worked: 0.60 (LB 0.45) True; 0.35 False.
    pred_c_less_one_sided  will side / would side <= 3 (v98's 5.4x was the P control's doing). Worked: 0.9 / 0.5 True; 1.0 / 0.19 False.
    pred_d_collateral   |C removal| <= 0.02 AND |cross A1 removal| <= 0.02 on all six v80 families. Worked: max 0.013 True; 0.06 False.
    pred_e_p_is_a1_side  P ODD removal within +-0.15 of the A1 will-side removal (P is a same-side A1 row under removal).
                        Worked: P 0.95 vs will 1.02 True; P 0.40 vs will 1.02 False.
    Prior: a ~70%; b ~75%; c ~60%; d ~70%; e ~75%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import run_unit_polarity_selective_removal_v50 as v50
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_modal_no_p_control_v99_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, EXT_LB, REM_MIN, SIDE_RATIO, C_ABS, CROSS_ABS, P_TOL = 0.80, 0.60, 0.40, 3.0, 0.02, 0.02, 0.15
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 30000


def _plan():
    return {"candidate_id": "corpus.unit_modal_no_p_control_v99", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * STEPS, "model_updates": 0, "fit_parameters": 10 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def sides(torch, prep, d):
    k = len(prep.base_batch.row_ids)
    return {"base_side": v51.summary(torch, {kk: v[:k] for kk, v in d.items()}), "donor_side": v51.summary(torch, {kk: v[k:] for kk, v in d.items()})}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    units = json.loads(V97.read_text())["final"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    cross_odd = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}

    a1 = g.rows_of(m_modal, "A1")
    pool = g.prepare(backend, a1[0::2])
    even_c = g.prepare(backend, g.rows_of(m_modal, "C")[0::2])
    odd = {f: g.prepare(backend, g.rows_of(m_modal, f)[1::2]) for f in ("A1", "A2", "P", "C")}
    mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache) for rid in pool.base_batch.row_ids]).mean(0) for u in units}
    controls = (even_c,) + tuple(cross_even.values())
    q, hist = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                               controls=controls, control_weight=LAM * len(controls), mu=mu)
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)

    # extraction on ODD A1: per-row recovery fractions for the paired bootstrap
    ex = g.patched_axis(backend, odd["A1"], units)
    sub = g.patched_axis(backend, odd["A1"], units, q=q)
    per_row = [kernel.signed_pairwise_donor_recovery(b, d, s) / max(kernel.signed_pairwise_donor_recovery(b, d, e), 1e-6)
               for b, d, e, s in zip(odd["A1"].base_axis, odd["A1"].donor_axis, ex, sub)]
    battery = g.block_direction_battery(backend, odd["A1"], units, q, q_rand=q_rand)
    ext_point, ext_lb, ext_ub = v50._boot(torch, per_row)

    rem = {f: v51.removal(backend, p, units, q, mu) for f, p in odd.items()}
    R = {f: v51.summary(torch, d) for f, d in rem.items()}
    R["A1"].update(sides(torch, odd["A1"], rem["A1"]))
    R["random_A1"] = v51.summary(torch, v51.removal(backend, odd["A1"], units, q_rand, mu))
    cross = {n: v51.summary(torch, v51.removal(backend, p, units, q, mu))["ce_damage"] for n, p in cross_odd.items()}
    ans = {"base": a1[1]["base_answer"].strip(), "donor": a1[1]["donor_answer"].strip()}
    # flips at the mean point on ODD A1 (unregistered observation)
    F = torch.nn.functional
    flips = {}
    for side in ("base", "donor"):
        batch = odd["A1"].base_batch if side == "base" else odd["A1"].donor_batch
        cache = odd["A1"].base_cache if side == "base" else odd["A1"].donor_cache
        bg = dict(cache)
        for rid in batch.row_ids:
            for u in units:
                bg[(rid, u)] = mu[u]
        _, out = g.forward_units(backend, batch, units=units, donor_cache=bg, base_cache=cache, q=q, return_logits=True)
        lp = F.log_softmax(out.float(), -1)
        i = torch.arange(len(batch.row_ids), device=backend.device)
        flips[ans[side]] = (lp[i, torch.tensor(batch.answer_ids, device=backend.device)] < lp[i, torch.tensor(batch.foil_ids, device=backend.device)]).float().mean().item()

    a1r = R["A1"]["ce_damage"]
    will, would = R["A1"]["base_side"]["ce_damage"], R["A1"]["donor_side"]["ce_damage"]
    assert ans == {"base": "will", "donor": "would"}, ans
    predictions = {
        'pred_a_extraction': ext_point >= EXT_MIN and ext_lb >= EXT_LB,
        'pred_b_removal': a1r >= REM_MIN and R["A1"]["ce_lb975"] > 0,
        'pred_c_less_one_sided': max(will, would) / max(min(will, would), 1e-6) <= SIDE_RATIO,
        'pred_d_collateral': abs(R["C"]["ce_damage"]) <= C_ABS and all(abs(x) <= CROSS_ABS for x in cross.values()),
        'pred_e_p_is_a1_side': abs(R["P"]["ce_damage"] - will) <= P_TOL,
    }
    summary = {"extraction": {"point": round(ext_point, 3), "lb": round(ext_lb, 3), "ub": round(ext_ub, 3)},
               "battery": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in battery.items()},
               "removal": {f: (round(R[f]["ce_damage"], 3), round(R[f]["ce_lb975"], 3), round(R[f]["ce_ub975"], 3)) for f in ("A1", "A2", "P", "C", "random_A1")},
               "sides": {ans["base"]: round(R["A1"]["base_side"]["ce_damage"], 3), ans["donor"]: round(R["A1"]["donor_side"]["ce_damage"], 3)},
               "flips": {k: round(v, 3) for k, v in flips.items()}, "cross": {k: round(v, 3) for k, v in cross.items()}}
    result = {"predictions": predictions, "schema": "circuit_unit_new_behaviour_no_p_control_result_v1", "candidate_id": "corpus.unit_modal_no_p_control_v99",
              "units": units, "answers": ans, "summary": summary, "removal": R, "cross": cross, "extraction_per_row": per_row, "history": hist,
              "bars": {"ext_min": EXT_MIN, "ext_lb": EXT_LB, "rem_min": REM_MIN, "side_ratio": SIDE_RATIO, "c_abs": C_ABS, "cross_abs": CROSS_ABS, "p_tol": P_TOL},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
