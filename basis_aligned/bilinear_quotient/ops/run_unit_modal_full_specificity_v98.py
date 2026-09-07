#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v97 final), recipe, controls and two-sided bars fixed before the run.
"""v98: modal_remoteness hub+8 full-specificity direction, with TWO-SIDED collateral bars.

v97 found a 10-head set for would/will (hub 09:04 + 11:03, then +8) with exact extraction 0.927 and diff-in-means removal
0.516 on ODD A1 -- but the same direction IMPROVES the same-answer P rows by 0.29 nat and quantifier A1 by 0.22 nat.
Both passed v97's `<=` bars, which is the eleventh predicate-construction miss: a one-sided bar on a signed collateral
statistic passes a large negative effect. (Checked: the six v80 sets' xctl cross values are all |x| <= 0.011, so the tier
table is unaffected.) Here the standard full-specificity recipe (rank 1 per block, fit on EVEN A1, complement 1.0, 120
steps, lr 0.05, seed 0) is run with controls = own P EVEN + own C EVEN + the six v80 families' A1 EVEN at weight 30 each,
evaluated on ODD rows only, and every collateral bar is on |CE damage|. Side split (base 'will' | donor 'would' on ODD
rows) and flip fractions at the mean are reported as unregistered observations.

REGISTERED BEFORE THE RUN (ODD rows; removal = mean-ablation CE damage in nat; extraction = rank-1 / exact-set recovery)
    pred_a_extraction   ODD A1 extraction fraction >= 0.80 with paired-bootstrap LB >= 0.60 (rubric row 2). Worked: 0.86 (LB 0.74) True; 0.76 False.
    pred_b_removal      ODD A1 removal >= 0.40 with LB > 0. Worked: 0.48 (LB 0.22) True; 0.35 False.
    pred_c_inert_own    |P removal| <= 0.05 AND |C removal| <= 0.02 (two-sided). Worked: P -0.03, C 0.01 True; P -0.12 False.
    pred_d_cross_clean  |cross A1 removal| <= 0.02 on every one of the six v80 families (quantifier included). Worked: max |x| 0.012 True; quantifier -0.06 False.
    pred_e_a2           ODD A2 removal >= 0.50 x ODD A1 removal. Worked: 0.30 vs 0.48 True; 0.18 vs 0.48 False.
    Prior: a ~70% (v97's exact 0.927 minus the usual 0.05-0.07 DAS shortfall); b ~70%; c ~55% (P is in the controls now,
    but the P improvement was 0.29 with a 10-head set); d ~70%; e ~75%.
    Reading: a+b+c+d+e = rows 2/3/5 with P as the in-construction control; the matched-sibling C for row 4 is a later design.
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
OUT = ROOT / "circuits/followups/unit_modal_full_specificity_v98_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, EXT_LB, REM_MIN, P_ABS, C_ABS, CROSS_ABS, A2_FRAC = 0.80, 0.60, 0.40, 0.05, 0.02, 0.02, 0.50
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 30000


def _plan():
    return {"candidate_id": "corpus.unit_modal_full_specificity_v98", "lambda": LAM,
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
    even_p, even_c = g.prepare(backend, g.rows_of(m_modal, "P")[0::2]), g.prepare(backend, g.rows_of(m_modal, "C")[0::2])
    odd = {f: g.prepare(backend, g.rows_of(m_modal, f)[1::2]) for f in ("A1", "A2", "P", "C")}
    mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache) for rid in pool.base_batch.row_ids]).mean(0) for u in units}
    controls = (even_p, even_c) + tuple(cross_even.values())
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
    predictions = {
        'pred_a_extraction': ext_point >= EXT_MIN and ext_lb >= EXT_LB,
        'pred_b_removal': a1r >= REM_MIN and R["A1"]["ce_lb975"] > 0,
        'pred_c_inert_own': abs(R["P"]["ce_damage"]) <= P_ABS and abs(R["C"]["ce_damage"]) <= C_ABS,
        'pred_d_cross_clean': all(abs(x) <= CROSS_ABS for x in cross.values()),
        'pred_e_a2': R["A2"]["ce_damage"] >= A2_FRAC * a1r,
    }
    summary = {"extraction": {"point": round(ext_point, 3), "lb": round(ext_lb, 3), "ub": round(ext_ub, 3)},
               "battery": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in battery.items()},
               "removal": {f: (round(R[f]["ce_damage"], 3), round(R[f]["ce_lb975"], 3), round(R[f]["ce_ub975"], 3)) for f in ("A1", "A2", "P", "C", "random_A1")},
               "sides": {ans["base"]: round(R["A1"]["base_side"]["ce_damage"], 3), ans["donor"]: round(R["A1"]["donor_side"]["ce_damage"], 3)},
               "flips": {k: round(v, 3) for k, v in flips.items()}, "cross": {k: round(v, 3) for k, v in cross.items()}}
    result = {"predictions": predictions, "schema": "circuit_unit_new_behaviour_full_specificity_result_v1", "candidate_id": "corpus.unit_modal_full_specificity_v98",
              "units": units, "answers": ans, "summary": summary, "removal": R, "cross": cross, "extraction_per_row": per_row, "history": hist,
              "bars": {"ext_min": EXT_MIN, "ext_lb": EXT_LB, "rem_min": REM_MIN, "p_abs": P_ABS, "c_abs": C_ABS, "cross_abs": CROSS_ABS, "a2_frac": A2_FRAC},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
