#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v97 final), recipe, split and bars fixed before the run.
"""v100: split-swap confirmation of the modal direction -- fit on ODD A1, evaluate on EVEN.

v99 (fit EVEN, evaluate ODD; own C + six A1 controls) gave modal_remoteness extraction 1.007 (LB 0.99), removal 0.615,
A2 0.509, C -0.002, cross |x| <= 0.004, and an UNREGISTERED side asymmetry: would side 0.998 / will side 0.231, flips at
the mean would 0.69 / will 0.00 ('would' is the marked value, 'will' the default -- the opposite of v98's P-control
artefact). Standing rule: unregistered observations get a split-swap confirmation before any row is claimed. Same recipe
with the halves exchanged: fit pool = ODD A1, controls = own C ODD + six v80 A1 ODD, evaluation on EVEN rows. On EVEN A1
rows the base answer is 'would' and the donor 'will'; P EVEN rows are would-rows.

REGISTERED BEFORE THE RUN (EVEN rows; removal = mean-ablation CE damage in nat; extraction = rank-1 / exact-set recovery)
    pred_a_extraction   EVEN A1 extraction fraction >= 0.80 with paired-bootstrap LB >= 0.60.
    pred_b_removal      EVEN A1 removal >= 0.40 with LB > 0.
    pred_c_would_marked would side >= 3 x will side. Worked: 0.9 / 0.2 True; 0.6 / 0.3 False.
    pred_d_will_default flip fraction at the mean: will side <= 0.10 AND would side >= 0.50. Worked: 0.0 / 0.6 True; 0.0 / 0.3 False.
    pred_e_p_is_a1_side P EVEN removal within +-0.15 of the A1 would-side removal. Worked: 0.95 vs 1.0 True; 0.5 vs 1.0 False.
    Prior: a ~85%; b ~80%; c ~70%; d ~70%; e ~75%.
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
OUT = ROOT / "circuits/followups/unit_modal_split_swap_v100_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, EXT_LB, REM_MIN, SIDE_RATIO, FLIP_LO, FLIP_HI, P_TOL = 0.80, 0.60, 0.40, 3.0, 0.10, 0.50, 0.15
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 30000


def _plan():
    return {"candidate_id": "corpus.unit_modal_split_swap_v100", "lambda": LAM,
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
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}  # FIT half (odd rows)
    cross_odd = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}   # EVAL half (even rows)

    a1 = g.rows_of(m_modal, "A1")
    pool = g.prepare(backend, a1[1::2])
    even_c = g.prepare(backend, g.rows_of(m_modal, "C")[1::2])
    odd = {f: g.prepare(backend, g.rows_of(m_modal, f)[0::2]) for f in ("A1", "A2", "P", "C")}  # EVAL half (even rows)
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
    ans = {"base": a1[0]["base_answer"].strip(), "donor": a1[0]["donor_answer"].strip()}
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
    would, will = R["A1"]["base_side"]["ce_damage"], R["A1"]["donor_side"]["ce_damage"]
    assert ans == {"base": "would", "donor": "will"}, ans
    predictions = {
        'pred_a_extraction': ext_point >= EXT_MIN and ext_lb >= EXT_LB,
        'pred_b_removal': a1r >= REM_MIN and R["A1"]["ce_lb975"] > 0,
        'pred_c_would_marked': would >= SIDE_RATIO * will,
        'pred_d_will_default': flips["will"] <= FLIP_LO and flips["would"] >= FLIP_HI,
        'pred_e_p_is_a1_side': abs(R["P"]["ce_damage"] - would) <= P_TOL,
    }
    summary = {"extraction": {"point": round(ext_point, 3), "lb": round(ext_lb, 3), "ub": round(ext_ub, 3)},
               "battery": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in battery.items()},
               "removal": {f: (round(R[f]["ce_damage"], 3), round(R[f]["ce_lb975"], 3), round(R[f]["ce_ub975"], 3)) for f in ("A1", "A2", "P", "C", "random_A1")},
               "sides": {ans["base"]: round(R["A1"]["base_side"]["ce_damage"], 3), ans["donor"]: round(R["A1"]["donor_side"]["ce_damage"], 3)},
               "flips": {k: round(v, 3) for k, v in flips.items()}, "cross": {k: round(v, 3) for k, v in cross.items()}}
    result = {"predictions": predictions, "schema": "circuit_unit_new_behaviour_split_swap_result_v1", "candidate_id": "corpus.unit_modal_split_swap_v100",
              "units": units, "answers": ans, "summary": summary, "removal": R, "cross": cross, "extraction_per_row": per_row, "history": hist,
              "bars": {"ext_min": EXT_MIN, "ext_lb": EXT_LB, "rem_min": REM_MIN, "side_ratio": SIDE_RATIO, "flip_lo": FLIP_LO, "flip_hi": FLIP_HI, "p_tol": P_TOL},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
