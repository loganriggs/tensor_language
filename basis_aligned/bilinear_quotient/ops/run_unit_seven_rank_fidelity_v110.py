#!/usr/bin/env python3
# BQGATE: five frozen predictions; ranks (dim 1, DAS 1 loaded, DAS 2, DAS 4), controls and bars fixed before the run.
"""v110: the specificity-fidelity tradeoff of the seven set directions, with the ranks FIXED IN ADVANCE.

v109: the rank-1 directions reproduce the exact set's final-layer residual change only partly (cos 0.69-0.98, norm ratio
0.44-0.97) while the complement carries up to 0.71 of the norm at cos 0.45-0.86 -- margin-inert but residual-active.
The question here is a fidelity question, NOT a margin repair (the margin rows are already ~1.0 at rank 1): does a
higher registered rank buy residual fidelity, and does it cost the specificity that the rank-1 recipe has (v80: cross
|collateral| <= 0.05, own-C UB <= 0.05)? Ranks compared, per block: diff-in-means rank 1 (no fit), DAS rank 1 (LOADED from
the v109 receipt, no refit -- instrument), DAS rank 2, DAS rank 4 (same full-specificity recipe: pooled EVEN A1, own C EVEN +
six other A1 EVEN controls at 30 each, complement 1.0, 120 steps, lr 0.05, seed 0, mu = pooled EVEN mean). Everything is
evaluated on ODD rows. Fidelity = v109's instrument at the final layer (17): mean cos and norm ratio of the residual delta
against the exact set's delta. Specificity = v51 mean-removal CE damage on own C ODD (UB) and max |CE damage| on the other
six behaviours' A1 ODD.

REGISTERED BEFORE THE RUN (final = layer 17; all counts over the seven sets)
    pred_a_fidelity_rises   cos_17(DAS r4) >= cos_17(DAS r1) + 0.05 on >= 5 of 7.       Worked: 0.80 -> 0.90 True; 0.80 -> 0.83 False.
    pred_b_norm_rises       ratio_17(DAS r4) >= ratio_17(DAS r1) + 0.10 on >= 5 of 7.   Worked: 0.44 -> 0.70 True; 0.44 -> 0.50 False.
    pred_c_specificity_free max|cross|(r4) <= max|cross|(r1) + 0.03 AND ownC_ub(r4) <= ownC_ub(r1) + 0.02 on >= 5 of 7.
                            Worked: cross 0.010 -> 0.025, C_ub 0.020 -> 0.030 True; cross 0.010 -> 0.060 False.
    pred_d_dim_more_faithful cos_17(dim r1) >= cos_17(DAS r1) on >= 5 of 7 (the mean delta is the fidelity-optimal rank-1 axis;
                            DAS optimises a 1-d readout). Worked: dim 0.85 vs DAS 0.80 True; 0.75 vs 0.80 False.
    pred_e_instrument       cos_17 of the LOADED v109 direction equals v109's recorded cos_17 within 0.02 on 7/7.
                            Worked: 0.981 vs 0.975 True; 0.981 vs 0.950 False.
    Prior: a 60%; b 65%; c 55%; d 50%; e 90%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_selective_removal_four_sets_v51 as v51
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_rank_fidelity_v110_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
V109 = ROOT / "circuits/followups/unit_seven_residual_fidelity_v109_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
RANKS = (2, 4)                       # fitted here; rank 1 is loaded from v109
FINAL = 17
COS_GAIN, NORM_GAIN, CROSS_TOL, C_TOL, INSTR_TOL, K = 0.05, 0.10, 0.03, 0.02, 0.02, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 240000


def _plan():
    return {"candidate_id": "corpus.unit_seven_rank_fidelity_v110", "lambda": LAM, "ranks": [1] + list(RANKS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 7 * len(RANKS) * 2 * STEPS, "model_updates": 0, "fit_parameters": 7 * 13 * 128 * sum(RANKS),
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    v109 = json.loads(V109.read_text())
    even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    even_c = {n: g.prepare(backend, g.rows_of(m, "C")[0::2]) for n, m in modules.items()}
    odd = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}
    odd_c = {n: g.prepare(backend, g.rows_of(m, "C")[1::2]) for n, m in modules.items()}

    def resid(O, units, q=None):
        cap = {}
        g.forward_units(backend, O.base_batch, units=units, donor_cache=O.donor_cache, base_cache=O.base_cache, q=q, capture_resid=cap)
        return torch.stack([cap[(rid, FINAL)] for rid in O.base_batch.row_ids])

    def fidelity(d_x, d_ref):
        return {"cos": round(torch.nn.functional.cosine_similarity(d_x, d_ref, dim=1).mean().item(), 3),
                "norm_ratio": round((d_x.norm(dim=1).sum() / d_ref.norm(dim=1).sum()).item(), 3)}

    def load_q(n):
        return {(int(k.split(":")[0]), k.split(":")[1]): torch.tensor(v, device=backend.device, dtype=torch.float32) for k, v in v109["q"][n].items()}

    report = {}
    for n, units in sets.items():
        pool = even[n]
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache)
                              for rid in pool.base_batch.row_ids]).mean(0) for u in units}
        controls = (even_c[n],) + tuple(p for k, p in even.items() if k != n)
        qs = {"dim1": g.block_diff_in_means(backend, pool, units), "das1": load_q(n)}
        for r in RANKS:
            qs[f"das{r}"], _ = g.fit_block_subspace_constrained(backend, pool, units, rank=r, steps=STEPS, lr=LR, seed=0,
                                                               complement_weight=CW, controls=controls, control_weight=LAM * len(controls), mu=mu)
        O = odd[n]
        base = resid(O, [])
        d_exact = resid(O, units) - base
        exact_rec = g.recovery(O, g.patched_axis(backend, O, units))
        arms = {}
        for arm, q in qs.items():
            arms[arm] = {**fidelity(resid(O, units, q=q) - base, d_exact),
                         "extraction": round(g.recovery(O, g.patched_axis(backend, O, units, q=q)) / exact_rec, 3),
                         "own_c": v51.summary(torch, v51.removal(backend, odd_c[n], units, q, mu)),
                         "cross": {m: round(v51.summary(torch, v51.removal(backend, odd[m], units, q, mu))["ce_damage"], 4) for m in modules if m != n}}
            arms[arm]["cross_abs_max"] = round(max(abs(v) for v in arms[arm]["cross"].values()), 3)
            arms[arm]["own_c_ub"] = round(arms[arm]["own_c"]["ce_ub975"], 3)
        report[n] = {"units": units, "arms": arms, "v109_cos17": v109["summary"][n]["final"]["cos"]}
        print(n, {a: (v["cos"], v["norm_ratio"], v["extraction"], v["cross_abs_max"], v["own_c_ub"]) for a, v in arms.items()}, flush=True)

    R = report
    A = lambda n, a: R[n]["arms"][a]
    predictions = {
        'pred_a_fidelity_rises': sum(A(n, "das4")["cos"] >= A(n, "das1")["cos"] + COS_GAIN for n in R) >= K,
        'pred_b_norm_rises': sum(A(n, "das4")["norm_ratio"] >= A(n, "das1")["norm_ratio"] + NORM_GAIN for n in R) >= K,
        'pred_c_specificity_free': sum(A(n, "das4")["cross_abs_max"] <= A(n, "das1")["cross_abs_max"] + CROSS_TOL
                                       and A(n, "das4")["own_c_ub"] <= A(n, "das1")["own_c_ub"] + C_TOL for n in R) >= K,
        'pred_d_dim_more_faithful': sum(A(n, "dim1")["cos"] >= A(n, "das1")["cos"] for n in R) >= K,
        'pred_e_instrument': all(abs(A(n, "das1")["cos"] - R[n]["v109_cos17"]) <= INSTR_TOL for n in R),
    }
    summary = {n: {a: {k: v for k, v in A(n, a).items() if k in ("cos", "norm_ratio", "extraction", "cross_abs_max", "own_c_ub")}
                   for a in R[n]["arms"]} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_rank_fidelity_result_v1",
              "candidate_id": "corpus.unit_seven_rank_fidelity_v110", "summary": summary, "sets": report,
              "recipe": {"lambda": LAM, "steps": STEPS, "lr": LR, "complement_weight": CW, "ranks": [1] + list(RANKS), "seed": 0, "rank1_source": "v109"},
              "bars": {"cos_gain": COS_GAIN, "norm_gain": NORM_GAIN, "cross_tol": CROSS_TOL, "c_tol": C_TOL, "instr_tol": INSTR_TOL, "k": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
