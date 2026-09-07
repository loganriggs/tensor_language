#!/usr/bin/env python3
# BQGATE: five frozen predictions; recipe, layers, controls and bars fixed before the run; fits are the v99/v80 recipe.
"""v109: do the rank-1 directions reproduce the exact set's RESIDUAL change, not only its margin?

The margin is a 1-d readout: two interventions can agree on it while producing different residual states (v7: DAS seeds
agree on margin at |cos| 0.73-0.97 on direction). v107 showed the directions route through the same readers as the sets.
Here the object is the prediction-position residual (after attention, entering the MLP; `capture_resid[(rid, L)]`) at
layers 7-17 and the final layer, per row: delta(intervention) = resid(intervention) - resid(base).
    fidelity cos_L  = mean over rows of cos(delta_direction, delta_exact) at layer L
    norm ratio_L    = sum ||delta_direction|| / sum ||delta_exact||
Controls that can fail: a random rank-1 per-block direction (seed 1) and the fitted direction's COMPLEMENT (I - qq^T).
Directions: v99/v80 full-specificity recipe (rank 1 per block, pooled EVEN A1, own C EVEN + six A1 EVEN controls at 30,
complement 1.0, 120 steps, lr 0.05, seed 0, mu = pooled EVEN mean); evaluation on ODD A1. Fitted q saved in the receipt.

REGISTERED BEFORE THE RUN (L_first = earliest set layer, final = layer 17)
    pred_a_final_cos       cos_17(direction) >= 0.90 on >= 5 of 7. Worked: 0.95 True; 0.7 False.
    pred_b_final_norm      0.80 <= ratio_17(direction) <= 1.20 on >= 5 of 7.
    pred_c_random_fails    cos_17(random) <= 0.30 on 7/7 (control capable of failing).
    pred_d_depth_cleans    cos_17(direction) >= cos_{L_first}(direction) + 0.05 on >= 5 of 7 (readers respond to the rank-1 part).
    pred_e_complement_inert ratio_17(complement) <= 0.30 on >= 5 of 7.
    Prior: a 45%; b 60%; c 85%; d 50%; e 50%.
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
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_residual_fidelity_v109_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
LAYERS = list(range(7, g.N_LAYERS))
COS_MIN, NORM_LO, NORM_HI, RAND_MAX, CLEAN, COMP_MAX, K = 0.90, 0.80, 1.20, 0.30, 0.05, 0.30, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 120000


def _plan():
    return {"candidate_id": "corpus.unit_seven_residual_fidelity_v109", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 7 * 2 * STEPS, "model_updates": 0, "fit_parameters": 7 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


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
    even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    even_c = {n: g.prepare(backend, g.rows_of(m, "C")[0::2]) for n, m in modules.items()}

    def resid(O, units, q=None, complement=False):
        cap = {}
        g.forward_units(backend, O.base_batch, units=units, donor_cache=O.donor_cache, base_cache=O.base_cache,
                        q=q, complement=complement, capture_resid=cap)
        return {L: torch.stack([cap[(rid, L)] for rid in O.base_batch.row_ids]) for L in LAYERS}

    def compare(d_x, d_ref):
        out = {}
        for L in LAYERS:
            a, b = d_x[L], d_ref[L]
            cos = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
            out[L] = {"cos": round(cos, 3), "norm_ratio": round((a.norm(dim=1).sum() / b.norm(dim=1).sum()).item(), 3)}
        return out

    report, qs = {}, {}
    for n, units in sets.items():
        pool = even[n]
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache)
                              for rid in pool.base_batch.row_ids]).mean(0) for u in units}
        controls = (even_c[n],) + tuple(p for k, p in even.items() if k != n)
        q, _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0,
                                                complement_weight=CW, controls=controls, control_weight=LAM * len(controls), mu=mu)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        qs[n] = {f"{k[0]}:{k[1]}": v.detach().cpu().tolist() for k, v in q.items()}
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        base = resid(O, [])
        d_exact = {L: resid(O, units)[L] - base[L] for L in LAYERS}
        d_dir = {L: resid(O, units, q=q)[L] - base[L] for L in LAYERS}
        d_rand = {L: resid(O, units, q=q_rand)[L] - base[L] for L in LAYERS}
        d_comp = {L: resid(O, units, q=q, complement=True)[L] - base[L] for L in LAYERS}
        first = min(g.unit_layer(u) for u in units)
        Lf = max(first, LAYERS[0])
        report[n] = {"units": units, "first_layer": first, "compare_layer": Lf,
                     "direction": compare(d_dir, d_exact), "random": compare(d_rand, d_exact), "complement": compare(d_comp, d_exact),
                     "extraction": round(g.recovery(O, g.patched_axis(backend, O, units, q=q)) / g.recovery(O, g.patched_axis(backend, O, units)), 3)}
        print(n, "ext", report[n]["extraction"], "dir", {L: report[n]["direction"][L] for L in (Lf, 12, 17)},
              "rand17", report[n]["random"][17], "comp17", report[n]["complement"][17], flush=True)

    R = report
    predictions = {
        'pred_a_final_cos': sum(R[n]["direction"][17]["cos"] >= COS_MIN for n in R) >= K,
        'pred_b_final_norm': sum(NORM_LO <= R[n]["direction"][17]["norm_ratio"] <= NORM_HI for n in R) >= K,
        'pred_c_random_fails': all(R[n]["random"][17]["cos"] <= RAND_MAX for n in R),
        'pred_d_depth_cleans': sum(R[n]["direction"][17]["cos"] >= R[n]["direction"][R[n]["compare_layer"]]["cos"] + CLEAN for n in R) >= K,
        'pred_e_complement_inert': sum(R[n]["complement"][17]["norm_ratio"] <= COMP_MAX for n in R) >= K,
    }
    summary = {n: {"extraction": R[n]["extraction"], "first": R[n]["direction"][R[n]["compare_layer"]], "final": R[n]["direction"][17],
                   "random17": R[n]["random"][17], "complement17": R[n]["complement"][17]} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_residual_fidelity_result_v1",
              "candidate_id": "corpus.unit_seven_residual_fidelity_v109", "summary": summary, "sets": report, "q": qs,
              "recipe": {"lambda": LAM, "steps": STEPS, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "random_seed": 1},
              "bars": {"cos_min": COS_MIN, "norm": [NORM_LO, NORM_HI], "rand_max": RAND_MAX, "clean": CLEAN, "comp_max": COMP_MAX, "k": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
