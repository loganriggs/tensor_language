#!/usr/bin/env python3
# BQGATE: five frozen predictions; directions loaded from v109 (no fits), controls and bars fixed before the run.
"""v111: the complement is margin-inert (v80/v109) -- is it OUTPUT-inert, over the whole vocabulary?

v109: under the exact set interchange the fitted rank-1 direction carries the margin and most of the final-layer residual
delta, but the complement (I - qq^T) still writes 0.18-0.71 of the residual norm at cos 0.45-0.86 to the exact delta. The
answer/foil margin cannot see whatever that write does to the rest of the distribution. Here the readout is the whole
next-token distribution at the prediction position: per row KL(base || x) for x in {exact set, direction, complement,
random rank-1 (seed 1), full-rank block patch} and the top-1 change rate. Directions are the v109 receipt's q (loaded, no
refit); rows are ODD A1. Shares are ratio-of-sums over rows: share(x) = sum KL_x / sum KL_exact.

REGISTERED BEFORE THE RUN (all counts over the seven sets)
    pred_a_direction_carries   share(direction) >= 0.70 on >= 5 of 7.                 Worked: 0.85 True; 0.55 False.
    pred_b_complement_inert    share(complement) <= 0.15 on >= 5 of 7.                Worked: 0.08 True; 0.30 False.
    pred_c_complement_top1     top-1 change rate(complement) <= 0.05 on >= 5 of 7.    Worked: 0.02 True; 0.12 False.
    pred_d_random_fails        share(random) <= 0.10 on 7/7 (control capable of failing). Worked: 0.02 True; 0.15 False.
    pred_e_instrument          |share(full rank) - 1| <= 0.01 on 7/7 (block-live full rank == exact). Worked: 1.003 True; 1.04 False.
    Prior: a 65%; b 40%; c 60%; d 85%; e 95%.
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
OUT = ROOT / "circuits/followups/unit_seven_complement_output_v111_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
V109 = ROOT / "circuits/followups/unit_seven_residual_fidelity_v109_result.json"
DIR_MIN, COMP_MAX, TOP1_MAX, RAND_MAX, INSTR_TOL, K = 0.70, 0.15, 0.05, 0.10, 0.01, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 24000


def _plan():
    return {"candidate_id": "corpus.unit_seven_complement_output_v111",
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    F = torch.nn.functional
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    v109 = json.loads(V109.read_text())

    def load_q(n):
        return {(int(k.split(":")[0]), k.split(":")[1]): torch.tensor(v, device=backend.device, dtype=torch.float32) for k, v in v109["q"][n].items()}

    def logits(O, units, q=None, complement=False):
        _, lg = g.forward_units(backend, O.base_batch, units=units, donor_cache=O.donor_cache, base_cache=O.base_cache,
                                q=q, complement=complement, return_logits=True)
        return F.log_softmax(lg.float(), -1)

    report = {}
    for n, units in sets.items():
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        q = load_q(n)
        q_full = g.block_identity(backend, units)
        lp_base = logits(O, [])
        arms = {"exact": logits(O, units), "direction": logits(O, units, q=q), "complement": logits(O, units, q=q, complement=True),
                "random": logits(O, units, q=g.block_random_subspace(backend, units, rank=1, seed=1))}
        arms["full_rank"] = logits(O, units, q=q_full)
        kl = {a: (lp_base.exp() * (lp_base - lp)).sum(-1) for a, lp in arms.items()}
        top1 = {a: (lp.argmax(-1) != lp_base.argmax(-1)).float().mean().item() for a, lp in arms.items()}
        ex = kl["exact"].sum().item()
        report[n] = {"units": units, "rows": len(O.base_batch.row_ids), "kl_exact_mean": round(ex / len(O.base_batch.row_ids), 4),
                     "share": {a: round(v.sum().item() / ex, 4) for a, v in kl.items()}, "top1_change": {a: round(v, 4) for a, v in top1.items()},
                     "top1_exact_vs_direction": round((arms["exact"].argmax(-1) != arms["direction"].argmax(-1)).float().mean().item(), 4)}
        print(n, report[n]["share"], report[n]["top1_change"], flush=True)

    R = report
    predictions = {
        'pred_a_direction_carries': sum(R[n]["share"]["direction"] >= DIR_MIN for n in R) >= K,
        'pred_b_complement_inert': sum(R[n]["share"]["complement"] <= COMP_MAX for n in R) >= K,
        'pred_c_complement_top1': sum(R[n]["top1_change"]["complement"] <= TOP1_MAX for n in R) >= K,
        'pred_d_random_fails': all(R[n]["share"]["random"] <= RAND_MAX for n in R),
        'pred_e_instrument': all(abs(R[n]["share"]["full_rank"] - 1) <= INSTR_TOL for n in R),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_complement_output_result_v1",
              "candidate_id": "corpus.unit_seven_complement_output_v111", "summary": {n: {"share": R[n]["share"], "top1": R[n]["top1_change"]} for n in R},
              "sets": report, "bars": {"dir_min": DIR_MIN, "comp_max": COMP_MAX, "top1_max": TOP1_MAX, "rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "k": K},
              "q_source": "v109", "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": result["summary"], "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
