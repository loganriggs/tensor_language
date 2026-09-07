#!/usr/bin/env python3
# BQGATE: five frozen predictions; set, pairs, rank (1 per block) and bars fixed before the run.
"""v73: does a POOLED lexical-variant direction meet dative's row 5 (A1-derived direction on A2) for hub+8?

v69: dative hub+8 fails row 5 only -- the A1-fit direction removes A2 at 0.218 nat = 0.43x its A1 damage (A2's own refit
0.629). v62 (5-head set, pooled diff-in-means incl. A2) reached 0.60x of the ceiling on an unseen pair. v72 showed for the
complementizer that pooling three lexical pairs of A1 recovers a shared rank-1 axis (0.56-0.67x of the unseen pair's
refit vs 0.32x single-pair). Same design here, with the row-5 target: fit on pooled EVEN rows of A1 + its two v15
variants {sent->handed, reserved->bought}, {sent->gave, reserved->kept} (A2 NEVER enters any fit), evaluate by v51
removal on the ODD rows of A2, of the unseen fourth variant {sent->passed, reserved->cooked}, and of own C.
Estimators, rank 1 per block, fixed: pdim (pooled diff-in-means), das (fit_block_subspace, complement_weight 1.0),
das0 (complement_weight 0). Set = v67 dative hub+8 (13 heads); the 5-head v15 set is reported alongside.

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows, hub+8)
    pred_a_das_row5         das on A2 odd >= 0.50 x das on A1 odd AND its lb975 > 0. Worked: 0.35 vs 0.60 True; 0.20 vs 0.60 False.
    pred_b_pdim_row5        same for pdim. Worked: 0.35 vs 0.60 True; 0.20 vs 0.60 False.
    pred_c_pool_beats_single  max(das, pdim, das0) on A2 odd >= 1.5 x the single-pair A1-fit on A2 odd measured here
                            (v69 0.218 -> >= 0.33). Worked: 0.40 True; 0.25 False.
    pred_d_transfer_v3      das on the unseen fourth pair odd >= 0.50 x that pair's own refit. Worked: 0.30 vs 0.50 True; 0.20 False.
    pred_e_controls         random rank-1 on A2 odd <= 0.05 x A2 own refit AND das own-C CE ub975 <= 0.01 nat.
                            Worked: 0.004, -0.02 True; 0.004, +0.05 False.
    Prior: c, d, e True; a, b uncertain (A2 is a different FRAME, not only a different verb pair -- v62's variants were
    verb pairs). a True re-registers dative's row 5 with the pooled direction; a, b False with d True means the A2 frame
    is keyed independently of the verb pair and row 5 stays open for dative.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_dative_pooled_das_row5_v73_result.json"
SRC = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
NAME = "dative"
TRANSFER, BEAT, RAND_FRAC, C_UB = 0.50, 1.5, 0.05, 0.01
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 12000


def _plan():
    return {"candidate_id": "corpus.unit_dative_pooled_das_row5_v73", "pairs": ["orig"] + [str(m) for m in v15.SETS[NAME][2]] + [str(v15.SETS[NAME][3])],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * 3 * STEPS, "model_updates": 0, "fit_parameters": 2 * 2 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, hub, maps, fourth = v15.SETS[NAME]
    hub = list(hub)
    hub8 = json.loads(SRC.read_text())["sets"][NAME]["final"]
    a1 = g.rows_of(module, "A1")
    c_rows = g.rows_of(module, "C")
    a2 = g.rows_of(module, "A2")
    pairs = {"orig": a1, "v1": g.lexical_variant(a1, maps[0]), "v2": g.lexical_variant(a1, maps[1]), "v3": g.lexical_variant(a1, fourth)}
    fitted = ("orig", "v1", "v2")
    even = {k: g.prepare(backend, r[0::2]) for k, r in pairs.items()}
    odd = {k: g.prepare(backend, r[1::2]) for k, r in pairs.items()}
    odd["C"] = g.prepare(backend, c_rows[1::2])
    odd["A2"] = g.prepare(backend, a2[1::2])
    even["A2"] = g.prepare(backend, a2[0::2])
    pool = g.prepare(backend, [r for k in fitted for r in pairs[k][0::2]])

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    R = {}
    for label, units in (("hub", hub), ("hub8", hub8)):
        q = {k: g.block_diff_in_means(backend, even[k], units) for k in pairs}
        mu = {k: mu_of(even[k], units) for k in pairs}
        q["pdim"] = g.block_diff_in_means(backend, pool, units)
        q["das"], hist = g.fit_block_subspace(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
        q["das0"], hist0 = g.fit_block_subspace(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=0.0)
        q["random"] = g.block_random_subspace(backend, units, rank=1, seed=1)
        mu_pool = mu_of(pool, units)
        for k in ("pdim", "das", "das0", "random"):
            mu[k] = mu_pool
        with torch.no_grad():
            q = {k: {b: m.detach() for b, m in v.items()} for k, v in q.items()}
        dmg = {}
        for src in ("pdim", "das", "das0", "random", "orig"):
            for tgt in ("orig", "v1", "v2", "v3", "A2", "C"):
                dmg[f"{src}->{tgt}"] = v51.summary(torch, v51.removal(backend, odd[tgt], units, q[src], mu[src]))
        for t in list(pairs) + ["A2"]:
            if t == "A2":
                q["A2"] = g.block_diff_in_means(backend, even["A2"], units)
                mu["A2"] = mu_of(even["A2"], units)
            dmg[f"{t}->{t}"] = v51.summary(torch, v51.removal(backend, odd[t], units, q[t], mu[t]))
        cos = {f"{a}|{b}": g.block_cosines(q[a], q[b]) for a, b in (("das", "pdim"), ("das", "das0"), ("das", "v3"), ("pdim", "v3"), ("das", "orig"), ("das", "A2"), ("pdim", "A2"), ("orig", "A2"))}
        R[label] = {"units": units, "damage": dmg, "block_cosines": cos,
                    "das_history": {"cw1": hist, "cw0": hist0}}
        print(label, json.dumps({k: round(v["ce_damage"], 3) for k, v in dmg.items()}), flush=True)

    d = lambda k: R["hub8"]["damage"][k]["ce_damage"]
    ref3 = d("v3->v3")
    D = R["hub8"]["damage"]
    predictions = {
        'pred_a_das_row5': d("das->A2") >= TRANSFER * d("das->orig") and D["das->A2"]["ce_lb975"] > 0,
        'pred_b_pdim_row5': d("pdim->A2") >= TRANSFER * d("pdim->orig") and D["pdim->A2"]["ce_lb975"] > 0,
        'pred_c_pool_beats_single': max(d("das->A2"), d("pdim->A2"), d("das0->A2")) >= BEAT * d("orig->A2"),
        'pred_d_transfer_v3': d("das->v3") >= TRANSFER * ref3,
        'pred_e_controls': d("random->A2") <= RAND_FRAC * d("A2->A2") and D["das->C"]["ce_ub975"] <= C_UB,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_pooled_das_transfer_result_v1",
              "candidate_id": "corpus.unit_dative_pooled_das_row5_v73",
              "pairs": {"v1": maps[0], "v2": maps[1], "v3": fourth}, "fitted_pairs": list(fitted), "pool_rows": len(pool.rows),
              "rows_odd": {k: len(p.rows) for k, p in odd.items()},
              "bars": {"transfer": TRANSFER, "beat": BEAT, "rand_frac": RAND_FRAC, "c_ub": C_UB, "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW}},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
