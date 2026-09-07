#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, pairs, rank (1 per block) and bars fixed before the run.
"""v72: does a POOLED cross-pair direction transfer to an unseen verb pair for verb_complementizer hub+8?

v71: single-pair diff-in-means directions transfer at only 0.26-0.41x across verb pairs (hub+8: orig-fit on
questioned/declared 0.360 vs that pair's own refit 1.110); block |cos| 0.43. Two readings remain open: (i) the
directions are genuinely verb-pair keyed, or (ii) each single-pair direction is a noisy sample of a shared axis plus a
pair-specific part, and pooling three pairs recovers the shared part. v17 found a pooled DAS+inertness axis that
transferred to an unseen pair for the 3-head hub under INTERCHANGE; never tested under removal, never on hub+8.

Estimators (rank 1 per block, fixed in advance; no rank raise on a null), fitted on the pooled EVEN rows of
orig+v1+v2 (48 documents), evaluated by v51 removal on ODD rows:
    pdim  pooled diff-in-means (no fit)
    das   fit_block_subspace on the exact-set margin, complement_weight 1.0 (the user's constrained DAS)
    das0  same with complement_weight 0 (what the inertness term buys)
    v3 refit (from v71 design, even rows of v3) bounds what a pair-specific direction can do; random rank-1 floor.

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows; v3 = questioned/declared, unseen by every pooled fit)
    pred_a_das_transfer     das on v3 odd >= 0.50 x v3 own refit (v71: 1.110 -> >= 0.555). Worked: 0.70 True; 0.36 False.
    pred_b_pdim_transfer    pdim on v3 odd >= 0.50 x v3 own refit. Worked: 0.60 True; 0.40 False.
    pred_c_pool_beats_single  max(das, pdim, das0) on v3 odd >= 1.5 x the v71 single-pair transfer measured here (orig-fit
                            on v3 odd; v71 0.360 -> >= 0.54). Worked: 0.60 True; 0.45 False.
    pred_d_das_keeps_fitted   das on each fitted pair's odd rows >= 0.50 x that pair's own refit
                            (v71: orig 1.119, v1 1.294, v2 0.756 -> >= 0.56 / 0.65 / 0.38). Worked: 0.8/0.9/0.5 True; 0.8/0.9/0.3 False.
    pred_e_random           random rank-1 on v3 odd <= 0.05 x v3 own refit. Worked: 0.004 True.
    Prior: b and c True, a True (reading ii); d True. If a, b False with c True the pooled axis is partial; if all of a, b, c
    False the directions are pair-keyed and a shared rank-1 axis does not exist at this set (reading i).
Own-C (noted/replied) damage of each direction is reported as information (v69: the hub itself damages own C).
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
OUT = ROOT / "circuits/followups/unit_complementizer_pooled_das_transfer_v72_result.json"
SRC = ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json"
NAME = "verb_complementizer"
TRANSFER, BEAT, RAND_FRAC = 0.50, 1.5, 0.05
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 12000


def _plan():
    return {"candidate_id": "corpus.unit_complementizer_pooled_das_transfer_v72", "pairs": ["orig"] + [str(m) for m in v15.SETS[NAME][2]] + [str(v15.SETS[NAME][3])],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * 3 * STEPS, "model_updates": 0, "fit_parameters": 2 * 2 * 11 * 128, "gpu_accessed": False,
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
    pairs = {"orig": a1, "v1": g.lexical_variant(a1, maps[0]), "v2": g.lexical_variant(a1, maps[1]), "v3": g.lexical_variant(a1, fourth)}
    fitted = ("orig", "v1", "v2")
    even = {k: g.prepare(backend, r[0::2]) for k, r in pairs.items()}
    odd = {k: g.prepare(backend, r[1::2]) for k, r in pairs.items()}
    odd["C"] = g.prepare(backend, c_rows[1::2])
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
            for tgt in ("orig", "v1", "v2", "v3", "C"):
                dmg[f"{src}->{tgt}"] = v51.summary(torch, v51.removal(backend, odd[tgt], units, q[src], mu[src]))
        for t in pairs:
            dmg[f"{t}->{t}"] = v51.summary(torch, v51.removal(backend, odd[t], units, q[t], mu[t]))
        cos = {f"{a}|{b}": g.block_cosines(q[a], q[b]) for a, b in (("das", "pdim"), ("das", "das0"), ("das", "v3"), ("pdim", "v3"), ("das", "orig"))}
        R[label] = {"units": units, "damage": dmg, "block_cosines": cos,
                    "das_history": {"cw1": hist, "cw0": hist0}}
        print(label, json.dumps({k: round(v["ce_damage"], 3) for k, v in dmg.items()}), flush=True)

    d = lambda k: R["hub8"]["damage"][k]["ce_damage"]
    ref3 = d("v3->v3")
    predictions = {
        'pred_a_das_transfer': d("das->v3") >= TRANSFER * ref3,
        'pred_b_pdim_transfer': d("pdim->v3") >= TRANSFER * ref3,
        'pred_c_pool_beats_single': max(d("das->v3"), d("pdim->v3"), d("das0->v3")) >= BEAT * d("orig->v3"),
        'pred_d_das_keeps_fitted': all(d(f"das->{t}") >= TRANSFER * d(f"{t}->{t}") for t in fitted),
        'pred_e_random': d("random->v3") <= RAND_FRAC * ref3,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_pooled_das_transfer_result_v1",
              "candidate_id": "corpus.unit_complementizer_pooled_das_transfer_v72",
              "pairs": {"v1": maps[0], "v2": maps[1], "v3": fourth}, "fitted_pairs": list(fitted), "pool_rows": len(pool.rows),
              "rows_odd": {k: len(p.rows) for k, p in odd.items()},
              "bars": {"transfer": TRANSFER, "beat": BEAT, "rand_frac": RAND_FRAC, "das": {"rank": 1, "steps": STEPS, "lr": LR, "cw": CW}},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
