#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_rank2_keep_retains_most pred_c_rank1_keep_replays_v55 pred_d_random_rank2_keep_retains_little
"""Number family DoD, step 4 (v58): keep-only with a declared rank-2 readout span per head.

Lane: Claude circuit lane. Parent: v55 (keep-only of the were−was direction alone retains 0.47), v56/v57 (the has−had
direction shares the number heads' output subspace). Keep, at each of {11.3, 5.7, 7.8, 9.7}, only the projection of the
head's write onto span{O_h^T(u_were−u_was), O_h^T(u_has−u_had)} (two weight objects, no fit); null: a random rank-2
subspace of the 128-d slice (16 seeds). Retention against zeroing the four slices.

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native  identity (keep everything) replays native <= 1e-3
    pred_b_rank2_keep_retains_most        rank-2 keep retention >= 0.70
    pred_c_rank1_keep_replays_v55         rank-1 (were−was only) keep retention within 0.05 of v55's 0.469
    pred_d_random_rank2_keep_retains_little   every random rank-2 keep <= 0.30

PRICE (registered maximum): native 2 + producer 2 + zero 2 + identity (capture 2 + arm 2) + rank1 (2+2) + rank2 (2+2) + 16 random
(2+2 each = 64) = 84 forwards; 0 backwards; 0 fits. Bar <= 92.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_keep_rank2_v58_result.json"
V55 = ROOT / "circuits/followups/number_family_dod_battery_v55_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_keep_rank2_v58"
SEEDS = tuple(range(1801, 1817))
RETAIN_MIN, RANDOM_MAX, REPLAY_TOL, INSTRUMENT_TOL = 0.70, 0.30, 0.05, 1e-3
FORWARDS_MAX = 92


def keep_table(store, basis_by_key, torch):
    """Replacement slices: projection of the captured write onto the (orthonormalized) basis per (comp, head)."""
    table = {}
    for (rid, name, pos, head), w in store.items():
        Q = basis_by_key[(name, head)].to(w.device)          # (128, r) orthonormal columns
        wf = w.float()
        table[(rid, name, pos, head)] = (Q @ (Q.T @ wf)).to(w.dtype)
    return table


def main() -> None:
    rows, _ = v55.build()
    ref_ret = json.loads(V55.read_text())["keep_only"]["retention"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    fw = L.ManualForward(backend)
    num = L.readout_directions(backend.model, v55.SINGLES, v55.WERE, v55.WAS)
    tense = L.readout_directions(backend.model, v55.SINGLES, L._single(" has"), L._single(" had"))
    def orth(vectors):
        M = torch.stack([v.float() for v in vectors], dim=1)
        Q, _ = torch.linalg.qr(M); return Q
    rank1 = {k: orth([num[k]]) for k in num}; rank2 = {k: orth([num[k], tense[k]]) for k in num}
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=v55.SET, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    store = {}
    for start in range(0, len(rows), v1.BATCH):
        store.update(fw.capture(rows[start:start + v1.BATCH], v55.SET)); forwards += 1
    def keep_arm(basis):
        nonlocal forwards
        fw.subtract = keep_table(store, basis, torch)
        arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="replace"); forwards += n
        return 1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd, arm
    identity = {k: torch.eye(128) for k in num}
    ret_id, arm_id = keep_arm(identity)
    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(arm_id, native))
    ret1, _ = keep_arm(rank1); ret2, _ = keep_arm(rank2)
    randoms = []
    for seed in SEEDS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        basis = {k: orth([torch.randn(128, generator=g), torch.randn(128, generator=g)]) for k in num}
        r, _ = keep_arm(basis); randoms.append(r)
    fw.use_subtract = False
    forwards -= 0
    print("zero", round(zd, 3), "identity retention", round(ret_id, 4), "rank1", round(ret1, 3), "rank2", round(ret2, 3), "random rank2 max", round(max(randoms), 3))
    predictions = {"pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_rank2_keep_retains_most": ret2 >= RETAIN_MIN,
                   "pred_c_rank1_keep_replays_v55": abs(ret1 - ref_ret) <= REPLAY_TOL, "pred_d_random_rank2_keep_retains_little": all(r <= RANDOM_MAX for r in randoms)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "number_family_dod_keep_rank2_result_v58", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "zero_damage": zd, "retention": {"identity": ret_id, "rank1_were_was": ret1, "rank2_were_was_has_had": ret2, "random_rank2": randoms},
              "v55_rank1_retention": ref_ret, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
