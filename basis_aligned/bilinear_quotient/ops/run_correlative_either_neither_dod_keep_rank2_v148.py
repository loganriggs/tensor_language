#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rank1_span_replays_v147 pred_b_rank2_keep_retains_most pred_c_random_rank2_keep_retains_little
"""Correlative either/neither DoD (v148): is a RANK-2 keep sufficient where the rank-1 readout keep was not (v147 retention 0.53)?

v58b's construction on the v147 fresh rows and set {14.8, 8.1, 16.8, 5.7}: in-forward `keep_span` per head of (i) the or-nor readout direction
alone (replay of v147's keep-only), (ii) the span of or-nor and the family's other contrast and-nor, and (iii) 16 random rank-2 spans (null).
Retention = 1 - damage(keep) / damage(zero). If (ii) reaches 0.70 the heads' or/nor service lives in the two correlative directions; if not, the
line stays "necessary, sufficiency open".
PREDICTIONS (scored as written; failures preserved)
    pred_a_rank1_span_replays_v147           rank-1 keep_span retention within 0.03 of v147's keep-only (0.525)
    pred_b_rank2_keep_retains_most           rank-2 retention >= 0.70. Prior: unsure.
    pred_c_random_rank2_keep_retains_little  every random rank-2 keep <= 0.30
PRICE (registered maximum): 3 batches x (native + producer + zero + rank1 + rank2 + 16 random) = 63 forwards; 0 backwards; 0 fits. Bar <= 66.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_correlative_either_neither_dod_battery_v147 as line
import dod_battery

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/correlative_either_neither_dod_keep_rank2_v148_result.json"
V147 = ROOT / "circuits/followups/correlative_either_neither_dod_battery_v147_result.json"
CANDIDATE_ID = "correlative_state.either_vs_neither.dod_keep_rank2_v148"
SEEDS = tuple(range(4801, 4817))
RETAIN_MIN, RANDOM_MAX, REPLAY_TOL = 0.70, 0.30, 0.03
FORWARDS_MAX = 66
PREDICTIONS = {"pred_a_rank1_span_replays_v147": "+/- 0.03", "pred_b_rank2_keep_retains_most": ">= 0.70", "pred_c_random_rank2_keep_retains_little": "<= 0.30 x 16"}


def main() -> None:
    rows, pos, neg, agents, objects = line.build()
    ref_ret = json.loads(V147.read_text())["keep_only"]["retention"]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "set": list(line.HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"retain_min": RETAIN_MIN, "random_max": RANDOM_MAX, "replay_tol": REPLAY_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fw = L.ManualForward(backend)
    SET = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, pos, neg, line.HEADS).set_components()
    own = L.readout_directions(backend.model, SET, pos, neg)
    other = L.readout_directions(backend.model, SET, L._single(" and"), L._single(" nor"))
    fw.directions = own
    orth = lambda vs: torch.linalg.qr(torch.stack([v.float() for v in vs], dim=1))[0]
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=SET, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    def keep(spans):
        nonlocal forwards
        fw.spans = spans
        arm, n = v1._run_arm(fw, rows, components=SET, mode="keep_span"); forwards += n
        return 1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd
    r1 = keep({k: orth([own[k]]) for k in own}); r2 = keep({k: orth([own[k], other[k]]) for k in own})
    randoms = []
    for seed in SEEDS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        randoms.append(keep({k: orth([torch.randn(128, generator=g), torch.randn(128, generator=g)]) for k in own}))
    fw.use_span = False
    print("zero", round(zd, 3), "rank1", round(r1, 3), "(v147", round(ref_ret, 3), ") rank2 or-nor + and-nor", round(r2, 3), "random max", round(max(randoms), 3))
    predictions = {"pred_a_rank1_span_replays_v147": abs(r1 - ref_ret) <= REPLAY_TOL, "pred_b_rank2_keep_retains_most": r2 >= RETAIN_MIN, "pred_c_random_rank2_keep_retains_little": all(r <= RANDOM_MAX for r in randoms)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "correlative_either_neither_dod_keep_rank2_result_v148", "candidate_id": CANDIDATE_ID, "plan": plan, "zero_damage": zd, "retention": {"rank1": r1, "rank2_or_nor_and_nor": r2, "random_rank2": randoms},
                               "v147_rank1_retention": ref_ret, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
