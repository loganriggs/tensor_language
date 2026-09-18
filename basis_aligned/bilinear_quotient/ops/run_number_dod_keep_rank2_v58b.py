#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rank1_span_replays_v55 pred_b_rank2_keep_retains_most pred_c_random_rank2_keep_retains_little
"""Number family DoD, step 4b (v58b): in-forward rank-2 keep (span of were−was and has−had per head) via `keep_span`.

Parent: v58 (replace-protocol mismatch). Same rows, heads and bases; now each head keeps the projection of the slice the
edited forward actually produces (the v55 keep-only semantics).

PREDICTIONS: pred_a rank-1 keep_span retention within 0.02 of v55's keep-only (0.469) — instrument; pred_b rank-2 retention
>= 0.70; pred_c every random rank-2 keep_span <= 0.30 (16 seeds).
PRICE (registered maximum): native 2 + producer 2 + zero 2 + rank1 2 + rank2 2 + 16 x 2 = 42 forwards; bar <= 48.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_keep_rank2_v58b_result.json"
V55 = ROOT / "circuits/followups/number_family_dod_battery_v55_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_keep_rank2_v58b"
SEEDS = tuple(range(1901, 1917))
RETAIN_MIN, RANDOM_MAX, REPLAY_TOL = 0.70, 0.30, 0.02
FORWARDS_MAX = 48


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
    orth = lambda vs: torch.linalg.qr(torch.stack([v.float() for v in vs], dim=1))[0]
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=v55.SET, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    def keep(spans):
        nonlocal forwards
        fw.spans = spans
        arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="keep_span"); forwards += n
        return 1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd
    r1 = keep({k: orth([num[k]]) for k in num}); r2 = keep({k: orth([num[k], tense[k]]) for k in num})
    randoms = []
    for seed in SEEDS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        randoms.append(keep({k: orth([torch.randn(128, generator=g), torch.randn(128, generator=g)]) for k in num}))
    fw.use_span = False
    print("zero", round(zd, 3), "rank1", round(r1, 3), "(v55", round(ref_ret, 3), ") rank2", round(r2, 3), "random max", round(max(randoms), 3))
    predictions = {"pred_a_rank1_span_replays_v55": abs(r1 - ref_ret) <= REPLAY_TOL, "pred_b_rank2_keep_retains_most": r2 >= RETAIN_MIN, "pred_c_random_rank2_keep_retains_little": all(r <= RANDOM_MAX for r in randoms)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_family_dod_keep_rank2_result_v58b", "candidate_id": CANDIDATE_ID, "zero_damage": zd, "retention": {"rank1": r1, "rank2": r2, "random_rank2": randoms}, "v55_rank1_retention": ref_ret,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
