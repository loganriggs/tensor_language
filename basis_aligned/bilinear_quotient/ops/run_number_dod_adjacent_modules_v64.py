#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_some_mlp_block_carries_at_least_030 pred_c_no_attention_block_exceeds_015 pred_d_the_leading_mlp_is_in_blocks_0_to_4
"""Number family, step 10 (v64): module-grain census on the subject-adjacent frame — which block's output at the final query
carries was/were when the subject noun is the final token? Zero each attention block output and each MLP output at the
final position (36 arms) on the v63 rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_some_mlp_block_carries_at_least_030   max over MLP blocks of damage fraction >= 0.30
    pred_c_no_attention_block_exceeds_015        every attention block's damage fraction <= 0.15
    pred_d_the_leading_mlp_is_in_blocks_0_to_4   the largest MLP damage is at a block <= 4 (prior: token-local early MLP)
PRICE (registered maximum): 1 batch: native 1 + producer 1 + 36 arms = 38 forwards; bar <= 42.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_adjacent_sweep_v63 as v63

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_adjacent_modules_v64_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_adjacent_modules_v64"
MLP_MIN, ATTN_MAX, INSTRUMENT_TOL = 0.30, 0.15, 1e-4
FORWARDS_MAX = 42
MODULES = tuple(L.Component(f"{kind}{l}_final", l, kind, tuple(range(9)) if kind == "attn" else (), "final") for l in range(18) for kind in ("attn", "mlp"))


def main() -> None:
    rows = v63.rows
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    table = {}
    for comp in MODULES:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="zero"); forwards += n
        s = L.summarize(rows, native, arm); table[comp.name] = {"damage": s["target_damage_mean"], "fraction": s["target_damage_fraction"], "positive": s["target_damage_positive_fraction"]}
    mlps = {k: v for k, v in table.items() if k.startswith("mlp")}; attns = {k: v for k, v in table.items() if k.startswith("attn")}
    lead = max(mlps, key=lambda k: mlps[k]["fraction"])
    print("mlp fractions", {k: round(v["fraction"], 2) for k, v in mlps.items()}); print("attn fractions", {k: round(v["fraction"], 2) for k, v in attns.items()})
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_some_mlp_block_carries_at_least_030": mlps[lead]["fraction"] >= MLP_MIN,
                   "pred_c_no_attention_block_exceeds_015": all(v["fraction"] <= ATTN_MAX for v in attns.values()), "pred_d_the_leading_mlp_is_in_blocks_0_to_4": int(lead[3:].split("_")[0]) <= 4}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_family_dod_adjacent_modules_result_v64", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "modules": table, "leading_mlp": lead,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
