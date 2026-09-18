#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_one_block5_head_carries_most pred_c_leading_head_beats_norm_matched_null pred_d_leading_head_is_5_7
"""Number family, step 11 (v65): block-5 heads on the subject-adjacent frame, whole-slice zeroing with a norm-matched null.
Arms on the v63 rows: each block-5 head zeroed at the final query (9), the leading head's 16 equal-norm random-direction
nulls, native, producer replay.
PREDICTIONS: pred_a instrument <= 1e-4; pred_b the largest single-head damage >= 0.60 x the whole-block damage (v64: 0.59
fraction); pred_c that head's damage > max of its 16 nulls; pred_d that head is 5.7 (prior: yes).
PRICE (registered maximum): 1 batch: native 1 + producer 1 + 9 heads + 16 nulls = 27 forwards; bar <= 30.
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
OUT = ROOT / "circuits/followups/number_family_dod_adjacent_block5_v65_result.json"
V64 = ROOT / "circuits/followups/number_family_dod_adjacent_modules_v64_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_adjacent_block5_v65"
NULL_SEEDS = tuple(range(2001, 2017))
SHARE_MIN, INSTRUMENT_TOL = 0.60, 1e-4
FORWARDS_MAX = 30
HEADS = tuple(L.Component(f"attn5_h{h}", 5, "attn", (h,), "final") for h in range(9))


def main() -> None:
    rows = v63.rows
    block = json.loads(V64.read_text())["modules"]["attn5_final"]["damage"]
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
    heads = {}
    for comp in HEADS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="zero"); forwards += n
        heads[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    lead = max(heads, key=heads.get); lead_comp = next(c for c in HEADS if c.name == lead)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=(lead_comp,), mode="random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm)["target_damage_mean"])
    print("heads", {k: round(v, 3) for k, v in heads.items()}, "block", round(block, 3), "lead", lead, "null max", round(max(nulls), 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_one_block5_head_carries_most": heads[lead] >= SHARE_MIN * block,
                   "pred_c_leading_head_beats_norm_matched_null": heads[lead] > max(nulls), "pred_d_leading_head_is_5_7": lead == "attn5_h7"}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_family_dod_adjacent_block5_result_v65", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "heads": heads, "block_damage": block, "leading": lead,
                               "null_damages": nulls, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
