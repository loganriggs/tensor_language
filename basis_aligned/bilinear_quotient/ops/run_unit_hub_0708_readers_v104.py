#!/usr/bin/env python3
# BQGATE: five frozen predictions; the head, the five behaviours and every bar are fixed before the run; no fitting.
"""v104: who reads hub head 07:08? Later-block freezing while 07:08 alone is interchanged.

v103 measured that 07:08's single-head interchange effect is mostly INDIRECT for quantifier (direct share 0.21), voice
(0.18), polarity (0.34) and preposition (0.37) but direct for dative (0.77). The indirect part must pass through later
attention or MLP blocks that read 07:08's write. Instrument (producer forward, `g.forward_units` with a merged cache):
07:08's c_proj input is set to the donor's value (exact single-head interchange) while ONE later block -- the attention
of layer L >= 8 (all nine heads' c_proj inputs) or the MLP output of layer L >= 7 -- is clamped to its BASE value.
The drop in 07:08's effect when block b is clamped = the share of the effect routed through b (first-order in paths;
shares over blocks can sum past 1 when paths overlap). 'all clamped' = every later block at base = the direct path only,
which must agree with v103's measured direct share (instrument bar).

REGISTERED BEFORE THE RUN (ODD A1 rows of each behaviour; effect = summed axis change over rows; shares = drop / single)
    pred_a_instrument    |all-clamped share - v103 direct share| <= 0.15 on all five behaviours. Worked: 0.25 vs 0.21 True; 0.5 vs 0.21 False.
    pred_b_dominant      each of the four indirect behaviours has a later block with share >= 0.25. Worked: top 0.4 each True; a top of 0.15 False.
    pred_c_shared_relay  the top reader block is the same for >= 3 of the 4 indirect behaviours. Worked: mlp:08 x3 True; all different False.
    pred_d_specific      the top reader blocks are pairwise distinct across the 4 (no repeats). Exclusive with c.
    pred_e_mlp_carries   sum of MLP-block shares > sum of attention-block shares in >= 3 of the 4 indirect behaviours.
    Prior: a ~70%; b ~65%; c ~35%; d ~35%; e ~55% (bilin18's mlp8-11 'increment' role suggests MLP readers).
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
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_hub_0708_readers_v104_result.json"
V103 = ROOT / "circuits/followups/unit_seven_head_direct_v103_result.json"
HEAD = "attn:07:head:08"
BEHAVIOURS = ("quantifier_number", "polarity_licensing", "voice_frame", "verb_preposition", "dative")
INDIRECT = BEHAVIOURS[:4]
INSTR_TOL, DOMINANT, SHARED_K, MLP_K = 0.15, 0.25, 3, 3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_hub_0708_readers_v104", "head": HEAD,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    v103 = json.loads(V103.read_text())["sets"]
    L0 = g.unit_layer(HEAD)
    blocks = [("attn", L) for L in range(L0 + 1, g.N_LAYERS)] + [("mlp", L) for L in range(L0, g.N_LAYERS)]

    def block_units(kind, L):
        return [f"attn:{L:02d}:head:{h:02d}" for h in range(g.N_HEADS)] if kind == "attn" else [f"mlp:{L:02d}"]

    def effect(O, clamp):
        """07:08 from the donor, the units in `clamp` from the base; summed axis change over rows."""
        units = [HEAD] + clamp
        merged = dict(O.base_cache)
        for rid in O.base_batch.row_ids:
            merged[(rid, HEAD)] = O.donor_cache[(rid, HEAD)]
            for u in clamp:
                assert (rid, u) in O.base_cache, u
        out = g.forward_units(backend, O.base_batch, units=units, donor_cache=merged, base_cache=O.base_cache)
        return sum(-(float(a) - float(f)) - b for (a, f), b in zip(out.tolist(), O.base_axis))

    report = {}
    for n in BEHAVIOURS:
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        single = effect(O, [])
        ref = sum(e - b for e, b in zip(g.patched_axis(backend, O, [HEAD]), O.base_axis))
        assert abs(single - ref) <= 1e-3 * max(abs(ref), 1.0), (single, ref)
        all_clamped = effect(O, [u for k, L in blocks for u in block_units(k, L)])
        shares = {}
        for kind, L in blocks:
            e = effect(O, block_units(kind, L))
            shares[f"{kind}:{L:02d}"] = round((single - e) / single, 3) if abs(single) > 1e-6 else None
        top = max(shares, key=lambda k: shares[k])
        report[n] = {"single": round(single, 3), "all_clamped_share": round(all_clamped / single, 3),
                     "v103_direct_share": v103[n]["heads"][HEAD]["share"], "v103_single": v103[n]["heads"][HEAD]["single"],
                     "shares": shares, "top": top, "top_share": shares[top],
                     "mlp_sum": round(sum(v for k, v in shares.items() if k.startswith("mlp")), 3),
                     "attn_sum": round(sum(v for k, v in shares.items() if k.startswith("attn")), 3)}
        print(n, {k: report[n][k] for k in ("single", "all_clamped_share", "v103_direct_share", "top", "top_share", "mlp_sum", "attn_sum")},
              sorted(shares.items(), key=lambda kv: -kv[1])[:4], flush=True)

    tops = [report[n]["top"] for n in INDIRECT]
    predictions = {
        'pred_a_instrument': all(abs(report[n]["all_clamped_share"] - report[n]["v103_direct_share"]) <= INSTR_TOL for n in BEHAVIOURS),
        'pred_b_dominant': all(report[n]["top_share"] >= DOMINANT for n in INDIRECT),
        'pred_c_shared_relay': max(tops.count(t) for t in tops) >= SHARED_K,
        'pred_d_specific': len(set(tops)) == len(tops),
        'pred_e_mlp_carries': sum(report[n]["mlp_sum"] > report[n]["attn_sum"] for n in INDIRECT) >= MLP_K,
    }
    summary = {n: {k: report[n][k] for k in ("all_clamped_share", "v103_direct_share", "top", "top_share", "mlp_sum", "attn_sum")} for n in BEHAVIOURS}
    result = {"predictions": predictions, "schema": "circuit_unit_hub_readers_result_v1", "candidate_id": "corpus.unit_hub_0708_readers_v104",
              "head": HEAD, "summary": summary, "behaviours": report,
              "bars": {"instr_tol": INSTR_TOL, "dominant": DOMINANT, "shared_k": SHARED_K, "mlp_k": MLP_K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
