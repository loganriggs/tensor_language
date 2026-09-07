#!/usr/bin/env python3
# BQGATE: six frozen predictions; sets from v115, partitions from the v121 receipt, key groups by rule.
"""Tier-5: exact column accounting of what each SET's heads read at the answer position t, all 14 sets.

v121 asked this with synthetic replays (near_none / far_none) and v127 built the exact instrument: replace the
donor term P[q,c] v[c] of a head by the base run's term for the same key -- all keys <= q equals the head clamp
to 0.000, groups are additive, and each carrier read ONE key. This rung applies it to the greedy head sets
themselves at q = t (their live c_proj-input slices at every layer of the set, block-live: a later head sees the
earlier clamps).

Design (donor side, ODD A1 rows, equal-length rows with equal semantic positions):
  reference   the set's heads clamped to the base run at t (cached, = exact interchange of the set toward base)
  groups      before (p < cue, equal tokens) | cue (all differing p < t) | between (cue < p < t-3, equal) |
              near (t-3..t-1, equal) | self (p = t);  all = every key <= t
  partitions  far-cue sets = v121 far_none <= 0.10; near sets = v121 far_none >= 0.20 (as v122)

Registered before the run (bars fixed):
  pred_a_instrument   |all - reference| <= 0.02 and |none| <= 0.02 on every set that runs
  pred_b_additive     |sum of the five group clamps - all| <= 0.1 x reference on >= 10/14
  pred_c_far_cue      cue >= 0.5 x reference on >= 5 of the 6 far-cue sets (they read the cue token's column)
  pred_d_near_sets    near >= 0.3 x reference on >= 5 of the 7 near sets            Worked: 0.35 of 1.0 True.
  pred_e_before_inert |before| <= 0.15 on >= 12/14 (control, capable of failing)
  pred_f_self_small   self <= 0.15 x reference on >= 10/14 (t is the same token; its residual holds earlier writes
                      that read the cue -- prior unsure, registered as a question)
If a fails the block-live column clamp is not the cached clamp somewhere (it must be, algebraically) -- stop, fix,
re-enqueue as v128b. Reference here is the set clamped TOWARD BASE on the donor side, so the loss scale is the
set's own (v115 exact ~1.0 on most sets).

Smoke: V128_SMOKE=<out.json> runs on CPU with one set (V128_SMOKE_SET) and 4 rows.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier4_expansion_batch_v115 as v115
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_writer_band_v122 as v122
import run_unit_tier5_carrier_read_accounting_v127 as v127

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_set_read_columns_v128_result.json"
V121 = ROOT / "circuits/followups/unit_tier4_near_offset_pattern_v121_result.json"
FAR_GAP = 3
INSTR_TOL, ADD_FRAC, CUE_FRAC, NEAR_FRAC, BEFORE_MAX, SELF_FRAC = 0.02, 0.1, 0.5, 0.3, 0.15, 0.15
K_B, K_C, K_D, K_E, K_F = 10, 5, 5, 12, 10
FAR_MAX, NEAR_MIN = 0.10, 0.20
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 12000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_set_read_columns_v128", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V128_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    fn = {n: v[5] for n, v in json.loads(V121.read_text())["summary"].items()}
    far_sets = sorted(n for n, x in fn.items() if x is not None and x <= FAR_MAX)
    near_sets = sorted(n for n, x in fn.items() if x is not None and x >= NEAR_MIN)
    if smoke:
        pick = os.environ.get("V128_SMOKE_SET") or list(S)[0]
        S = {pick: S[pick]}
    R = {}
    for n, heads in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["donor_semantic_position"] >= FAR_GAP
                and r["base_semantic_position"] == r["donor_semantic_position"]]
        if len(rows) < 4:
            R[n] = {"rows": len(rows), "skipped": "fewer than 4 rows"}; print(n, "skipped", flush=True); continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        T = max(len(x) for x in D.token_rows)
        layers = sorted({g.unit_layer(u) for u in heads})
        geo = []
        for r in rows:
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            cue = diff[-1] if diff else 0
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            geo.append({"t": t, "cue": diff, "between": [p for p in range(cue + 1, t - FAR_GAP) if eq(p)],
                        "before": [p for p in range(cue) if eq(p)], "near": [p for p in range(max(cue + 1, t - FAR_GAP), t) if eq(p)]})

        def mask_for(kind):
            masks = {l: torch.zeros(len(rows), g.N_HEADS, T, T, dtype=torch.bool) for l in layers}
            if kind == "none":
                return masks
            for u in heads:
                l, h = g.unit_layer(u), int(u.rsplit(":", 1)[1])
                for i, ge in enumerate(geo):
                    q = ge["t"]
                    keys = (ge["before"] if kind == "before" else ge["cue"] if kind == "cue" else ge["between"] if kind == "between"
                            else ge["near"] if kind == "near" else [q] if kind == "self" else list(range(q + 1)))
                    for c in keys:
                        masks[l][i, h, q, c] = True
            return masks

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(prep.donor_axis, prep.base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        hb = v120.head_cache(backend, B, layers)
        ref = loss(v122.clamped_margins(backend, D, [(u, 0, hb) for u in heads], []))
        base_attn = v127.capture_attention(backend, B, layers)
        L = {"reference": ref}
        for kind in ("none", "before", "cue", "between", "near", "self", "all"):
            L[kind] = loss(v127.clamped_column_margins(backend, D, mask_for(kind), base_attn))
        L["group_sum"] = round(sum(L[k] for k in ("before", "cue", "between", "near", "self") if L[k] is not None), 3)
        R[n] = {"heads": heads, "rows": len(rows), "loss": L, "kind": "far" if n in far_sets else "near" if n in near_sets else "other",
                "cue_offset_mean": round(sum(ge["t"] - ge["cue"][-1] for ge in geo if ge["cue"]) / len(geo), 2),
                "seconds": round(time.perf_counter() - t1, 1)}
        print(n, R[n]["kind"], L, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    L_ = lambda n, k: R[n]["loss"].get(k)
    ok = lambda n, k: L_(n, k) is not None and L_(n, "reference")
    inst = [n for n in ran if ok(n, "all") and abs(L_(n, "all") - L_(n, "reference")) <= INSTR_TOL and L_(n, "none") is not None and abs(L_(n, "none")) <= INSTR_TOL]
    additive = [n for n in ran if ok(n, "all") and abs(L_(n, "group_sum") - L_(n, "all")) <= ADD_FRAC * L_(n, "reference")]
    far = [n for n in far_sets if n in ran and ok(n, "cue") and L_(n, "cue") >= CUE_FRAC * L_(n, "reference")]
    near = [n for n in near_sets if n in ran and ok(n, "near") and L_(n, "near") >= NEAR_FRAC * L_(n, "reference")]
    before = [n for n in ran if L_(n, "before") is not None and abs(L_(n, "before")) <= BEFORE_MAX]
    selfs = [n for n in ran if ok(n, "self") and L_(n, "self") <= SELF_FRAC * L_(n, "reference")]
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_additive": len(additive) >= K_B,
        "pred_c_far_cue": len(far) >= K_C,
        "pred_d_near_sets": len(near) >= K_D,
        "pred_e_before_inert": len(before) >= K_E,
        "pred_f_self_small": len(selfs) >= K_F,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_set_read_columns_v128", "candidate_id": "corpus.unit_tier5_set_read_columns_v128",
              "bars": {"instr_tol": INSTR_TOL, "add_frac": ADD_FRAC, "cue_frac": CUE_FRAC, "near_frac": NEAR_FRAC, "before_max": BEFORE_MAX,
                       "self_frac": SELF_FRAC, "K": [K_B, K_C, K_D, K_E, K_F], "far_gap": FAR_GAP, "far_max": FAR_MAX, "near_min": NEAR_MIN},
              "partition": {"far": far_sets, "near": near_sets},
              "counts": {"ran": len(ran), "instrument": len(inst), "additive": len(additive), "far_cue": len(far), "near": len(near),
                         "before_inert": len(before), "self_small": len(selfs), "n": len(R)},
              "summary": {n: R[n].get("loss") for n in R}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
