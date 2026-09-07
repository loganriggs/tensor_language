#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets from v115, groups by rule, per-head arms for every head of every set (none chosen after results).
"""Tier-5: the per-HEAD read map at t -- which head of each set reads which key column?

v128 (exact column clamps, set level) found the adjacent-cue sets read the cue column alone, the number family
reads the cue column AND the near columns redundantly (single-group losses sum to 0.54-0.79 of the joint), and
the possessives read cue / cue+1 copy / near about equally. This rung asks whether that redundancy lives INSIDE
heads or ACROSS heads: every head of every set gets its own column clamps at t (cue, between, near, self, all),
with the other heads live.

Design (donor side, ODD A1 rows, equal-length rows; v127 instrument; groups as v128):
  own(u)        all keys <= t clamped for head u alone (= u's cached clamp toward base, block-live for the rest)
  share(u, G)   loss of clamping group G for u alone, over own(u)   (reported for heads with own(u) >= 0.10)
  set reference the whole set clamped (v128's number)

Registered before the run (bars fixed):
  pred_a_instrument     |own_all(u) - own_cached(u)| <= 0.02 for >= 95% of the heads (own_cached via the c_proj cache)
  pred_b_single_column  among heads with own >= 0.10, the largest group reaches >= 0.7 x own for >= 60% of them
                        (heads read one column each; the redundancy is across heads)
  pred_c_number_split   in each of the 3 number sets the head with the largest cue share and the head with the
                        largest near share are DIFFERENT heads (3/3)
  pred_d_hub_0708_cue   attn 07:08 belongs to 8 sets; on the adjacent-cue sets it belongs to (correlative, degree,
                        interrogative, preposition) its cue share is >= 0.5 on >= 3 of them
  pred_e_heads_additive |sum over heads of own(u) - set reference| <= 0.2 x reference on >= 8/12 (head effects at t
                        add; v-series head_additivity found near-additivity only where one head dominated)
Prior: a firm; b, c genuine questions; d is the multiplexing guess (memory: hub heads carry near-orthogonal
directions per behaviour -- this asks whether they also read different columns); e unsure.

Smoke: V129_SMOKE=<out.json> runs on CPU with one set (V129_SMOKE_SET) and 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
FAR_GAP = 3
INSTR_TOL, OWN_MIN, SINGLE_FRAC, HUB_FRAC, ADD_FRAC = 0.02, 0.10, 0.7, 0.5, 0.2
A_FRAC, B_FRAC, K_D, K_E = 0.95, 0.6, 3, 8
NUMBER = ("lexical_number_pp", "perfect_number", "quantifier_number")
ADJACENT = ("correlative_either_neither", "degree_frame", "interrogative_licensing", "preposition_selection")
HUB = "attn:07:head:08"
GROUPS = ("cue", "between", "near", "self")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 900, 30000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_head_read_map_v129", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V129_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    if smoke:
        pick = os.environ.get("V129_SMOKE_SET") or list(S)[0]
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
                        "near": [p for p in range(max(cue + 1, t - FAR_GAP), t) if eq(p)]})

        def mask_for(u, kind):
            l, h = g.unit_layer(u), int(u.rsplit(":", 1)[1])
            masks = {l: torch.zeros(len(rows), g.N_HEADS, T, T, dtype=torch.bool)}
            for i, ge in enumerate(geo):
                q = ge["t"]
                keys = [q] if kind == "self" else list(range(q + 1)) if kind == "all" else ge[kind]
                for c in keys:
                    masks[l][i, h, q, c] = True
            return masks

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(prep.donor_axis, prep.base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        hb = v120.head_cache(backend, B, layers)
        ref = loss(v122.clamped_margins(backend, D, [(u, 0, hb) for u in heads], []))
        base_attn = v127.capture_attention(backend, B, layers)
        H = {}
        for u in heads:
            own_cached = loss(v122.clamped_margins(backend, D, [(u, 0, hb)], []))
            Lu = {"own_cached": own_cached}
            for kind in ("all",) + GROUPS:
                Lu[kind] = loss(v127.clamped_column_margins(backend, D, mask_for(u, kind), base_attn))
            H[u] = Lu
        R[n] = {"heads": heads, "rows": len(rows), "reference": ref, "per_head": H,
                "sum_own": round(sum(H[u]["all"] for u in heads if H[u]["all"] is not None), 3), "seconds": round(time.perf_counter() - t1, 1)}
        print(n, "ref", ref, "sum_own", R[n]["sum_own"], {u.replace("attn:", ""): (H[u]["all"], H[u]["cue"], H[u]["near"]) for u in heads}, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    all_heads = [(n, u) for n in ran for u in R[n]["heads"]]
    inst_ok = [x for x in all_heads if R[x[0]]["per_head"][x[1]]["all"] is not None and R[x[0]]["per_head"][x[1]]["own_cached"] is not None
               and abs(R[x[0]]["per_head"][x[1]]["all"] - R[x[0]]["per_head"][x[1]]["own_cached"]) <= INSTR_TOL]
    strong = [x for x in all_heads if (R[x[0]]["per_head"][x[1]]["all"] or 0) >= OWN_MIN]

    def share(n, u, kind):
        Lu = R[n]["per_head"][u]
        return (Lu[kind] / Lu["all"]) if Lu.get(kind) is not None and Lu["all"] else 0.0

    single = [x for x in strong if max(share(x[0], x[1], k) for k in GROUPS) >= SINGLE_FRAC]
    split = []
    for n in NUMBER:
        if n not in ran:
            continue
        top_cue = max(R[n]["heads"], key=lambda u: R[n]["per_head"][u]["cue"] or -9)
        top_near = max(R[n]["heads"], key=lambda u: R[n]["per_head"][u]["near"] or -9)
        if top_cue != top_near:
            split.append(n)
    hub = [n for n in ADJACENT if n in ran and HUB in R[n]["heads"] and share(n, HUB, "cue") >= HUB_FRAC]
    additive = [n for n in ran if R[n]["reference"] and abs(R[n]["sum_own"] - R[n]["reference"]) <= ADD_FRAC * R[n]["reference"]]
    predictions = {
        "pred_a_instrument": bool(all_heads) and len(inst_ok) >= A_FRAC * len(all_heads),
        "pred_b_single_column": bool(strong) and len(single) >= B_FRAC * len(strong),
        "pred_c_number_split": len(split) == len([n for n in NUMBER if n in ran]) and bool(split),
        "pred_d_hub_0708_cue": len(hub) >= K_D,
        "pred_e_heads_additive": len(additive) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_head_read_map_v129", "candidate_id": "corpus.unit_tier5_head_read_map_v129",
              "bars": {"instr_tol": INSTR_TOL, "own_min": OWN_MIN, "single_frac": SINGLE_FRAC, "hub_frac": HUB_FRAC, "add_frac": ADD_FRAC,
                       "a_frac": A_FRAC, "b_frac": B_FRAC, "K": [K_D, K_E], "far_gap": FAR_GAP},
              "counts": {"ran": len(ran), "heads": len(all_heads), "instrument_ok": len(inst_ok), "strong": len(strong), "single": len(single),
                         "number_split": len(split), "hub_cue": len(hub), "additive": len(additive)},
              "hub_shares": {n: {k: round(share(n, HUB, k), 3) for k in GROUPS} for n in ran if HUB in R[n]["heads"]},
              "summary": {n: {u: [R[n]["per_head"][u][k] for k in ("all",) + GROUPS] for u in R[n]["heads"]} for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "hub_shares": result["hub_shares"]}, indent=2))


if __name__ == "__main__":
    main()
