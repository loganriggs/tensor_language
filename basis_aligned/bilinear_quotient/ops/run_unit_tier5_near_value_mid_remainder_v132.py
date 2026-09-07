#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms are exhaustive over layers 0-10 (no unit chosen after results).
"""Tier-5: who writes the OTHER fifth of the near value the number readers take?

v131: clamping every head and MLP of layers 0-4 at the near positions removes only 0.78 / 0.76 / 0.83 of the
reader's near-column contribution vector (11:03 on lexical / perfect, 07:08 on quantifier). The value a layer-11
head reads at position c is the residual at c after layers 0-10, so the remainder must be written by layers 5-10
at the near positions. Same instrument as v131 (`capture_with_clamp`, cue-excluded offsets t-1..t-3), arms:
  all_0_10           every head and MLP of layers 0-10 (instrument: the value at c is then the base value)
  early_all          layers 0-4 (v131 replication)
  mid_all            layers 5-10, heads and MLPs
  mid_heads / mid_mlps
  layer_05 .. layer_10   one layer's heads + MLP
For 07:08 (layer 7) the mid band is layers 5-6 only, so its arms stop at layer_06.

Registered before the run:
  pred_a_instrument   all_0_10 removes >= 0.95 of the near delta on 3/3
  pred_b_mid_present  mid_all alone removes >= 0.15 on 3/3 (the remainder is a mid-band write, not an interaction)
  pred_c_additive     |removed(early_all) + removed(mid_all) - removed(all_0_10)| <= 0.10 on 3/3
  pred_d_mid_is_mlp   removed(mid_mlps) >= removed(mid_heads) on >= 2/3 (the mid band increments rather than copies)
  pred_e_one_layer    the largest single mid layer reaches >= 0.5 of mid_all on 3/3
Prior: a firm (a floating-point identity up to the query at t); b likely; c unsure (later layers read the
early writes, so the mid write may be a relay of the early one -- then the two are NOT additive and c fails);
d, e questions.
Smoke: V132_SMOKE=<out.json>, V132_SMOKE_SET=<set> -> CPU, 4 rows.
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
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_near_value_mid_remainder_v132_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
INSTR_MIN, MID_MIN, ADD_TOL, ONE_FRAC, K_D = 0.95, 0.15, 0.10, 0.5, 2
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_near_value_mid_remainder_v132", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def units_of(layers, heads=True, mlps=True):
    out = []
    for l in layers:
        if heads:
            out += [f"attn:{l:02d}:head:{h:02d}" for h in range(g.N_HEADS)]
        if mlps:
            out.append(f"mlp:{l:02d}")
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V132_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V132_SMOKE_SET") or "lexical_number_pp"
        sets = {pick: READERS[pick]}
    R = {}
    for n, reader in sets.items():
        t1 = time.perf_counter()
        layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
        below = list(range(0, layer))
        mid = [l for l in below if l >= 5]
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows = v123.rows_of(m, 1, smoke)
        side = v123.Side(backend, rows, below)
        D, B = side.D, side.B
        geo = []
        for r in rows:
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            cue = diff[-1] if diff else 0
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            geo.append({"t": t, "near": [p for p in range(max(cue + 1, t - FAR_GAP), t) if eq(p)]})
        PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)

        def delta(P, V):
            out = []
            for i, ge in enumerate(geo):
                cs = torch.tensor(ge["near"])
                out.append(P[i, h, ge["t"], cs] @ V[i, cs, h, :] - PB[i, h, ge["t"], cs] @ VB[i, cs, h, :] if len(cs) else None)
            return out

        def items(units):
            return ([(u, -k, side.hb[k]) for k in v123.OFFSETS for u in units if u.startswith("attn")],
                    [(u, -k, side.mb[k]) for k in v123.OFFSETS for u in units if u.startswith("mlp")])

        arms = {"none": [], "all_below": units_of(below), "early_all": units_of(range(0, 5)), "mid_all": units_of(mid),
                "mid_heads": units_of(mid, mlps=False), "mid_mlps": units_of(mid, heads=False)}
        for l in mid:
            arms[f"layer_{l:02d}"] = units_of([l])
        live, A = None, {}
        for arm, units in arms.items():
            hi, mi = items(units)
            P, V = v131.capture_with_clamp(backend, D, hi, mi, layer)
            dn = delta(P, V)
            if arm == "none":
                live = dn
            fr = [float(c @ r / (r @ r)) for c, r in zip(dn, live) if c is not None and r is not None and float(r @ r) > 1e-8]
            keep = round(sum(fr) / len(fr), 3) if fr else None
            A[arm] = {"removed_near": None if keep is None else round(1 - keep, 3), "margin_loss": side.loss(backend, units) if units else 0.0}
        R[n] = {"reader": reader, "mid_layers": mid, "rows": len(rows), "arms": A, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, reader, {a: x["removed_near"] for a, x in A.items()}, round(time.perf_counter() - t0), "s", flush=True)

    def rem(n, arm):
        x = R[n]["arms"].get(arm, {}).get("removed_near")
        return -9 if x is None else x

    ran = list(R)
    a = [n for n in ran if rem(n, "all_below") >= INSTR_MIN]
    b = [n for n in ran if rem(n, "mid_all") >= MID_MIN]
    c = [n for n in ran if abs(rem(n, "early_all") + rem(n, "mid_all") - rem(n, "all_below")) <= ADD_TOL]
    d = [n for n in ran if rem(n, "mid_mlps") >= rem(n, "mid_heads")]
    e = [n for n in ran if rem(n, "mid_all") > 0 and max(rem(n, f"layer_{l:02d}") for l in R[n]["mid_layers"]) >= ONE_FRAC * rem(n, "mid_all")]
    full = len(READERS)
    predictions = {"pred_a_instrument": len(a) == full, "pred_b_mid_present": len(b) == full, "pred_c_additive": len(c) == full,
                   "pred_d_mid_is_mlp": len(d) >= K_D, "pred_e_one_layer": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_near_value_mid_remainder_v132", "candidate_id": "corpus.unit_tier5_near_value_mid_remainder_v132",
              "bars": {"instr_min": INSTR_MIN, "mid_min": MID_MIN, "add_tol": ADD_TOL, "one_frac": ONE_FRAC, "K_D": K_D},
              "counts": {"ran": len(ran), "instrument": len(a), "mid_present": len(b), "additive": len(c), "mid_is_mlp": len(d), "one_layer": len(e)},
              "summary": {n: {arm: R[n]["arms"][arm]["removed_near"] for arm in R[n]["arms"]} for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
