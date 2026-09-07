#!/usr/bin/env python3
# BQGATE: five frozen predictions; carriers and readers fixed from the v123 / v129 receipts; random control seeded.
"""Tier-5: is the near-column VALUE that 11:03 / 07:08 read the early-band carriers' write?

v130: the number-family reader heads take their near-column read through the value term (P_B * dv): the pattern
sits still, the content under it changed. v123 named the early-band heads whose clamp at the near positions
removes the near effect at the margin (lexical / perfect: 04:05, 02:06, 04:07, 03:06; quantifier: 00:03, 01:03,
02:08, 03:06). This rung closes the chain at the VECTOR level: clamp the carriers at the cue-excluded near
offsets (t-1..t-3, v123's Side) and re-measure the reader's near-column contribution vector
    delta_near = sum_{c in near} P_D[t,c] v_D[c] - P_B[t,c] v_B[c]      (128-d, per row)
in the head's value space. removed = 1 - <delta_clamped, delta_live> / <delta_live, delta_live>, mean over rows.

Arms per (set, reader): none (instrument) / carriers / random4 (4 early-band heads outside the carrier set,
seed 0) / early_all (every head and MLP of layers 0-4). The reader's CUE-column delta is measured under the
carrier clamp as a position-specificity control.

Registered before the run:
  pred_a_instrument   the no-item clamp path reproduces the live delta (removed within 0.02 of 0) on 3/3
  pred_b_carriers     carriers remove >= 0.7 of the near delta on lexical and perfect (11:03) and >= 0.5 on
                      quantifier (07:08)
  pred_c_random       the seeded random quartet removes <= 0.2 on 3/3
  pred_d_early_all    early_all removes >= 0.9 on 3/3 and carriers / early_all >= 0.7 on 3/3
  pred_e_cue_spared   the reader's CUE-column delta keeps >= 0.8 of its projection under the carrier clamp, 3/3
Prior: a firm; b is the chain claim (the margin result says yes; the vector may disagree if the carriers act
through the query side); c should hold; d likely; e likely (the query at t may move a little).
Smoke: V131_SMOKE=<out.json>, V131_SMOKE_SET=<set> -> CPU, 4 rows.
"""
from __future__ import annotations

import importlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier5_near_carrier_heads_v123 as v123

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_near_value_source_v131_result.json"
V123 = ROOT / "circuits/followups/unit_tier5_near_carrier_heads_v123_result.json"
FAR_GAP = 3
EARLY = range(0, 5)
READERS = {"lexical_number_pp": "attn:11:head:03", "perfect_number": "attn:11:head:03", "quantifier_number": "attn:07:head:08"}
CARRIER_MIN = {"lexical_number_pp": 0.7, "perfect_number": 0.7, "quantifier_number": 0.5}
INSTR_TOL, RANDOM_MAX, ALL_MIN, RATIO_MIN, CUE_KEEP = 0.02, 0.2, 0.9, 0.7, 0.8
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_near_value_source_v131", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def capture_with_clamp(backend, batch, head_items, mlp_items, layer):
    """v122's exact multi-offset clamp (c_proj pre-hooks / Down hooks) plus v127-style capture of (P, v) at `layer`."""
    torch = backend.torch
    positions = list(batch.semantic_positions)
    heads, mlps = {}, {}
    for u, off, cache in head_items:
        heads.setdefault(g.unit_layer(u), []).append((u, off, cache))
    for u, off, cache in mlp_items:
        mlps.setdefault(g.unit_layer(u), []).append((u, off, cache))
    handles = []
    for l, items in heads.items():
        def pre(_m, args, items=items):
            v = args[0].clone()
            for u, off, cache in items:
                h = int(u.rsplit(":", 1)[1])
                for i, rid in enumerate(batch.row_ids):
                    if (rid, u) in cache:
                        v[i, positions[i] + off, h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype)
            return (v,) + tuple(args[1:])
        handles.append(backend.model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre))
    for l, items in mlps.items():
        bias = backend.model.transformer.h[l].mlp.Down_bias
        def post(_m, _args, out, items=items, bias=bias):
            v = out.clone()
            for u, off, cache in items:
                for i, rid in enumerate(batch.row_ids):
                    if (rid, u) in cache:
                        v[i, positions[i] + off] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype) - bias
            return v
        handles.append(backend.model.transformer.h[l].mlp.Down.register_forward_hook(post))
    attn = backend.model.transformer.h[layer].attn
    orig = attn.squared_attention
    got = {}

    def patched(q, k, v, q2, k2):
        D = q.shape[-1]
        pattern = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
        T = q.shape[1]
        causal = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
        got["P"], got["V"] = pattern.masked_fill(causal.logical_not(), 0.0).detach().clone(), v.detach().clone()
        return orig(q, k, v, q2, k2)

    attn.squared_attention = patched
    try:
        g.forward_units(backend, batch)
    finally:
        attn.squared_attention = orig
        for h in handles:
            h.remove()
    return got["P"], got["V"]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V131_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    v123r = json.load(open(V123))["summary"]
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V131_SMOKE_SET") or "lexical_number_pp"
        sets = {pick: READERS[pick]}
    early_heads = [f"attn:{l:02d}:head:{h:02d}" for l in EARLY for h in range(g.N_HEADS)]
    early_mlps = [f"mlp:{l:02d}" for l in EARLY]
    R = {}
    for n, reader in sets.items():
        t1 = time.perf_counter()
        carriers = v123r[n][2]
        rng = random.Random(0)
        rand4 = rng.sample([u for u in early_heads if u not in carriers], 4)
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows = v123.rows_of(m, 1, smoke)
        side = v123.Side(backend, rows, list(EARLY))
        D, B = side.D, side.B
        layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
        geo = []
        for r in rows:
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            cue = diff[-1] if diff else 0
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            geo.append({"t": t, "cue": diff, "near": [p for p in range(max(cue + 1, t - FAR_GAP), t) if eq(p)]})
        PB, VB = capture_with_clamp(backend, B, [], [], layer)

        def delta(P, V, kind):
            out = []
            for i, ge in enumerate(geo):
                cs = torch.tensor(ge[kind])
                out.append(P[i, h, ge["t"], cs] @ V[i, cs, h, :] - PB[i, h, ge["t"], cs] @ VB[i, cs, h, :] if len(cs) else None)
            return out

        def items(units):
            return ([(u, -k, side.hb[k]) for k in v123.OFFSETS for u in units if u.startswith("attn")],
                    [(u, -k, side.mb[k]) for k in v123.OFFSETS for u in units if u.startswith("mlp")])

        arms = {"none": [], "carriers": carriers, "random4": rand4, "early_all": early_heads + early_mlps}
        live = None
        A = {}
        for arm, units in arms.items():
            hi, mi = items(units)
            P, V = capture_with_clamp(backend, D, hi, mi, layer)
            dn, dc = delta(P, V, "near"), delta(P, V, "cue")
            if arm == "none":
                live = (dn, dc)
            keep = {}
            for kind, cur, ref in (("near", dn, live[0]), ("cue", dc, live[1])):
                fr = [float(c @ r / (r @ r)) for c, r in zip(cur, ref) if c is not None and r is not None and float(r @ r) > 1e-8]
                keep[kind] = round(sum(fr) / len(fr), 3) if fr else None
            A[arm] = {"keep_near": keep["near"], "keep_cue": keep["cue"], "removed_near": None if keep["near"] is None else round(1 - keep["near"], 3),
                      "margin_loss": side.loss(backend, units) if units else 0.0}
        R[n] = {"reader": reader, "carriers": carriers, "random4": rand4, "rows": len(rows), "cue_rows": side.cue_rows, "arms": A,
                "seconds": round(time.perf_counter() - t1, 1)}
        print(n, reader, {a: (x["removed_near"], x["keep_cue"], x["margin_loss"]) for a, x in A.items()}, round(time.perf_counter() - t0), "s", flush=True)

    def rem(n, arm):
        x = R[n]["arms"][arm]["removed_near"]
        return -9 if x is None else x

    ran = list(R)
    a = [n for n in ran if abs(rem(n, "none")) <= INSTR_TOL]
    b = [n for n in ran if rem(n, "carriers") >= CARRIER_MIN[n]]
    c = [n for n in ran if rem(n, "random4") <= RANDOM_MAX]
    d = [n for n in ran if rem(n, "early_all") >= ALL_MIN and rem(n, "carriers") >= RATIO_MIN * rem(n, "early_all")]
    e = [n for n in ran if (R[n]["arms"]["carriers"]["keep_cue"] or 0) >= CUE_KEEP]
    full = len(READERS)
    predictions = {"pred_a_instrument": len(a) == full, "pred_b_carriers": len(b) == full, "pred_c_random": len(c) == full,
                   "pred_d_early_all": len(d) == full, "pred_e_cue_spared": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_near_value_source_v131", "candidate_id": "corpus.unit_tier5_near_value_source_v131",
              "bars": {"carrier_min": CARRIER_MIN, "instr_tol": INSTR_TOL, "random_max": RANDOM_MAX, "all_min": ALL_MIN, "ratio_min": RATIO_MIN, "cue_keep": CUE_KEEP},
              "counts": {"ran": len(ran), "instrument": len(a), "carriers": len(b), "random": len(c), "early_all": len(d), "cue_spared": len(e)},
              "summary": {n: {arm: R[n]["arms"][arm]["removed_near"] for arm in R[n]["arms"]} for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
