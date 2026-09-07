#!/usr/bin/env python3
# BQGATE: five frozen predictions; carrier heads from the v123 receipt, positions by rule; column CLAMPS (pre-enqueue redesign from knockouts, disclosed).
"""Tier-5: close the read-source accounting for the near-position carrier heads (v126 follow-up), with COLUMN CLAMPS.

v126 knocked out the carriers' reads from the cue and the between positions and recovered only 0.28-0.83 of their
clamp (perfect 0.83, lexical 0.68, verbfinal 0.73, argument 0.55, narrative 0.44, medial 0.35, quantifier 0.28).
Before-cue columns were inert 7/7. The CPU smoke of this rung's first draft (quantifier, 4 rows) showed why a
knockout cannot close the accounting: removing a column deletes the whole term P[q,c] v[c], including the part that
is identical in base and donor, so knocking out MORE columns lowered the loss (cue 0.15 -> cue+near+self 0.09).
Disclosed pre-enqueue redesign: this rung CLAMPS columns instead -- z_D[q] -= sum_{c in K} (P_D v_D - P_B v_B)[q, c],
i.e. each key's term is replaced by the base run's term for the same (q, c). Because z is the plain sum of the
column terms, clamping ALL keys <= q equals the head clamp exactly (the instrument), and every key group's
contribution is a well-defined share of the reference.

Design (identical rows, carriers, offsets to v126; base pattern and v captured at the carrier layers):
  groups     before (p < cue, equal tokens) | cue (all differing p < t) | between (cue < p < t-3, equal) |
             near (other near positions) | self (key = q)
  arms       one clamp per group; source = cue + between; all = every key <= q
Registered before the run (bars fixed):
  pred_a_instrument     |all - reference| <= 0.02 on 7/7 (the column decomposition is exact)
  pred_b_additive       |sum of the five group clamps - all| <= 0.1 x reference on >= 5/7 (downstream linearity of
                        the column contributions is NOT guaranteed -- the MLPs are bilinear)
  pred_c_near_relay     near >= 0.2 x reference on >= 4/7 (the near positions relay among themselves)
  pred_d_self_small     self <= 0.15 x reference on >= 5/7 -- prior genuinely unsure: a layer-4 carrier's own
                        position already holds layer 0-3 writes that read the cue
  pred_e_before_inert   |before| <= 0.15 on >= 5/7 (control, capable of failing)

Smoke: V127_SMOKE=<out.json> runs on CPU with one set (V127_SMOKE_SET) and 4 rows.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_carrier_read_accounting_v127_result.json"
V123 = ROOT / "circuits/followups/unit_tier5_near_carrier_heads_v123_result.json"
OFFSETS, FAR_GAP = (1, 2, 3), 3
INSTR_TOL, ADD_FRAC, NEAR_FRAC, SELF_FRAC, BEFORE_MAX = 0.02, 0.1, 0.2, 0.15, 0.15
K_B, K_C, K_D, K_E = 5, 4, 5, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 12000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_carrier_read_accounting_v127", "behaviours": 7,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def capture_attention(backend, batch, layers):
    """Base-run pattern (B,H,T,T) and mixed v (B,T,H,D) at each layer, via a monkeypatched squared_attention."""
    torch = backend.torch
    got, saved = {}, {}
    for l in layers:
        attn = backend.model.transformer.h[l].attn
        orig = attn.squared_attention

        def patched(q, k, v, q2, k2, l=l, orig=orig):
            D = q.shape[-1]
            pattern = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            T = q.shape[1]
            causal = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
            got[l] = (pattern.masked_fill(causal.logical_not(), 0.0).detach().clone(), v.detach().clone())
            return orig(q, k, v, q2, k2)

        saved[l] = orig
        attn.squared_attention = patched
    try:
        g.forward_units(backend, batch)
    finally:
        for l, orig in saved.items():
            backend.model.transformer.h[l].attn.squared_attention = orig
    return got


def clamped_column_margins(backend, batch, masks, base_attn):
    """masks: {layer: bool (B,H,T,T)}; for masked (b,h,q,c) the donor term P v is replaced by the base run's term."""
    torch = backend.torch
    saved = {}
    for l, M in masks.items():
        attn = backend.model.transformer.h[l].attn
        orig = attn.squared_attention
        PB, VB = base_attn[l]

        def patched(q, k, v, q2, k2, M=M, PB=PB, VB=VB):
            D = q.shape[-1]
            pattern = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            T = q.shape[1]
            causal = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
            pattern = pattern.masked_fill(causal.logical_not(), 0.0)
            Md = M.to(pattern.device)
            z = torch.einsum("bhqk,bkhd->bhqd", pattern, v)
            z_d = torch.einsum("bhqk,bkhd->bhqd", pattern.masked_fill(~Md, 0.0), v)
            z_b = torch.einsum("bhqk,bkhd->bhqd", PB.to(pattern.device, pattern.dtype).masked_fill(~Md, 0.0), VB.to(v.device, v.dtype))
            return z - z_d + z_b

        saved[l] = orig
        attn.squared_attention = patched
    try:
        out = g.forward_units(backend, batch)
    finally:
        for l, orig in saved.items():
            backend.model.transformer.h[l].attn.squared_attention = orig
    return [float(a) - float(f) for a, f in out.tolist()]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V127_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    v123r = json.load(open(V123))
    carriers = {n: b["greedy"] for n, b in v123r["behaviours"].items() if "greedy" in b}
    if smoke:
        pick = os.environ.get("V127_SMOKE_SET") or sorted(carriers)[0]
        carriers = {pick: carriers[pick]}
    R = {}
    for n, heads in carriers.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["donor_semantic_position"] >= max(OFFSETS)
                and r["base_semantic_position"] == r["donor_semantic_position"]]
        if len(rows) < 4:
            R[n] = {"rows": len(rows), "skipped": "fewer than 4 rows"}; continue
        prep = g.prepare(backend, rows)
        D = prep.donor_batch
        T = max(len(x) for x in D.token_rows)
        layers = sorted({g.unit_layer(u) for u in heads})
        # per row: cue / between / before / near-query positions
        geo = []
        for r in rows:
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            cue = diff[-1] if diff else None
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            near = [t - k for k in OFFSETS if eq(t - k)]
            geo.append({"cue": diff, "between": [p for p in range(cue + 1, t - FAR_GAP) if eq(p)] if diff else [],
                        "before": [p for p in range(cue) if eq(p)] if diff else [], "near": near})

        def mask_for(kind):
            masks = {l: torch.zeros(len(rows), g.N_HEADS, T, T, dtype=torch.bool) for l in layers}
            if kind == "none":
                return masks
            for u in heads:
                l, h = g.unit_layer(u), int(u.rsplit(":", 1)[1])
                for i, ge in enumerate(geo):
                    for q in ge["near"]:
                        keys = (ge["cue"] if kind == "cue" else ge["between"] if kind == "between" else ge["before"] if kind == "before"
                                else ge["cue"] + ge["between"] if kind == "source" else [p for p in ge["near"] if p != q] if kind == "near"
                                else [q] if kind == "self" else list(range(q + 1)))
                        for c in keys:
                            masks[l][i, h, q, c] = True
            return masks

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(prep.donor_axis, prep.base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        # reference: the carrier heads clamped to base at the near non-cue positions (v123's cumulative), same rows
        cue_off = {rid: {k for k in OFFSETS if (r["donor_semantic_position"] - k) in ge["cue"]} for rid, r, ge in zip(D.row_ids, rows, geo)}
        hb = {}
        for k in OFFSETS:
            cache = v120.head_cache(backend, v120.shifted(prep.base_batch, k), layers)
            hb[k] = {key: v for key, v in cache.items() if k not in cue_off[key[0]]}
        ref = loss(v122.clamped_margins(backend, D, [(u, -k, hb[k]) for k in OFFSETS for u in heads], []))
        base_attn = capture_attention(backend, prep.base_batch, layers)
        L = {"reference": ref}
        for kind in ("none", "before", "cue", "between", "near", "self", "source", "all"):
            L[kind] = loss(clamped_column_margins(backend, D, mask_for(kind), base_attn))
        L["group_sum"] = round(sum(L[k] for k in ("before", "cue", "between", "near", "self") if L[k] is not None), 3)
        R[n] = {"carriers": heads, "rows": len(rows), "loss": L,
                "cue_offset_mean": round(sum(r["donor_semantic_position"] - ge["cue"][-1] for r, ge in zip(rows, geo) if ge["cue"]) / len(rows), 2),
                "between_len_mean": round(sum(len(ge["between"]) for ge in geo) / len(geo), 2), "seconds": round(time.perf_counter() - t1, 1)}
        print(n, heads, L, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    L_ = lambda n, k: R[n]["loss"].get(k)
    inst = [n for n in ran if L_(n, "all") is not None and L_(n, "reference") is not None and abs(L_(n, "all") - L_(n, "reference")) <= INSTR_TOL
            and L_(n, "none") is not None and abs(L_(n, "none")) <= INSTR_TOL]
    additive = [n for n in ran if L_(n, "reference") and abs(L_(n, "group_sum") - L_(n, "all")) <= ADD_FRAC * L_(n, "reference")]
    nearr = [n for n in ran if L_(n, "reference") and L_(n, "near") is not None and L_(n, "near") >= NEAR_FRAC * L_(n, "reference")]
    selfs = [n for n in ran if L_(n, "reference") and L_(n, "self") is not None and L_(n, "self") <= SELF_FRAC * L_(n, "reference")]
    before = [n for n in ran if L_(n, "before") is not None and abs(L_(n, "before")) <= BEFORE_MAX]
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_additive": len(additive) >= K_B,
        "pred_c_near_relay": len(nearr) >= K_C,
        "pred_d_self_small": len(selfs) >= K_D,
        "pred_e_before_inert": len(before) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_carrier_read_accounting_v127",
              "candidate_id": "corpus.unit_tier5_carrier_read_accounting_v127",
              "bars": {"instr_tol": INSTR_TOL, "add_frac": ADD_FRAC, "near_frac": NEAR_FRAC, "self_frac": SELF_FRAC, "before_max": BEFORE_MAX,
                       "K": [K_B, K_C, K_D, K_E], "offsets": list(OFFSETS), "far_gap": FAR_GAP},
              "counts": {"ran": len(ran), "instrument": len(inst), "additive": len(additive), "near_relay": len(nearr),
                         "self_small": len(selfs), "before_inert": len(before)},
              "summary": {n: R[n].get("loss") for n in R}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
