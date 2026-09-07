#!/usr/bin/env python3
# BQGATE: five frozen predictions; (head, column) pairs selected by a rule from the v129 receipt, terms exhaustive.
"""Tier-5: what a head's column read is made of -- pattern shift, value shift, or both?

v129 gave each head of each set its column read at t. The donor-minus-base contribution of key column c to
head h at query t is an exact three-term identity in the head's 128-d output:

    P_D v_D - P_B v_B = (P_D - P_B) v_B  +  P_B (v_D - v_B)  +  (P_D - P_B)(v_D - v_B)
                        pattern term        value term          cross term

This rung clamps each term separately (subtracting it from the live donor head output, other heads live) for
every (set, head, column group) with head own >= 0.10 and group loss >= 0.05 in v129, groups cue / between / near.
It also measures, for the number-family heads that read BOTH the cue and the near columns (v129: 11:03 on
lexical / perfect, 07:08 on quantifier), the cosine between the cue-column and near-column contribution vectors.

Registered before the run (bars fixed):
  pred_a_instrument      clamping all three terms = v129's stored group loss for that (head, group), within 0.02,
                         on >= 95% of the pairs (new code path reproduces the old digest)
  pred_b_near_is_value   on near-column pairs the VALUE term carries >= 0.6 of the group loss for >= 70% of pairs
                         (near content is written upstream -- v122 -- and read through an unchanged pattern)
  pred_c_cue_is_value    on cue-column pairs the value term carries >= 0.5 of the group loss for >= 60% of pairs
                         (the cue token's own value differs; the pattern to an adjacent cue barely moves)
  pred_d_same_direction  cos(cue contribution, near contribution) >= 0.5 for 3/3 number-family two-column heads
                         (the within-head redundancy is the same information from two keys)
  pred_e_cross_small     |cross term| <= 0.15 x group loss on >= 80% of pairs
Prior: a firm; b likely; c a real question (a pattern shift toward the cue would be a query-side mechanism);
d the hypothesis this rung exists for; e likely.

Smoke: V130_SMOKE=<out.json>, V130_SMOKE_SET=<set> -> CPU, 4 rows.
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
import run_unit_tier5_carrier_read_accounting_v127 as v127

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_column_term_split_v130_result.json"
V129 = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
FAR_GAP = 3
OWN_MIN, GROUP_MIN = 0.10, 0.05
INSTR_TOL, VALUE_NEAR, VALUE_CUE, COS_MIN, CROSS_MAX = 0.02, 0.6, 0.5, 0.5, 0.15
A_FRAC, B_FRAC, C_FRAC, E_FRAC = 0.95, 0.7, 0.6, 0.8
TWO_COLUMN = (("lexical_number_pp", "attn:11:head:03"), ("perfect_number", "attn:11:head:03"), ("quantifier_number", "attn:07:head:08"))
GROUPS = ("cue", "between", "near")
TERMS = ("pattern", "value", "cross", "all")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 700, 25000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_column_term_split_v130", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def term_clamped_margins(backend, batch, masks, base_attn, term):
    """masks: {layer: bool (B,H,T,T)}; subtract the named term of (P v - PB VB) at masked (b,h,q,c) from the live output."""
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
            PBd, VBd = PB.to(pattern.device, pattern.dtype), VB.to(v.device, v.dtype)
            dP, dV = (pattern - PBd).masked_fill(~Md, 0.0), v - VBd
            z = torch.einsum("bhqk,bkhd->bhqd", pattern, v)
            if term == "pattern":
                sub = torch.einsum("bhqk,bkhd->bhqd", dP, VBd)
            elif term == "value":
                sub = torch.einsum("bhqk,bkhd->bhqd", PBd.masked_fill(~Md, 0.0), dV)
            elif term == "cross":
                sub = torch.einsum("bhqk,bkhd->bhqd", dP, dV)
            else:
                sub = torch.einsum("bhqk,bkhd->bhqd", pattern.masked_fill(~Md, 0.0), v) - torch.einsum("bhqk,bkhd->bhqd", PBd.masked_fill(~Md, 0.0), VBd)
            return z - sub

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
    smoke = os.environ.get("V130_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    prev = json.load(open(V129))["behaviours"]
    S = v115.sets()
    if smoke:
        pick = os.environ.get("V130_SMOKE_SET") or list(S)[0]
        S = {pick: S[pick]}
    R = {}
    for n, heads in S.items():
        if n not in prev or "skipped" in prev[n]:
            R[n] = {"skipped": "not in v129"}; continue
        pairs = [(u, k) for u in heads for k in GROUPS
                 if (prev[n]["per_head"][u]["all"] or 0) >= OWN_MIN and (prev[n]["per_head"][u][k] or 0) >= GROUP_MIN]
        if not pairs:
            R[n] = {"skipped": "no strong (head, group) pair in v129"}; print(n, "skipped", flush=True); continue
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["donor_semantic_position"] >= FAR_GAP
                and r["base_semantic_position"] == r["donor_semantic_position"]]
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        T = max(len(x) for x in D.token_rows)
        layers = sorted({g.unit_layer(u) for u, _ in pairs})
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
                for c in ge[kind]:
                    masks[l][i, h, ge["t"], c] = True
            return masks

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(prep.donor_axis, prep.base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        base_attn = v127.capture_attention(backend, B, layers)
        donor_attn = v127.capture_attention(backend, D, layers)
        P = {}
        for u, k in pairs:
            P[f"{u}|{k}"] = {"v129": prev[n]["per_head"][u][k],
                             **{term: loss(term_clamped_margins(backend, D, mask_for(u, k), base_attn, term)) for term in TERMS}}
        # contribution vectors (128-d, mean over rows) for the two-column heads
        cos = {}
        for u in {u for u, _ in pairs}:
            l, h = g.unit_layer(u), int(u.rsplit(":", 1)[1])
            PB, VB = base_attn[l]
            PD, VD = donor_attn[l]
            vec = {}
            for k in GROUPS:
                acc = []
                for i, ge in enumerate(geo):
                    if not ge[k]:
                        continue
                    cs = torch.tensor(ge[k])
                    acc.append(PD[i, h, ge["t"], cs] @ VD[i, cs, h, :] - PB[i, h, ge["t"], cs] @ VB[i, cs, h, :])
                vec[k] = torch.stack(acc).mean(0) if acc else None
            if vec["cue"] is not None and vec["near"] is not None:
                c = torch.nn.functional.cosine_similarity(vec["cue"], vec["near"], dim=0)
                cos[u] = {"cos_cue_near": round(float(c), 3), "norm_cue": round(float(vec["cue"].norm()), 3), "norm_near": round(float(vec["near"].norm()), 3)}
        R[n] = {"rows": len(rows), "pairs": P, "cosine": cos, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, {p.replace("attn:", ""): (x["v129"], x["all"], x["pattern"], x["value"], x["cross"]) for p, x in P.items()},
              {u.replace("attn:", ""): c["cos_cue_near"] for u, c in cos.items()}, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    allp = [(n, p, x) for n in ran for p, x in R[n]["pairs"].items()]
    ok = [x for x in allp if x[2]["all"] is not None and abs(x[2]["all"] - x[2]["v129"]) <= INSTR_TOL]

    def frac(x, term):
        return (x[term] or 0) / x["all"] if x["all"] else 0.0

    near = [x for x in allp if x[1].endswith("|near")]
    cue = [x for x in allp if x[1].endswith("|cue")]
    near_ok = [x for x in near if frac(x[2], "value") >= VALUE_NEAR]
    cue_ok = [x for x in cue if frac(x[2], "value") >= VALUE_CUE]
    cross_ok = [x for x in allp if abs(frac(x[2], "cross")) <= CROSS_MAX]
    same = [(n, u) for n, u in TWO_COLUMN if n in ran and u in R[n]["cosine"] and R[n]["cosine"][u]["cos_cue_near"] >= COS_MIN]
    testable = [(n, u) for n, u in TWO_COLUMN if n in ran and u in R[n]["cosine"]]
    predictions = {
        "pred_a_instrument": bool(allp) and len(ok) >= A_FRAC * len(allp),
        "pred_b_near_is_value": bool(near) and len(near_ok) >= B_FRAC * len(near),
        "pred_c_cue_is_value": bool(cue) and len(cue_ok) >= C_FRAC * len(cue),
        "pred_d_same_direction": len(testable) == 3 and len(same) == 3,
        "pred_e_cross_small": bool(allp) and len(cross_ok) >= E_FRAC * len(allp),
    }
    result = {"predictions": predictions, "schema": "unit_tier5_column_term_split_v130", "candidate_id": "corpus.unit_tier5_column_term_split_v130",
              "bars": {"own_min": OWN_MIN, "group_min": GROUP_MIN, "instr_tol": INSTR_TOL, "value_near": VALUE_NEAR, "value_cue": VALUE_CUE,
                       "cos_min": COS_MIN, "cross_max": CROSS_MAX, "fracs": [A_FRAC, B_FRAC, C_FRAC, E_FRAC]},
              "counts": {"ran": len(ran), "pairs": len(allp), "instrument_ok": len(ok), "near": len(near), "near_value": len(near_ok),
                         "cue": len(cue), "cue_value": len(cue_ok), "cross_ok": len(cross_ok), "two_column_testable": len(testable), "same_direction": len(same)},
              "term_shares": {n: {p: {t: round(frac(x, t), 3) for t in ("pattern", "value", "cross")} for p, x in R[n]["pairs"].items()} for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
