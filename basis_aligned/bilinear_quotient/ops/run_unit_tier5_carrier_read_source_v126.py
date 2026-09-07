#!/usr/bin/env python3
# BQGATE: five frozen predictions; carrier heads taken from the v123 receipt (greedy lists), positions by rule.
"""Tier-5: what do the near-position CARRIER heads read FROM -- the cue token, or the relay positions?

v123 named the heads that write the near non-cue positions (t-1..t-3): the early quartet 04:05/02:06/04:07/03:06
for lexical_number_pp and perfect_number, 00:03+01:03 for quantifier_number, 05:03 for possessive_argument and
possessive_verbfinal, 02:06/04:05/01:03/02:08 for possessive_medial, 05:01/05:06/07:07/07:00 for narrative_tense.
v124/v125 showed that the cue's information also sits at the cue position itself (early writes) and, for the
long-offset sets, on the token right after the cue (early copy). A head at a near position can read the cue from
either. This rung knocks out single attention COLUMNS of the carrier heads at the near positions. bilin18's
pattern is the unnormalised product (q.k/D)(q2.k2/D) with no softmax, so zeroing an entry removes exactly that
key's value term and renormalises nothing: z[q] -= P[q, c] v[c].

Design (donor side, ODD A1 rows, equal-length rows, cue-excluded near offsets 1..3 as v122/v123):
  carrier(set)   = the v123 greedy list                       reference = those heads clamped to base (v123's number)
  cue_ko         = the carrier heads at the near positions lose their read from the CUE positions (row-wise differing)
  between_ko     = ... lose their read from BETWEEN positions (cue < p < t-3, equal tokens)
  before_ko      = ... lose their read from BEFORE-cue positions (equal tokens) -- the control, capable of failing
  source_ko      = cue_ko + between_ko together
  loss           = (donor - knocked) / (donor - base) on the donor answer axis

Registered before the run (bars fixed):
  pred_a_instrument      the knockout with an all-False mask reproduces the donor (|loss| <= 0.02) on every set
  pred_b_source_reads    source_ko >= 0.7 x reference on >= 5/7            Worked: 0.16 of 0.20 True; 0.12 False.
  pred_c_before_inert    |before_ko| <= 0.15 on >= 5/7
  pred_d_possessive_relay between_ko >= cue_ko on both possessive_argument and possessive_verbfinal (they read the
                         cue+1 copy, v125)
  pred_e_number_direct   cue_ko >= 0.5 x reference on >= 2 of the 3 number sets (they read the cue token itself)
Prior: b likely (what else could a same-token position read that differs?), c is the control, d and e are the
registered split. If b fails with c inert the carriers read a differing position I have not listed (t itself is
excluded by causality at t-1..t-3; the remaining candidates are the near positions reading each other) -- report.

Smoke: V126_SMOKE=<out.json> runs on CPU with one set (V126_SMOKE_SET) and 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_carrier_read_source_v126_result.json"
V123 = ROOT / "circuits/followups/unit_tier5_near_carrier_heads_v123_result.json"
OFFSETS, FAR_GAP = (1, 2, 3), 3
INSTR_TOL, SOURCE_FRAC, BEFORE_MAX, DIRECT_FRAC = 0.02, 0.7, 0.15, 0.5
K_B, K_C, K_E = 5, 5, 2
NUMBER = ("lexical_number_pp", "perfect_number", "quantifier_number")
POSSESSIVE = ("possessive_argument", "possessive_verbfinal")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 10000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_carrier_read_source_v126", "behaviours": 7,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def knocked_margins(backend, batch, masks):
    """masks: {layer: bool tensor (B, H, T, T)} of pattern entries to zero; monkeypatches squared_attention."""
    torch = backend.torch
    saved = {}
    for l, M in masks.items():
        attn = backend.model.transformer.h[l].attn
        orig = attn.squared_attention

        def patched(q, k, v, q2, k2, M=M):
            D = q.shape[-1]
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k)
            scores2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2)
            pattern = (scores / D) * (scores2 / D)
            T = q.shape[1]
            causal = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
            pattern = pattern.masked_fill(causal.logical_not(), 0.0)
            pattern = pattern.masked_fill(M.to(pattern.device), 0.0)
            return torch.einsum("bhqk,bkhd->bhqd", pattern, v)

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
    smoke = os.environ.get("V126_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    v123r = json.load(open(V123))
    carriers = {n: b["greedy"] for n, b in v123r["behaviours"].items() if "greedy" in b}
    if smoke:
        pick = os.environ.get("V126_SMOKE_SET") or sorted(carriers)[0]
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
                    keys = ge["cue"] if kind == "cue" else ge["between"] if kind == "between" else ge["before"] if kind == "before" else ge["cue"] + ge["between"]
                    for q in ge["near"]:
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
        L = {"reference": ref}
        for kind in ("none", "cue", "between", "before", "source"):
            L[kind] = loss(knocked_margins(backend, D, mask_for(kind)))
        R[n] = {"carriers": heads, "rows": len(rows), "loss": L,
                "cue_offset_mean": round(sum(r["donor_semantic_position"] - ge["cue"][-1] for r, ge in zip(rows, geo) if ge["cue"]) / len(rows), 2),
                "between_len_mean": round(sum(len(ge["between"]) for ge in geo) / len(geo), 2), "seconds": round(time.perf_counter() - t1, 1)}
        print(n, heads, L, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    L_ = lambda n, k: R[n]["loss"].get(k)
    inst = [n for n in ran if L_(n, "none") is not None and abs(L_(n, "none")) <= INSTR_TOL]
    source = [n for n in ran if L_(n, "reference") and L_(n, "source") is not None and L_(n, "source") >= SOURCE_FRAC * L_(n, "reference")]
    before = [n for n in ran if L_(n, "before") is not None and abs(L_(n, "before")) <= BEFORE_MAX]
    poss = [n for n in POSSESSIVE if n in ran and L_(n, "between") is not None and L_(n, "cue") is not None and L_(n, "between") >= L_(n, "cue")]
    direct = [n for n in NUMBER if n in ran and L_(n, "reference") and L_(n, "cue") is not None and L_(n, "cue") >= DIRECT_FRAC * L_(n, "reference")]
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_source_reads": len(source) >= K_B,
        "pred_c_before_inert": len(before) >= K_C,
        "pred_d_possessive_relay": len(poss) == len([n for n in POSSESSIVE if n in ran]) and bool(poss),
        "pred_e_number_direct": len(direct) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_carrier_read_source_v126",
              "candidate_id": "corpus.unit_tier5_carrier_read_source_v126",
              "bars": {"instr_tol": INSTR_TOL, "source_frac": SOURCE_FRAC, "before_max": BEFORE_MAX, "direct_frac": DIRECT_FRAC,
                       "K": [K_B, K_C, K_E], "offsets": list(OFFSETS), "far_gap": FAR_GAP},
              "counts": {"ran": len(ran), "instrument": len(inst), "source": len(source), "before_inert": len(before),
                         "possessive_relay": len(poss), "number_direct": len(direct)},
              "summary": {n: R[n].get("loss") for n in R}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
