#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the noun-invariant number vector -- does it transfer across behaviours, reverse on the donor, and need its slot?

v143: at every writes reader of the 16-noun sets the leave-one-out mean of the other nouns' writes-route vectors
reproduces each noun's own margin (11:03: 1.07 / 1.11; possessive readers 1.00-1.13), cos among the w_i 0.58-0.90.
So per (set, reader) there is ONE vector. Three questions, same instrument (v139 slot-add at t into head h's c_proj slice):
  transfer   the mean vector of SOURCE set S added to the rows of TARGET set T (same reader): share of T's own m_w
  reversal   -w_bar (LOO) added on the DONOR batch at t: does the donor margin fall (a value axis, not a one-way push)?
  slot       the same vector added into head (h+1) mod 9's slice at the same layer (W_O slice specificity control)
Sets = the writes readers of v143 (kind == writes) on the number family (lexical_number_pp, perfect_number, quantifier_number,
possessive x5). w_bar_S = mean over S's rows. Shares: base batch (-raw - b)/(d - b); donor batch (raw - b)/(d - b).

Registered before the run (a pair is a (set, reader) with |m_w| >= 0.03; worked example: possessive_medial|10:01 m_w 0.039 is a pair):
  pred_a_instrument   m_none = 0 and |m_rand| <= 0.03 on every pair; own LOO w_bar reproduces v143 loo_w within 0.02;
                      donor batch with a zero add gives share 1.0 within 0.02
  pred_b_cross_set    over ordered (S != T) pairs sharing a reader, m_cross / m_w_own(T) >= 0.6 on >= 0.8 of them
  pred_c_reversal     on every pair with |m_w| >= 0.3 (the two 11:03 pairs), -LOO w_bar on the donor batch gives donor share <= 0.4
  pred_d_slot_control on every pair, own LOO w_bar into slot (h+1) mod 9 gives |share| <= 0.15
  pred_e_quantifier_source  quantifier_number's 11:03 w_bar (fixed each/all cue) added to lexical_number_pp and perfect_number
                      recovers >= 0.6 of each target's own m_w (both)
Prior: b likely (hub-heads note: the number axis is shared within the number family); c open -- the readout-default
finding (v96) says the model reads the unmarked answer at the mean, so subtracting the vector should land there; d is the
control that would expose a generic push; e is the cross-cue-class case (fixed-pair cue, writes reader) and is the least sure.
Smoke: V144_SMOKE=<out.json> -> CPU, 4 rows per set, two sets.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_vector_transfer_v144_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
PAIR_MIN, RAND_MAX, INSTR_TOL, CROSS_MIN, CROSS_FRAC, BIG, REV_MAX, SLOT_MAX = 0.03, 0.03, 0.02, 0.6, 0.8, 0.3, 0.4, 0.15
FAMILY = ("lexical_number_pp", "perfect_number", "quantifier_number", "possessive_adjacent", "possessive_argument",
          "possessive_long_simple", "possessive_medial", "possessive_verbfinal")
V143_RECEIPT = ROOT / "circuits/followups/unit_tier5_cue_route_noun_invariance_v143_result.json"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def pairs_from_v143():
    r = json.loads(V143_RECEIPT.read_text())["pairs_detail"]
    out = {}
    for k, A in r.items():
        n, reader = k.split("|")
        if n in FAMILY and A.get("kind") == "writes":
            out.setdefault(n, []).append(reader)
    return out


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_vector_transfer_v144", "behaviours": 8, "pairs": 10,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V144_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = pairs_from_v143()
    if smoke:
        sets = {k: sets[k] for k in ("lexical_number_pp", "perfect_number")}
    S, skipped = {}, {}
    for n, readers in sets.items():
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows, geo = [], []
        for r in v123.rows_of(m, 1, smoke):
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            if len(diff) != 1:
                continue
            rows.append(r); geo.append({"t": t, "cue": diff[0]})
        if len(rows) < 4:
            skipped[n] = f"{len(rows)} one-cue rows"; continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        top = max(g.unit_layer(h) for h in readers)
        below_all = list(range(0, top))
        cuepos = tuple(ge["cue"] for ge in geo)
        Bc = dataclasses.replace(B, semantic_positions=cuepos)
        hd, md = v120.head_cache(backend, dataclasses.replace(D, semantic_positions=cuepos), below_all), v120.mlp_cache(backend, dataclasses.replace(D, semantic_positions=cuepos), below_all)
        for reader in readers:
            layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
            units = v132.units_of(range(0, layer))
            PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)
            hi = [(u, 0, hd) for u in units if u.startswith("attn")]
            mi = [(u, 0, md) for u in units if u.startswith("mlp")]
            P, V = v131.capture_with_clamp(backend, Bc, hi, mi, layer)
            w_cue = [P[i, h, ge["t"], ge["cue"]] * V[i, ge["cue"], h, :] - PB[i, h, ge["t"], ge["cue"]] * VB[i, ge["cue"], h, :] for i, ge in enumerate(geo)]
            S[(n, reader)] = {"B": B, "D": D, "d_ax": prep.donor_axis, "b_ax": prep.base_axis, "tpos": [ge["t"] for ge in geo],
                              "layer": layer, "h": h, "w": w_cue, "wbar": sum(w_cue) / len(w_cue),
                              "loo": [(sum(w_cue) - w) / (len(w_cue) - 1) for w in w_cue]}
        print(n, "vectors ready", round(time.perf_counter() - t0), "s", flush=True)

    def add_margin(E, vecs, slot=None, donor=False):
        layer, h = E["layer"], E["h"] if slot is None else slot
        batch = E["D"] if donor else E["B"]

        def pre(_m, args):
            v = args[0].clone()
            for i, vec in enumerate(vecs):
                v[i, E["tpos"][i], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] += vec.to(v.device, v.dtype)
            return (v,) + tuple(args[1:])
        hdl = backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre)
        try:
            out = g.forward_units(backend, batch)
        finally:
            hdl.remove()
        vals = [float(a) - float(f) for a, f in out.tolist()]
        sgn = 1.0 if donor else -1.0
        per = [(sgn * p - bb) / (dd - bb) for dd, bb, p in zip(E["d_ax"], E["b_ax"], vals) if abs(dd - bb) > 1e-6]
        return round(sum(per) / len(per), 3)

    R = {}
    for key, E in S.items():
        n, reader = key
        k = len(E["w"])
        gen = torch.Generator().manual_seed(0)
        rand = [torch.randn(w.shape, generator=gen).to(w.device) * (w.norm() / g.HEAD_DIM ** 0.5) for w in E["w"]]
        A = {"rows": k, "m_none": add_margin(E, [torch.zeros_like(w) for w in E["w"]]), "m_w": add_margin(E, E["w"]),
             "loo_w": add_margin(E, E["loo"]), "m_rand": add_margin(E, rand),
             "donor_none": add_margin(E, [torch.zeros_like(w) for w in E["w"]], donor=True),
             "donor_minus_loo": add_margin(E, [-v for v in E["loo"]], donor=True),
             "slot_shift": add_margin(E, E["loo"], slot=(E["h"] + 1) % 9)}
        A["cross"] = {}
        for (n2, r2), E2 in S.items():
            if r2 == reader and n2 != n:
                A["cross"][n2] = add_margin(E, [E2["wbar"]] * k)
        A["pair"] = abs(A["m_w"]) >= PAIR_MIN
        R[key] = A
        print(n, reader, A, round(time.perf_counter() - t0), "s", flush=True)

    pairs = {f"{n}|{r}": A for (n, r), A in R.items() if A["pair"]}
    v143 = json.loads(V143_RECEIPT.read_text())["pairs_detail"] if not smoke else {}
    repro = [abs(A["loo_w"] - v143[k]["loo_w"]) <= INSTR_TOL for k, A in pairs.items() if k in v143]
    pred_a = bool(pairs) and all(A["m_none"] == 0 and abs(A["m_rand"]) <= RAND_MAX and abs(A["donor_none"] - 1.0) <= INSTR_TOL for A in pairs.values()) \
        and (smoke or (len(repro) == len(pairs) and all(repro)))
    cross = [(k, n2, round(v / A["m_w"], 3)) for k, A in pairs.items() for n2, v in A["cross"].items() if f"{n2}|{k.split('|')[1]}" in pairs]
    cross_ok = [c for c in cross if c[2] >= CROSS_MIN]
    pred_b = bool(cross) and len(cross_ok) / len(cross) >= CROSS_FRAC
    big = {k: A for k, A in pairs.items() if abs(A["m_w"]) >= BIG}
    pred_c = bool(big) and all(A["donor_minus_loo"] <= REV_MAX for A in big.values())
    pred_d = bool(pairs) and all(abs(A["slot_shift"]) <= SLOT_MAX for A in pairs.values())
    q = [(k, round(A["cross"]["quantifier_number"] / A["m_w"], 3)) for k, A in pairs.items()
         if k in ("lexical_number_pp|attn:11:head:03", "perfect_number|attn:11:head:03") and "quantifier_number" in A["cross"]]
    pred_e = len(q) == 2 and all(v >= CROSS_MIN for _, v in q)
    predictions = {"pred_a_instrument": pred_a, "pred_b_cross_set": pred_b, "pred_c_reversal": pred_c,
                   "pred_d_slot_control": pred_d, "pred_e_quantifier_source": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_vector_transfer_v144", "candidate_id": "corpus.unit_tier5_cue_vector_transfer_v144",
              "bars": {"pair_min": PAIR_MIN, "rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "cross_min": CROSS_MIN, "cross_frac": CROSS_FRAC,
                       "big": BIG, "rev_max": REV_MAX, "slot_max": SLOT_MAX},
              "counts": {"pairs": len(pairs), "cross": len(cross), "cross_ok": len(cross_ok), "big": sorted(big), "v143_repro": [sum(repro), len(repro)]},
              "cross_ratios": cross, "quantifier_source": q, "skipped": skipped, "pairs_detail": {f"{n}|{r}": A for (n, r), A in R.items()},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "cross": cross, "q": q}, indent=2))


if __name__ == "__main__":
    main()
