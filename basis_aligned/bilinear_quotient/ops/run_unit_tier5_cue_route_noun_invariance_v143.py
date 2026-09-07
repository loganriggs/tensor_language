#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: is the layer-writes route a NOUN-INVARIANT number vector, and the embedding route a token-specific one?

v141/v142 (corrected 07:14): the writes-route cue columns vary over many token pairs (the number/possessive sets swap 16
different nouns; interrogative column 2 did->noun 16 pairs), the embedding-route cue columns are one fixed token pair
(each/all, either/neither, so/too, stands/stood) or nearly (preposition: 3 verbs -> "depend"). Hypothesis under test:
the writes route carries a feature abstracted across tokens (one number vector shared by the 16 nouns), the embedding
route a fixed-token identity. Instrument: v141's per-row w_i (clamp donor writes at the cue, base batch) and e_i (base
writes on the donor batch) at each reader's cue column, v139 slot-adds at t. New arms: leave-one-out means
w_bar_(-i) and e_bar_(-i) over the other rows of the set (rows of odd parity share the interchange direction), added to
row i; pairwise cosines among the w_i and among the e_i. Readers/sets: the v141 pairs (one-cue rows).

Registered before the run (a pair counts when |m_we| >= 0.05; a WRITES reader has ratio_w >= 0.6, an EMBEDDING reader ratio_e >= 0.6):
  pred_a_instrument   m_none = 0, |m_rand| <= 0.03 on every pair; m_w and m_e reproduce v141 within 0.02 on every pair
  pred_b_noun_invariant_w   on >= 0.8 of counting WRITES readers of the 16-noun sets, LOO w_bar recovers >= 0.6 of the
                      pair's own m_w (loo_w / m_w >= 0.6)
  pred_c_w_more_consistent  on every counting WRITES reader of the 16-noun sets, mean pairwise cos(w_i, w_j) >= mean pairwise cos(e_i, e_j) + 0.1
  pred_d_fixed_pair_e  on >= 0.8 of counting EMBEDDING readers (fixed-pair sets), LOO e_bar recovers >= 0.6 of own m_e and mean cos(e_i, e_j) >= 0.5
  pred_e_e_noun_specific  on every counting WRITES reader of the 16-noun sets, mean pairwise cos(e_i, e_j) <= 0.3
Prior: b, c firm if the reader reads an abstracted number axis (v79-v85 rank-1 directions transfer across nouns); d near-
tautological (same token, different frames) and is the contrast, not a discovery; e is the open one -- 16 noun embeddings
may still share a plural direction. If b fails the writes route is itself noun-specific and "abstracted" is wrong.
Smoke: V143_SMOKE=<out.json>, V143_SMOKE_SET=<set> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_cue_route_noun_invariance_v143_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
CUE_MIN, COUNT_MIN, RAND_MAX, INSTR_TOL, ROUTE_MAJ, LOO_MIN, FRAC_MIN, COS_GAP, E_COS_MIN, E_COS_MAX = 0.04, 0.05, 0.03, 0.02, 0.6, 0.6, 0.8, 0.1, 0.5, 0.3
NOUN_SETS = ("lexical_number_pp", "perfect_number", "possessive_adjacent", "possessive_argument", "possessive_long_simple",
             "possessive_medial", "possessive_verbfinal")
V141_RECEIPT = ROOT / "circuits/followups/unit_tier5_cue_route_readout_map_v141_result.json"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def pairs_from_v129():
    r = json.loads(V129_RECEIPT.read_text())["behaviours"]
    return {n: [h for h, v in b["per_head"].items() if v.get("cue", 0) >= CUE_MIN] for n, b in r.items() if "skipped" not in b}


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_route_noun_invariance_v143", "behaviours": 9, "pairs": 23,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V143_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = pairs_from_v129()
    if smoke:
        pick = os.environ.get("V143_SMOKE_SET") or "lexical_number_pp"
        sets = {pick: sets[pick]}
    R, skipped = {}, {}
    for n, readers in sets.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows, geo = [], []
        for r in v123.rows_of(m, 1, smoke):
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            if len(diff) != 1:
                continue
            rows.append(r); geo.append({"t": t, "cue": diff[0]})
        if len(rows) < 4:
            skipped[n] = f"{len(rows)} one-cue rows"; print(n, "skipped", skipped[n], flush=True); continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        d_ax, b_ax = prep.donor_axis, prep.base_axis
        top = max(g.unit_layer(h) for h in readers)
        below_all = list(range(0, top))
        cuepos = tuple(ge["cue"] for ge in geo)
        Dc, Bc = dataclasses.replace(D, semantic_positions=cuepos), dataclasses.replace(B, semantic_positions=cuepos)
        hd, md = v120.head_cache(backend, Dc, below_all), v120.mlp_cache(backend, Dc, below_all)
        hb, mb = v120.head_cache(backend, Bc, below_all), v120.mlp_cache(backend, Bc, below_all)
        tpos = [ge["t"] for ge in geo]

        def cue_items(units, donor):
            hc, mc = (hd, md) if donor else (hb, mb)
            return ([(u, 0, hc) for u in units if u.startswith("attn")], [(u, 0, mc) for u in units if u.startswith("mlp")])

        for reader in readers:
            layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
            units = v132.units_of(range(0, layer))
            PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)
            PD, VD = v131.capture_with_clamp(backend, D, [], [], layer)

            def delta(P, V):
                return [P[i, h, ge["t"], ge["cue"]] * V[i, ge["cue"], h, :] - PB[i, h, ge["t"], ge["cue"]] * VB[i, ge["cue"], h, :] for i, ge in enumerate(geo)]

            live = delta(PD, VD)
            hi, mi = cue_items(units, True)
            P, V = v131.capture_with_clamp(backend, Bc, hi, mi, layer)
            w_cue = delta(P, V)
            hi, mi = cue_items(units, False)
            P, V = v131.capture_with_clamp(backend, Dc, hi, mi, layer)
            e_cue = delta(P, V)
            gen = torch.Generator().manual_seed(0)
            rand = [torch.randn(w.shape, generator=gen).to(w.device) * (w.norm() / g.HEAD_DIM ** 0.5) for w in w_cue]

            def add_margin(vecs):
                def pre(_m, args):
                    v = args[0].clone()
                    for i, vec in enumerate(vecs):
                        v[i, tpos[i], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] += vec.to(v.device, v.dtype)
                    return (v,) + tuple(args[1:])
                hdl = backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre)
                try:
                    out = g.forward_units(backend, B)
                finally:
                    hdl.remove()
                vals = [float(a) - float(f) for a, f in out.tolist()]
                per = [(-p - bb) / (dd - bb) for dd, bb, p in zip(d_ax, b_ax, vals) if abs(dd - bb) > 1e-6]
                return round(sum(per) / len(per), 3)

            def loo(vecs):
                tot = sum(vecs)
                return [(tot - v) / (len(vecs) - 1) for v in vecs]

            def pairwise_cos(vecs):
                M = torch.stack(vecs)
                M = M / (M.norm(dim=1, keepdim=True) + 1e-8)
                Cm = M @ M.T
                k = len(vecs)
                return round(float((Cm.sum() - Cm.diagonal().sum()) / (k * (k - 1))), 3)

            A = {"m_none": add_margin([torch.zeros_like(w) for w in w_cue]), "m_live": add_margin(live), "m_w": add_margin(w_cue),
                 "m_e": add_margin(e_cue), "m_we": add_margin([w + e for w, e in zip(w_cue, e_cue)]), "m_rand": add_margin(rand),
                 "loo_w": add_margin(loo(w_cue)), "loo_e": add_margin(loo(e_cue)),
                 "cos_ww": pairwise_cos(w_cue), "cos_ee": pairwise_cos(e_cue), "rows": len(rows), "noun_set": n in NOUN_SETS}
            fr = [float(c @ r / (r @ r)) for c, r in zip(w_cue, live) if float(r @ r) > 1e-8]
            A["vec_writes"] = round(sum(fr) / len(fr), 3) if fr else None
            cos = [float(torch.nn.functional.cosine_similarity(w, e, dim=0)) for w, e in zip(w_cue, e_cue)]
            A["rowcos_mean"] = round(sum(cos) / len(cos), 3)
            ok = abs(A["m_we"]) >= COUNT_MIN
            A["counts"] = ok
            A["ratio_w"] = round(A["m_w"] / A["m_we"], 3) if ok else None
            A["ratio_e"] = round(A["m_e"] / A["m_we"], 3) if ok else None
            A["loo_w_ratio"] = round(A["loo_w"] / A["m_w"], 3) if abs(A["m_w"]) >= COUNT_MIN else None
            A["loo_e_ratio"] = round(A["loo_e"] / A["m_e"], 3) if abs(A["m_e"]) >= COUNT_MIN else None
            A["kind"] = ("writes" if A["ratio_w"] >= ROUTE_MAJ else "embedding" if A["ratio_e"] >= ROUTE_MAJ else "mixed") if ok else None
            R[(n, reader)] = A
            print(n, reader, A, round(time.perf_counter() - t0), "s", flush=True)

    pairs = {f"{n}|{r}": A for (n, r), A in R.items()}
    v141 = json.loads(V141_RECEIPT.read_text())["pairs_detail"] if V141_RECEIPT.exists() and not smoke else {}
    a_ok = all(A["m_none"] == 0 and abs(A["m_rand"]) <= RAND_MAX for A in R.values())
    repro = [abs(A["m_w"] - v141[k]["m_w"]) <= INSTR_TOL and abs(A["m_e"] - v141[k]["m_e"]) <= INSTR_TOL for k, A in pairs.items() if k in v141]
    pred_a = a_ok and (smoke or (len(repro) == len(pairs) and all(repro)))
    counting = {k: A for k, A in R.items() if A["counts"]}
    wn = [A for A in counting.values() if A["kind"] == "writes" and A["noun_set"]]
    em = [A for A in counting.values() if A["kind"] == "embedding"]
    frac = lambda L, f: (sum(1 for A in L if f(A)) / len(L)) if L else 0.0
    pred_b = bool(wn) and frac(wn, lambda A: A["loo_w_ratio"] is not None and A["loo_w_ratio"] >= LOO_MIN) >= FRAC_MIN
    pred_c = bool(wn) and all(A["cos_ww"] >= A["cos_ee"] + COS_GAP for A in wn)
    pred_d = bool(em) and frac(em, lambda A: A["loo_e_ratio"] is not None and A["loo_e_ratio"] >= LOO_MIN and A["cos_ee"] >= E_COS_MIN) >= FRAC_MIN
    pred_e = bool(wn) and all(A["cos_ee"] <= E_COS_MAX for A in wn)
    predictions = {"pred_a_instrument": pred_a, "pred_b_noun_invariant_w": pred_b, "pred_c_w_more_consistent": pred_c,
                   "pred_d_fixed_pair_e": pred_d, "pred_e_e_noun_specific": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_route_noun_invariance_v143", "candidate_id": "corpus.unit_tier5_cue_route_noun_invariance_v143",
              "bars": {"cue_min": CUE_MIN, "count_min": COUNT_MIN, "rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "route_maj": ROUTE_MAJ, "loo_min": LOO_MIN,
                       "frac_min": FRAC_MIN, "cos_gap": COS_GAP, "e_cos_min": E_COS_MIN, "e_cos_max": E_COS_MAX},
              "counts": {"pairs": len(R), "counting": len(counting), "writes_noun": len(wn), "embedding": len(em), "v141_repro": [sum(repro), len(repro)],
                         "loo_w_ok": sum(1 for A in wn if A["loo_w_ratio"] is not None and A["loo_w_ratio"] >= LOO_MIN),
                         "loo_e_ok": sum(1 for A in em if A["loo_e_ratio"] is not None and A["loo_e_ratio"] >= LOO_MIN)},
              "skipped": skipped, "pairs_detail": pairs, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
