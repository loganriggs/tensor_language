#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: across every cue reader of the corpus -- which of the two cue routes does each head's margin read?

v136-v140 on three reader-set pairs: the cue-column delta a head carries splits into two near-orthogonal head-space
vectors -- the layer-writes route w (donor writes below the reader clamped at the cue position on the base batch)
and the token-embedding route e (base writes clamped on the donor batch) -- and the margin reads ONE of them per
reader: 11:03 reads w (e inert), 07:08 reads e (w inert). Is one-route readout general, and is the route a property
of the HEAD (hub 07:08 = embedding reader everywhere, 11:03 = writes reader everywhere) or of the LAYER (early
readers see few writes)? Pairs: every (set, head) with cue-column loss >= 0.04 in v129 (32 pairs, 12 sets).
Arms per pair (v139 method): add w, e, w + e, the live cue delta, and a seeded Gaussian at |w| into the head's
own c_proj slot at t on the base batch; margin shares on the donor axis. ratio_w = m_w / m_we, ratio_e = m_e / m_we.

Registered before the run (a pair "counts" when |m_we| >= 0.05; sets with < 4 one-cue rows are skipped and listed):
  pred_a_instrument   m_none = 0 and |m_rand| <= 0.03 on every pair; the three v139 pairs reproduce m_w and m_e
                      within 0.02
  pred_b_one_route    >= 0.8 of counting pairs have min(|m_w|, |m_e|) / |m_we| <= 0.25
  pred_c_hub_e        07:08 has ratio_e >= 0.6 on every counting set (>= 2 such sets, else untested = False)
  pred_d_1103_w       11:03 has ratio_w >= 0.6 on every counting set (>= 2 such sets, else untested = False)
  pred_e_layer_rule   among counting pairs, the e-majority fraction (ratio_e >= 0.5) is >= 0.7 for reader layer <= 8
                      and <= 0.3 for layer >= 10 (>= 3 pairs in each band, else False)
Prior: a firm; b likely (3/3 so far); c, d the head hypothesis; e the layer hypothesis -- c/d and e can both hold
(07:08 is layer 7, 11:03 layer 11), so the discriminating cases are 08:01 / 14:08 / 16:08 (correlative, degree),
03:00 / 02:06 (interrogative), 04:05 / 10:01 / 12:04 (possessives).
Smoke: V141_SMOKE=<out.json>, V141_SMOKE_SET=<set> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_cue_route_readout_map_v141_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
V139_RECEIPT = ROOT / "circuits/followups/unit_tier5_cue_readout_alignment_v139_result.json"
CUE_MIN, COUNT_MIN, RAND_MAX, INSTR_TOL, ONE_ROUTE, ONE_FRAC, ROUTE_MAJ, E_MAJ_LO, E_MAJ_HI = 0.04, 0.05, 0.03, 0.02, 0.25, 0.8, 0.6, 0.7, 0.3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def pairs_from_v129():
    r = json.loads(V129_RECEIPT.read_text())["behaviours"]
    return {n: [h for h, v in b["per_head"].items() if v.get("cue", 0) >= CUE_MIN] for n, b in r.items() if "skipped" not in b}


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_route_readout_map_v141", "behaviours": 12, "pairs": 32,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V141_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = pairs_from_v129()
    if smoke:
        pick = os.environ.get("V141_SMOKE_SET") or "lexical_number_pp"
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

            A = {"m_none": add_margin([torch.zeros_like(w) for w in w_cue]), "m_live": add_margin(live), "m_w": add_margin(w_cue),
                 "m_e": add_margin(e_cue), "m_we": add_margin([w + e for w, e in zip(w_cue, e_cue)]), "m_rand": add_margin(rand)}
            fr = [float(c @ r / (r @ r)) for c, r in zip(w_cue, live) if float(r @ r) > 1e-8]
            A["vec_writes"] = round(sum(fr) / len(fr), 3) if fr else None
            cos = [float(torch.nn.functional.cosine_similarity(w, e, dim=0)) for w, e in zip(w_cue, e_cue)]
            A["rowcos_mean"] = round(sum(cos) / len(cos), 3)
            ok = abs(A["m_we"]) >= COUNT_MIN
            A["counts"] = ok
            A["ratio_w"] = round(A["m_w"] / A["m_we"], 3) if ok else None
            A["ratio_e"] = round(A["m_e"] / A["m_we"], 3) if ok else None
            A["inert_share"] = round(min(abs(A["m_w"]), abs(A["m_e"])) / abs(A["m_we"]), 3) if ok else None
            R[(n, reader)] = A
            print(n, reader, A, round(time.perf_counter() - t0), "s", flush=True)

    pairs = {f"{n}|{r}": A for (n, r), A in R.items()}
    v139 = json.loads(V139_RECEIPT.read_text())["summary"] if V139_RECEIPT.exists() and not smoke else {}
    a_ok = all(A["m_none"] == 0 and abs(A["m_rand"]) <= RAND_MAX for A in R.values())
    repro = []
    for (n, r), A in R.items():
        if n in v139 and v131.READERS.get(n) == r:
            repro.append(abs(A["m_w"] - v139[n]["m_w"]) <= INSTR_TOL and abs(A["m_e"] - v139[n]["m_e"]) <= INSTR_TOL)
    pred_a = a_ok and len(repro) == 3 and all(repro)
    counting = {k: A for k, A in R.items() if A["counts"]}
    one = [k for k, A in counting.items() if A["inert_share"] <= ONE_ROUTE]
    pred_b = bool(counting) and len(one) / len(counting) >= ONE_FRAC
    hub = {n: A for (n, r), A in counting.items() if r == "attn:07:head:08"}
    pred_c = len(hub) >= 2 and all(A["ratio_e"] >= ROUTE_MAJ for A in hub.values())
    h1103 = {n: A for (n, r), A in counting.items() if r == "attn:11:head:03"}
    pred_d = len(h1103) >= 2 and all(A["ratio_w"] >= ROUTE_MAJ for A in h1103.values())
    early = [A for (n, r), A in counting.items() if g.unit_layer(r) <= 8]
    late = [A for (n, r), A in counting.items() if g.unit_layer(r) >= 10]
    e_frac = lambda L: sum(1 for A in L if A["ratio_e"] >= 0.5) / len(L)
    pred_e = len(early) >= 3 and len(late) >= 3 and e_frac(early) >= E_MAJ_LO and e_frac(late) <= E_MAJ_HI
    predictions = {"pred_a_instrument": pred_a, "pred_b_one_route": pred_b, "pred_c_hub_e": pred_c, "pred_d_1103_w": pred_d, "pred_e_layer_rule": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_route_readout_map_v141", "candidate_id": "corpus.unit_tier5_cue_route_readout_map_v141",
              "bars": {"cue_min": CUE_MIN, "count_min": COUNT_MIN, "rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "one_route": ONE_ROUTE, "one_frac": ONE_FRAC,
                       "route_maj": ROUTE_MAJ, "e_maj_band": [E_MAJ_LO, E_MAJ_HI]},
              "counts": {"pairs": len(R), "counting": len(counting), "one_route": len(one), "hub_sets": len(hub), "h1103_sets": len(h1103),
                         "early": len(early), "late": len(late), "e_frac_early": round(e_frac(early), 3) if early else None,
                         "e_frac_late": round(e_frac(late), 3) if late else None, "v139_repro": repro},
              "skipped": skipped, "pairs_detail": pairs, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
