#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms fixed; readers from v131; random control seeded; one-cue rows by rule.
"""Tier-5: inside the reader, which component of the cue-column delta does the margin read?

v136/v137: the two routes into the cue key -- layer writes (w) and the token embedding (e) -- are near-orthogonal
vectors in the reader's head space (per-row cos median -0.11 / -0.12 / +0.39) that add in projection on the live
cue delta (0.98 / 0.95 / 0.90). At the margin the writes route carries 0.91 / 0.88 of 11:03's behaviours although
it is only 0.66 / 0.56 of 11:03's own cue-column delta. Two explanations: (i) the downstream readout is aligned
with w and reads e weakly; (ii) heads other than 11:03 consume the cue-key writes. This rung separates them by
ADDING each component into the reader's own c_proj slot at t on the BASE batch (per-head delta addition is exact
at the head's own layer, v103) and reading the margin: m_live (full cue delta), m_w, m_e, m_we (w + e), m_rand
(seeded Gaussian at |w|). Margin shares on the donor axis, (m - b) / (d - b).

Registered before the run:
  pred_a_instrument   |m_rand| <= 0.03 and 0.8 <= m_we / m_live <= 1.2 on 3/3 (w + e reproduce the live cue delta)
  pred_b_readout_w    m_w / m_we >= 0.8 on lexical and perfect (the readout is aligned with the writes component)
  pred_c_consumers    m_w / m_we <= 0.7 on lexical and perfect (11:03 reads e as well as its projection says;
                      v137's excess margin comes from other consumers of the cue writes) -- exclusive with b
  pred_d_additive     |(m_w + m_e) / m_we - 1| <= 0.2 on 3/3
  pred_e_quantifier   m_e / m_we >= 0.6 on quantifier (07:08 is embedding-majority inside the reader too)
Prior: a firm; b vs c genuinely open (v129: 11:03's cue share is 0.22 of its own 0.42 on lexical, so other
consumers exist); d likely; e the v137 reading.
Smoke: V138_SMOKE=<out.json>, V138_SMOKE_SET=<set> -> CPU, 4 rows.
"""
from __future__ import annotations

import dataclasses
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
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_readout_alignment_v138_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
RAND_MAX, REPRO_LO, REPRO_HI, W_MAJ, W_MIN, ADD_TOL, E_MAJ = 0.03, 0.8, 1.2, 0.8, 0.7, 0.2, 0.6
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_readout_alignment_v138", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V138_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V138_SMOKE_SET") or "lexical_number_pp"
        sets = {pick: READERS[pick]}
    R = {}
    for n, reader in sets.items():
        t1 = time.perf_counter()
        layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
        below = list(range(0, layer))
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows, geo = [], []
        for r in v123.rows_of(m, 1, smoke):
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            if len(diff) != 1:
                continue
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            rows.append(r); geo.append({"t": t, "cue": diff[0], "near": [p for p in range(max(diff[0] + 1, t - FAR_GAP), t) if eq(p)]})
        if len(rows) < 4:
            R[n] = {"skipped": f"{len(rows)} one-cue rows"}; print(n, "skipped", flush=True); continue
        side = v123.Side(backend, rows, below)
        D, B = side.D, side.B
        cuepos = tuple(ge["cue"] for ge in geo)
        Dc, Bc = dataclasses.replace(D, semantic_positions=cuepos), dataclasses.replace(B, semantic_positions=cuepos)
        hd, md = v120.head_cache(backend, Dc, below), v120.mlp_cache(backend, Dc, below)
        hb, mb = v120.head_cache(backend, Bc, below), v120.mlp_cache(backend, Bc, below)
        all_units = v132.units_of(below)
        rand4 = random.Random(0).sample(v132.units_of(range(0, 5), mlps=False), 4)

        def cue_items(units, donor):
            hc, mc = (hd, md) if donor else (hb, mb)
            return ([(u, 0, hc) for u in units if u.startswith("attn")], [(u, 0, mc) for u in units if u.startswith("mlp")])

        def near_items(units, donor):
            hs, ms = (side.hd, side.md) if donor else (side.hb, side.mb)
            return ([(u, -k, hs[k]) for k in v123.OFFSETS for u in units if u.startswith("attn")],
                    [(u, -k, ms[k]) for k in v123.OFFSETS for u in units if u.startswith("mlp")])

        PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)
        PD, VD = v131.capture_with_clamp(backend, D, [], [], layer)

        def delta(P, V, kind):
            out = []
            for i, ge in enumerate(geo):
                cs = torch.tensor([ge["cue"]] if kind == "cue" else ge["near"])
                out.append(P[i, h, ge["t"], cs] @ V[i, cs, h, :] - PB[i, h, ge["t"], cs] @ VB[i, cs, h, :] if len(cs) else None)
            return out

        live = {k: delta(PD, VD, k) for k in ("cue", "near")}

        def proj(cur, ref):
            fr = [float(c @ r / (r @ r)) for c, r in zip(cur, ref) if c is not None and r is not None and float(r @ r) > 1e-8]
            return round(sum(fr) / len(fr), 3) if fr else None

        def mean_vec(vs):
            vs = [v for v in vs if v is not None]
            return torch.stack(vs).mean(0) if vs else None

        A = {}
        hi, mi = cue_items(all_units, True)
        P, V = v131.capture_with_clamp(backend, Bc, hi, mi, layer)
        w_cue = delta(P, V, "cue")
        hi, mi = cue_items(all_units, False)
        P, V = v131.capture_with_clamp(backend, Dc, hi, mi, layer)
        e_cue = delta(P, V, "cue")
        A["vec_writes"], A["vec_embedding"] = proj(w_cue, live["cue"]), proj(e_cue, live["cue"])
        d_ax, b_ax = side.donor_axis, side.base_axis
        gen = torch.Generator().manual_seed(0)
        rand = [torch.randn(w.shape, generator=gen) * (w.norm() / g.HEAD_DIM ** 0.5) for w in w_cue]
        tpos = [ge["t"] for ge in geo]

        def add_margin(vecs):
            # add vecs[i] into head h's c_proj input slot at t_i on the base batch (delta addition at the head's own layer)
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

        A["m_none"] = add_margin([torch.zeros_like(w) for w in w_cue])
        A["m_live"] = add_margin(live["cue"])
        A["m_w"] = add_margin(w_cue)
        A["m_e"] = add_margin(e_cue)
        A["m_we"] = add_margin([w + e for w, e in zip(w_cue, e_cue)])
        A["m_rand"] = add_margin(rand)
        A["ratio_w"] = round(A["m_w"] / A["m_we"], 3) if abs(A["m_we"]) > 1e-6 else None
        A["ratio_e"] = round(A["m_e"] / A["m_we"], 3) if abs(A["m_we"]) > 1e-6 else None
        R[n] = {"reader": reader, "rows": len(rows), "random4": rand4, "arms": A, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, A, round(time.perf_counter() - t0), "s", flush=True)

    def v(n, k):
        x = R[n]["arms"].get(k)
        return -9 if x is None else x

    ran = [n for n in R if "skipped" not in R[n]]
    eleven = [n for n in ran if n != "quantifier_number"]
    a = [n for n in ran if abs(v(n, "m_rand")) <= RAND_MAX and abs(v(n, "m_none")) <= RAND_MAX and abs(v(n, "m_live")) > 1e-6 and REPRO_LO <= v(n, "m_we") / v(n, "m_live") <= REPRO_HI]
    b = [n for n in eleven if v(n, "ratio_w") >= W_MAJ]
    c = [n for n in eleven if v(n, "ratio_w") != -9 and v(n, "ratio_w") <= W_MIN]
    d = [n for n in ran if abs(v(n, "m_we")) > 1e-6 and abs((v(n, "m_w") + v(n, "m_e")) / v(n, "m_we") - 1) <= ADD_TOL]
    e = [n for n in ran if n == "quantifier_number" and v(n, "ratio_e") >= E_MAJ]
    full = len(READERS)
    predictions = {"pred_a_instrument": len(a) == full, "pred_b_readout_w": len(b) == 2, "pred_c_consumers": len(c) == 2,
                   "pred_d_additive": len(d) == full, "pred_e_quantifier": len(e) == 1}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_readout_alignment_v138", "candidate_id": "corpus.unit_tier5_cue_readout_alignment_v138",
              "bars": {"rand_max": RAND_MAX, "repro": [REPRO_LO, REPRO_HI], "w_maj": W_MAJ, "w_min": W_MIN, "add_tol": ADD_TOL, "e_maj": E_MAJ},
              "counts": {"ran": len(ran), "instrument": len(a), "readout_w": len(b), "consumers": len(c), "additive": len(d), "quantifier": len(e)},
              "summary": {n: R[n]["arms"] for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
