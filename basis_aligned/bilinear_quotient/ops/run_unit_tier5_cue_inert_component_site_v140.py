#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms fixed; readers from v131; random control seeded; one-cue rows by rule.
"""Tier-5: where is the reader's inert cue component discarded -- in c_proj, or downstream of it?

v139: each reader's cue-column delta is two near-orthogonal head-space vectors, and the margin reads only one of
them -- 11:03 the layer-writes component w (e inert: -0.02 / -0.05), 07:08 the embedding component e (w inert:
0.09). Three places the inert component could die: (i) W_O (c_proj) maps it to nothing; (ii) its residual image
is orthogonal to the unembedding margin direction; (iii) layers above the reader cancel it. Arms: the c_proj
image norms of w and e; `resid_add` of each image at the reader's own layer (must reproduce v139's slot add --
c_proj is linear); and the same image added at layer 17 (after attn:17, only mlp:17 and the final norm remain --
the direct path). Margin shares on the donor axis.

Registered before the run:
  pred_a_instrument   |m_x_resid - v139 m_x| <= 0.02 for x in (w, e) on 3/3, and |m_rand_late| <= 0.03
  pred_b_wo_passes    the INERT component's c_proj image has >= 0.5 of the ACTIVE component's image norm on 3/3
                      (inert = e on lexical/perfect, w on quantifier) -- W_O is not the filter
  pred_c_direct_blind |m_inert_late| <= 0.15 * |m_active_late| on 3/3 (the unembedding is blind to the inert
                      image: the discrimination is W_O -> W_U geometry)
  pred_d_cancelled    |m_inert_late| >= 0.5 * |m_active_late| on lexical and perfect (the direct path DOES read
                      the inert image and layers 12-17 cancel it) -- exclusive with c
  pred_e_direct_share m_active_late / m_active_resid >= 0.3 on 3/3 (the direct path carries a third or more of
                      the active route's margin; the rest is amplified by later layers)
Prior: a firm; b likely (v136's projections say both components are sizeable in head space); c vs d open;
e unsure (mlp:08-11 increment the number, memory; layers 12-17 may carry most of it).
Smoke: V140_SMOKE=<out.json>, V140_SMOKE_SET=<set> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_cue_inert_component_site_v140_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
INSTR_TOL, RAND_MAX, PASS_MIN, BLIND_MAX, CANCEL_MIN, DIRECT_MIN = 0.02, 0.03, 0.5, 0.15, 0.5, 0.3
LATE = 17
V139_RECEIPT = ROOT / "circuits/followups/unit_tier5_cue_readout_alignment_v139_result.json"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_inert_component_site_v140", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V140_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V140_SMOKE_SET") or "lexical_number_pp"
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
        rand = [torch.randn(w.shape, generator=gen).to(w.device) * (w.norm() / g.HEAD_DIM ** 0.5) for w in w_cue]  # v138 died here on cuda (randn on cpu)
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

        W = backend.model.transformer.h[layer].attn.c_proj.weight[:, h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM].detach().float()
        img = lambda vecs: torch.stack([vec.float().to(W.device) @ W.T for vec in vecs])          # (n, N_EMBD)
        Iw, Ie, Ir = img(w_cue), img(e_cue), img(rand)
        A["norm_w_head"] = round(float(torch.stack(w_cue).norm(dim=1).mean()), 4)
        A["norm_e_head"] = round(float(torch.stack(e_cue).norm(dim=1).mean()), 4)
        A["norm_w_img"], A["norm_e_img"] = round(float(Iw.norm(dim=1).mean()), 4), round(float(Ie.norm(dim=1).mean()), 4)

        def resid_margin(image, at):
            out = g.forward_units(backend, B, resid_add={at: image})
            vals = [float(a) - float(f) for a, f in out.tolist()]
            per = [(-p - bb) / (dd - bb) for dd, bb, p in zip(d_ax, b_ax, vals) if abs(dd - bb) > 1e-6]
            return round(sum(per) / len(per), 3)

        A["m_w_resid"], A["m_e_resid"] = resid_margin(Iw, layer), resid_margin(Ie, layer)
        A["m_w_late"], A["m_e_late"], A["m_rand_late"] = resid_margin(Iw, LATE), resid_margin(Ie, LATE), resid_margin(Ir, LATE)
        act, ine = ("e", "w") if n == "quantifier_number" else ("w", "e")
        A["active"] = act
        A["pass_ratio"] = round(A[f"norm_{ine}_img"] / A[f"norm_{act}_img"], 3)
        A["late_inert_over_active"] = round(abs(A[f"m_{ine}_late"]) / abs(A[f"m_{act}_late"]), 3) if abs(A[f"m_{act}_late"]) > 1e-6 else None
        A["direct_share"] = round(A[f"m_{act}_late"] / A[f"m_{act}_resid"], 3) if abs(A[f"m_{act}_resid"]) > 1e-6 else None
        R[n] = {"reader": reader, "rows": len(rows), "random4": rand4, "arms": A, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, A, round(time.perf_counter() - t0), "s", flush=True)

    def v(n, k):
        x = R[n]["arms"].get(k)
        return -9 if x is None else x

    ran = [n for n in R if "skipped" not in R[n]]
    eleven = [n for n in ran if n != "quantifier_number"]
    v139 = json.loads(V139_RECEIPT.read_text())["summary"] if V139_RECEIPT.exists() and not smoke else {}
    for n in ran:
        R[n]["arms"]["v139_m_w"], R[n]["arms"]["v139_m_e"] = v139.get(n, {}).get("m_w"), v139.get(n, {}).get("m_e")
    a = [n for n in ran if v(n, "v139_m_w") != -9 and abs(v(n, "m_w_resid") - v(n, "v139_m_w")) <= INSTR_TOL
         and abs(v(n, "m_e_resid") - v(n, "v139_m_e")) <= INSTR_TOL and abs(v(n, "m_rand_late")) <= RAND_MAX]
    b = [n for n in ran if v(n, "pass_ratio") >= PASS_MIN]
    c = [n for n in ran if v(n, "late_inert_over_active") != -9 and v(n, "late_inert_over_active") <= BLIND_MAX]
    d = [n for n in eleven if v(n, "late_inert_over_active") >= CANCEL_MIN]
    e = [n for n in ran if v(n, "direct_share") >= DIRECT_MIN]
    full = len(READERS)
    predictions = {"pred_a_instrument": len(a) == full, "pred_b_wo_passes": len(b) == full, "pred_c_direct_blind": len(c) == full,
                   "pred_d_cancelled": len(d) == 2, "pred_e_direct_share": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_inert_component_site_v140", "candidate_id": "corpus.unit_tier5_cue_inert_component_site_v140",
              "bars": {"instr_tol": INSTR_TOL, "rand_max": RAND_MAX, "pass_min": PASS_MIN, "blind_max": BLIND_MAX, "cancel_min": CANCEL_MIN, "direct_min": DIRECT_MIN, "late": LATE},
              "counts": {"ran": len(ran), "instrument": len(a), "wo_passes": len(b), "direct_blind": len(c), "cancelled": len(d), "direct_share": len(e)},
              "summary": {n: R[n]["arms"] for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
