#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms fixed; readers from v131; random control seeded; one-cue rows by rule.
"""Tier-5: the two routes into the CUE key -- the token embedding itself and the layer writes -- do they add?

v134: clamping every layer write below the reader at the cue position removes only 0.68 / 0.61 / 0.34 of what
11:03 (lexical / perfect) and 07:08 (quantifier) read from the cue column; the remainder rides on the token
embedding, which bilin18 carries both in the residual chain and in the x0 re-mix at every layer. The mirror
experiment isolates the writes route: run the BASE batch with the DONOR's layer writes (every head and MLP of
layers < reader) clamped in at the cue position, and measure how much of the live (donor) cue-column delta the
reader recovers. Same mirror at the NEAR keys (t-1..t-3, cue-excluded; token equal on both sides) as the
instrument, where the writes are the only route. Reader-space vectors are compared by projection and cosine.

  keep_emb(cue)   = 1 - removed(all_below) from v134's arm, re-measured here      (embedding route alone)
  rec_writes(cue) = projection of the writes-only delta on the live delta            (writes route alone)

Registered before the run:
  pred_a_instrument   rec_writes(near) >= 0.9 on 3/3
  pred_b_routes_add   |rec_writes(cue) + keep_emb(cue) - 1| <= 0.15 on 3/3
  pred_c_reader_type  rec_writes(cue) >= 0.5 on lexical and perfect (11:03: writes-majority) and <= 0.4 on
                      quantifier (07:08: embedding-majority) -- 3/3
  pred_d_same_axis    cos(writes-only delta, embedding-only delta) >= 0.7 on 3/3 (the layers re-write the axis
                      the embedding already carries)
  pred_e_control      a seeded random early quartet's donor writes recover <= 0.15 on 3/3
Prior: a, e firm; b unsure (rms_norm and the q2k2 pattern make the value non-additive); c the v134 reading;
d the hypothesis this rung exists for.
Smoke: V135_SMOKE=<out.json>, V135_SMOKE_SET=<set> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_cue_two_routes_v135_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
INSTR_MIN, ADD_TOL, WRITES_MAJ, EMB_MAJ, COS_MIN, CTRL_MAX = 0.9, 0.15, 0.5, 0.4, 0.7, 0.15
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_two_routes_v135", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V135_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V135_SMOKE_SET") or "lexical_number_pp"
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
        # cue key: writes-only (base batch + donor writes), embedding-only (donor batch + base writes), random control
        hi, mi = cue_items(all_units, True)
        P, V = v131.capture_with_clamp(backend, B, hi, mi, layer)
        w_cue = delta(P, V, "cue")
        A["cue_writes_only"] = proj(w_cue, live["cue"])
        hi, mi = cue_items(all_units, False)
        P, V = v131.capture_with_clamp(backend, D, hi, mi, layer)
        e_cue = delta(P, V, "cue")
        A["cue_embedding_only"] = proj(e_cue, live["cue"])
        hi, mi = cue_items(rand4, True)
        P, V = v131.capture_with_clamp(backend, B, hi, mi, layer)
        A["cue_random4_writes"] = proj(delta(P, V, "cue"), live["cue"])
        wv, ev = mean_vec(w_cue), mean_vec(e_cue)
        A["cos_writes_vs_embedding"] = round(float(torch.nn.functional.cosine_similarity(wv, ev, dim=0)), 3) if wv is not None and ev is not None else None
        # near keys: writes-only (instrument) and embedding-only (should be ~0: tokens equal)
        hi, mi = near_items(all_units, True)
        P, V = v131.capture_with_clamp(backend, B, hi, mi, layer)
        A["near_writes_only"] = proj(delta(P, V, "near"), live["near"])
        hi, mi = near_items(all_units, False)
        P, V = v131.capture_with_clamp(backend, D, hi, mi, layer)
        A["near_embedding_only"] = proj(delta(P, V, "near"), live["near"])
        R[n] = {"reader": reader, "rows": len(rows), "random4": rand4, "arms": A, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, A, round(time.perf_counter() - t0), "s", flush=True)

    def v(n, k):
        x = R[n]["arms"].get(k)
        return -9 if x is None else x

    ran = [n for n in R if "skipped" not in R[n]]
    a = [n for n in ran if v(n, "near_writes_only") >= INSTR_MIN]
    b = [n for n in ran if abs(v(n, "cue_writes_only") + v(n, "cue_embedding_only") - 1) <= ADD_TOL]
    c = [n for n in ran if (v(n, "cue_writes_only") <= EMB_MAJ if n == "quantifier_number" else v(n, "cue_writes_only") >= WRITES_MAJ)]
    d = [n for n in ran if v(n, "cos_writes_vs_embedding") >= COS_MIN]
    e = [n for n in ran if abs(v(n, "cue_random4_writes")) <= CTRL_MAX]
    full = len(READERS)
    predictions = {"pred_a_instrument": len(a) == full, "pred_b_routes_add": len(b) == full, "pred_c_reader_type": len(c) == full,
                   "pred_d_same_axis": len(d) == full, "pred_e_control": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_two_routes_v135", "candidate_id": "corpus.unit_tier5_cue_two_routes_v135",
              "bars": {"instr_min": INSTR_MIN, "add_tol": ADD_TOL, "writes_maj": WRITES_MAJ, "emb_maj": EMB_MAJ, "cos_min": COS_MIN, "ctrl_max": CTRL_MAX},
              "counts": {"ran": len(ran), "instrument": len(a), "routes_add": len(b), "reader_type": len(c), "same_axis": len(d), "control": len(e)},
              "summary": {n: R[n]["arms"] for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
