#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms fixed; readers from v131; random control seeded; one-cue rows by rule.
"""Tier-5: the two routes into the CUE key at the MARGIN, and per-row cosine.

v136: the writes route (layer writes below the reader, clamped at the cue position) and the embedding route
(the token itself, via the x0 re-mix and the residual chain) add in projection on the reader's live cue
delta (0.98 / 0.95 / 0.90) but their mean vectors are near-orthogonal (cos -0.14 / -0.17 / +0.39). Two open
questions. (1) Is the orthogonality real per row, or a mean-vector artefact? (2) Does the answer MARGIN split
the same way -- the reader's column is one consumer of the cue key; every head above the clamp reads it too.

Arms (v122 clamps at the cue position through Bc/Dc; margin read at the last token; donor axis):
  rec_writes_margin = (m(base + donor writes at cue) - b) / (d - b)     writes route alone, at the margin
  keep_emb_margin   = (m(donor + base writes at cue) - b) / (d - b)     embedding route alone, at the margin
  random4_margin    = (m(base + random early quartet's donor writes at cue) - b) / (d - b)
  rowcos_median     = median over rows of cos(writes-only reader delta, embedding-only reader delta)
  (the v136 projections are recomputed in-run as `vec_*` and must reproduce the v136 receipt: instrument)

Registered before the run:
  pred_a_instrument   |vec_writes - v136 cue_writes_only| <= 0.02, no-clamp base/donor margins at 0/1 within 0.02,
                      and |random4_margin| <= 0.15 on 3/3
  pred_b_rowwise      rowcos_median <= 0.5 on 3/3 (the near-orthogonality is per-row, not a mean artefact)
  pred_c_margin_add   |rec_writes_margin + keep_emb_margin - 1| <= 0.15 on 3/3
  pred_d_margin_match |rec_writes_margin - vec_writes| <= 0.15 on 3/3 (the reader's column accounts for the
                      margin split; fails if heads other than the reader consume the cue-key writes)
  pred_e_majority     rec_writes_margin >= 0.5 on lexical and perfect, <= 0.4 on quantifier -- 3/3
Prior: a firm; b unsure; c likely; d unsure (07:08's set has four other heads reading the cue, v129); e likely.
Smoke: V137_SMOKE=<out.json>, V137_SMOKE_SET=<set> -> CPU, 4 rows.
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
import run_unit_tier5_near_writer_band_v122 as v122
import statistics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_routes_margin_v137_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
INSTR_TOL, CTRL_MAX, ROWCOS_MAX, ADD_TOL, MATCH_TOL, WRITES_MAJ, EMB_MAJ = 0.02, 0.15, 0.5, 0.15, 0.15, 0.5, 0.4
V136_RECEIPT = ROOT / "circuits/followups/unit_tier5_cue_two_routes_v136_result.json"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_routes_margin_v137", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V137_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V137_SMOKE_SET") or "lexical_number_pp"
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
        A["vec_writes"] = proj(w_cue, live["cue"])
        hi, mi = cue_items(all_units, False)
        P, V = v131.capture_with_clamp(backend, Dc, hi, mi, layer)
        e_cue = delta(P, V, "cue")
        A["vec_embedding"] = proj(e_cue, live["cue"])
        rc = [float(torch.nn.functional.cosine_similarity(w, e, dim=0)) for w, e in zip(w_cue, e_cue) if w is not None and e is not None]
        A["rowcos_median"] = round(statistics.median(rc), 3) if rc else None
        A["rowcos_frac_ge_0.5"] = round(sum(1 for c in rc if c >= 0.5) / len(rc), 3) if rc else None
        # margins on the donor axis, clamps at the cue position (Bc/Dc carry the cue positions; the margin is read at the last token)
        d_ax, b_ax = side.donor_axis, side.base_axis

        def share(vals, sign):
            # clamped_margins returns answer - foil on the batch's OWN answer axis; the base batch's is the donor axis negated
            per = [(sign * p - b) / (d - b) for d, b, p in zip(d_ax, b_ax, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        hi, mi = cue_items(all_units, True)
        A["rec_writes_margin"] = share(v122.clamped_margins(backend, Bc, hi, mi), -1)
        hi, mi = cue_items(all_units, False)
        A["keep_emb_margin"] = share(v122.clamped_margins(backend, Dc, hi, mi), 1)
        hi, mi = cue_items(rand4, True)
        A["random4_margin"] = share(v122.clamped_margins(backend, Bc, hi, mi), -1)
        A["margin_check_none"] = share(v122.clamped_margins(backend, Bc, [], []), -1)
        A["margin_check_donor"] = share(v122.clamped_margins(backend, Dc, [], []), 1)
        R[n] = {"reader": reader, "rows": len(rows), "random4": rand4, "arms": A, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, A, round(time.perf_counter() - t0), "s", flush=True)

    def v(n, k):
        x = R[n]["arms"].get(k)
        return -9 if x is None else x

    ran = [n for n in R if "skipped" not in R[n]]
    v136 = json.loads(V136_RECEIPT.read_text())["summary"] if V136_RECEIPT.exists() and not smoke else {}
    for n in ran:
        R[n]["arms"]["v136_vec_writes"] = v136.get(n, {}).get("cue_writes_only")
    a = [n for n in ran if v(n, "v136_vec_writes") != -9 and abs(v(n, "vec_writes") - v(n, "v136_vec_writes")) <= INSTR_TOL and abs(v(n, "random4_margin")) <= CTRL_MAX and abs(v(n, "margin_check_none")) <= INSTR_TOL and abs(v(n, "margin_check_donor") - 1) <= INSTR_TOL]
    b = [n for n in ran if v(n, "rowcos_median") != -9 and v(n, "rowcos_median") <= ROWCOS_MAX]
    c = [n for n in ran if abs(v(n, "rec_writes_margin") + v(n, "keep_emb_margin") - 1) <= ADD_TOL]
    d = [n for n in ran if abs(v(n, "rec_writes_margin") - v(n, "vec_writes")) <= MATCH_TOL]
    e = [n for n in ran if (v(n, "rec_writes_margin") <= EMB_MAJ if n == "quantifier_number" else v(n, "rec_writes_margin") >= WRITES_MAJ)]
    full = len(READERS)
    predictions = {"pred_a_instrument": len(a) == full, "pred_b_rowwise": len(b) == full, "pred_c_margin_add": len(c) == full,
                   "pred_d_margin_match": len(d) == full, "pred_e_majority": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_routes_margin_v137", "candidate_id": "corpus.unit_tier5_cue_routes_margin_v137",
              "bars": {"instr_tol": INSTR_TOL, "ctrl_max": CTRL_MAX, "rowcos_max": ROWCOS_MAX, "add_tol": ADD_TOL, "match_tol": MATCH_TOL, "writes_maj": WRITES_MAJ, "emb_maj": EMB_MAJ},
              "counts": {"ran": len(ran), "instrument": len(a), "rowwise": len(b), "margin_add": len(c), "margin_match": len(d), "majority": len(e)},
              "summary": {n: R[n]["arms"] for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
