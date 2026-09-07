#!/usr/bin/env python3
# BQGATE: five frozen predictions; bands and positions fixed by rule (row-wise differing positions), no set chosen after results.
"""Tier-5 for the CUE positions: which layer band's writes AT the cue position(s) do the set's heads read?

v121/v122 covered the near NON-cue positions (t-1..t-3 with the same token in base and donor): their content is
written by the early band (0-4) for the number family + possessive_medial and by the mid band for the possessive
argument/verbfinal + narrative sets, and the six "far-cue" sets do not use them at all. What every set uses is the
CUE position(s) -- the positions p < t where base and donor tokens differ (v121 far_none = 0.0-0.1 on the far-cue
sets means the cue positions carry everything there). At a cue position the raw embedding differs by construction
and bilin18 re-mixes the embedding into every layer (live = lambda0 * x + lambda1 * x0), so the set's heads can read
the cue either from the RAW embedding or from what the layers WROTE at that position. This rung clamps the writes.

Design (donor side, ODD A1 rows, equal-length rows, base/donor semantic position equal):
  cue positions   row-wise {p < t : base_ids[p] != donor_ids[p]}   (any offset; rows with 0 cues are dropped)
  far non-cue     row-wise {p < t-3 : base_ids[p] == donor_ids[p]} (the same clamp on equal-token far positions)
  band clamp      every head + MLP output of the band at the cue positions set to the BASE run's values at those
                  positions (early 0-4 / mid 5-8 / late 9-11 / top 12-14 / all 0-14); the embedding is never touched
  instrument      the same clamp with the DONOR's own values (|loss| <= 0.02)
  loss            (donor - clamped) / (donor - base) on the donor answer axis

Registered before the run (bars fixed):
  pred_a_instrument      |loss| <= 0.02 on every set that runs
  pred_b_written_content all-band clamp at the cue positions loses >= 0.50 on >= 10 of 14 (the heads read written
                         content, not the raw embedding alone)                          Worked: 0.62 True; 0.41 False.
  pred_c_early_writers   the early band is the largest single band on >= 8 of 14      Worked: early 0.4 > mid 0.3 True.
  pred_d_localized       max single band >= 0.5 x all on >= 10 of 14                   Worked: 0.35 of 0.60 True; 0.25 False.
  pred_e_far_noncue_inert the all-band clamp at far NON-cue positions has |loss| <= 0.15 on >= 10 of 14 (capable of
                         failing: v122 found near non-cue content matters for 7 sets; far non-cue is the open case)
Prior: b likely (v118/v119 showed the late heads read structured content); c is the question -- early for token
identity, or mid if the cue is composed (e.g. 'each of' + noun); e uncertain.
If b fails the heads read the raw embedding through the lambda re-mix -- I would report that as the mechanism.

Smoke: V124_SMOKE=<out.json> runs on CPU with one set (V124_SMOKE_SET) and 4 rows.
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
import run_unit_tier4_expansion_batch_v115 as v115
import run_unit_tier5_carrier_relay_v120 as v120

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_writer_band_v124_result.json"
BANDS = {"early": range(0, 5), "mid": range(5, 9), "late": range(9, 12), "top": range(12, 15), "all": range(0, 15)}
FAR_GAP = 3
INSTR_TOL, WRITTEN_MIN, LOCAL_FRAC, FAR_MAX = 0.02, 0.50, 0.5, 0.15
K_B, K_C, K_D, K_E = 10, 8, 10, 10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 30000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_writer_band_v124", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def slot_caches(backend, batch, posmaps, layers):
    """For each slot j (a {rid: position} map), native caches of every head slice and MLP output at that position."""
    heads, mlps = [], []
    for pm in posmaps:
        pos = tuple(pm.get(rid, 0) for rid in batch.row_ids)
        b = dataclasses.replace(batch, semantic_positions=pos)
        hc = {k: v for k, v in v120.head_cache(backend, b, layers).items() if k[0] in pm}
        mc = {k: v for k, v in v120.mlp_cache(backend, b, layers).items() if k[0] in pm}
        heads.append((pm, hc)); mlps.append((pm, mc))
    return heads, mlps


def clamped_margins(backend, batch, layers, head_slots, mlp_slots):
    """Clamp every head slice / MLP output of `layers` at each slot's row-wise position to the slot's cache."""
    torch = backend.torch
    handles = []
    for l in layers:
        block = backend.model.transformer.h[l]
        def pre(_m, args, l=l):
            v = args[0].clone()
            for pm, cache in head_slots:
                for i, rid in enumerate(batch.row_ids):
                    if rid not in pm:
                        continue
                    for h in range(g.N_HEADS):
                        u = f"attn:{l:02d}:head:{h:02d}"
                        v[i, pm[rid], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype)
            return (v,) + tuple(args[1:])
        handles.append(block.attn.c_proj.register_forward_pre_hook(pre))
        bias = block.mlp.Down_bias
        def post(_m, _args, out, l=l, bias=bias):
            v = out.clone()
            u = f"mlp:{l:02d}"
            for pm, cache in mlp_slots:
                for i, rid in enumerate(batch.row_ids):
                    if rid in pm:
                        v[i, pm[rid]] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype) - bias
            return v
        handles.append(block.mlp.Down.register_forward_hook(post))
    try:
        out = g.forward_units(backend, batch)
    finally:
        for h in handles:
            h.remove()
    return [float(a) - float(f) for a, f in out.tolist()]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V124_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    if smoke:
        pick = os.environ.get("V124_SMOKE_SET") or list(S)[0]
        S = {pick: S[pick]}
    layers_all = list(BANDS["all"])
    R = {}
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["base_semantic_position"] == r["donor_semantic_position"]]
        cues = [[p for p in range(r["donor_semantic_position"]) if r["base_ids"][p] != r["donor_ids"][p]] for r in rows]
        keep = [i for i, c in enumerate(cues) if c]
        rows, cues = [rows[i] for i in keep], [cues[i] for i in keep]
        if len(rows) < 4:
            R[n] = {"units": units, "rows": len(rows), "skipped": "fewer than 4 equal-length rows with a cue"}
            print(n, "skipped", flush=True); continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        n_slots = max(len(c) for c in cues)
        cue_maps = [{rid: c[j] for rid, c in zip(D.row_ids, cues) if len(c) > j} for j in range(n_slots)]
        fars = [[p for p in range(r["donor_semantic_position"] - FAR_GAP) if r["base_ids"][p] == r["donor_ids"][p]] for r in rows]
        n_far = max(len(f) for f in fars)
        far_maps = [{rid: f[j] for rid, f in zip(D.row_ids, fars) if len(f) > j} for j in range(n_far)]
        hd, md = slot_caches(backend, D, cue_maps, layers_all)
        hb, mb = slot_caches(backend, B, cue_maps, layers_all)
        fhb, fmb = slot_caches(backend, B, far_maps, layers_all) if far_maps else ([], [])

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(prep.donor_axis, prep.base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        L = {"instrument": loss(clamped_margins(backend, D, layers_all, hd, md))}
        for band, ls in BANDS.items():
            L[band] = loss(clamped_margins(backend, D, list(ls), hb, mb))
        L["far_noncue_all"] = loss(clamped_margins(backend, D, layers_all, fhb, fmb)) if far_maps else None
        R[n] = {"units": units, "rows": len(rows), "cue_slots": n_slots, "far_slots": n_far,
                "cue_offsets_mean": round(sum(r["donor_semantic_position"] - c[-1] for r, c in zip(rows, cues)) / len(rows), 2),
                "loss": L, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, L, "rows", len(rows), "slots", n_slots, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    L_ = lambda n, k: R[n]["loss"].get(k)
    inst = [n for n in ran if L_(n, "instrument") is not None and abs(L_(n, "instrument")) <= INSTR_TOL]
    written = [n for n in ran if L_(n, "all") is not None and L_(n, "all") >= WRITTEN_MIN]
    single_band = {n: max((b for b in ("early", "mid", "late", "top") if L_(n, b) is not None), key=lambda b: L_(n, b)) for n in ran}
    early = [n for n in ran if single_band[n] == "early"]
    local = [n for n in ran if L_(n, "all") and L_(n, single_band[n]) >= LOCAL_FRAC * L_(n, "all")]
    far = [n for n in ran if L_(n, "far_noncue_all") is not None and abs(L_(n, "far_noncue_all")) <= FAR_MAX]
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_written_content": len(written) >= K_B,
        "pred_c_early_writers": len(early) >= K_C,
        "pred_d_localized": len(local) >= K_D,
        "pred_e_far_noncue_inert": len(far) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_cue_writer_band_v124",
              "candidate_id": "corpus.unit_tier5_cue_writer_band_v124",
              "bars": {"instr_tol": INSTR_TOL, "written_min": WRITTEN_MIN, "local_frac": LOCAL_FRAC, "far_max": FAR_MAX,
                       "far_gap": FAR_GAP, "K": [K_B, K_C, K_D, K_E], "bands": {b: [min(r), max(r)] for b, r in BANDS.items()}},
              "counts": {"ran": len(ran), "instrument": len(inst), "written": len(written), "early": len(early),
                         "localized": len(local), "far_inert": len(far), "n": len(R)},
              "largest_band": single_band,
              "summary": {n: R[n].get("loss") for n in R}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
