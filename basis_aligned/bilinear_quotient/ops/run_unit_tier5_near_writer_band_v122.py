#!/usr/bin/env python3
# BQGATE: five frozen predictions; set partition by a fixed rule on the v121 receipt; bands, offsets and bars fixed before the run.
"""v122: WHICH layers write the changed content at the near non-cue positions? (Tier-5 step, second attempt)
v121: at the near non-cue offsets (1-3) the set's heads read CONTENT change (near_content = full on 14/14; pattern ~0),
and that content carries 0.29-0.52 of the effect on seven sets and 0.00 on four. v120 clamped only layers <= 8 (+
mlp_6-8) at those positions and found <= 0.19 loss, so the writers are elsewhere. This rung base-clamps, on the DONOR
run, every attention head AND every MLP output of one layer band at the near non-cue positions (t-1..t-3, excluding
offsets whose base/donor tokens differ, rows with equal base/donor length), and measures the donor-margin loss:
    bands   early 0-4 | mid 5-8 | late 9-11 | top 12-14 | all 0-14        instrument: all 0-14 clamped to the donor's own values
    loss := (donor - patched) / (donor - base) on the donor's answer axis
Partition (fixed rule on the v121 receipt): NEAR sets = far_none >= 0.20 (7: lexical_number_pp, quantifier_number,
perfect_number, possessive_medial, possessive_argument, possessive_verbfinal, additive_scope); CONTROL sets =
far_none <= 0.10 (6: interrogative_licensing, degree_frame, preposition_selection, possessive_adjacent,
correlative_either_neither, possessive_long_simple); narrative_tense (0.17) is reported unclassified. The controls get
the SAME clamp and can fail (v120's did: 0.35/0.42) -- a failure there says the clamp is a generic perturbation.
REGISTERED BEFORE THE RUN (ODD rows; losses as fractions of the donor margin)
    pred_a_instrument      instrument loss within 0.02 of 0 on 14/14.                       Worked: 0.000 True; 0.03 False.
    pred_b_near_necessary  all-band loss >= 0.30 on >= 5 of the 7 NEAR sets.                Worked: 0.44 True; 0.21 False.
    pred_c_controls_spared all-band loss <= 0.15 on >= 4 of the 6 CONTROL sets.             Worked: 0.09 True; 0.35 False.
    pred_d_band_localized  the largest single-band loss >= 0.5 x the all-band loss on >= 4 of 7 NEAR sets.
                                                                                            Worked: 0.30 vs 0.44 True; 0.15 vs 0.44 False.
    pred_e_late_writers    (late + top) band losses summed > (early + mid) summed on >= 5 of 7 NEAR sets.
                                                                                            Worked: 0.35 > 0.12 True; 0.10 < 0.20 False.
    Prior: a 85%; b 55%; c 40%; d 50%; e 55%.
    Reading: b+c+d True name the band as the near-position writer of the set (Tier-5 step 1); b True with c False is
    uninterpretable (generic damage) and is reported as such; e settles the v120 reconciliation.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_near_writer_band_v122_result.json"
V121 = ROOT / "circuits/followups/unit_tier4_near_offset_pattern_v121_result.json"
BANDS = {"early": range(0, 5), "mid": range(5, 9), "late": range(9, 12), "top": range(12, 15), "all": range(0, 15)}
OFFSETS = (1, 2, 3)
NEAR_MIN, CTRL_MAX_FN, INSTR_TOL, LOSS_MIN, CTRL_MAX, LOCAL_FRAC, K_B, K_C, K_D, K_E = 0.20, 0.10, 0.02, 0.30, 0.15, 0.5, 5, 4, 4, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_near_writer_band_v122", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def clamped_margins(backend, batch, head_items, mlp_items):
    """v120's multi-offset exact clamp, but a row is skipped when its (rid, unit) key is absent from the cache
    (row-wise cue exclusion)."""
    torch = backend.torch
    positions = list(batch.semantic_positions)
    heads, mlps = {}, {}
    for u, off, cache in head_items:
        heads.setdefault(g.unit_layer(u), []).append((u, off, cache))
    for u, off, cache in mlp_items:
        mlps.setdefault(g.unit_layer(u), []).append((u, off, cache))
    handles = []
    for l, items in heads.items():
        def pre(_m, args, items=items):
            v = args[0].clone()
            for u, off, cache in items:
                h = int(u.rsplit(":", 1)[1])
                for i, rid in enumerate(batch.row_ids):
                    if (rid, u) in cache:
                        v[i, positions[i] + off, h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype)
            return (v,) + tuple(args[1:])
        handles.append(backend.model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre))
    for l, items in mlps.items():
        bias = backend.model.transformer.h[l].mlp.Down_bias
        def post(_m, _args, out, items=items, bias=bias):
            v = out.clone()
            for u, off, cache in items:
                for i, rid in enumerate(batch.row_ids):
                    if (rid, u) in cache:
                        v[i, positions[i] + off] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype) - bias
            return v
        handles.append(backend.model.transformer.h[l].mlp.Down.register_forward_hook(post))
    try:
        out = g.forward_units(backend, batch)
    finally:
        for h in handles:
            h.remove()
    return [float(a) - float(f) for a, f in out.tolist()]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V122_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    fn = {n: v[5] for n, v in json.loads(V121.read_text())["summary"].items()}
    near_sets = sorted(n for n, x in fn.items() if x is not None and x >= NEAR_MIN)
    ctrl_sets = sorted(n for n, x in fn.items() if x is not None and x <= CTRL_MAX_FN)
    if smoke:
        pick = os.environ.get("V122_SMOKE_SET") or list(S)[0]
        S = {pick: S[pick]}
    layers_all = list(BANDS["all"])
    heads_of = lambda ls: [f"attn:{l:02d}:head:{h:02d}" for l in ls for h in range(g.N_HEADS)]
    mlps_of = lambda ls: [f"mlp:{l:02d}" for l in ls]
    R = {}
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["donor_semantic_position"] >= max(OFFSETS)
                and r["base_semantic_position"] == r["donor_semantic_position"]]
        dropped = len(a1) - len(rows)
        if len(rows) < 4:
            R[n] = {"units": units, "rows": len(rows), "dropped_unequal": dropped, "loss": {}, "skipped": "fewer than 4 equal-length rows"}
            print(n, "skipped", len(rows), "rows", flush=True); continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        donor_axis, base_axis = prep.donor_axis, prep.base_axis
        # row-wise cue exclusion: offset k is a cue for a row when the base/donor tokens at t-k differ
        cue = {rid: {k for k in OFFSETS if r["base_ids"][r["base_semantic_position"] - k] != r["donor_ids"][r["donor_semantic_position"] - k]}
               for rid, r in zip(D.row_ids, rows)}
        hd, hb, md, mb = {}, {}, {}, {}
        for k in OFFSETS:
            keep = lambda cache, k=k: {key: v for key, v in cache.items() if k not in cue[key[0]]}
            hd[k] = keep(v120.head_cache(backend, v120.shifted(D, k), layers_all))
            hb[k] = keep(v120.head_cache(backend, v120.shifted(B, k), layers_all))
            md[k] = keep(v120.mlp_cache(backend, v120.shifted(D, k), layers_all))
            mb[k] = keep(v120.mlp_cache(backend, v120.shifted(B, k), layers_all))

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(donor_axis, base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        def run(ls, hsrc, msrc):
            hi = [(u, -k, hsrc[k]) for k in OFFSETS for u in heads_of(ls)]
            mi = [(u, -k, msrc[k]) for k in OFFSETS for u in mlps_of(ls)]
            return loss(clamped_margins(backend, D, hi, mi))

        L = {"instrument": run(layers_all, hd, md)}
        for band, ls in BANDS.items():
            L[band] = run(list(ls), hb, mb)
        R[n] = {"units": units, "rows": len(rows), "dropped_unequal": dropped, "loss": L,
                "kind": "near" if n in near_sets else "control" if n in ctrl_sets else "unclassified",
                "cue_offsets_excluded_rows": sum(1 for c in cue.values() if c), "seconds": round(time.perf_counter() - t1, 1)}
        print(n, R[n]["kind"], L, "rows", len(rows), round(time.perf_counter() - t0), "s", flush=True)

    ok = lambda n, k: n in R and R[n]["loss"].get(k) is not None
    inst = [n for n in R if ok(n, "instrument") and abs(R[n]["loss"]["instrument"]) <= INSTR_TOL]
    near = [n for n in near_sets if ok(n, "all") and R[n]["loss"]["all"] >= LOSS_MIN]
    ctrl = [n for n in ctrl_sets if ok(n, "all") and abs(R[n]["loss"]["all"]) <= CTRL_MAX]
    local, late = [], []
    for n in near_sets:
        if not ok(n, "all"):
            continue
        L = R[n]["loss"]
        single = max(L[b] for b in ("early", "mid", "late", "top") if L.get(b) is not None)
        if L["all"] > 0 and single >= LOCAL_FRAC * L["all"]:
            local.append(n)
        if (L["late"] or 0) + (L["top"] or 0) > (L["early"] or 0) + (L["mid"] or 0):
            late.append(n)
    ran = [n for n in R if "skipped" not in R[n]]
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_near_necessary": len(near) >= K_B,
        "pred_c_controls_spared": len(ctrl) >= K_C,
        "pred_d_band_localized": len(local) >= K_D,
        "pred_e_late_writers": len(late) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_near_writer_band_v122",
              "candidate_id": "corpus.unit_tier5_near_writer_band_v122",
              "bars": {"near_min_far_none": NEAR_MIN, "ctrl_max_far_none": CTRL_MAX_FN, "instr_tol": INSTR_TOL, "loss_min": LOSS_MIN,
                       "ctrl_max": CTRL_MAX, "local_frac": LOCAL_FRAC, "K": [K_B, K_C, K_D, K_E], "offsets": list(OFFSETS),
                       "bands": {b: [min(r), max(r)] for b, r in BANDS.items()}},
              "partition": {"near": near_sets, "control": ctrl_sets},
              "counts": {"instrument": len(inst), "near": len(near), "controls_spared": len(ctrl), "localized": len(local),
                         "late": len(late), "ran": len(ran), "n": len(R)},
              "localized_sets": local, "late_sets": late,
              "summary": {n: R[n]["loss"] for n in R}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
