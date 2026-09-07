#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms fixed; readers and carriers from the v131 / v123 receipts; random control seeded.
"""Tier-5: is the mid-band write at the near positions a RELAY of the early write?

v132: at the near positions (t-1..t-3) the early band (0-4) and the mid band (5-10) each remove ~0.8 of the
number value 11:03 / 07:08 read, and the two are not additive. The relay reading: the mid MLPs' donor-minus-base
output at those positions exists only because the early heads wrote the number there first. Two tests:
  (1) dependence: the mid MLPs' (donor - base) output at the cue-excluded near positions, projected on its live
      value, under the early-band clamp (early_all), the four carriers alone, and a seeded random early quartet
  (2) direction: in the reader's 128-d value space, the component of its near-column contribution removed by the
      early clamp vs the component removed by the mid clamp -- a relay re-writes the SAME direction

Registered before the run:
  pred_a_control       the seeded random quartet keeps >= 0.8 of the pooled mid-MLP near delta on 3/3
  pred_b_early_removes early_all removes >= 0.7 of the pooled mid-MLP near delta on 3/3
  pred_c_carriers      the four carriers alone remove >= 0.5 on 3/3
  pred_d_uniform       every mid MLP whose near delta norm is >= 0.2 x the largest one loses >= 0.5 under early_all,
                       on >= 80% of such MLPs pooled over the 3 sets
  pred_e_same_axis     cos(removed_by_early, removed_by_mid) in the reader's value space >= 0.8 on 3/3
Prior: b, e are the relay claim; a, c should hold; d unsure (a chain within the mid band would make the
deeper MLPs depend on the early write only through the shallower ones and still lose it, so d is weak).
Smoke: V133_SMOKE=<out.json>, V133_SMOKE_SET=<set> -> CPU, 4 rows.
"""
from __future__ import annotations

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
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_relay_dependence_v133_result.json"
V123 = ROOT / "circuits/followups/unit_tier5_near_carrier_heads_v123_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
CTRL_KEEP, EARLY_MIN, CARRIER_MIN, UNIT_MIN, UNIT_NORM, UNIT_FRAC, COS_MIN = 0.8, 0.7, 0.5, 0.5, 0.2, 0.8, 0.8
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_relay_dependence_v133", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def mlp_out_with_clamp(backend, batch, head_items, mlp_items, layers, offsets_by_row, reader_layer):
    """v131's clamp + capture, additionally recording each `layers` MLP's Down output at the row's near offsets."""
    torch = backend.torch
    positions = list(batch.semantic_positions)
    got, handles = {}, []
    for l in layers:
        def post(_m, _args, out, l=l):
            for i in range(len(positions)):
                for k in offsets_by_row[i]:
                    got[(l, i, k)] = out[i, positions[i] - k].detach().clone()
        handles.append(backend.model.transformer.h[l].mlp.Down.register_forward_hook(post))
    try:
        P, V = v131.capture_with_clamp(backend, batch, head_items, mlp_items, reader_layer)
    finally:
        for h in handles:
            h.remove()
    return P, V, got


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V133_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    v123r = json.load(open(V123))["summary"]
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V133_SMOKE_SET") or "lexical_number_pp"
        sets = {pick: READERS[pick]}
    early_heads = v132.units_of(range(0, 5), mlps=False)
    R = {}
    for n, reader in sets.items():
        t1 = time.perf_counter()
        layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
        mid = [l for l in range(5, layer)]
        carriers = v123r[n][2]
        rand4 = random.Random(0).sample([u for u in early_heads if u not in carriers], 4)
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows = v123.rows_of(m, 1, smoke)
        side = v123.Side(backend, rows, list(range(0, layer)))
        D, B = side.D, side.B
        geo, offs = [], []
        for r in rows:
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            cue = diff[-1] if diff else 0
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            near = [p for p in range(max(cue + 1, t - FAR_GAP), t) if eq(p)]
            geo.append({"t": t, "near": near})
            offs.append([t - p for p in near])
        PB, VB, MB = mlp_out_with_clamp(backend, B, [], [], mid, offs, layer)

        def reader_delta(P, V):
            out = []
            for i, ge in enumerate(geo):
                cs = torch.tensor(ge["near"])
                out.append(P[i, h, ge["t"], cs] @ V[i, cs, h, :] - PB[i, h, ge["t"], cs] @ VB[i, cs, h, :] if len(cs) else None)
            return out

        def items(units):
            return ([(u, -k, side.hb[k]) for k in v123.OFFSETS for u in units if u.startswith("attn")],
                    [(u, -k, side.mb[k]) for k in v123.OFFSETS for u in units if u.startswith("mlp")])

        arms = {"none": [], "early_all": v132.units_of(range(0, 5)), "carriers": carriers, "random4": rand4, "mid_all": v132.units_of(mid)}
        live_reader, live_mlp, A = None, None, {}
        for arm, units in arms.items():
            hi, mi = items(units)
            P, V, M = mlp_out_with_clamp(backend, D, hi, mi, mid, offs, layer)
            rd = reader_delta(P, V)
            md = {key: M[key] - MB[key] for key in M}
            if arm == "none":
                live_reader, live_mlp = rd, md
            # pooled mid-MLP near delta: concatenate over (layer, offset) per row
            keeps, per_layer = [], {}
            for i in range(len(rows)):
                keys = [(l, i, k) for l in mid for k in offs[i]]
                if not keys:
                    continue
                cur = torch.cat([md[key] for key in keys]); ref = torch.cat([live_mlp[key] for key in keys])
                if float(ref @ ref) > 1e-8:
                    keeps.append(float(cur @ ref / (ref @ ref)))
            for l in mid:
                fr, norms = [], []
                for i in range(len(rows)):
                    keys = [(l, i, k) for k in offs[i]]
                    if not keys:
                        continue
                    cur = torch.cat([md[key] for key in keys]); ref = torch.cat([live_mlp[key] for key in keys])
                    norms.append(float(ref.norm()))
                    if float(ref @ ref) > 1e-8:
                        fr.append(float(cur @ ref / (ref @ ref)))
                per_layer[f"mlp:{l:02d}"] = {"keep": round(sum(fr) / len(fr), 3) if fr else None, "live_norm": round(sum(norms) / len(norms), 2) if norms else None}
            rk = [float(c @ r / (r @ r)) for c, r in zip(rd, live_reader) if c is not None and r is not None and float(r @ r) > 1e-8]
            removed_vec = [r - c for c, r in zip(rd, live_reader) if c is not None and r is not None]
            A[arm] = {"mlp_keep_pooled": round(sum(keeps) / len(keeps), 3) if keeps else None, "mlp_per_layer": per_layer,
                      "reader_removed_near": round(1 - sum(rk) / len(rk), 3) if rk else None,
                      "_removed": torch.stack(removed_vec).mean(0) if removed_vec else None}
        ce, cm = A["early_all"]["_removed"], A["mid_all"]["_removed"]
        cos = round(float(torch.nn.functional.cosine_similarity(ce, cm, dim=0)), 3) if ce is not None and cm is not None else None
        for arm in A:
            A[arm].pop("_removed")
        R[n] = {"reader": reader, "mid_layers": mid, "carriers": carriers, "random4": rand4, "rows": len(rows), "arms": A,
                "cos_removed_early_vs_mid": cos, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, {a: (x["mlp_keep_pooled"], x["reader_removed_near"]) for a, x in A.items()}, "cos", cos,
              {l: (x["keep"], x["live_norm"]) for l, x in A["early_all"]["mlp_per_layer"].items()}, round(time.perf_counter() - t0), "s", flush=True)

    def keep(n, arm):
        x = R[n]["arms"][arm]["mlp_keep_pooled"]
        return 9 if x is None else x

    ran = list(R)
    a = [n for n in ran if keep(n, "random4") >= CTRL_KEEP]
    b = [n for n in ran if 1 - keep(n, "early_all") >= EARLY_MIN]
    c = [n for n in ran if 1 - keep(n, "carriers") >= CARRIER_MIN]
    units, lost = [], []
    for n in ran:
        pl = R[n]["arms"]["early_all"]["mlp_per_layer"]
        top = max((x["live_norm"] or 0) for x in pl.values())
        for l, x in pl.items():
            if (x["live_norm"] or 0) >= UNIT_NORM * top and x["keep"] is not None:
                units.append((n, l))
                if 1 - x["keep"] >= UNIT_MIN:
                    lost.append((n, l))
    e = [n for n in ran if (R[n]["cos_removed_early_vs_mid"] or -9) >= COS_MIN]
    full = len(READERS)
    predictions = {"pred_a_control": len(a) == full, "pred_b_early_removes": len(b) == full, "pred_c_carriers": len(c) == full,
                   "pred_d_uniform": bool(units) and len(lost) >= UNIT_FRAC * len(units), "pred_e_same_axis": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_relay_dependence_v133", "candidate_id": "corpus.unit_tier5_relay_dependence_v133",
              "bars": {"ctrl_keep": CTRL_KEEP, "early_min": EARLY_MIN, "carrier_min": CARRIER_MIN, "unit_min": UNIT_MIN, "unit_norm": UNIT_NORM, "unit_frac": UNIT_FRAC, "cos_min": COS_MIN},
              "counts": {"ran": len(ran), "control": len(a), "early": len(b), "carriers": len(c), "units": len(units), "units_lost": len(lost), "same_axis": len(e)},
              "summary": {n: {"mlp_keep": {arm: R[n]["arms"][arm]["mlp_keep_pooled"] for arm in R[n]["arms"]}, "cos": R[n]["cos_removed_early_vs_mid"]} for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
