#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms fixed; readers from v131; random control seeded; rows restricted by a rule (one cue token).
"""Tier-5: the CUE key -- is what 11:03 / 07:08 read there also written twice (early band, then mid-MLP relay)?

v131-v133 closed the near-key chain: early carriers write the number at t-1..t-3, mlp:05-10 re-write it on the
same axis, the reader takes it by value. The reader takes the same axis from the CUE key too (v130 cos 0.7-0.8).
At the cue position the token itself differs, and bilin18 re-mixes the raw embedding x0 into every layer, so
some of the cue-column value can never be clamped away by layer writes (v124: the raw-embedding share is
0.14-0.29 on four sets). This rung repeats v132/v133 at the cue position, rows with exactly one cue token:
  reader cue-column delta  sum_{c in cue} P_D v_D - P_B v_B   (128-d), projected on its live value
  arms  none / all_below (every head+MLP of layers < reader at the cue position) / early (0-4) / mid (5..reader-1)
        / random4 (seeded early quartet)
  plus the mid MLPs' (donor - base) output at the cue position under the early clamp (relay dependence)

Registered before the run:
  pred_a_layers_write   all_below removes >= 0.7 of the reader's cue-column delta on 3/3 (the rest is x0)
  pred_b_early          early removes >= 0.5 on 3/3
  pred_c_mid            mid removes >= 0.5 on 3/3
  pred_d_relay          early + mid - all_below >= 0.3 on 3/3 (the non-additivity that marks a relay)
  pred_e_control        random4 removes <= 0.15 on 3/3
Prior: a, b, e likely; c, d the question -- at the cue the mid MLPs may add to an x0-carried token identity
rather than relay an early copy, in which case c fails and the cue chain differs from the near chain.
Smoke: V134_SMOKE=<out.json>, V134_SMOKE_SET=<set> -> CPU, 4 rows.
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
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_value_relay_v134_result.json"
FAR_GAP = 3
READERS = dict(v131.READERS)
ALL_MIN, EARLY_MIN, MID_MIN, RELAY_MIN, CTRL_MAX = 0.7, 0.5, 0.5, 0.3, 0.15
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_value_relay_v134", "behaviours": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V134_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = dict(READERS)
    if smoke:
        pick = os.environ.get("V134_SMOKE_SET") or "lexical_number_pp"
        sets = {pick: READERS[pick]}
    R = {}
    for n, reader in sets.items():
        t1 = time.perf_counter()
        layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
        below = list(range(0, layer))
        mid = [l for l in below if l >= 5]
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows, geo = [], []
        for r in a1:
            if len(r["base_ids"]) != len(r["donor_ids"]) or r["base_semantic_position"] != r["donor_semantic_position"]:
                continue
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            if len(diff) != 1:
                continue
            rows.append(r); geo.append({"t": t, "cue": diff[0]})
        if len(rows) < 4:
            R[n] = {"skipped": f"{len(rows)} one-cue rows"}; print(n, "skipped", flush=True); continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        cuepos = tuple(ge["cue"] for ge in geo)
        Dc, Bc = dataclasses.replace(D, semantic_positions=cuepos), dataclasses.replace(B, semantic_positions=cuepos)
        hb, mb = v120.head_cache(backend, Bc, below), v120.mlp_cache(backend, Bc, below)
        early_heads = v132.units_of(range(0, 5), mlps=False)
        rand4 = random.Random(0).sample(early_heads, 4)

        def items(units):
            return ([(u, 0, hb) for u in units if u.startswith("attn")], [(u, 0, mb) for u in units if u.startswith("mlp")])

        mids = {}
        handles = []
        for l in mid:
            def post(_m, _args, out, l=l):
                for i, ge in enumerate(geo):
                    mids[(l, i)] = out[i, ge["cue"]].detach().clone()
            handles.append(backend.model.transformer.h[l].mlp.Down.register_forward_hook(post))

        def run(units, batch):
            hi, mi = items(units)
            mids.clear()
            P, V = v131.capture_with_clamp(backend, batch, hi, mi, layer)
            return P, V, dict(mids)

        try:
            PB, VB, MB = run([], Bc)

            def delta(P, V):
                return [P[i, h, ge["t"], ge["cue"]] * V[i, ge["cue"], h, :] - PB[i, h, ge["t"], ge["cue"]] * VB[i, ge["cue"], h, :] for i, ge in enumerate(geo)]

            arms = {"none": [], "all_below": v132.units_of(below), "early": v132.units_of(range(0, 5)), "mid": v132.units_of(mid), "random4": rand4}
            live, live_m, A = None, None, {}
            for arm, units in arms.items():
                P, V, M = run(units, Dc)
                d = delta(P, V)
                md = {k: M[k] - MB[k] for k in M}
                if arm == "none":
                    live, live_m = d, md
                fr = [float(c @ r / (r @ r)) for c, r in zip(d, live) if float(r @ r) > 1e-8]
                mk = []
                for i in range(len(rows)):
                    cur = torch.cat([md[(l, i)] for l in mid]); ref = torch.cat([live_m[(l, i)] for l in mid])
                    if float(ref @ ref) > 1e-8:
                        mk.append(float(cur @ ref / (ref @ ref)))
                A[arm] = {"removed_cue": round(1 - sum(fr) / len(fr), 3) if fr else None, "mid_mlp_keep_at_cue": round(sum(mk) / len(mk), 3) if mk else None}
        finally:
            for hd in handles:
                hd.remove()
        R[n] = {"reader": reader, "rows": len(rows), "mid_layers": mid, "random4": rand4, "arms": A, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, {a: (x["removed_cue"], x["mid_mlp_keep_at_cue"]) for a, x in A.items()}, round(time.perf_counter() - t0), "s", flush=True)

    def rem(n, arm):
        x = R[n]["arms"][arm]["removed_cue"]
        return -9 if x is None else x

    ran = [n for n in R if "skipped" not in R[n]]
    a = [n for n in ran if rem(n, "all_below") >= ALL_MIN]
    b = [n for n in ran if rem(n, "early") >= EARLY_MIN]
    c = [n for n in ran if rem(n, "mid") >= MID_MIN]
    d = [n for n in ran if rem(n, "early") + rem(n, "mid") - rem(n, "all_below") >= RELAY_MIN]
    e = [n for n in ran if 0 <= rem(n, "random4") <= CTRL_MAX or -CTRL_MAX <= rem(n, "random4") < 0]
    full = len(READERS)
    predictions = {"pred_a_layers_write": len(a) == full, "pred_b_early": len(b) == full, "pred_c_mid": len(c) == full,
                   "pred_d_relay": len(d) == full, "pred_e_control": len(e) == full}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_value_relay_v134", "candidate_id": "corpus.unit_tier5_cue_value_relay_v134",
              "bars": {"all_min": ALL_MIN, "early_min": EARLY_MIN, "mid_min": MID_MIN, "relay_min": RELAY_MIN, "ctrl_max": CTRL_MAX},
              "counts": {"ran": len(ran), "layers_write": len(a), "early": len(b), "mid": len(c), "relay": len(d), "control": len(e)},
              "summary": {n: {arm: R[n]["arms"][arm]["removed_cue"] for arm in R[n]["arms"]} for n in ran},
              "mid_mlp_keep_under_early": {n: R[n]["arms"]["early"]["mid_mlp_keep_at_cue"] for n in ran},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"], "mid_mlp_keep_under_early": result["mid_mlp_keep_under_early"]}, indent=2))


if __name__ == "__main__":
    main()
