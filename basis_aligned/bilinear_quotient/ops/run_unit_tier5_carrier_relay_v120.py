#!/usr/bin/env python3
# BQGATE: five frozen predictions; relay and control sets named from the v117 receipt before the run; bars fixed.
"""v120: the CARRIER RELAY of the three far-cue sets -- first Tier-5 recursion step, tested on the donor side.
v117 left three sets at the head-write level: narrative_tense (struct 0.31), possessive_argument (0.64),
possessive_verbfinal (0.64). Their v115 expansions show WHY: the early heads (attn 4/5) read the cue offsets, but the
late heads (attn 9:06, 9:07, 12:04, 15:01; narrative 9:01, 9:07, 11:03) read offsets 1-3 -- the positions just before
the read position -- and the dominant writers there are mlp_6..mlp_8 (possessives) / mlp_7..mlp_8 (narrative). So the
donor's cue reaches those heads through a CARRIER: attention at the carrier positions (layers <= 8) copies the cue
there, mlp_6-8 transform it, the late heads read it at offsets 1-3 (possessives 1-2, narrative 2-4; the window
{t-1, t-2, t-3} is fixed here for all six sets). Executable test, donor batch, rows with equal base/donor length (so
the carrier positions are the same token slots on both sides):
    instrument   donor run, carrier heads (layers 0-8, positions t-1..t-3) clamped to the DONOR's own values (= donor margin)
    attn_clamp   donor run, carrier heads (layers 0-8, t-1..t-3) clamped to the BASE run's values at the same positions
    mlp_clamp    donor run, mlp_6..mlp_8 outputs at t-1..t-3 clamped to the BASE run's values
    restore      attn_clamp AND the set's own heads at t clamped to the donor's native values
(The CPU code-path smoke, 4 narrative rows with a t-1-only window, gave attn_clamp loss 0.02; the window was widened to
1-3 BEFORE enqueue to match the v117 offsets and is registered as such -- one design change, disclosed.)
    loss := (donor - patched) / (donor - base) on the donor's answer axis (1 = the whole donor margin is lost)
Control sets (relay NOT expected: their heads read the cue directly, v117 struct >= 0.80, cue far from t-1):
additive_scope, degree_frame, interrogative_licensing. A control clamp here removes the same positions' processing, so
the control can fail (lesson 4) -- it is the same intervention on sets whose expansion says the carrier is unused.
REGISTERED BEFORE THE RUN (ODD rows; losses as fractions of the donor margin)
    pred_a_instrument         instrument loss within 0.02 of 0 on 6/6 sets.                         Worked: 0.003 True; 0.03 False.
    pred_b_relay_necessary    attn_clamp loss >= 0.30 on >= 2 of the 3 relay sets.                  Worked: 0.45 True; 0.18 False.
    pred_c_controls_spared    attn_clamp loss <= 0.15 on >= 2 of the 3 control sets.                Worked: 0.08 True; 0.22 False.
    pred_d_mlp_carrier        mlp_clamp loss >= 0.30 on >= 2 of the 3 relay sets.                   Worked: 0.35 True; 0.12 False.
    pred_e_through_the_set    restore brings the loss back to <= 0.20 on 3/3 relay sets (the carrier acts only
                              through the set's heads at t).                                        Worked: 0.10 True; 0.31 False.
    Prior: a 85%; b 55%; c 60%; d 45%; e 55%.
    Reading: b+e True name the relay (carrier position -> mlp_6-8 -> late heads) as the sets' Tier-5 first step; b True
    with e False says other heads at t also read the carrier (report the residual loss; no new arm). c False says the
    clamp damages the donor prompt generically and b is uninterpretable -- reported as such.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_carrier_relay_v120_result.json"
RELAY = ("narrative_tense", "possessive_argument", "possessive_verbfinal")
CONTROL = ("additive_scope", "degree_frame", "interrogative_licensing")
CARRIER_LAYERS, CARRIER_MLPS, CARRIER_OFFSETS = range(0, 9), (6, 7, 8), (1, 2, 3)
INSTR_TOL, LOSS_MIN, CTRL_MAX, RESTORE_MAX, K_B, K_C, K_D = 0.02, 0.30, 0.15, 0.20, 2, 2, 2
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_carrier_relay_v120", "behaviours": 6,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def shifted(batch, k=1):
    return dataclasses.replace(batch, semantic_positions=tuple(p - k for p in batch.semantic_positions))


def head_cache(backend, batch, layers):
    """Every head's c_proj-input slice at batch.semantic_positions in a native run of `batch`."""
    got, handles = {}, []
    positions = list(batch.semantic_positions)
    for l in layers:
        block = backend.model.transformer.h[l]
        def pre(_m, args, l=l):
            v = args[0]
            for i, rid in enumerate(batch.row_ids):
                for h in range(g.N_HEADS):
                    got[(rid, f"attn:{l:02d}:head:{h:02d}")] = v[i, positions[i], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM].detach().cpu().clone()
        handles.append(block.attn.c_proj.register_forward_pre_hook(pre))
    try:
        g.forward_units(backend, batch)
    finally:
        for h in handles:
            h.remove()
    return got


def mlp_cache(backend, batch, layers):
    hid = {}
    g.forward_units(backend, batch, capture_hidden=hid)
    out = {}
    with backend.torch.no_grad():
        for l in layers:
            mlp = backend.model.transformer.h[l].mlp
            for rid in batch.row_ids:
                h = backend.torch.as_tensor(hid[(rid, g.hidden_key(l))]).to(backend.device).float()
                out[(rid, f"mlp:{l:02d}")] = (mlp.Down(h) + mlp.Down_bias).detach().cpu()
    return out


def margins(backend, batch, units, cache):
    out = g.forward_units(backend, batch, units=units, donor_cache=cache)
    return [float(a) - float(f) for a, f in out.tolist()]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V120_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    order = [n for n in RELAY + CONTROL if n in S]
    if smoke:
        order = order[:1]
    carrier_heads = [f"attn:{l:02d}:head:{h:02d}" for l in CARRIER_LAYERS for h in range(g.N_HEADS)]
    carrier_mlps = [f"mlp:{l:02d}" for l in CARRIER_MLPS]
    R = {}
    for n in order:
        units = S[n]
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["donor_semantic_position"] >= max(CARRIER_OFFSETS)
                and r["base_semantic_position"] == r["donor_semantic_position"]]
        dropped = len(a1) - len(rows)
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        donor_axis, base_axis = prep.donor_axis, prep.base_axis            # both on the donor's answer axis
        hd, hb, mb = {}, {}, {}
        for k in CARRIER_OFFSETS:
            hd[k] = head_cache(backend, shifted(D, k), CARRIER_LAYERS)
            hb[k] = head_cache(backend, shifted(B, k), CARRIER_LAYERS)
            mb[k] = mlp_cache(backend, shifted(B, k), CARRIER_MLPS)
        own = {k: v for k, v in prep.donor_cache.items() if k[1] in units}

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(donor_axis, base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        heads_from = lambda src: [(u, -k, src[k]) for k in CARRIER_OFFSETS for u in carrier_heads]
        mlps_from = lambda src: [(u, -k, src[k]) for k in CARRIER_OFFSETS for u in carrier_mlps]
        inst = loss(clamped_margins(backend, D, heads_from(hd), []))
        attn = loss(clamped_margins(backend, D, heads_from(hb), []))
        mlpc = loss(clamped_margins(backend, D, [], mlps_from(mb)))
        restore = loss(clamped_margins(backend, D, heads_from(hb) + [(u, 0, own) for u in units], []))
        R[n] = {"units": units, "kind": "relay" if n in RELAY else "control", "rows": len(rows), "dropped_unequal": dropped,
                "loss": {"instrument": inst, "attn_clamp": attn, "mlp_clamp": mlpc, "restore": restore},
                "donor_minus_base_median": round(float(sorted(d - b for d, b in zip(donor_axis, base_axis))[len(rows) // 2]), 3),
                "seconds": round(time.perf_counter() - t1, 1)}
        print(n, R[n]["kind"], R[n]["loss"], "rows", len(rows), round(time.perf_counter() - t0), "s", flush=True)

    ok = lambda x: x is not None
    inst = [n for n, v in R.items() if ok(v["loss"]["instrument"]) and abs(v["loss"]["instrument"]) <= INSTR_TOL]
    relay = [n for n in RELAY if n in R and ok(R[n]["loss"]["attn_clamp"]) and R[n]["loss"]["attn_clamp"] >= LOSS_MIN]
    ctrl = [n for n in CONTROL if n in R and ok(R[n]["loss"]["attn_clamp"]) and abs(R[n]["loss"]["attn_clamp"]) <= CTRL_MAX]
    mlpr = [n for n in RELAY if n in R and ok(R[n]["loss"]["mlp_clamp"]) and R[n]["loss"]["mlp_clamp"] >= LOSS_MIN]
    rest = [n for n in RELAY if n in R and ok(R[n]["loss"]["restore"]) and abs(R[n]["loss"]["restore"]) <= RESTORE_MAX]
    predictions = {
        "pred_a_instrument": len(inst) == len(R),
        "pred_b_relay_necessary": len(relay) >= K_B,
        "pred_c_controls_spared": len(ctrl) >= K_C,
        "pred_d_mlp_carrier": len(mlpr) >= K_D,
        "pred_e_through_the_set": len(rest) == len([n for n in RELAY if n in R]),
    }
    result = {"predictions": predictions, "schema": "unit_tier5_carrier_relay_v120",
              "candidate_id": "corpus.unit_tier5_carrier_relay_v120",
              "bars": {"instr_tol": INSTR_TOL, "loss_min": LOSS_MIN, "ctrl_max": CTRL_MAX, "restore_max": RESTORE_MAX,
                       "K": [K_B, K_C, K_D], "carrier_layers": [0, 8], "carrier_mlps": list(CARRIER_MLPS), "carrier_offsets": list(CARRIER_OFFSETS)},
              "relay_sets": list(RELAY), "control_sets": list(CONTROL),
              "counts": {"instrument": len(inst), "relay": len(relay), "controls_spared": len(ctrl), "mlp": len(mlpr), "restored": len(rest)},
              "summary": {n: v["loss"] for n, v in R.items()}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


def clamped_margins(backend, batch, head_items, mlp_items):
    """One forward with exact clamps at arbitrary offsets from the semantic position. head_items: (unit, offset, cache)
    replacing the head's c_proj-input slice; mlp_items: (unit, offset, cache) replacing the MLP output vector. Same
    operations `forward_units` performs at offset 0 (c_proj pre-hook; MLP output replacement via a Down forward hook)."""
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
                    v[i, positions[i] + off, h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] = torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype)
            return (v,) + tuple(args[1:])
        handles.append(backend.model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre))
    for l, items in mlps.items():
        bias = backend.model.transformer.h[l].mlp.Down_bias
        def post(_m, _args, out, items=items, bias=bias):
            v = out.clone()
            for u, off, cache in items:
                for i, rid in enumerate(batch.row_ids):
                    v[i, positions[i] + off] = (torch.as_tensor(cache[(rid, u)]).to(v.device, v.dtype) - bias)
            return v
        handles.append(backend.model.transformer.h[l].mlp.Down.register_forward_hook(post))
    try:
        out = g.forward_units(backend, batch)
    finally:
        for h in handles:
            h.remove()
    return [float(a) - float(f) for a, f in out.tolist()]


if __name__ == "__main__":
    main()
