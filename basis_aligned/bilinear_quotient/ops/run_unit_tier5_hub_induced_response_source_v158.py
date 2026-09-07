#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: what 11:03 responds to in quantifier's patched upstream state -- which units, pattern or value?

v157: under {mlp:07, mlp:08, mlp:09, mlp:10, 07:08, 08:01 exact} 11:03's output at t shifts by a 23-norm write whose
off-q part lands on the layer-11 number axis (cos 0.41) and adds +0.052 to the margin; the exact patch of 11:03 clamps
it away. Source: capture 11:03's (P, V) at layer 11 (v131 capture_with_clamp, v122 exact clamps) under base and under
each clamp set U -- six, the four MLPs, the two heads, each single MLP -- out_U = P_U[t] @ V_U[:, 11:03]; shift_U =
out_U - out_base; effect(U) = recovery({seven exact} + resid_add of the complement image of shift_U) - exact. For U = six,
split shift = P_base (V_six - V_base) [value-only] + (P_six - P_base) V_base [pattern-only] + cross.
Quantifier seven (v152 order), q = 11:03 diff-in-means from parity 0, evaluated on held-out parity 1.
Row population (ops/row_population.py): quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_instrument   out_base equals the cached base slice (max |diff| <= 0.05 x its norm) and effect(six) = v157's 0.052 within 0.02
  pred_b_mlps         effect(four MLPs) >= 0.7 x effect(six); |effect(two heads)| <= 0.02
  pred_c_value        effect(value-only) >= 0.8 x effect(six) and |effect(pattern-only)| <= 0.2 x effect(six)   (a value effect, not re-routing)
  pred_d_additive     the four single-MLP effects sum to effect(four MLPs) within 0.02
  pred_e_mlp08        the largest single-MLP effect is mlp:08 (v155: the one MLP with a negative own complement)
Reported, unregistered: shift norms per U, q-part / complement split per U, cross term.
Prior: b/c likely (v157: the shift is tiny relative to delta and the MLPs dominate the patched state); d open (the Bilinear MLPs'
writes add in the residual, but 11:03's value read is linear in its input only after rms_norm); e open.
Smoke: V158_SMOKE=<out.json> -> CPU, 4 rows. 4-row smoke: instrument 0.000, six 0.047, mlp4 0.042, heads2 0.027 (bar b's head
half may fail), value-only 0.046 / pattern-only 0.003, mlp:07 0.019 > mlp:08 0.016 -- bars kept as registered.
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_induced_response_source_v158_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
INSTR_REL, REPRO_TOL, MLP_FRAC, HEAD_MAX, VALUE_FRAC, PATTERN_FRAC, ADD_TOL = 0.05, 0.02, 0.7, 0.02, 0.8, 0.2, 0.02
V157_EFFECT = 0.052
SUBSETS = {"six": ("mlp:07", "mlp:08", "mlp:09", "mlp:10", "attn:07:head:08", "attn:08:head:01"),
           "mlp4": ("mlp:07", "mlp:08", "mlp:09", "mlp:10"), "heads2": ("attn:07:head:08", "attn:08:head:01"),
           "mlp:07": ("mlp:07",), "mlp:08": ("mlp:08",), "mlp:09": ("mlp:09",), "mlp:10": ("mlp:10",)}
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_induced_response_source_v158", "behaviours": 1, "targets": 1,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V158_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    P = {}
    for n in ("lexical_number_pp", "quantifier_number"):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        P[n] = {"p0": g.prepare(backend, cut(a1[0::2])), "p1": g.prepare(backend, cut(a1[1::2]))}
    W = backend.model.transformer.h[LAYER].attn.c_proj.weight
    Wh = W[:, HUB_H * g.HEAD_DIM:(HUB_H + 1) * g.HEAD_DIM]
    n, units, _ = SETS["quant_seven"]
    units = list(units)
    hub_q = g.block_diff_in_means(backend, P[n]["p0"], [HUB])
    qv = hub_q[(LAYER, "heads")][:, 0]
    prep = P[n]["p1"]
    batch = prep.base_batch
    positions = list(batch.semantic_positions)
    rows = len(batch.row_ids)
    def clamp_items(U):
        heads = [(u, 0, prep.donor_cache) for u in U if u.startswith("attn")]
        mlps = [(u, 0, prep.donor_cache) for u in U if u.startswith("mlp")]
        return heads, mlps
    def out_of(Pm, Vm):
        return torch.stack([Pm[i, HUB_H, positions[i], :] @ Vm[i, :, HUB_H, :] for i in range(rows)]).float()
    with torch.no_grad():
        PB, VB = v131.capture_with_clamp(backend, batch, [], [], LAYER)
        out_base = out_of(PB, VB).to(backend.device)
        base_cached = torch.stack([torch.as_tensor(prep.base_cache[(rid, HUB)]) for rid in batch.row_ids]).float().to(backend.device)
        instr_rel = float((out_base - base_cached).abs().max() / base_cached.norm(dim=1).mean())
        shifts, PV = {}, {}
        for name, U in SUBSETS.items():
            hi, mi = clamp_items(U)
            Pu, Vu = v131.capture_with_clamp(backend, batch, hi, mi, LAYER)
            PV[name] = (Pu, Vu)
            shifts[name] = out_of(Pu, Vu).to(backend.device) - out_base
        P6, V6 = PV["six"]
        shifts["value_only"] = out_of(PB, V6 - VB).to(backend.device)
        shifts["pattern_only"] = out_of(P6 - PB, VB).to(backend.device)
        shifts["cross"] = shifts["six"] - shifts["value_only"] - shifts["pattern_only"]
        qpart = lambda t: (t @ qv)[:, None] * qv[None, :]
        img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
    def rec(us, q=None, add=None):
        out = g.forward_units(backend, batch, units=us, donor_cache=prep.donor_cache, base_cache=prep.base_cache, q=q,
                              resid_add=None if add is None else {LAYER: add})
        return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
    exact = rec(units)
    R = {"exact": exact, "hub_rank1": rec(units, hub_q), "instr_rel": round(instr_rel, 4), "rows": rows, "arms": {}}
    for name, sh in shifts.items():
        comp = sh - qpart(sh)
        R["arms"][name] = {"effect": round(rec(units, None, img(comp)) - exact, 3), "shift_norm": round(float(sh.norm(dim=1).mean()), 2),
                           "q_part_norm": round(float(qpart(sh).norm(dim=1).mean()), 2), "comp_norm": round(float(comp.norm(dim=1).mean()), 2)}
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    E = {k: v["effect"] for k, v in R["arms"].items()}
    pred_a = instr_rel <= INSTR_REL and abs(E["six"] - V157_EFFECT) <= REPRO_TOL
    pred_b = E["mlp4"] >= MLP_FRAC * E["six"] and abs(E["heads2"]) <= HEAD_MAX
    pred_c = E["value_only"] >= VALUE_FRAC * E["six"] and abs(E["pattern_only"]) <= PATTERN_FRAC * E["six"]
    singles = {k: E[k] for k in ("mlp:07", "mlp:08", "mlp:09", "mlp:10")}
    pred_d = abs(sum(singles.values()) - E["mlp4"]) <= ADD_TOL
    pred_e = max(singles, key=singles.get) == "mlp:08"
    predictions = {"pred_a_instrument": pred_a, "pred_b_mlps": pred_b, "pred_c_value": pred_c, "pred_d_additive": pred_d, "pred_e_mlp08": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_induced_response_source_v158", "candidate_id": "corpus.unit_tier5_hub_induced_response_source_v158",
              "bars": {"instr_rel": INSTR_REL, "repro_tol": REPRO_TOL, "mlp_frac": MLP_FRAC, "head_max": HEAD_MAX, "value_frac": VALUE_FRAC, "pattern_frac": PATTERN_FRAC, "add_tol": ADD_TOL, "v157_effect": V157_EFFECT},
              "units": units, "subsets": {k: list(v) for k, v in SUBSETS.items()}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
