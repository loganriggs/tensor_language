#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: 11:03's self-position read is a functional part of quantifier's BASE circuit, not only a patch artefact.

v158/v159: under upstream patches 11:03's induced response is a value read of position t. Before writing this runner I
printed the base-forward magnitudes on 4 held-out rows: quantifier's 11:03 self-term P[t,t] V_h[t] has norm 49 of the
head's 95 output norm, P[t,t] is NEGATIVE on every base and donor row (-0.09 .. -0.15; squared attention is unnormalised
and can be negative), and resid_add of -image(self-term) on the UNPATCHED base forward moves the margin 0.073 toward the
donor (removing the whole head output: 0.115). Lexical: self-term 5.6 of 151, P[t,t] +0.006 .. +0.02, removal 0.000.
Instrument: (P, V) via v131.capture_with_clamp with no clamps; self-term image via W_O's 11:03 slice; effects via resid_add
on the base forward (units=()) -- the same path v157-v159 validated (closure to exact 0.000). Sets: quantifier seven and
lexical seven held-out p1 (16 rows each).
Row population (ops/row_population.py): lexical A1 p1 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 16}';
quantifier A1 p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_removal      quantifier: resid_add of -image(self-term) on the base forward gives recovery within 0.03-0.15 toward the donor
  pred_b_linear       quantifier: doubling the self-term (+image) gives -(removal) within 0.03
  pred_c_share        quantifier: removal(self-term) / removal(full 11:03 output) within 0.4-0.9
  pred_d_sign         quantifier: P[t,t] < 0 on >= 75% of rows on BOTH the base and the donor side
  pred_e_lex_control  lexical: |removal(self-term)| <= 0.02 and |self-term| / |full output| within 0-0.1
Reported, unregistered: per-row P[t,t] (base, donor), norms, removal of the full output on both sets, cos of the oriented
self-term image with the post-attention layer-11 number axis (v157 recipe), row-max |P[t,:]| for scale.
Prior: a/c/d hold on the smoke rows (4 of 16), so a full pass is expected; the information is in the 16-row sign count (d)
and the share (c) -- a hub head that spends half its output on re-reading its own position with a negative weight.
Smoke: V160_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_self_read_in_base_v160_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
REM_LO, REM_HI, LIN_TOL, SHARE_LO, SHARE_HI, SIGN_FRAC, LEX_TOL, LEX_SHARE_HI = 0.03, 0.15, 0.03, 0.4, 0.9, 0.75, 0.02, 0.1
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_self_read_in_base_v160", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V160_SMOKE")
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
    attn = backend.model.transformer.h[LAYER].attn
    W = attn.c_proj.weight
    Wh = W[:, HUB_H * g.HEAD_DIM:(HUB_H + 1) * g.HEAD_DIM]
    R = {}
    for sname, (n, units, _) in SETS.items():
        if sname == "shared_four":
            continue
        prep = P[n]["p1"]
        batch = prep.base_batch
        positions = list(batch.semantic_positions)
        rows = len(batch.row_ids)
        img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
        with torch.no_grad():
            PB, VB = v131.capture_with_clamp(backend, batch, [], [], LAYER)
            PD, VD = v131.capture_with_clamp(backend, prep.donor_batch, [], [], LAYER)
            ptt = torch.stack([PB[i, HUB_H, positions[i], positions[i]] for i in range(rows)]).float()
            pttD = torch.stack([PD[i, HUB_H, positions[i], positions[i]] for i in range(rows)]).float()
            selfterm = torch.stack([PB[i, HUB_H, positions[i], positions[i]] * VB[i, positions[i], HUB_H, :] for i in range(rows)]).float().to(backend.device)
            full = torch.stack([PB[i, HUB_H, positions[i], :] @ VB[i, :, HUB_H, :] for i in range(rows)]).float().to(backend.device)
            rowmax = torch.stack([PB[i, HUB_H, positions[i], :].abs().max() for i in range(rows)]).float()
            dres, bres = {}, {}
            g.forward_units(backend, prep.donor_batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=dres)
            g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=bres)
            rdelta = (torch.stack([dres[(rid, LAYER)] for rid in prep.donor_batch.row_ids]) - torch.stack([bres[(rid, LAYER)] for rid in batch.row_ids])).to(backend.device)
            s2 = g._orientation(rdelta)
            a_out = (rdelta * s2[:, None]).mean(0); a_out = a_out / a_out.norm()
            m = (img(selfterm) * s2[:, None]).mean(0)
            self_axis_cos = float(m @ a_out / m.norm())
        def rec(add=None):
            out = g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        base = rec()
        eff = {"remove_self": round(rec(-img(selfterm)) - base, 3), "double_self": round(rec(img(selfterm)) - base, 3),
               "remove_full": round(rec(-img(full)) - base, 3)}
        S = {"base": base, "effect": eff, "p_tt_base": [round(float(x), 4) for x in ptt], "p_tt_donor": [round(float(x), 4) for x in pttD],
             "neg_frac_base": round(float((ptt < 0).float().mean()), 3), "neg_frac_donor": round(float((pttD < 0).float().mean()), 3),
             "self_norm": round(float(selfterm.norm(dim=1).mean()), 2), "full_norm": round(float(full.norm(dim=1).mean()), 2),
             "rowmax_abs_p": round(float(rowmax.mean()), 4), "self_axis_cos": round(self_axis_cos, 3), "rows": rows}
        S["norm_share"] = round(S["self_norm"] / S["full_norm"], 3)
        S["share_of_full_removal"] = round(eff["remove_self"] / eff["remove_full"], 3) if abs(eff["remove_full"]) > 1e-6 else None
        R[sname] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    Qs, Lx = R["quant_seven"], R["lex_seven"]
    pred_a = REM_LO <= Qs["effect"]["remove_self"] <= REM_HI
    pred_b = abs(Qs["effect"]["double_self"] + Qs["effect"]["remove_self"]) <= LIN_TOL
    pred_c = Qs["share_of_full_removal"] is not None and SHARE_LO <= Qs["share_of_full_removal"] <= SHARE_HI
    pred_d = Qs["neg_frac_base"] >= SIGN_FRAC and Qs["neg_frac_donor"] >= SIGN_FRAC
    pred_e = abs(Lx["effect"]["remove_self"]) <= LEX_TOL and 0.0 <= Lx["norm_share"] <= LEX_SHARE_HI
    predictions = {"pred_a_removal": pred_a, "pred_b_linear": pred_b, "pred_c_share": pred_c, "pred_d_sign": pred_d, "pred_e_lex_control": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_self_read_in_base_v160", "candidate_id": "corpus.unit_tier5_hub_self_read_in_base_v160",
              "bars": {"removal_band": [REM_LO, REM_HI], "linear_tol": LIN_TOL, "share_band": [SHARE_LO, SHARE_HI], "sign_frac": SIGN_FRAC, "lex_tol": LEX_TOL, "lex_share_max": LEX_SHARE_HI},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
