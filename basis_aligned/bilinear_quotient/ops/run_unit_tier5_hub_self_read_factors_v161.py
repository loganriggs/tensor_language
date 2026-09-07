#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: which squared-attention factor makes 11:03's self-weight negative, and where its self-term acts in the base forward.

v160: quantifier's 11:03 has P[t,t] < 0 on 32/32 rows and its self-term is 0.61 of the head's base effect. Squared attention:
P = (q.k/D)(q2.k2/D) with QK-norm and rotary; at (t,t) the rotary cancels, so both factors follow from the rms-normed input
x_t alone (captured at c_q's input). Printed on 4 rows before registering: q2.k2/D = +0.46..+0.49 (quantifier) and
+0.42..+0.54 (lexical) -- a positive near-constant gate; q.k/D = -0.21..-0.30 (quantifier) vs +0.01..+0.04 (lexical) -- the
sign and the magnitude; the product matches P[t,t] to 3 decimals. Splitting the self-term's W_O image (base rows) into its
post-attention layer-11 number-axis component and the off-axis remainder (v157 axis recipe): removal effects 0.026 (axis,
35% of norm) and 0.045 (off-axis) of 0.073 -- the remainder carries more. Sets: quantifier seven and lexical seven held-out
p1 (16 rows each; the same 4 smoke rows are 4 of the 16).
Row population (ops/row_population.py): lexical A1 p1 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 16}';
quantifier A1 p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_instrument   (q.k/D)(q2.k2/D) at t reproduces P[t,t] within 0.005 on every row of both sets, and removal(axis) + removal(off) = removal(full) within 0.02
  pred_b_sign_factor  quantifier: q.k/D < 0 on >= 75% of rows and q2.k2/D > 0 on >= 75% of rows (base side)
  pred_c_gate         q2.k2/D mean is within 0.7-1.4 x between quantifier and lexical (the gate does not distinguish the sets)
  pred_d_off_axis     quantifier: removal(off-axis) / removal(full) within 0.4-0.8 and removal(axis) / removal(full) within 0.2-0.5
  pred_e_lex_control  lexical: |q.k/D| mean <= 0.1 and |removal(axis)|, |removal(off)| <= 0.02
Reported, unregistered: per-row factors (base and donor sides), corr(q.k/D, P[t,t]) across rows, |axis| / |image| norm share,
the donor-side sign counts, and the rank of the self-key among the row's |P[t,:]|.
Prior: a/b/c/e should hold (smoke); d is the informative bar -- if the off-axis remainder carries the effect, the number axis
(diff-in-means at layer 11) is not the coordinate in which 11:03's self-read defends the base answer.
Smoke: V161_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_self_read_factors_v161_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
P_TOL, ADD_TOL, SIGN_FRAC, GATE_LO, GATE_HI, OFF_LO, OFF_HI, AX_LO, AX_HI, LEX_F1, LEX_TOL = 0.005, 0.02, 0.75, 0.7, 1.4, 0.4, 0.8, 0.2, 0.5, 0.1, 0.02
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_self_read_factors_v161", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V161_SMOKE")
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
    D = g.HEAD_DIM
    for sname, (n, units, _) in SETS.items():
        if sname == "shared_four":
            continue
        prep = P[n]["p1"]
        batch = prep.base_batch
        positions = list(batch.semantic_positions)
        rows = len(batch.row_ids)
        img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
        def factors(bt):
            got = {}
            def pre(_m, args):
                got["x"] = torch.stack([args[0][i, positions[i]] for i in range(rows)]).detach().clone()
            hnd = attn.c_q.register_forward_pre_hook(pre)
            try:
                Pm, Vm = v131.capture_with_clamp(backend, bt, [], [], LAYER)
            finally:
                hnd.remove()
            x = got["x"]
            hv = lambda lin: lin(x).view(rows, g.N_HEADS, D)[:, HUB_H, :].float()
            q, k, q2, k2 = [torch.nn.functional.rms_norm(hv(l), (D,)) for l in (attn.c_q, attn.c_k, attn.c_q2, attn.c_k2)]
            f1 = (q * k).sum(1) / D
            f2 = (q2 * k2).sum(1) / D
            ptt = torch.stack([Pm[i, HUB_H, positions[i], positions[i]] for i in range(rows)]).float().to(f1.device)
            rank = torch.stack([(Pm[i, HUB_H, positions[i], :positions[i] + 1].abs() > Pm[i, HUB_H, positions[i], positions[i]].abs()).sum() for i in range(rows)]).float()
            return f1, f2, ptt, rank, Pm, Vm
        with torch.no_grad():
            f1, f2, ptt, rank, PB, VB = factors(batch)
            f1d, f2d, pttd, _, _, _ = factors(prep.donor_batch)
            selfterm = torch.stack([PB[i, HUB_H, positions[i], positions[i]] * VB[i, positions[i], HUB_H, :] for i in range(rows)]).float().to(backend.device)
            image = img(selfterm)
            dres, bres = {}, {}
            g.forward_units(backend, prep.donor_batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=dres)
            g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=bres)
            rdelta = (torch.stack([dres[(rid, LAYER)] for rid in prep.donor_batch.row_ids]) - torch.stack([bres[(rid, LAYER)] for rid in batch.row_ids])).to(backend.device)
            s2 = g._orientation(rdelta)
            a_out = (rdelta * s2[:, None]).mean(0); a_out = a_out / a_out.norm()
            ax = (image @ a_out)[:, None] * a_out[None, :]
            off = image - ax
            corr = float(torch.corrcoef(torch.stack([f1, ptt]))[0, 1]) if rows > 2 else None
        def rec(add=None):
            out = g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        base = rec()
        eff = {"remove_full": round(rec(-image) - base, 3), "remove_axis": round(rec(-ax) - base, 3), "remove_off": round(rec(-off) - base, 3)}
        S = {"base": base, "effect": eff, "f1_base": [round(float(x), 4) for x in f1], "f2_base": [round(float(x), 4) for x in f2],
             "f1_donor": [round(float(x), 4) for x in f1d], "f2_donor": [round(float(x), 4) for x in f2d],
             "p_tt_base": [round(float(x), 4) for x in ptt], "max_abs_product_err": round(float(((f1 * f2) - ptt).abs().max()), 5),
             "max_abs_product_err_donor": round(float(((f1d * f2d) - pttd).abs().max()), 5),
             "f1_neg_frac_base": round(float((f1 < 0).float().mean()), 3), "f2_pos_frac_base": round(float((f2 > 0).float().mean()), 3),
             "f1_neg_frac_donor": round(float((f1d < 0).float().mean()), 3), "f2_pos_frac_donor": round(float((f2d > 0).float().mean()), 3),
             "f1_abs_mean": round(float(f1.abs().mean()), 4), "f2_mean": round(float(f2.mean()), 4), "corr_f1_ptt": None if corr is None else round(corr, 3),
             "self_key_rank_in_row": [int(x) for x in rank], "axis_norm_share": round(float((ax.norm(dim=1) / image.norm(dim=1)).mean()), 3), "rows": rows}
        if abs(eff["remove_full"]) > 1e-6:
            S["off_share"] = round(eff["remove_off"] / eff["remove_full"], 3); S["axis_share"] = round(eff["remove_axis"] / eff["remove_full"], 3)
        else:
            S["off_share"] = S["axis_share"] = None
        S["additivity_gap"] = round(eff["remove_axis"] + eff["remove_off"] - eff["remove_full"], 3)
        R[sname] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    Qs, Lx = R["quant_seven"], R["lex_seven"]
    pred_a = all(R[k]["max_abs_product_err"] <= P_TOL and R[k]["max_abs_product_err_donor"] <= P_TOL and abs(R[k]["additivity_gap"]) <= ADD_TOL for k in R)
    pred_b = Qs["f1_neg_frac_base"] >= SIGN_FRAC and Qs["f2_pos_frac_base"] >= SIGN_FRAC
    ratio = Qs["f2_mean"] / Lx["f2_mean"] if Lx["f2_mean"] else None
    pred_c = ratio is not None and GATE_LO <= ratio <= GATE_HI
    pred_d = Qs["off_share"] is not None and OFF_LO <= Qs["off_share"] <= OFF_HI and AX_LO <= Qs["axis_share"] <= AX_HI
    pred_e = Lx["f1_abs_mean"] <= LEX_F1 and abs(Lx["effect"]["remove_axis"]) <= LEX_TOL and abs(Lx["effect"]["remove_off"]) <= LEX_TOL
    predictions = {"pred_a_instrument": pred_a, "pred_b_sign_factor": pred_b, "pred_c_gate": pred_c, "pred_d_off_axis": pred_d, "pred_e_lex_control": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_self_read_factors_v161", "candidate_id": "corpus.unit_tier5_hub_self_read_factors_v161",
              "bars": {"p_tol": P_TOL, "add_tol": ADD_TOL, "sign_frac": SIGN_FRAC, "gate_band": [GATE_LO, GATE_HI], "off_band": [OFF_LO, OFF_HI], "axis_band": [AX_LO, AX_HI], "lex_f1": LEX_F1, "lex_tol": LEX_TOL},
              "gate_ratio_q_over_lex": None if ratio is None else round(ratio, 3),
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
