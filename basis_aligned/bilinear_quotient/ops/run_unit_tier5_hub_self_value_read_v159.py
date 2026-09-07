#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: 11:03's induced response is a self-attention value read of the number axis at t (positive feedback) -- and why lexical's is inert.

v158: under quantifier's six upstream writers exact-patched at t, 11:03's output shifts through its VALUE path (0.050
of 0.052), pattern change inert; only position t is clamped, so the value change sits at t alone and the shift is
P[t,t] x dV[t]. Analytic read: dn = rms_norm(live)[t] under {six exact} minus base (captured at c_v's input);
dV_h = (1 - lamb) (W_V dn)_h (the value residual v1 is layer 0's and unchanged); shift_pred = P_base[t,t] dV_h. Split
dn into its component along the layer-11 input number axis a_in (donor - base of the same rms-normed input, oriented,
unit norm) and the remainder; effects via resid_add of the complement W_O image on {seven exact}, as in v157/v158.
Lexical seven run the same way as the control (v157: lexical's induced response is inert, -0.003).
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 'rows 16 ... {1: 16}'; quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_instrument   quantifier: cos(shift_pred(dn), measured value-only shift) >= 0.99 and effect(shift_pred(dn)) = v158's 0.050 within 0.01
  pred_b_axis_share   quantifier: effect(shift_pred(dn_axis)) / effect(shift_pred(dn)) within 0.5-0.9
  pred_c_feedback     quantifier: cos(oriented mean W_O image of shift_pred(dn_axis), post-attention layer-11 axis of v157) >= 0.5
  pred_d_self_attn    |P_base[t,t]| of 11:03 (mean over rows) quantifier within 2-20 x lexical  (lexical's 11:03 attends elsewhere, so it cannot re-read t)
  pred_e_lex_control  lexical: |effect(shift_pred(dn))| <= 0.02 through the same analytic path
Reported, unregistered: |dn|, |dn_axis| / |dn|, SIGNED P[t,t] per set (the 4-row smoke showed quantifier's self-weight is NEGATIVE,
-0.12, lexical +0.016 -- squared attention is unnormalised; the sign is part of the mechanism and is reported, not registered),
effect(remainder), lexical's axis share. Smoke hints: feedback cos 0.44 on 4 rows (bar kept at 0.5).
Prior: a should hold by algebra (it validates the capture); b/c are the feedback hypothesis; d is my explanation of the lexical
null -- if d fails and e holds, the difference is in W_V's read (dn_axis inert in lexical), which the next rung would test.
Smoke: V159_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_self_value_read_v159_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
COS_MIN, REPRO_TOL, SHARE_LO, SHARE_HI, FEEDBACK_MIN, SELF_LO, SELF_HI, LEX_TOL = 0.99, 0.01, 0.5, 0.9, 0.5, 2.0, 20.0, 0.02
V158_VALUE = 0.050
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_self_value_read_v159", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V159_SMOKE")
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
    WV = attn.c_v.weight
    lamb = float(attn.lamb)
    R = {}
    for sname, (n, units, _) in SETS.items():
        if sname == "shared_four":
            continue
        units = list(units)
        six = [u for u in units if u != HUB]
        hub_q = g.block_diff_in_means(backend, P[n]["p0"], [HUB])
        qv = hub_q[(LAYER, "heads")][:, 0]
        prep = P[n]["p1"]
        batch = prep.base_batch
        positions = list(batch.semantic_positions)
        rows = len(batch.row_ids)
        def capture(bt, U):
            got = {}
            def pre(_m, args):
                x = args[0]
                got["n"] = torch.stack([x[i, positions[i]] for i in range(rows)]).detach().float().clone()
            h = attn.c_v.register_forward_pre_hook(pre)
            try:
                heads = [(u, 0, prep.donor_cache) for u in U if u.startswith("attn")]
                mlps = [(u, 0, prep.donor_cache) for u in U if u.startswith("mlp")]
                Pm, Vm = v131.capture_with_clamp(backend, bt, heads, mlps, LAYER)
            finally:
                h.remove()
            return got["n"].to(backend.device), Pm, Vm
        def out_of(Pm, Vm):
            return torch.stack([Pm[i, HUB_H, positions[i], :] @ Vm[i, :, HUB_H, :] for i in range(rows)]).float().to(backend.device)
        with torch.no_grad():
            nB, PB, VB = capture(batch, [])
            n6, P6, V6 = capture(batch, six)
            nD, _, _ = capture(prep.donor_batch, [])
            measured_value = out_of(PB, V6 - VB)
            dn = n6 - nB
            din = nD - nB
            sgn = g._orientation(din)
            a_in = (din * sgn[:, None]).mean(0); a_in = a_in / a_in.norm()
            dn_axis = (dn @ a_in)[:, None] * a_in[None, :]
            dn_rem = dn - dn_axis
            ptt = torch.stack([PB[i, HUB_H, positions[i], positions[i]] for i in range(rows)]).float().to(backend.device)
            def shift_pred(d):
                dv = ((1.0 - lamb) * (d.to(WV.dtype) @ WV.T).float())[:, HUB_H * g.HEAD_DIM:(HUB_H + 1) * g.HEAD_DIM]
                return ptt[:, None] * dv
            preds = {"full": shift_pred(dn), "axis": shift_pred(dn_axis), "rem": shift_pred(dn_rem)}
            cos_full = float(torch.nn.functional.cosine_similarity(preds["full"], measured_value, dim=1).mean())
            qpart = lambda t: (t @ qv)[:, None] * qv[None, :]
            img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
            # post-attention layer-11 axis (v157) for the feedback cosine
            dres, bres = {}, {}
            g.forward_units(backend, prep.donor_batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=dres)
            g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=bres)
            rdelta = (torch.stack([dres[(rid, LAYER)] for rid in prep.donor_batch.row_ids]) - torch.stack([bres[(rid, LAYER)] for rid in batch.row_ids])).to(backend.device)
            s2 = g._orientation(rdelta)
            a_out = (rdelta * s2[:, None]).mean(0); a_out = a_out / a_out.norm()
            ax_img = img(preds["axis"] - qpart(preds["axis"]))
            m = (ax_img * s2[:, None]).mean(0)
            feedback_cos = float(m @ a_out / m.norm())
        def rec(us, q=None, add=None):
            out = g.forward_units(backend, batch, units=us, donor_cache=prep.donor_cache, base_cache=prep.base_cache, q=q,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        exact = rec(units)
        eff = {k: round(rec(units, None, img(v - qpart(v))) - exact, 3) for k, v in preds.items()}
        eff["measured_value"] = round(rec(units, None, img(measured_value - qpart(measured_value))) - exact, 3)
        S = {"exact": exact, "effect": eff, "cos_pred_vs_measured": round(cos_full, 4), "p_tt_mean": round(float(ptt.mean()), 4), "p_tt_abs_mean": round(float(ptt.abs().mean()), 4),
             "dn_norm": round(float(dn.norm(dim=1).mean()), 2), "dn_axis_share": round(float((dn_axis.norm(dim=1) / dn.norm(dim=1)).mean()), 3),
             "shift_pred_norm": round(float(preds["full"].norm(dim=1).mean()), 2), "measured_value_norm": round(float(measured_value.norm(dim=1).mean()), 2),
             "feedback_cos": round(feedback_cos, 3), "lamb": round(lamb, 4), "rows": rows}
        S["axis_share_of_effect"] = round(eff["axis"] / eff["full"], 3) if abs(eff["full"]) > 1e-6 else None
        R[sname] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    Qs, Lx = R["quant_seven"], R["lex_seven"]
    pred_a = Qs["cos_pred_vs_measured"] >= COS_MIN and abs(Qs["effect"]["full"] - V158_VALUE) <= REPRO_TOL
    pred_b = Qs["axis_share_of_effect"] is not None and SHARE_LO <= Qs["axis_share_of_effect"] <= SHARE_HI
    pred_c = Qs["feedback_cos"] >= FEEDBACK_MIN
    pred_d = SELF_LO * Lx["p_tt_abs_mean"] <= Qs["p_tt_abs_mean"] <= SELF_HI * Lx["p_tt_abs_mean"]
    pred_e = abs(Lx["effect"]["full"]) <= LEX_TOL
    predictions = {"pred_a_instrument": pred_a, "pred_b_axis_share": pred_b, "pred_c_feedback": pred_c, "pred_d_self_attn": pred_d, "pred_e_lex_control": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_self_value_read_v159", "candidate_id": "corpus.unit_tier5_hub_self_value_read_v159",
              "bars": {"cos_min": COS_MIN, "repro_tol": REPRO_TOL, "share_band": [SHARE_LO, SHARE_HI], "feedback_min": FEEDBACK_MIN, "self_ratio_band": [SELF_LO, SELF_HI], "lex_tol": LEX_TOL, "v158_value": V158_VALUE},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
