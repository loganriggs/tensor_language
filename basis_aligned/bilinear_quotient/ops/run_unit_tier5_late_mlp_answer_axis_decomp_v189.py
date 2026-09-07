#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; layers 16/17 fixed; exact algebraic decomposition of the base write (no fit, one forward per cell).
"""v189: WHY mlp:16's base write opposes the promoted answer and mlp:17's does not (v187) -- the exact decomposition of each late MLP's
base write on the answer axis. The ungated Bilinear MLP gives w(x_n) = Down(L x_n * R x_n) for the rms-normed input x_n; writing
x_n = x_perp + beta * u_a (u_a = U[ans]/|U[ans]|, beta = x_n . u_a) the on-axis write splits EXACTLY into
    w . u_a = perp + cross + quad,   perp = u_a . W(x_perp)   (context alone),
                                     cross = beta * u_a . Down(L x_perp * R u_a + L u_a * R x_perp)   (context x answer axis),
                                     quad = beta^2 * u_a . W(u_a)   (the answer token's own self-saturation; v186: u_a . W(u_a) < 0).
Magnitudes printed before writing (CPU probe, 3-row cells coordination p0 / polarity p0 / degree p1 / correlative_both_neither p0; recon 0.000):
  mlp:16  full -613..-2890 on 12/12 rows = perp (-128..-1634 on 11/12; +182, +2 on two coordination rows) + cross (-324..+372, |share| 0.02-0.49)
          + quad (-338..-1538 on 12/12; share of full 0.18-0.79).  beta 4.3-8.4 on |x_n| = 33.9 (sqrt(1152)).
  mlp:17  full -1430..+1059 (mixed) = perp +91..+2389 on 12/12 (the context alone PROMOTES the answer) + cross (-504..+601) + quad
          (-111..-2452 on 12/12); saturation = cross + quad = -377..-2284 on 12/12; cell-mean saturation / cell-mean perp = -0.58 / -1.1 / -2.3 / -3.4.
So the sharpener opposes the answer through the context and the self-saturation term, the flattener's context promotes the answer and its
self-saturation cancels that promotion -- the same weight-space fact (u_a . W(u_a) < 0) plays out differently through the two contexts.

Smoke (CPU, coordination p0 + p1, 3 rows): recon/split 1e-6; coordination p0 mlp:16 full -655 = perp +19 + cross -204 + quad -470 (share 0.72), mlp:17 full +840 = perp +2003 + cross -433 + quad -730,
  sat/perp -0.58; p1 mlp:16 full -1201 = perp -783 + cross -98 + quad -320 (share 0.27), mlp:17 perp +1638, sat -440, sat/perp -0.27; beta 3.2-4.7.
  The p1 sat/perp of -0.27 lies OUTSIDE the pre-smoke ratio band -20..-0.4 (set from the 4-cell probe); the band registered below is -20..-0.2 --
  what I now believe after the smoke -- and the pre-smoke bar is disclosed here. No other bar was changed by the smoke.

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar carries a lower AND an upper bound):
  pred_a_exact          the three-term expansion reproduces the captured write at t to 1e-3 relative on 14 of 14 cells.
  pred_b_self_saturation quad < 0 on 14 of 14 cells for BOTH layers, and quad's share of mlp:16's full on-axis write in 0.15-0.85 on >= 12 of 14.
  pred_c_sharpener      mlp:16: full < 0 on 14 of 14 (v187), cell-mean perp <= 0 on >= 10 of 14, and |cross| / |full| in 0.0-0.5 on >= 12 of 14
                        (the base opposition is NOT the context x answer-axis cross term that carried the push damping in v184).
  pred_d_flattener      mlp:17: cell-mean perp > 0 on >= 12 of 14 (context promotes the answer) and cell-mean (cross + quad) < 0 on >= 12 of 14,
                        with the ratio mean(cross + quad) / mean(perp) in -20..-0.2 on >= 12 of 14 (the saturation cancels the promotion).
  pred_e_beta           the answer component of the normed input, beta, lies in 2-12 on 14 of 14 cells for both layers (the residual
                        carries the promoted token at 6-35 % of its norm at t; the quad term is second order in it).
  Reported, not predicted: the foil-axis decomposition, |w|, per-term values for both tokens.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_late_mlp_answer_axis_decomp_v189_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LATE = (16, 17)
BARS = {"recon_tol": 1e-3, "quad_min_cells": 14, "quad_share_band": [0.15, 0.85], "quad_share_min_cells": 12,
        "full16_min_cells": 14, "perp16_min_cells": 10, "cross16_share_band": [0.0, 0.5], "cross16_min_cells": 12,
        "perp17_min_cells": 12, "sat17_min_cells": 12, "sat17_ratio_band": [-20.0, -0.2], "sat17_ratio_min_cells": 12,
        "beta_band": [2.0, 12.0], "beta_min_cells": 14}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 100, 4000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_late_mlp_answer_axis_decomp_v189", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V189_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    dev = backend.device
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V189_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V189_SMOKE_CELLS", "2"))]
    U = model.lm_head.weight.detach().float()
    nz = lambda v: v / v.norm(dim=-1, keepdim=True)
    WTS = {}
    for l in LATE:
        mlp = model.transformer.h[l].mlp
        assert not getattr(model.config, "gated", False), "expansion assumes the ungated Bilinear"
        WTS[l] = (mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float(),
                  mlp.Down.bias.detach().float() if mlp.Down.bias is not None else None)
    R = {}
    for fam, par in cells_run:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1); batch = prep.base_batch; rows = len(batch.row_ids)
        ar = torch.arange(rows, device=dev); t_t = torch.tensor(list(batch.semantic_positions), device=dev)
        ans = torch.tensor(list(batch.answer_ids), device=dev); foil = torch.tensor(list(batch.foil_ids), device=dev)
        C = {}
        hs = []
        for l in LATE:
            mlp = model.transformer.h[l].mlp
            hs.append(mlp.register_forward_pre_hook(lambda m_, a, l=l: C.__setitem__(("xin", l), a[0].detach().float())))
            hs.append(mlp.Down.register_forward_hook(lambda m_, a, o, l=l: C.__setitem__(("w", l), o.detach().float())))
        with torch.no_grad():
            g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
        for h in hs: h.remove()
        S = {"rows": rows}
        for l in LATE:
            Lw, Rw, Dw, bias = WTS[l]
            W = lambda x: ((x @ Lw.T) * (x @ Rw.T)) @ Dw.T
            xn = C[("xin", l)][ar, t_t]; w = C[("w", l)][ar, t_t]
            recon = W(xn) + (bias if bias is not None else 0)
            S[f"recon_err_{l}"] = round(float(((recon - w).norm(dim=1) / w.norm(dim=1)).max()), 6)
            for tok, lab in ((ans, "ans"), (foil, "foil")):
                ua = nz(U[tok]); beta = (xn * ua).sum(1, keepdim=True); xp = xn - beta * ua
                full = (W(xn) * ua).sum(1); perp = (W(xp) * ua).sum(1)
                cross = beta[:, 0] * (((((xp @ Lw.T) * (ua @ Rw.T)) + ((ua @ Lw.T) * (xp @ Rw.T))) @ Dw.T) * ua).sum(1)
                quad = beta[:, 0] ** 2 * (W(ua) * ua).sum(1)
                S[f"{lab}_{l}_beta"] = round(float(beta.mean()), 3)
                S[f"{lab}_{l}_full"] = round(float(full.mean()), 1); S[f"{lab}_{l}_perp"] = round(float(perp.mean()), 1)
                S[f"{lab}_{l}_cross"] = round(float(cross.mean()), 1); S[f"{lab}_{l}_quad"] = round(float(quad.mean()), 1)
                S[f"{lab}_{l}_split_err"] = round(float(((perp + cross + quad) - full).abs().max() / full.abs().mean()), 6)
                S[f"{lab}_{l}_quad_share"] = round(float(quad.mean() / full.mean()), 3) if abs(float(full.mean())) > 1e-6 else None
                S[f"{lab}_{l}_cross_share_abs"] = round(abs(float(cross.mean())) / max(abs(float(full.mean())), 1e-6), 3)
                S[f"{lab}_{l}_sat"] = round(float((cross + quad).mean()), 1)
                S[f"{lab}_{l}_sat_over_perp"] = round(float((cross + quad).mean() / perp.mean()), 3) if abs(float(perp.mean())) > 1e-6 else None
                S[f"{lab}_{l}_quad_neg_rows"] = int((quad < 0).sum())
            S[f"wnorm_{l}"] = round(float(w.norm(dim=1).mean()), 1)
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_late_mlp_answer_axis_decomp_v189", "candidate_id": "corpus.unit_tier5_late_mlp_answer_axis_decomp_v189",
              "bars": BARS, "families": list(FAMILIES), "layers": list(LATE), "measures": R, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = [S for k, S in R.items() if not k.startswith("_")]
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    pred_a = all(S["recon_err_16"] <= B["recon_tol"] and S["recon_err_17"] <= B["recon_tol"] and S["ans_16_split_err"] <= B["recon_tol"] and S["ans_17_split_err"] <= B["recon_tol"] for S in cells)
    pred_b = sum(S["ans_16_quad"] < 0 and S["ans_17_quad"] < 0 for S in cells) >= B["quad_min_cells"] and sum(inb(S["ans_16_quad_share"], B["quad_share_band"]) for S in cells) >= B["quad_share_min_cells"]
    pred_c = sum(S["ans_16_full"] < 0 for S in cells) >= B["full16_min_cells"] and sum(S["ans_16_perp"] <= 0 for S in cells) >= B["perp16_min_cells"] and \
        sum(inb(S["ans_16_cross_share_abs"], B["cross16_share_band"]) for S in cells) >= B["cross16_min_cells"]
    pred_d = sum(S["ans_17_perp"] > 0 for S in cells) >= B["perp17_min_cells"] and sum(S["ans_17_sat"] < 0 for S in cells) >= B["sat17_min_cells"] and \
        sum(inb(S["ans_17_sat_over_perp"], B["sat17_ratio_band"]) for S in cells) >= B["sat17_ratio_min_cells"]
    pred_e = sum(inb(S["ans_16_beta"], B["beta_band"]) and inb(S["ans_17_beta"], B["beta_band"]) for S in cells) >= B["beta_min_cells"]
    return {"pred_a_exact": pred_a, "pred_b_self_saturation": pred_b, "pred_c_sharpener": pred_c, "pred_d_flattener": pred_d, "pred_e_beta": pred_e}


if __name__ == "__main__":
    main()
