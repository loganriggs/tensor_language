#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; layers 13-17 fixed; token classes seeded (seed 1, new); random-write controls seeded; no fit.
"""v187: the late MLP pair as a CALIBRATION pair — mlp:16 sharpens the next-token distribution, mlp:17 flattens it — and the per-token
self-saturation of v186 is a weight-space property that GROWS over layers 14 -> 17. v186 established that mlp:17's gate vector f(U[a]) is
anti-aligned with U[a] itself for the task's function-word tokens (14/14 cells, class controls in band) and that this is causal for a push.
Three questions here, all in the BASE forward (no push): (i) do the earlier late MLPs carry the same weight structure, and from which layer;
(ii) what does each late MLP's own base write at t do to the promoted answer token; (iii) what does each do to the whole distribution at t
(entropy, top-1 minus mean logit, ans-foil margin) when its write at t is zeroed, compared with a random write of the same norm (dilution
control) and with the write doubled.

Magnitudes printed before writing (CPU probes; weights over 11 task tokens and seed-0 classes of 300; base forward on 3-row cells of
coordination p0 / polarity p0 / degree p1 / correlative_both_neither p0):
  weights, task tokens: mean cos(f_l(U[a]), U[a]) = +0.155 / -0.005 / -0.286 / -0.476 / -0.589 and bottom-0.1% rank fraction 0.00 / 0.27 / 0.91 /
  1.00 / 1.00 for l = 13 / 14 / 15 / 16 / 17; frequent tokens (id<1000): 0.027 / 0.11 / 0.167 / 0.287 / 0.323; random 0.013 / 0.02 / 0.043 /
  0.063 / 0.067; rare (id>20000) 0.003 / 0.01 / 0.02 / 0.023 / 0.023.
  base writes at t: mlp:16 cos(w, U[ans]) = -0.019..-0.107 on 12/12 rows (w.U[ans] < 0 on 12/12), mlp:15 -0.002..-0.088, mlp:17 -0.030..+0.015
  (w.U[ans] of mixed sign). Over all positions >= 1: cos(w16, U[top-1]) < 0 on 0.83-1.00 of positions vs 0.17-0.33 for a random token;
  mlp:17 0.33-0.73.
  zero mlp:17 at t: d entropy -1.15..-3.21 nats, d(top1 - mean) +7.3..+10.6, d margin(ans-foil) +0.53..+6.06 on 12/12 rows; double it:
  d entropy +3.2..+7.6, d margin -0.35..-5.8. zero mlp:16: d entropy +3.05..+7.55, d(top1 - mean) -4.0..-7.9, d margin -0.07..-6.06 on 12/12.
  random write of the same norm in place of mlp:17 (2 seeds): d(top1 - mean) +1.3..+8.1 (mean of seeds 2.8-6.6, below zeroing on 12/12 rows);
  in place of mlp:16: d entropy +0.7..+4.5 (mean 1.4-4.2, below zeroing on 12/12).
|w17| at t is 39k-88k against a residual of ~58k, so a pure-dilution account was possible; the random-same-norm control is the test of it.

Smoke (CPU, coordination p0 + p1, 3 rows, 72 s): seed-1 classes, bottom-0.1% fraction over layers 13..17: frequent 0.007 / 0.073 / 0.117 / 0.183 /
0.223, random 0.007 / 0.010 / 0.023 / 0.053 / 0.080, rare 0.010 / 0.017 / 0.020 / 0.003 / 0.003, task 0.00 / 0.27 / 0.91 / 1.00 / 1.00 -- these
weight quantities are deterministic and the run reproduces them exactly, so pred_a is decided by the smoke (registered before it, seed 1 new)
and is NOT an independent run outcome; the cell predictions b-e are. Cells: cos_ans_16 -0.022 / -0.046, dl_ans_16 < 0, cos_ans_17 +0.012 /
+0.020; zero mlp:17 d entropy -2.97 / -2.17, d(top1 - mean) +9.5 / +9.0, d margin +1.29 / +1.43; zero mlp:16 d entropy +3.58 / +3.70,
d(top1 - mean) -4.6 / -5.4, d margin -0.43 / -1.01; beyond-dilution 17: +3.90 / +3.95, 16: +1.27 / +2.34. All within the registered bands.

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar carries a lower AND an upper bound):
  pred_a_weight_gradient   with NEW seed-1 classes of 300 tokens: the bottom-0.1% fraction of frequent tokens (id<1000) is non-decreasing over
                           layers 13..17 (each layer >= the previous - 0.02), <= 0.10 at layer 13 and in 0.15-0.50 at layer 17; random tokens
                           in 0.00-0.15 at every layer; rare tokens in 0.00-0.06 at every layer.
  pred_b_base_writes       mlp:16's base write at t has cos(w, U[ans]) <= -0.015 on >= 12 of 14 cells and w.U[ans] < 0 on >= 12 of 14;
                           mlp:17's base write has |cos(w, U[ans])| <= 0.05 on >= 12 of 14 (its opposition is push-conditional, v186).
  pred_c_calibration_pair  zeroing mlp:17's write at t: mean d entropy in -6..-0.5 nats AND mean d(top1 - mean) in +4..+15 on >= 12 of 14;
                           zeroing mlp:16's write at t: mean d entropy in +1.5..+10 AND mean d(top1 - mean) in -12..-2 on >= 12 of 14.
  pred_d_beyond_dilution   the real write does more than dilute: for mlp:17, d(top1 - mean)[zero] - d(top1 - mean)[random same norm, mean of
                           2 seeds] in +1..+10 on >= 10 of 14; for mlp:16, d entropy[zero] - d entropy[random same norm] in +0.5..+8 on >= 10 of 14.
  pred_e_margin_sign       zeroing mlp:17 raises the ans-foil margin: mean d margin in +0.2..+8 on >= 12 of 14; zeroing mlp:16 lowers it:
                           mean d margin in -8..-0.05 on >= 12 of 14.
  Reported, not predicted: the doubled-write arms, mlp:15 everywhere, the per-position top-1 opposition fractions, task-token weight values.
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
OUT = ROOT / "circuits/followups/unit_tier5_late_mlp_calibration_pair_v187_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LAYERS = (13, 14, 15, 16, 17)
LATE = (15, 16, 17)
TASK_TOKENS = (547, 373, 290, 393, 4249, 621, 355, 284, 326, 597, 617)
BARS = {"class_n": 300, "class_seed": 1, "frequent_max_id": 1000, "rare_min_id": 20000, "bottom_quantile": 0.999, "mono_slack": 0.02,
        "frequent_l13_max": 0.10, "frequent_l17_band": [0.15, 0.50], "random_band": [0.0, 0.15], "rare_band": [0.0, 0.06],
        "cos16_max": -0.015, "cos16_min_cells": 12, "dl16_min_cells": 12, "cos17_abs_max": 0.05, "cos17_min_cells": 12,
        "ent17_band": [-6.0, -0.5], "top17_band": [4.0, 15.0], "ent16_band": [1.5, 10.0], "top16_band": [-12.0, -2.0], "calib_min_cells": 12,
        "dil17_band": [1.0, 10.0], "dil16_band": [0.5, 8.0], "dil_min_cells": 10, "random_seeds": 2,
        "margin17_band": [0.2, 8.0], "margin16_band": [-8.0, -0.05], "margin_min_cells": 12, "cap": 30.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_late_mlp_calibration_pair_v187", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V187_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    dev = backend.device
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V187_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V187_SMOKE_CELLS", "2"))]
    U = model.lm_head.weight.detach().float(); V = U.shape[0]
    nz = lambda v: v / v.norm(dim=-1, keepdim=True)
    R = {}

    # ---- weights: per-layer self-saturation over token classes (seed 1, new) and the task tokens ----
    def fgate_at(l):
        mlp = model.transformer.h[l].mlp
        Lw, Rw, Dw = mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float()
        def fgate(u):
            Du = u @ Dw
            return ((Du * (u @ Rw.T)) @ Lw) + ((Du * (u @ Lw.T)) @ Rw)
        return fgate
    def bottom_frac(fgate, ids):
        u = nz(U[ids]); f = fgate(u); lg = nz(f) @ U.T
        r = torch.stack([(lg[i] > lg[i, ids[i]]).sum() for i in range(len(ids))]).float() / V
        return round(float((r > BARS["bottom_quantile"]).float().mean()), 3), round(float((nz(f) * u).sum(1).mean()), 3)
    gen = torch.Generator(device="cpu").manual_seed(BARS["class_seed"])
    n_cls = BARS["class_n"]
    ids = {"random": torch.randint(0, V, (n_cls,), generator=gen), "frequent": torch.randint(0, BARS["frequent_max_id"], (n_cls,), generator=gen),
           "rare": torch.randint(BARS["rare_min_id"], V, (n_cls,), generator=gen), "task": torch.tensor(TASK_TOKENS)}
    W = {}
    for l in LAYERS:
        fg = fgate_at(l)
        W[l] = {k: bottom_frac(fg, v.to(dev)) for k, v in ids.items()}
        print("layer", l, "(bottom-0.1% fraction, mean cos):", W[l], flush=True)
    R["_weights"] = {str(l): W[l] for l in LAYERS}

    # ---- base forward per cell ----
    rgen = torch.Generator(device="cpu").manual_seed(BARS["class_seed"])
    for fam, par in cells_run:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1); batch = prep.base_batch; rows = len(batch.row_ids)
        ar = torch.arange(rows, device=dev); t_t = torch.tensor(list(batch.semantic_positions), device=dev)
        ans = torch.tensor(list(batch.answer_ids), device=dev); foil = torch.tensor(list(batch.foil_ids), device=dev)

        def run(layer=None, mode="base", scale=0.0, rvec=None):
            """base forward; optionally replace layer's mlp write at t: scale * write, or rvec (same-norm random)."""
            C = {}
            hs = [model.lm_head.register_forward_hook(lambda m_, a, o: C.__setitem__("logits", o.detach().float()))]
            for l in LATE:
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: C.__setitem__(l, o.detach().float())))
            if layer is not None:
                def zh(m_, a, o):
                    o = o.clone(); w = o[ar, t_t]
                    o[ar, t_t] = (scale * w) if mode == "scale" else (nz(rvec) * w.norm(dim=1, keepdim=True)).to(o.dtype)
                    return o
                hs.append(model.transformer.h[layer].mlp.Down.register_forward_hook(zh))
            with torch.no_grad():
                g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            for h in hs: h.remove()
            lg = BARS["cap"] * torch.tanh(C["logits"] / BARS["cap"])           # (rows, T, V) post-cap logits
            lt = lg[ar, t_t]; lp = lt.log_softmax(-1)
            stats = {"logp_ans": lp[ar, ans], "margin": lt[ar, ans] - lt[ar, foil], "entropy": -(lp.exp() * lp).sum(-1), "top_minus_mean": lt.max(-1).values - lt.mean(-1)}
            return stats, C, lg
        b, C0, lg0 = run()
        top1 = lg0.argmax(-1); T = top1.shape[1]
        mask = (torch.arange(T, device=dev)[None, :] >= 1).expand(rows, T)
        S = {"rows": rows, "base_logp_ans": round(float(b["logp_ans"].mean()), 3), "base_margin": round(float(b["margin"].mean()), 3),
             "base_entropy": round(float(b["entropy"].mean()), 3), "base_top_minus_mean": round(float(b["top_minus_mean"].mean()), 3)}
        for l in LATE:
            w = C0[l]; wt = w[ar, t_t]
            S[f"cos_ans_{l}"] = round(float((nz(wt) * nz(U[ans])).sum(1).mean()), 4)
            S[f"cos_foil_{l}"] = round(float((nz(wt) * nz(U[foil])).sum(1).mean()), 4)
            S[f"dl_ans_{l}"] = round(float((wt * U[ans]).sum(1).mean()), 1)
            S[f"dl_foil_{l}"] = round(float((wt * U[foil]).sum(1).mean()), 1)
            S[f"wnorm_{l}"] = round(float(wt.norm(dim=1).mean()), 1)
            c_top = (nz(w) * nz(U[top1])).sum(-1)
            rnd = torch.randint(0, V, top1.shape, generator=rgen).to(dev)
            c_rnd = (nz(w) * nz(U[rnd])).sum(-1)
            S[f"top1_opp_frac_{l}"] = round(float((c_top[mask] < 0).float().mean()), 3)
            S[f"rand_opp_frac_{l}"] = round(float((c_rnd[mask] < 0).float().mean()), 3)
        for l in (16, 17):
            for lab, mode, sc in (("zero", "scale", 0.0), ("double", "scale", 2.0)):
                z, _, _ = run(l, mode, sc)
                for k in ("logp_ans", "margin", "entropy", "top_minus_mean"):
                    S[f"{lab}_{l}_d_{k}"] = round(float((z[k] - b[k]).mean()), 3)
            acc = {k: 0.0 for k in ("logp_ans", "margin", "entropy", "top_minus_mean")}
            for s_ in range(BARS["random_seeds"]):
                rvec = torch.randn((rows, U.shape[1]), generator=rgen).to(dev)
                z, _, _ = run(l, "random", 0.0, rvec)
                for k in acc: acc[k] += float((z[k] - b[k]).mean()) / BARS["random_seeds"]
            for k in acc: S[f"random_{l}_d_{k}"] = round(acc[k], 3)
            S[f"beyond_dilution_{l}"] = round(S[f"zero_{l}_d_top_minus_mean"] - S[f"random_{l}_d_top_minus_mean"], 3) if l == 17 else round(S[f"zero_{l}_d_entropy"] - S[f"random_{l}_d_entropy"], 3)
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_late_mlp_calibration_pair_v187", "candidate_id": "corpus.unit_tier5_late_mlp_calibration_pair_v187",
              "bars": BARS, "families": list(FAMILIES), "layers": list(LAYERS), "task_tokens": list(TASK_TOKENS),
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    Wt = R["_weights"]
    cells = [S for k, S in R.items() if not k.startswith("_")]
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    freq = [Wt[str(l)]["frequent"][0] for l in LAYERS]
    mono = all(freq[i] >= freq[i - 1] - B["mono_slack"] for i in range(1, len(freq)))
    pred_a = mono and freq[0] <= B["frequent_l13_max"] and inb(freq[-1], B["frequent_l17_band"]) and \
        all(inb(Wt[str(l)]["random"][0], B["random_band"]) for l in LAYERS) and all(inb(Wt[str(l)]["rare"][0], B["rare_band"]) for l in LAYERS)
    pred_b = sum(S["cos_ans_16"] <= B["cos16_max"] for S in cells) >= B["cos16_min_cells"] and sum(S["dl_ans_16"] < 0 for S in cells) >= B["dl16_min_cells"] and \
        sum(abs(S["cos_ans_17"]) <= B["cos17_abs_max"] for S in cells) >= B["cos17_min_cells"]
    pred_c = sum(inb(S["zero_17_d_entropy"], B["ent17_band"]) and inb(S["zero_17_d_top_minus_mean"], B["top17_band"]) for S in cells) >= B["calib_min_cells"] and \
        sum(inb(S["zero_16_d_entropy"], B["ent16_band"]) and inb(S["zero_16_d_top_minus_mean"], B["top16_band"]) for S in cells) >= B["calib_min_cells"]
    pred_d = sum(inb(S["beyond_dilution_17"], B["dil17_band"]) for S in cells) >= B["dil_min_cells"] and sum(inb(S["beyond_dilution_16"], B["dil16_band"]) for S in cells) >= B["dil_min_cells"]
    pred_e = sum(inb(S["zero_17_d_margin"], B["margin17_band"]) for S in cells) >= B["margin_min_cells"] and sum(inb(S["zero_16_d_margin"], B["margin16_band"]) for S in cells) >= B["margin_min_cells"]
    return {"pred_a_weight_gradient": pred_a, "pred_b_base_writes": pred_b, "pred_c_calibration_pair": pred_c, "pred_d_beyond_dilution": pred_d, "pred_e_margin_sign": pred_e}


if __name__ == "__main__":
    main()
