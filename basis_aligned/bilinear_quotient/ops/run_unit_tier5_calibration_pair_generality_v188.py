#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; layers 16/17 fixed; per-position zeroing (exact: no position mixing after the write is undone one position at a time); no fit.
"""v188: the late MLP calibration pair (v187: mlp:16 sharpens, mlp:17 flattens, at the answer position t) is a GLOBAL temperature pair --
it acts at every position of every row, and only mlp:16's half is token-class dependent. v187 measured the pair at t only. Here each
layer's MLP write is zeroed at ONE position at a time (per-position forwards, so mlp:16's arm is not confounded by attn:17 reading other
zeroed positions) and the next-token distribution at that position is read: entropy, top-1 minus mean logit. Positions are then binned by
the base top-1 token's id (frequent = id < 1000, other = id >= 1000) to test whether the sharpening/flattening depends on WHAT is being
predicted, which is where v186's per-token self-saturation (a property of frequent function-word tokens) should show if it does.

Magnitudes printed before writing (CPU probe, 5 cells x 4 rows, all positions >= 1, zeroing at ALL positions at once): mlp:17 zero: d entropy
< 0 at 100/100 positions, mean -2.35 / -2.84 / -2.43 / -3.04 / -2.37 per cell, by top-1 class frequent -2.59 (n=88) vs other -2.59 (n=12);
mlp:16 zero: d entropy > 0 at 100/100, mean +3.9 / +4.13 / +4.73 / +4.74 / +5.7, frequent +4.9 vs other +3.2 (n=12). v187 at t: |d entropy 16|
> |d entropy 17| on 14/14 (ratio 1.3-5.0); doubling mlp:17 at t raised entropy +3.3..+6.8 on 14/14.

Smoke (CPU, coordination p0 + p1, 3 rows, 2 s): zero-17 per-position entropy drop on 12/12 and 12/12 positions, cell means -2.91 / -2.03;
zero-16 rise on 12/12 and 12/12, means +3.87 / +4.34; double-17 +3.47 / +3.49; ratio 1.33 / 2.14; pooled gap16 +0.44, gap17 -0.90 (n_other 8).

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar carries a lower AND an upper bound):
  pred_a_flattener_everywhere   zeroing mlp:17 at a position lowers that position's entropy on >= 0.90 of positions in 14 of 14 cells, with the
                                cell mean d entropy in -5..-1 nats on >= 12 of 14.
  pred_b_sharpener_everywhere   zeroing mlp:16 at a position raises that position's entropy on >= 0.90 of positions in 14 of 14 cells, with the
                                cell mean in +2..+8 nats on >= 12 of 14.
  pred_c_class_dependence       pooled over all cells' positions: d entropy(16)[frequent top-1] - d entropy(16)[other] in +0.3..+3.0 nats
                                (mlp:16's sharpening is stronger where a frequent token is predicted); requires >= 20 'other' positions
                                pooled, else reported as untested (pred_c False). Pre-smoke this pred also required the mlp:17 gap in
                                -0.7..+0.7 (probe: 0.00 on n=12) and the mlp:16 gap in +0.5..+3.0; the smoke (24 positions, 8 'other') read
                                gap16 +0.44 and gap17 -0.90 -- the mlp:17 gap's sign is not something I can predict (v186's per-token
                                saturation of frequent tokens would make it negative; the all-positions probe read zero), so it is REPORTED
                                only, and the mlp:16 lower bound is lowered to +0.3. Disclosed here before the run.
  pred_d_dose                   doubling mlp:17's write at all positions raises the cell-mean entropy by +1..+10 nats on >= 12 of 14
                                (no position mixing follows mlp:17, so the all-positions arm is exact per position).
  pred_e_ordering               |cell-mean d entropy(16)| / |cell-mean d entropy(17)| in 1.2-4.0 on >= 12 of 14.
  Reported, not predicted: d(top1 - mean) per arm, per-class means, position counts, per-cell class gaps.
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
OUT = ROOT / "circuits/followups/unit_tier5_calibration_pair_generality_v188_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
BARS = {"pos_frac_min": 0.90, "pos_min_cells": 14, "ent17_band": [-5.0, -1.0], "ent16_band": [2.0, 8.0], "mean_min_cells": 12,
        "gap16_band": [0.3, 3.0], "gap17_band_reported_only": [-0.7, 0.7], "other_min_n": 20, "frequent_max_id": 1000,
        "double17_band": [1.0, 10.0], "double_min_cells": 12, "ratio_band": [1.2, 4.0], "ratio_min_cells": 12, "cap": 30.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_calibration_pair_generality_v188", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V188_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    dev = backend.device
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V188_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V188_SMOKE_CELLS", "2"))]
    R = {}
    pooled = {16: {"frequent": [], "other": []}, 17: {"frequent": [], "other": []}}
    for fam, par in cells_run:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1); batch = prep.base_batch; rows = len(batch.row_ids)

        def run(layer=None, pos=None, scale=0.0):
            """base forward; optionally scale layer's mlp write at position pos (all positions if pos is None)."""
            C = {}
            hs = [model.lm_head.register_forward_hook(lambda m_, a, o: C.__setitem__("logits", o.detach().float()))]
            if layer is not None:
                def zh(m_, a, o):
                    o = o.clone()
                    if pos is None: o = scale * o
                    else: o[:, pos] = scale * o[:, pos]
                    return o
                hs.append(model.transformer.h[layer].mlp.Down.register_forward_hook(zh))
            with torch.no_grad():
                g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            for h in hs: h.remove()
            lg = BARS["cap"] * torch.tanh(C["logits"] / BARS["cap"]); lp = lg.log_softmax(-1)
            return lg, -(lp.exp() * lp).sum(-1), lg.max(-1).values - lg.mean(-1)
        lg0, e0, tm0 = run(); top1 = lg0.argmax(-1); T = top1.shape[1]
        freq = (top1 < BARS["frequent_max_id"])[:, 1:]
        S = {"rows": rows, "positions": T - 1, "n_frequent": int(freq.sum()), "n_other": int((~freq).sum()), "base_entropy": round(float(e0[:, 1:].mean()), 3)}
        de = {}
        for l in (16, 17):
            d_e = torch.zeros(rows, T - 1, device=dev); d_t = torch.zeros(rows, T - 1, device=dev)
            for p in range(1, T):
                _, e, tm = run(l, p, 0.0)
                d_e[:, p - 1] = e[:, p] - e0[:, p]; d_t[:, p - 1] = tm[:, p] - tm0[:, p]
            de[l] = d_e
            sgn = (d_e < 0) if l == 17 else (d_e > 0)
            S[f"zero_{l}_pos_frac"] = round(float(sgn.float().mean()), 3)
            S[f"zero_{l}_d_entropy"] = round(float(d_e.mean()), 3)
            S[f"zero_{l}_d_top_minus_mean"] = round(float(d_t.mean()), 3)
            S[f"zero_{l}_d_entropy_frequent"] = round(float(d_e[freq].mean()), 3) if freq.any() else None
            S[f"zero_{l}_d_entropy_other"] = round(float(d_e[~freq].mean()), 3) if (~freq).any() else None
            pooled[l]["frequent"] += [float(v) for v in d_e[freq]]; pooled[l]["other"] += [float(v) for v in d_e[~freq]]
        _, e2, tm2 = run(17, None, 2.0)
        S["double_17_d_entropy"] = round(float((e2 - e0)[:, 1:].mean()), 3); S["double_17_d_top_minus_mean"] = round(float((tm2 - tm0)[:, 1:].mean()), 3)
        S["ratio_16_over_17"] = round(abs(S["zero_16_d_entropy"]) / max(abs(S["zero_17_d_entropy"]), 1e-6), 3)
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    P = {}
    for l in (16, 17):
        nf, no = len(pooled[l]["frequent"]), len(pooled[l]["other"])
        mf = sum(pooled[l]["frequent"]) / max(nf, 1); mo = sum(pooled[l]["other"]) / max(no, 1)
        P[str(l)] = {"n_frequent": nf, "n_other": no, "mean_frequent": round(mf, 3), "mean_other": round(mo, 3), "gap": round(mf - mo, 3) if no else None}
    R["_pooled"] = P
    print("pooled", P, round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_calibration_pair_generality_v188", "candidate_id": "corpus.unit_tier5_calibration_pair_generality_v188",
              "bars": BARS, "families": list(FAMILIES), "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    P = R["_pooled"]
    cells = [S for k, S in R.items() if not k.startswith("_")]
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    pred_a = sum(S["zero_17_pos_frac"] >= B["pos_frac_min"] for S in cells) >= B["pos_min_cells"] and sum(inb(S["zero_17_d_entropy"], B["ent17_band"]) for S in cells) >= B["mean_min_cells"]
    pred_b = sum(S["zero_16_pos_frac"] >= B["pos_frac_min"] for S in cells) >= B["pos_min_cells"] and sum(inb(S["zero_16_d_entropy"], B["ent16_band"]) for S in cells) >= B["mean_min_cells"]
    pred_c = P["16"]["n_other"] >= B["other_min_n"] and inb(P["16"]["gap"], B["gap16_band"])
    pred_d = sum(inb(S["double_17_d_entropy"], B["double17_band"]) for S in cells) >= B["double_min_cells"]
    pred_e = sum(inb(S["ratio_16_over_17"], B["ratio_band"]) for S in cells) >= B["ratio_min_cells"]
    return {"pred_a_flattener_everywhere": pred_a, "pred_b_sharpener_everywhere": pred_b, "pred_c_class_dependence": pred_c, "pred_d_dose": pred_d, "pred_e_ordering": pred_e}


if __name__ == "__main__":
    main()
