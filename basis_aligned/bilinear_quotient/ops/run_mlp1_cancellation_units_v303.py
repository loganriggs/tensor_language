#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_cancellation_is_spread pred_c_lookup_units_are_the_cancelling_units pred_d_per_unit_ratio_is_uniform pred_e_same_units_across_lengths
"""MLP 1: is the cancellation carried by a few units or by all 4,608? (v303). v301 / v302: MLP 1's write = gamma^2 T + cross + context^2 with the
token x context cross term projecting -0.44 to -0.9 on the lookup T. Both the lookup and the cross term are sums over the 4,608 bilinear units
j: lookup_j = (D_j . T^) gamma^2 (L_j t')(R_j t'), cross_j = (D_j . T^) [(L_j t')(R_j c) + (L_j c)(R_j t')], with T^ the unit lookup direction, so the
projections split exactly by unit. Registered question (Logan: sparse in some basis?): does a small set of units do the cancelling -- a gate --
or does every unit cancel its own lookup contribution by a common ratio (a function of the whole layer)? Phrase A, lengths 1 / 8 / 64, 224 targets.
Per unit: pooled lookup_j and cross_j over rows; per row: the per-unit ratio cross_j / lookup_j over the 200 largest-|lookup_j| units.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_unit_closure                      sum_j lookup_j = gamma^2 ||T|| and sum_j cross_j = proj_T(cross) ||T|| within relative 1e-2 on every row (float32)
    pred_b_cancellation_is_spread            the 200 units with the largest |pooled cross_j| carry <= 0.50 of the pooled cross at every length (no gate subset)
    pred_c_lookup_units_are_the_cancelling_units  Pearson r over units between pooled lookup_j and pooled cross_j <= -0.70 at every length (the units that write the lookup are the ones that cancel it)
    pred_d_per_unit_ratio_is_uniform         the median over rows of the interquartile range of cross_j / lookup_j over the top-200 lookup units is <= 0.50 at every length
    pred_e_same_units_across_lengths         the top-200 |cross_j| sets at lengths 1 and 64 share >= 100 units
PRICE (registered maximum): 1 table batch + 3 length batches = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_cancellation_units_v303_result.json"
CANDIDATE_ID = "mlp1.token_table.cancellation_units_v303"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, TOP200_MAX, R_MAX, IQR_MAX, SHARE_MIN = 1e-2, 0.50, -0.70, 0.50, 100
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-2", "pred_b_cancellation_is_spread": "<= 0.50 x 3", "pred_c_lookup_units_are_the_cancelling_units": "r <= -0.70 x 3", "pred_d_per_unit_ratio_is_uniform": "IQR <= 0.50 x 3", "pred_e_same_units_across_lengths": ">= 100 of 200"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top200_max": TOP200_MAX, "r_max": R_MAX, "iqr_max": IQR_MAX, "share_min": SHARE_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    closure, per_k, tops = 0.0, {}, {}
    That = T / T.norm(dim=1, keepdim=True); DT = That @ Dw          # [rows, 4608]: D_j . T^ per row
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; n = F.rms_norm(c["x1"], (T.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
        Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
        look_j = DT * (Lt2 * Rt2); cross_j = DT * (Lt2 * Rc + Lc * Rt2)                       # [rows, 4608], in units of ||T||
        quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T
        tl, tc = (quad * That).sum(1), (cross * That).sum(1)
        closure = max(closure, float(((look_j.sum(1) - tl).abs() / tl.abs()).max()), float(((cross_j.sum(1) - tc).abs() / tc.abs().clamp_min(1e-6)).max()))
        pl, pc = look_j.sum(0), cross_j.sum(0); order = torch.argsort(pc.abs(), descending=True); tops[k] = set(order[:200].tolist())
        top200 = float(pc[order[:200]].sum() / pc.sum())
        r = float(((pl - pl.mean()) * (pc - pc.mean())).sum() / ((pl - pl.mean()).norm() * (pc - pc.mean()).norm()))
        iqrs = []
        for i in range(W.shape[0]):
            top = torch.argsort(look_j[i].abs(), descending=True)[:200]; ratio = cross_j[i, top] / look_j[i, top]; q = ratio.quantile(torch.tensor([0.25, 0.75])); iqrs.append(float(q[1] - q[0]))
        iqr = float(torch.tensor(iqrs).median())
        shares = {str(m): float(pc[order[:m]].sum() / pc.sum()) for m in (10, 50, 200, 500, 1000, 2000)}
        per_k[k] = {"pooled_lookup_over_T": float(pl.sum() / W.shape[0]), "pooled_cross_over_T": float(pc.sum() / W.shape[0]), "top_shares_of_cross": shares, "top200_share": top200, "pearson_lookup_vs_cross_units": r, "ratio_iqr_median": iqr,
                    "ratio_median_of_medians": float(torch.tensor([float((cross_j[i, torch.argsort(look_j[i].abs(), descending=True)[:200]] / look_j[i, torch.argsort(look_j[i].abs(), descending=True)[:200]]).median()) for i in range(W.shape[0])]).median()),
                    "units_negative_fraction": float((pc < 0).float().mean()), "top12_units": [(int(j), float(pc[j]), float(pl[j])) for j in order[:12]]}
    overlap = len(tops[1] & tops[64])
    report = {"closure_max": closure, "top200_overlap_1_vs_64": overlap, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps({k: v for k, v in report.items()}, indent=1, default=float))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_cancellation_is_spread": all(v["top200_share"] <= TOP200_MAX for v in per_k.values()), "pred_c_lookup_units_are_the_cancelling_units": all(v["pearson_lookup_vs_cross_units"] <= R_MAX for v in per_k.values()),
                   "pred_d_per_unit_ratio_is_uniform": all(v["ratio_iqr_median"] <= IQR_MAX for v in per_k.values()), "pred_e_same_units_across_lengths": overlap >= SHARE_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_cancellation_units_result_v303", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
