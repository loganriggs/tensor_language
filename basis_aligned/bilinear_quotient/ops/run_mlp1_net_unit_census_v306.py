#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_net_closure pred_b_net_change_is_spread pred_c_net_change_units_are_lookup_units pred_d_head_units_not_net_cancellers pred_e_same_net_units_across_lengths
"""MLP 1: NET per-unit census of the context change (v306). v305 showed that a unit's cross term and context^2 term cancel inside the unit, so the
cross-term census (v303) named units 3289 / 624 that are not net cancellers (zeroing them lowered alpha). The causal per-unit quantity is the NET
change of the unit's along-entry write between context and isolation: delta_j = (D_j . T^) [h_j(context) - h_j(alone)], which sums exactly to
(alpha - 1) ||T|| because Down is linear (unit zeroing is additive in the write). Phrase A, lengths 1 / 8 / 64, 224 targets. Pooled over rows:
top-k shares of the net change, the sign fraction, the correlation with the unit's lookup contribution, the rank of 3289 / 624, and the stability of
the head set across lengths. The head of this census is what v307 edits.
PREDICTIONS (scored as written; failures preserved; priors unsure after v305)
    pred_a_net_closure                    sum_j delta_j = (alpha - 1) ||T|| within relative 1e-3 on every row
    pred_b_net_change_is_spread           the 200 units with the largest |pooled delta_j| carry <= 0.50 of the pooled net change at every length
    pred_c_net_change_units_are_lookup_units  Pearson r over units between pooled lookup_j (alone) and pooled delta_j <= -0.70 at every length (units lose in proportion to what they wrote)
    pred_d_head_units_not_net_cancellers  3289 and 624 are both outside the top-200 |pooled delta_j| at 8 tokens (consistent with v305)
    pred_e_same_net_units_across_lengths  the top-200 sets at 1 and 64 tokens share >= 100 units
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
OUT = ROOT / "circuits/followups/mlp1_net_unit_census_v306_result.json"
CANDIDATE_ID = "mlp1.token_table.net_unit_census_v306"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, TOP200_MAX, R_MAX, SHARE_MIN = 1e-3, 0.50, -0.70, 100; HEAD = (3289, 624)
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_net_closure": "<= 1e-3", "pred_b_net_change_is_spread": "<= 0.50 x 3", "pred_c_net_change_units_are_lookup_units": "r <= -0.70 x 3", "pred_d_head_units_not_net_cancellers": "both outside top-200 at 8", "pred_e_same_net_units_across_lengths": ">= 100 of 200"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top200_max": TOP200_MAX, "r_max": R_MAX, "share_min": SHARE_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    closure, per_k, tops, rank8 = 0.0, {}, {}, {}
    That = T / T.norm(dim=1, keepdim=True); DT = That @ Dw                     # [rows, 4608]
    h_tab = (n_tab @ Lw.T) * (n_tab @ Rw.T); look_j = DT * h_tab              # per-unit lookup along T^ (alone), in units of ||T||... (sum = ||T||)
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; n = F.rms_norm(c["x1"], (T.shape[-1],)); h_ctx = (n @ Lw.T) * (n @ Rw.T)
        delta_j = DT * (h_ctx - h_tab); alpha = (W * T).sum(1) / (T * T).sum(1)
        closure = max(closure, float(((delta_j.sum(1) - (alpha - 1) * T.norm(dim=1)).abs() / ((alpha - 1) * T.norm(dim=1)).abs().clamp_min(1e-6)).max()))
        pd, pl = delta_j.sum(0), look_j.sum(0); order = torch.argsort(pd.abs(), descending=True); tops[k] = set(order[:200].tolist())
        shares = {str(m): float(pd[order[:m]].sum() / pd.sum()) for m in (10, 50, 200, 500, 1000, 2000)}
        r = float(((pl - pl.mean()) * (pd - pd.mean())).sum() / ((pl - pl.mean()).norm() * (pd - pd.mean()).norm()))
        ranks = {str(u): int((pd.abs() > pd.abs()[u]).sum()) + 1 for u in HEAD}
        if k == 8: rank8 = ranks
        per_k[k] = {"alpha_median": float(alpha.median()), "pooled_net_over_T_per_row": float(pd.sum() / W.shape[0]), "top_shares_of_net": shares, "top200_share": shares["200"], "pearson_lookup_vs_net": r, "units_negative_fraction": float((pd < 0).float().mean()),
                    "head_ranks": ranks, "head_net_per_row": {str(u): float(pd[u] / W.shape[0]) for u in HEAD}, "top12_units": [(int(j), float(pd[j] / W.shape[0]), float(pl[j] / W.shape[0])) for j in order[:12]]}
    overlap = len(tops[1] & tops[64])
    report = {"closure_max": closure, "top200_overlap_1_vs_64": overlap, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps(report, indent=1, default=float))
    predictions = {"pred_a_net_closure": closure <= CLOSURE_TOL, "pred_b_net_change_is_spread": all(v["top200_share"] <= TOP200_MAX for v in per_k.values()), "pred_c_net_change_units_are_lookup_units": all(v["pearson_lookup_vs_net"] <= R_MAX for v in per_k.values()),
                   "pred_d_head_units_not_net_cancellers": all(rank8[str(u)] > 200 for u in HEAD), "pred_e_same_net_units_across_lengths": overlap >= SHARE_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_net_unit_census_result_v306", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
