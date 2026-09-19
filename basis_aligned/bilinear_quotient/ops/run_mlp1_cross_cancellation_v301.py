#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_normalised_table_term_near_full pred_c_cross_projection_negative pred_d_cross_projection_is_the_gain_loss pred_e_cross_projection_tracks_context_size
"""MLP 1: the gain is a cross-term cancellation (v301). v300: MLP 1's input x1 keeps >= 100% of the token's table-input direction at 1-64 context
tokens, so the output's loss of identity (alpha 0.55 -> 0.29) is not input identity loss. In the exact expansion with n = t' + c (t' = the table input
direction scaled to its coefficient in the normalised in-context input; c = the rest), W = Down[(L t')(R t')] + Down[(L t')(R c) + (L c)(R t')] +
Down[(L c)(R c)]. The first term is gamma^2 T with gamma = (n . t^) / ||t^|| (so alpha = gamma^2 + proj_T(cross) + proj_T(only)). Registered reading: gamma^2 stays
near 1 and proj_T(cross) is NEGATIVE and equals the gain loss. Phrase A, lengths 1 / 8 / 64, 7 classes x 32 targets; attention 0's write norm is
also reported (v300 found it writes nothing along the table direction).
PREDICTIONS (scored as written; failures preserved; priors from v289 / v300)
    pred_a_pair_closure                    gamma^2 T + cross + only = W within relative 1e-3 on every row
    pred_b_normalised_table_term_near_full  gamma^2 median >= 0.80 at every length
    pred_c_cross_projection_negative       proj_T(cross) < 0 for >= 0.90 of rows at every length
    pred_d_cross_projection_is_the_gain_loss  |(alpha - gamma^2) - proj_T(cross)| <= 0.10 median at every length (the context-only term's projection on T is small)
    pred_e_cross_projection_tracks_context_size  Pearson r over all rows between proj_T(cross) and -||c|| >= 0.50 (more context input, more cancellation). Prior: unsure.
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
OUT = ROOT / "circuits/followups/mlp1_cross_cancellation_v301_result.json"
CANDIDATE_ID = "mlp1.token_table.cross_cancellation_v301"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, GAMMA_MIN, NEG_MIN, LOSS_TOL, R_MIN = 1e-3, 0.80, 0.90, 0.10, 0.50
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_normalised_table_term_near_full": ">= 0.80 x 3", "pred_c_cross_projection_negative": ">= 0.90 x 3", "pred_d_cross_projection_is_the_gain_loss": "<= 0.10 x 3", "pred_e_cross_projection_tracks_context_size": "r >= 0.50"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gamma_min": GAMMA_MIN, "neg_min": NEG_MIN, "loss_tol": LOSS_TOL, "r_min": R_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    closure, per_k, allp, allc = 0.0, {}, [], []
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; n = F.rms_norm(c["x1"], (T.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
        Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
        quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T; only = (Lc * Rc) @ Dw.T
        closure = max(closure, float((((quad + cross + only) - W).norm(dim=1) / W.norm(dim=1)).max()))
        pT = lambda A: (A * T).sum(1) / (T * T).sum(1)
        alpha, g2, pc, po = pT(W), pT(quad), pT(cross), pT(only)
        per_k[k] = {"alpha_median": float(alpha.median()), "gamma2_median": float(g2.median()), "gamma_median": float(gamma.median()), "cross_proj_median": float(pc.median()), "cross_negative_fraction": float((pc < 0).float().mean()),
                    "only_proj_median": float(po.median()), "loss_gap_median": float(((alpha - g2) - pc).abs().median()), "c_norm_median": float(cc.norm(dim=1).median()), "attn0_norm_median": float(c["attn0"].norm(dim=1).median()), "attn1_norm_median": float(c["attn1"].norm(dim=1).median())}
        allp.append(pc); allc.append(-cc.norm(dim=1))
    p_, q_ = torch.cat(allp), torch.cat(allc); r = float(((p_ - p_.mean()) * (q_ - q_.mean())).sum() / ((p_ - p_.mean()).norm() * (q_ - q_.mean()).norm()))
    report = {"closure_max": closure, "pearson_crossproj_vs_neg_cnorm": r, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_normalised_table_term_near_full": all(v["gamma2_median"] >= GAMMA_MIN for v in per_k.values()), "pred_c_cross_projection_negative": all(v["cross_negative_fraction"] >= NEG_MIN for v in per_k.values()),
                   "pred_d_cross_projection_is_the_gain_loss": all(v["loss_gap_median"] <= LOSS_TOL for v in per_k.values()), "pred_e_cross_projection_tracks_context_size": r >= R_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_cross_cancellation_result_v301", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
