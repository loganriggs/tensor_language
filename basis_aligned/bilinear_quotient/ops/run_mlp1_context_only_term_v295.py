#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_context_input_is_token_independent pred_c_shared_direction_is_the_context_only_term pred_d_cross_term_has_no_shared_part pred_e_three_term_description_explains_most
"""MLP 1: the shared remainder is the context-only bilinear term (v295). v293: the non-table remainder R of MLP 1's in-context write splits ~35-40%
into one direction shared across tokens and ~60% into token-specific parts; v294: the shared direction is context-specific (not a register direction).
The exact expansion (v289) of R has two terms: cross = Down[(L t)(R c) + (L c)(R t)] and only = Down[(L c)(R c)], with c the normalised-input change
the context makes through attention 0 / 1 and MLP 0. If c is (nearly) the same for every target token, `only` is token-independent and is the shared
direction, and `cross` -- linear in t -- carries the token-specific class structure. Tested at phrase A, lengths 1 / 8 / 64, 7 classes x 32 targets.
Three-term description scored: W = alpha T + only_mean + cross (exact terms, no fit).
PREDICTIONS (scored as written; failures preserved; priors from v293 / v294)
    pred_a_pair_closure                       Down[cross + context-only] = W - T within relative 1e-3 on every row
    pred_b_context_input_is_token_independent median cosine of c with its mean over targets >= 0.90 at every length
    pred_c_shared_direction_is_the_context_only_term  |cos(grand-mean R, mean only)| >= 0.85 at every length
    pred_d_cross_term_has_no_shared_part      the grand mean of the (table-direction-removed) cross term carries <= 0.15 of its energy at every length
    pred_e_three_term_description_explains_most  median energy of W - (alpha T + mean only) that is NOT in the cross term is <= 0.10 of W's energy (i.e. the three named terms are the write)
PRICE (registered maximum): 3 context batches + 1 table batch = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
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
OUT = ROOT / "circuits/followups/mlp1_context_only_term_v295_result.json"
CANDIDATE_ID = "mlp1.token_table.context_only_term_v295"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, C_COS_MIN, ONLY_COS_MIN, CROSS_MEAN_MAX, LEFT_MAX = 1e-3, 0.90, 0.85, 0.15, 0.10
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_context_input_is_token_independent": ">= 0.90 x 3", "pred_c_shared_direction_is_the_context_only_term": ">= 0.85 x 3", "pred_d_cross_term_has_no_shared_part": "<= 0.15 x 3", "pred_e_three_term_description_explains_most": "<= 0.10 x 3"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "c_cos_min": C_COS_MIN, "only_cos_min": ONLY_COS_MIN, "cross_mean_max": CROSS_MEAN_MAX, "left_max": LEFT_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    cosim = torch.nn.functional.cosine_similarity; closure, per_k = 0.0, {}
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; cc = F.rms_norm(c["x1"], (T.shape[-1],)) - n_tab; Lc, Rc = cc @ Lw.T, cc @ Rw.T
        cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
        closure = max(closure, float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max()))
        alpha = (W * T).sum(1) / (T * T).sum(1); R = W - alpha[:, None] * T; g = R.mean(0)
        c_cos = float(cosim(cc, cc.mean(0, keepdim=True).expand_as(cc), dim=1).median())
        only_mean = only.mean(0); only_cos = float(abs(cosim(g, only_mean, dim=0))); only_spread = float(cosim(only, only_mean.unsqueeze(0).expand_as(only), dim=1).median())
        proj = lambda A: A - ((A * T).sum(1) / (T * T).sum(1))[:, None] * T
        Xc = proj(cross); cross_mean_share = float((Xc.mean(0).norm() ** 2 * len(Xc)) / (Xc * Xc).sum())
        left = W - alpha[:, None] * T - only_mean.unsqueeze(0); left_perp = left - ((left * cross).sum(1) / (cross * cross).sum(1))[:, None] * cross
        left_share = float(((left_perp * left_perp).sum(1) / (W * W).sum(1)).median())
        per_k[k] = {"alpha_median": float(alpha.median()), "c_cos_with_mean_median": c_cos, "c_norm_median": float(cc.norm(dim=1).median()), "only_cos_grand_mean": only_cos, "only_cos_with_its_mean_median": only_spread,
                    "only_energy_share_of_R": float(((only * only).sum(1) / (R * R).sum(1)).median()), "cross_mean_share": cross_mean_share, "left_perp_share_median": left_share,
                    "terms_energy_share_of_W": {"alpha_T": float(((alpha[:, None] * T).norm(dim=1) ** 2 / (W * W).sum(1)).median()), "only_mean": float((only_mean.norm() ** 2 / (W * W).sum(1)).median()), "cross": float(((cross * cross).sum(1) / (W * W).sum(1)).median())}}
    report = {"closure_max": closure, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps(report, indent=1, default=float))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_context_input_is_token_independent": all(v["c_cos_with_mean_median"] >= C_COS_MIN for v in per_k.values()), "pred_c_shared_direction_is_the_context_only_term": all(v["only_cos_grand_mean"] >= ONLY_COS_MIN for v in per_k.values()),
                   "pred_d_cross_term_has_no_shared_part": all(v["cross_mean_share"] <= CROSS_MEAN_MAX for v in per_k.values()), "pred_e_three_term_description_explains_most": all(v["left_perp_share_median"] <= LEFT_MAX for v in per_k.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_context_only_term_result_v295", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
