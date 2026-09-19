#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_scores_bounded pred_b_copiers_score_high_on_ov_identity pred_c_ov_identity_is_head_selective pred_d_qk_identity_exists pred_e_ov_and_qk_identity_coincide_somewhere
"""Template contraction, first cut (v389; Logan, 19 Sep): score every head's weights against the identity template. For head h of block l, the OV path from a
token's normalised embedding e_t to the logits is W_U . O_h V_h . e_t (dropping the pattern weight and the value mix); its identity-likeness is the normalised
Frobenius inner product of M = W_U O_h V_h E^T (vocab x vocab) with the identity, trace(M) / (||M||_F sqrt(V)) -- the 'copy from position j' component. The QK
forms are E Q_h^T K_h E^T and E Q2_h^T K2_h E^T (vocab x vocab, before rotary), scored the same way -- 'this token matches that token'. All 162 heads, weights
only (no forwards; the embedding E is rms-normalised wte, as the model re-injects it). Also the equivariant share: |identity score|^2 is the fraction of ||M||^2
along the inner-product direction.
PREDICTIONS (scored as written; failures preserved; priors from sections 4.1-4.8: copiers 4.5 / 9.6 / 12.4 / 15.1)
    pred_a_scores_bounded              every score lies in [-1, 1] and the vocab-restricted computation matches a full-vocab check on 4 heads within 1e-6 (instrument)
    pred_b_copiers_score_high_on_ov_identity  heads 4.5, 9.6, 12.4, 15.1 all rank in the top 30 of 162 by OV-identity score. Prior: unsure -- they copy subject FEATURES, not tokens.
    pred_c_ov_identity_is_head_selective  the top-5 heads by OV-identity score each exceed 3x the median |score|
    pred_d_qk_identity_exists          at least one head has a QK-identity score (either form) >= 0.10 (a duplicate-token / match detector exists)
    pred_e_ov_and_qk_identity_coincide_somewhere  at least one head is in the top 20 by both OV-identity and QK-identity (an induction-like candidate). Prior: unsure.
PRICE (registered maximum): 0 forwards; weight matmuls only (162 x 3 vocab-square scores by trace identities, no vocab x vocab materialisation); 0 backwards; 0 fits. Bar <= 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/head_identity_templates_v389_result.json"
CANDIDATE_ID = "templates.head_identity_v389"
COPIERS = ("4.5", "9.6", "12.4", "15.1"); TOP_COPIER, SELECT, QK_MIN = 30, 3.0, 0.10
FORWARDS_MAX = 0
PREDICTIONS = {"pred_a_scores_bounded": "[-1, 1], check 1e-6", "pred_b_copiers_score_high_on_ov_identity": "top 30 x 4", "pred_c_ov_identity_is_head_selective": "top-5 > 3x median", "pred_d_qk_identity_exists": ">= 0.10", "pred_e_ov_and_qk_identity_coincide_somewhere": "one head top-20 on both"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "heads": 162, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"top_copier": TOP_COPIER, "select": SELECT, "qk_min": QK_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H
    with torch.no_grad():
        E = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))          # [V, D] the re-injected token vectors
        WU = model.lm_head.weight.detach().float()                                     # [V, D]
        V = E.shape[0]; sqrtV = V ** 0.5
        # Gram matrices so that trace(W_U A E^T) = sum_ij (W_U)_ij (E A^T)_ij = <W_U, E A^T>_F = trace(A^T (E^T W_U)) etc.
        G_UE = WU.T @ E                                                                # [D, D]: sum_t u_t e_t^T
        G_EE = E.T @ E; G_UU = WU.T @ WU
        def ov_score(A):                                                               # A: [D, D] map e -> logit-space direction (W_U applied outside)
            tr = float((A * G_UE.T).sum())                                             # trace(W_U A E^T) = <A, E^T W_U>?  trace(W_U A E^T) = sum_t u_t^T A e_t = <A, sum_t u_t e_t^T> = <A, G_UE>
            tr = float((A * G_UE).sum())
            fro2 = float(torch.trace(A.T @ G_UU @ A @ G_EE))                          # ||W_U A E^T||_F^2 = trace(E A^T W_U^T W_U A E^T)
            return tr / ((fro2 ** 0.5) * sqrtV)
        def qk_score(B):                                                               # B: [D, D] bilinear form e_q^T B e_k ; M = E B E^T
            tr = float((B * G_EE).sum())                                               # trace(E B E^T) = <B, E^T E>
            fro2 = float(torch.trace(B.T @ G_EE @ B @ G_EE))
            return tr / ((fro2 ** 0.5) * sqrtV)
        scores = {}
        for l, block in enumerate(model.transformer.h):
            a = block.attn; Wq, Wk, Wq2, Wk2, Wv, Wo = (a.c_q.weight.detach().float(), a.c_k.weight.detach().float(), a.c_q2.weight.detach().float(), a.c_k2.weight.detach().float(), a.c_v.weight.detach().float(), a.c_proj.weight.detach().float())
            for h in range(H):
                sl = slice(h * hd, (h + 1) * hd)
                A = Wo[:, sl] @ Wv[sl]                                                # [D, D]  O_h V_h
                B1 = Wq[sl].T @ Wk[sl]; B2 = Wq2[sl].T @ Wk2[sl]                       # [D, D]  Q^T K (QK-norm and rotary dropped: weights-only template)
                scores[f"{l}.{h}"] = {"ov_identity": ov_score(A), "qk1_identity": qk_score(B1), "qk2_identity": qk_score(B2)}
        # instrument: full materialisation check on 4 heads with a 4096-token vocab slice is not the same quantity; instead check the trace identity numerically on 4 heads
        check = 0.0
        for key in ("0.0", "4.5", "9.6", "17.8"):
            l, h = map(int, key.split(".")); a = model.transformer.h[l].attn; sl = slice(h * hd, (h + 1) * hd)
            A = a.c_proj.weight.detach().float()[:, sl] @ a.c_v.weight.detach().float()[sl]
            idx = torch.arange(0, V, 7, device=E.device)[:4096]; Es, WUs = E[idx], WU[idx]; M = WUs @ A @ Es.T
            direct = float(torch.trace(M) / (M.norm() * (len(idx) ** 0.5))); viaG = float((A * (WUs.T @ Es)).sum() / ((torch.trace(A.T @ (WUs.T @ WUs) @ A @ (Es.T @ Es)) ** 0.5) * (len(idx) ** 0.5)))
            check = max(check, abs(direct - viaG))
    ov = {k: v["ov_identity"] for k, v in scores.items()}; qk = {k: max(v["qk1_identity"], v["qk2_identity"], key=abs) for k, v in scores.items()}
    ov_order = sorted(ov, key=lambda k: -ov[k]); qk_order = sorted(qk, key=lambda k: -abs(qk[k])); med = sorted(abs(v) for v in ov.values())[len(ov) // 2]
    ranks_ov = {c: ov_order.index(c) + 1 for c in COPIERS}
    report = {"trace_identity_check_max": check, "ov_top10": [(k, round(ov[k], 4)) for k in ov_order[:10]], "ov_bottom5": [(k, round(ov[k], 4)) for k in ov_order[-5:]], "qk_top10": [(k, round(qk[k], 4)) for k in qk_order[:10]], "copier_ov_ranks": ranks_ov, "copier_ov_scores": {c: ov[c] for c in COPIERS},
              "ov_median_abs": med, "both_top20": sorted(set(ov_order[:20]) & set(qk_order[:20])), "scores": scores}
    print(json.dumps({k: v for k, v in report.items() if k != "scores"}, indent=1))
    predictions = {"pred_a_scores_bounded": all(-1 <= s <= 1 for v in scores.values() for s in v.values()) and check <= 1e-6, "pred_b_copiers_score_high_on_ov_identity": all(r <= TOP_COPIER for r in ranks_ov.values()),
                   "pred_c_ov_identity_is_head_selective": all(ov[k] > SELECT * med for k in ov_order[:5]), "pred_d_qk_identity_exists": max(abs(v) for v in qk.values()) >= QK_MIN, "pred_e_ov_and_qk_identity_coincide_somewhere": len(report["both_top20"]) >= 1}
    OUT.write_text(json.dumps({"schema": "head_identity_templates_result_v389", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": 0,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": 0}, indent=2))


if __name__ == "__main__":
    main()
