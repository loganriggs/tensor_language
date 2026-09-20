#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_target_finite pred_b_tucker_converges pred_c_tucker_beats_independent_matrix pred_d_core_is_sparse pred_e_scale_constraint_holds
"""Sparse symmetric Tucker on the joint C,A,B tensor vs independent matrix decomposition (v609; Logan's redirect, 19 Sep). v608 built C=U@Down_17,
A=L_17@E, B=R_17@E and captured real z=[r;m;a] samples from 40 natural sentences (every position). MLP17's true weights-only contribution to the logits
is f(z) = C @ [(Az) * (Bz)] -- this defines, implicitly, an order-3 tensor T_vij = sum_k C_vk (A_ki B_kj + A_kj B_ki)/2 (never materialised: C is
[50304,4608], A and B are [4608,7872], so T would be [50304,7872,7872], ~3.1e12 entries).

TWO decompositions compared on the SAME real z-samples, at matched parameter budgets:
  INDEPENDENT MATRIX baseline: truncate C, A, B each to their OWN top-k SVD, independently, then compose the truncated versions:
    f_base(z) = C_k @ [(A_k z) * (B_k z)]. This is "matrix decomposition" -- exactly what v601/v602 did to C alone, extended here to also truncate
    A and B on their own terms, composed back together, and scored against the TRUE target on real data (not against C alone in isolation).
  JOINT SPARSE SYMMETRIC TUCKER: s = P^T z (P: [7872, r_feat], a SHARED input-feature basis, same P used for both A- and B- sides per Logan's
    symmetric-Tucker form), f_tucker(z) = W @ [s^T G_1 s; ...; s^T G_{r_out} s] (W: [V, r_out], G: [r_out, r_feat, r_feat] sparse, L1-penalised).
    Never materialises T: computed directly from real z-batches via A@z, B@z (cheap [4608,batch] matmuls) then P^T applied to... note P acts on z
    (7872-dim), not on Az/Bz (4608-dim) -- s = P^T z directly (P discovers structure in the INPUT z-space, per Logan's construction), and the model's
    OWN prediction of f_tucker is fit by regression against the true target f(z) on the same real samples, NOT by first forming s^TG s from A,B
    directly (that would require materialising per-k structure); instead G, P, W are fit end to end by gradient descent against the true targets,
    which is the standard way to fit a Tucker model when the full tensor cannot be materialised.
    Scale-evasion guard (Logan's explicit warning): after every optimiser step, P's columns and W's rows are projected back to unit norm; only G
    carries the L1 sparsity penalty, so shrinking-to-evade is not available to G alone (P and W are fixed-scale).
  Both are trained/truncated to the SAME nominal rank budget (r_feat = r_out = K, K in (16, 32, 64)) and compared by held-out (last 20%) relative
  reconstruction error against the true target f(z) on real natural-sentence samples.
PREDICTIONS (scored as written; failures preserved; priors: this is the first run of a brand-new method, most predictions are genuinely unsure)
    pred_a_baseline_target_finite       the true target f(z) on every sample is finite and has nonzero norm (instrument)
    pred_b_tucker_converges             the Tucker fit's training loss is monotonically non-increasing over the last half of training and its relative
                                         decrease over the final 10% of steps is < 1e-3, for every K (3/3)
    pred_c_tucker_beats_independent_matrix  at K=32, the joint Tucker's held-out relative error is <= the independent-matrix baseline's, on the real
                                         natural-sentence samples. Prior: unsure -- this is the actual question Logan asked
    pred_d_core_is_sparse               at K=32, at least 30% of G's entries are within 1e-4 of zero after training (the L1 penalty achieves real
                                         sparsity, not just small-but-dense values)
    pred_e_scale_constraint_holds       P's columns and W's rows have norm within 1e-3 of 1.0 after training, every K (the scale-evasion guard held)
PRICE (registered maximum): 0 forwards (trains on v608's already-captured real samples); GPU tensor ops only; 0 backwards in the BQGATE-price sense
  (autograd training is the deliverable, run to convergence and reported, not a nominating fit). Bar <= 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import dod_battery

ROOT = dod_battery.ROOT
IN_PT = ROOT / "circuits/followups/joint_tensor_capture_v608_tensors.pt"
OUT = ROOT / "circuits/followups/joint_tensor_tucker_fit_v609_result.json"
CANDIDATE_ID = "tucker.joint_tensor_fit_v609"
KS = (16, 32, 64)
STEPS, LR, L1, LOG_EVERY = 3000, 3e-3, 3e-4, 100
FORWARDS_MAX = 0
PREDICTIONS = {"pred_a_baseline_target_finite": "finite, nonzero", "pred_b_tucker_converges": "monotone + converged x 3", "pred_c_tucker_beats_independent_matrix": "tucker <= baseline @ K=32",
               "pred_d_core_is_sparse": ">= 30% near-zero @ K=32", "pred_e_scale_constraint_holds": "norms within 1e-3 of 1.0"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "Ks": list(KS), "steps": STEPS, "lr": LR, "l1": L1, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(IN_PT, map_location=dev)
    A, B, C = data["A"].to(dev).float(), data["B"].to(dev).float(), data["C"].to(dev).float()
    r, m, a = data["r"].to(dev).float(), data["m"].to(dev).float(), data["a"].to(dev).float()
    z = torch.cat([r, m, a], dim=-1)   # [N, 7872]
    N = z.shape[0]; Dz = z.shape[1]; V = C.shape[0]
    perm = torch.randperm(N, generator=torch.Generator(device=dev).manual_seed(0), device=dev)
    z = z[perm]; n_train = int(N * 0.8); z_tr, z_ho = z[:n_train], z[n_train:]

    def true_target(zb):   # zb: [B, Dz] -> [B, V]
        Az = zb @ A.T; Bz = zb @ B.T          # [B, HID17]
        return (Az * Bz) @ C.T                # [B, V]

    with torch.no_grad():
        tgt_tr_full = true_target(z_tr); tgt_ho_full = true_target(z_ho)
    finite_ok = bool(torch.isfinite(tgt_tr_full).all() and torch.isfinite(tgt_ho_full).all()) and float(tgt_tr_full.norm()) > 0

    def independent_matrix_baseline(k):
        Ua, Sa, Vha = torch.linalg.svd(A, full_matrices=False); Ak = (Ua[:, :k] * Sa[:k]) @ Vha[:k, :]
        Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False); Bk = (Ub[:, :k] * Sb[:k]) @ Vhb[:k, :]
        Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False); Ck = (Uc[:, :k] * Sc[:k]) @ Vhc[:k, :]
        with torch.no_grad():
            pred_ho = ((z_ho @ Ak.T) * (z_ho @ Bk.T)) @ Ck.T
        err = float((pred_ho - tgt_ho_full).norm() / tgt_ho_full.norm())
        return err

    def tucker_fit(k, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        P = torch.nn.Parameter((torch.randn(Dz, k, generator=g, device=dev) / Dz ** 0.5))
        Gc = torch.nn.Parameter(torch.randn(k, k, k, generator=g, device=dev) * 0.01)
        W = torch.nn.Parameter(torch.randn(V, k, generator=g, device=dev) / k ** 0.5)
        opt = torch.optim.Adam([P, Gc, W], lr=LR)
        curve = []; batch = min(2048, n_train)
        for step in range(STEPS):
            idx = torch.randint(0, n_train, (batch,), generator=g, device=dev)
            zb = z_tr[idx]; tgt = tgt_tr_full[idx]
            s = zb @ P   # [B, k]
            Gsym = (Gc + Gc.transpose(1, 2)) / 2
            core = torch.einsum("bp,apq,bq->ba", s, Gsym, s)   # [B, k]
            pred = core @ W.T   # [B, V]
            recon_loss = (pred - tgt).square().mean()
            sparsity = Gsym.abs().mean()
            loss = recon_loss + L1 * sparsity
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                P.data = P.data / P.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
                W.data = W.data / W.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
            if step % LOG_EVERY == 0 or step == STEPS - 1:
                curve.append(float(recon_loss.detach()))
        with torch.no_grad():
            Gsym = (Gc + Gc.transpose(1, 2)) / 2
            s_ho = z_ho @ P; core_ho = torch.einsum("bp,apq,bq->ba", s_ho, Gsym, s_ho); pred_ho = core_ho @ W.T
            err = float((pred_ho - tgt_ho_full).norm() / tgt_ho_full.norm())
            sparse_frac = float((Gsym.abs() < 1e-4).float().mean())
            p_norm_ok = bool((P.data.norm(dim=0) - 1).abs().max() < 1e-3); w_norm_ok = bool((W.data.norm(dim=1) - 1).abs().max() < 1e-3)
        window = curve[-max(1, len(curve) // 10):]
        monotone_tail = all(curve[i] >= curve[i + 1] - 1e-6 * max(1.0, curve[i]) for i in range(len(curve) // 2, len(curve) - 1))
        last_decrease = abs((window[0] - window[-1]) / max(window[0], 1e-9)) if len(window) > 1 else 0.0
        converged = monotone_tail and last_decrease < 1e-3
        return {"held_out_relative_error": err, "sparse_fraction": sparse_frac, "p_norm_ok": p_norm_ok, "w_norm_ok": w_norm_ok,
                "loss_curve": curve, "converged": converged, "final_train_loss": curve[-1]}

    results = {}
    for k in KS:
        base_err = independent_matrix_baseline(k)
        tuck = tucker_fit(k)
        results[k] = {"independent_matrix_error": base_err, "tucker": tuck}
        print(f"K={k}: independent_matrix_err={base_err:.4f}  tucker_held_out_err={tuck['held_out_relative_error']:.4f}  "
              f"sparse_frac={tuck['sparse_fraction']:.3f}  converged={tuck['converged']}")

    tucker_converges = all(results[k]["tucker"]["converged"] for k in KS)
    beats_at_32 = results[32]["tucker"]["held_out_relative_error"] <= results[32]["independent_matrix_error"]
    sparse_at_32 = results[32]["tucker"]["sparse_fraction"] >= 0.30
    scale_ok = all(results[k]["tucker"]["p_norm_ok"] and results[k]["tucker"]["w_norm_ok"] for k in KS)
    predictions = {"pred_a_baseline_target_finite": finite_ok, "pred_b_tucker_converges": bool(tucker_converges),
                   "pred_c_tucker_beats_independent_matrix": bool(beats_at_32), "pred_d_core_is_sparse": bool(sparse_at_32), "pred_e_scale_constraint_holds": bool(scale_ok)}
    print(json.dumps(predictions, indent=2))
    OUT.write_text(json.dumps({"schema": "joint_tensor_tucker_fit_result_v609", "candidate_id": CANDIDATE_ID, "plan": plan,
                                "results": {str(k): {"independent_matrix_error": results[k]["independent_matrix_error"], "tucker": results[k]["tucker"]} for k in KS},
                                "predictions": predictions, "forwards": 0, "serial_seconds": time.perf_counter() - t0,
                                "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
