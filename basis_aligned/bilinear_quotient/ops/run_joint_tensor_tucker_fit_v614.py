#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_target_finite pred_b_tucker_converges pred_c_tucker_beats_independent_matrix pred_d_core_is_sparse pred_e_scale_constraint_holds
"""Sparse symmetric Tucker on the joint C,A,B tensor vs independent matrix decomposition (v614; Logan's redirect, 19-20 Sep). Supersedes v609/v611/v613:

FIX (v614, after v613's optimisation instability): v613's Tucker fit DIVERGED as K grew (held-out error 1.27 at K=16, 2.82 at K=32, 24.1 at K=64,
none converged) while the independent-matrix baseline stayed sane (0.82-0.90) -- classic exploding-optimisation, not a capacity problem: the true
targets have a large, uncontrolled raw scale (MLP17's contribution before the model's own tanh/30 softcap, not itself bounded), G has k^3 parameters
that grow fast with K, and Adam at lr=3e-3 with no gradient clipping let the core blow up, worse at larger K. Fixed here: targets normalised by their
own training-set RMS before fitting (the relative-error metric is scale-invariant, so this changes only the optimisation's numerics, not what's
measured); gradient-norm clipping (1.0); G initialised at a K-dependent smaller scale (0.01/sqrt(K)) so the initial bilinear form's variance does not
grow with K; a short LR warmup-free cosine decay to steady the late steps.
those saved the fully-composed A,B,C to disk (~1.2GB) and helped fill the instance's disk to 0 bytes free. Here A,B,C are recomputed IN MEMORY from
the small factor weights (Down_17, L_17, R_17, D_prev, O_attn, W_U -- tens of MB total, 0 forwards, seconds) every run, and NOTHING large is written
to disk (disk_guard checked before the one save this script does, which is a small JSON). v612's samples file (57MB: real (r,m,a,h) quadruples from
40 natural sentences, captured before the outage) supplies the real, on-distribution training data.

Method (unchanged from v609/v611): f(z) = C @ [(Az)*(Bz)] / rms(h)^2 (the RMSNorm denominator kept explicit, per Logan's formula and the bug v608/v610
caught). TWO decompositions compared at matched K on real held-out samples:
  INDEPENDENT MATRIX baseline: truncate C, A, B each to their OWN top-k SVD independently, then compose: f_base(z) = C_k@[(A_k z)*(B_k z)] / rms(h)^2.
  JOINT SPARSE SYMMETRIC TUCKER: s=P^T z (P:[7872,K] shared input basis), f_tucker(z) = W@[s^T G_1 s; ...; s^T G_K s] (W:[V,K], G:[K,K,K] sparse,
    L1-penalised), fit end-to-end by SGD against the real targets (the full tensor is never materialised: [50304,7872,7872] ~3.1e12 entries).
    Scale-evasion guard: P's columns and W's rows projected to unit norm after every step; only G carries the sparsity penalty.
PREDICTIONS (unchanged from v609/v611; this is the same experiment, re-run disk-safely)
    pred_a_baseline_target_finite       the true target f(z) on every sample is finite and has nonzero norm (instrument)
    pred_b_tucker_converges             the Tucker fit's training loss is monotonically non-increasing over the last half of training and its relative
                                         decrease over the final 10% of steps is < 1e-3, for every K (3/3)
    pred_c_tucker_beats_independent_matrix  at K=32, the joint Tucker's held-out relative error is <= the independent-matrix baseline's. Prior: unsure
    pred_d_core_is_sparse               at K=32, at least 30% of G's entries are within 1e-4 of zero after training
    pred_e_scale_constraint_holds       P's columns and W's rows have norm within 1e-3 of 1.0 after training, every K
PRICE (registered maximum): 1 forward (to reload the small factor weights; 0 backward passes count toward the GPU-forward price; the model itself
  is loaded but not run on any input); 0 fits in the BQGATE-price sense (the trained decomposition is the deliverable, reported to convergence). Bar <= 2.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import disk_guard
import dod_battery

ROOT = dod_battery.ROOT
IN_PT = ROOT / "circuits/followups/joint_tensor_samples_v612.pt"
OUT = ROOT / "circuits/followups/joint_tensor_tucker_fit_v614_result.json"
CANDIDATE_ID = "tucker.joint_tensor_fit_v614"
KS = (16, 32, 64)
STEPS, LR, L1, LOG_EVERY = 3000, 3e-3, 3e-4, 100
FORWARDS_MAX = 2
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
    small = torch.load(IN_PT, map_location=dev)
    r, m, a, h = small["r"].to(dev).float(), small["m"].to(dev).float(), small["a"].to(dev).float(), small["h"].to(dev).float()
    D_prev, O_attn = small["D_prev"].to(dev).float(), small["O_attn"].to(dev).float()

    # recompute A, B, C IN MEMORY from the model's own weights -- 0 forwards, seconds, nothing large written to disk
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    with torch.no_grad():
        b17 = model.transformer.h[17]
        WU = model.lm_head.weight.detach().float()
        L17 = b17.mlp.Left.weight.detach().float(); R17 = b17.mlp.Right.weight.detach().float(); Down17 = b17.mlp.Down.weight.detach().float()
        C = WU @ Down17                                                    # [V, HID17] -- exists only in GPU memory, never saved
        A = torch.cat([L17, L17 @ D_prev, L17 @ O_attn], dim=1)            # [HID17, D+HID16+D]
        B = torch.cat([R17, R17 @ D_prev, R17 @ O_attn], dim=1)
    forwards = 0   # loading the backend touches the GPU but runs no input through the model; counted as 0 per the registered price

    z = torch.cat([r, m, a], dim=-1)   # [N, 7872]
    N = z.shape[0]; Dz = z.shape[1]; V = C.shape[0]
    s_h2 = h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6).square()   # [N, 1]  rms(h)^2, the explicit RMSNorm denominator
    perm = torch.randperm(N, generator=torch.Generator(device=dev).manual_seed(0), device=dev)
    z = z[perm]; s_h2 = s_h2[perm]; n_train = int(N * 0.8); z_tr, z_ho = z[:n_train], z[n_train:]; s2_tr, s2_ho = s_h2[:n_train], s_h2[n_train:]

    def true_target(zb, s2b):
        Az = zb @ A.T; Bz = zb @ B.T
        return ((Az * Bz) @ C.T) / s2b

    with torch.no_grad():
        tgt_tr_full = true_target(z_tr, s2_tr); tgt_ho_full = true_target(z_ho, s2_ho)
    finite_ok = bool(torch.isfinite(tgt_tr_full).all() and torch.isfinite(tgt_ho_full).all()) and float(tgt_tr_full.norm()) > 0

    def independent_matrix_baseline(k):
        Ua, Sa, Vha = torch.linalg.svd(A, full_matrices=False); Ak = (Ua[:, :k] * Sa[:k]) @ Vha[:k, :]
        Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False); Bk = (Ub[:, :k] * Sb[:k]) @ Vhb[:k, :]
        Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False); Ck = (Uc[:, :k] * Sc[:k]) @ Vhc[:k, :]
        with torch.no_grad():
            pred_ho = (((z_ho @ Ak.T) * (z_ho @ Bk.T)) @ Ck.T) / s2_ho
        return float((pred_ho - tgt_ho_full).norm() / tgt_ho_full.norm())

    tgt_scale = float(tgt_tr_full.std().clamp_min(1e-6))   # normalise for optimisation stability only; relative error below is scale-invariant

    def tucker_fit(k, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        P = torch.nn.Parameter((torch.randn(Dz, k, generator=g, device=dev) / Dz ** 0.5))
        Gc = torch.nn.Parameter(torch.randn(k, k, k, generator=g, device=dev) * (0.01 / k ** 0.5))
        W = torch.nn.Parameter(torch.randn(V, k, generator=g, device=dev) / k ** 0.5)
        opt = torch.optim.Adam([P, Gc, W], lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
        curve = []; batch = min(2048, n_train)
        for step in range(STEPS):
            idx = torch.randint(0, n_train, (batch,), generator=g, device=dev)
            zb = z_tr[idx]; tgt = tgt_tr_full[idx] / tgt_scale
            s = zb @ P
            Gsym = (Gc + Gc.transpose(1, 2)) / 2
            core = torch.einsum("bp,apq,bq->ba", s, Gsym, s)
            pred = core @ W.T
            recon_loss = (pred - tgt).square().mean()
            loss = recon_loss + L1 * Gsym.abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([P, Gc, W], max_norm=1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                P.data = P.data / P.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
                W.data = W.data / W.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
            if step % LOG_EVERY == 0 or step == STEPS - 1:
                curve.append(float(recon_loss.detach()))
        with torch.no_grad():
            Gsym = (Gc + Gc.transpose(1, 2)) / 2
            s_ho = z_ho @ P; core_ho = torch.einsum("bp,apq,bq->ba", s_ho, Gsym, s_ho); pred_ho = (core_ho @ W.T) * tgt_scale
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
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    payload = {"schema": "joint_tensor_tucker_fit_result_v614", "candidate_id": CANDIDATE_ID, "plan": plan,
               "results": {str(k): {"independent_matrix_error": results[k]["independent_matrix_error"], "tucker": results[k]["tucker"]} for k in KS},
               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    payload_bytes = len(json.dumps(payload).encode())
    disk_guard.guard_write(payload_bytes, label=f"OUT json ({payload_bytes/1e6:.1f} MB)")
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
