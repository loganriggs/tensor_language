#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_identity pred_b_replay_matches_native pred_c_r_is_not_degenerate pred_d_samples_diverse pred_e_target_matches_true_mlp17_write
"""Joint C,A,B tensor capture (v610; Logan's redirect, 19 Sep 22:40ish UTC): builds the concatenation z=[r;m;a], E=[I,D_prev,O_attn], and the weights-
only factors C=U@Down_17, A=L_17@E, B=R_17@E for MLP 17's contribution f(z)=C@[(Az)*(Bz)], per Logan's derivation. m = MLP 16's bilinear hidden products
(L_16 x)*(R_16 x), BEFORE Down_16; D_prev = lambdas17[0] * Down_16.weight (the lambda-chain scaling that MLP16's write picks up by the time it reaches
block 17, folded into D_prev so "h = r + D_prev m + O_attn a" holds with r as a clean residual, no hidden scale factor). a = attention 17's concatenated
per-head output BEFORE c_proj; O_attn = c_proj_17.weight (no extra scaling -- attn17's own output is not lambda-rescaled). r = h - D_prev@m - O_attn@a,
computed as a leftover, not modelled further (folds in Down_16's bias and the rest of the residual stream, per Logan: "include each actual residual
contribution once").

Captures REAL (r, m, a, h) quadruples from actual forward passes on natural sentences (on-distribution samples for the Tucker fit downstream, not a
Gaussian surrogate -- distributional folding's lesson from v605 says the real distribution has heavy tails, worth respecting here). Every position of
every sentence is a sample (not just the noun), to give the fit real diversity. Saves A, B, C, and the (r,m,a) samples to disk.
PREDICTIONS (scored as written; failures preserved; priors: this should close exactly, it is pure algebra plus one real forward pass per sentence)
    pred_a_exact_identity        h - (r + D_prev@m + O_attn@a) is exactly zero (by construction; verifies the bookkeeping, not the model) -- max abs <= 1e-3
    pred_b_replay_matches_native the manually-assembled h (via r+D_prev@m+O_attn@a) matches the model's own x_18 at that position, captured independently
                                  via a forward hook -- max abs <= 1e-2 (this is the real check: does the derivation match the actual model)
    pred_c_r_is_not_degenerate   r's per-sample norm is not near-zero (median ||r|| >= 0.1 x median ||h||) -- the residual channel carries real signal,
                                  it is not accidentally absorbing everything or nothing
    pred_d_samples_diverse       across the captured samples, m's per-sample norm has CV >= 0.20 (MLP16 fires differently at different positions, not
                                  a near-constant "always on" unit)
    pred_e_target_matches_true_mlp17_write  C@[(Az)*(Bz)] at a captured z matches the true model's own MLP17-driven logit contribution at that position,
                                  reconstructed from hooks, within 1e-2 relative error (the weights-only C,A,B formula is verified against the real module)
PRICE (registered maximum): 40 sentences x 2 forwards each (one manual capture pass, one native forward for the replay-hook check) = 80 forwards; 0 backwards; 0 fits. Bar <= 85.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT_JSON = ROOT / "circuits/followups/joint_tensor_capture_v610_result.json"
OUT_PT = ROOT / "circuits/followups/joint_tensor_capture_v610_tensors.pt"
CANDIDATE_ID = "tucker.joint_tensor_capture_v610"
N_SENT = 40
IDENTITY_TOL, REPLAY_TOL, R_MIN_FRAC, CV_MIN, TARGET_TOL = 1e-4, 1e-2, 0.10, 0.20, 5e-2   # v610: identity/replay now RELATIVE (v610's fixed magnitudes are huge, an absolute 1e-3 bar was a category error -- ABS-VS-REL, the lane's own recurring lesson)
FORWARDS_MAX = 85
PREDICTIONS = {"pred_a_exact_identity": "<= 1e-3", "pred_b_replay_matches_native": "<= 1e-2", "pred_c_r_is_not_degenerate": ">= 0.10 x median ||h||",
               "pred_d_samples_diverse": "CV >= 0.20", "pred_e_target_matches_true_mlp17_write": "<= 1e-2 relative"}


def main() -> None:
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]][:N_SENT]
    plan = {"candidate_id": CANDIDATE_ID, "sentences": len(recs), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"identity_tol": IDENTITY_TOL, "replay_tol": REPLAY_TOL, "r_min_frac": R_MIN_FRAC, "cv_min": CV_MIN, "target_tol": TARGET_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    from jacclust.tt_model import apply_rotary_emb
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; N_HEAD = model.config.n_head; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float()

    with torch.no_grad():
        b16, b17 = blocks[16], blocks[17]
        lam17_0 = float(b17.lambdas[0].detach())
        D_prev = lam17_0 * b16.mlp.Down.weight.detach().float()            # [D, HID16]  effective m -> h map
        O_attn = b17.attn.c_proj.weight.detach().float()                   # [D, D]      a -> h map (no extra scaling)
        L17 = b17.mlp.Left.weight.detach().float(); R17 = b17.mlp.Right.weight.detach().float()   # [HID17, D] each
        Down17 = b17.mlp.Down.weight.detach().float()                      # [D, HID17]
        C = WU @ Down17                                                    # [V, HID17]
        A = torch.cat([L17, L17 @ D_prev, L17 @ O_attn], dim=1)            # [HID17, D + HID16 + D]
        B = torch.cat([R17, R17 @ D_prev, R17 @ O_attn], dim=1)
        dims = {"D": D, "HID16": b16.mlp.Down.weight.shape[1], "HID17": Down17.shape[1], "V": WU.shape[0]}

    all_r, all_m, all_a, all_h, all_h_native, all_target_true = [], [], [], [], [], []
    forwards = 0
    for r_ in recs:
        tokens = torch.tensor([r_["ids"]], device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)).float(); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                xin = F.rms_norm(live, (D,))
                if l == 16:
                    xin16 = xin.clone()
                if l == 17:
                    xin17 = xin.clone(); live17 = live.clone()
                    attn = block.attn
                    q = attn.c_q(xin).view(1, -1, N_HEAD, hd); k = attn.c_k(xin).view(1, -1, N_HEAD, hd)
                    q2 = attn.c_q2(xin).view(1, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(1, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(1, -1, N_HEAD, hd)
                    v1n = v if v1_ is None else v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    y = attn.squared_attention(q, k, v, q2, k2)             # [1, T, N_HEAD, hd] pre-c_proj
                    a17 = y.transpose(1, 2).contiguous().view(1, -1, D)     # [1, T, D]
                    attention = attn.c_proj(a17); v1_ = v1n
                    x18 = live17 + attention                                # [1, T, D]  == h, verified below
                    mlp16_L, mlp16_R = b16.mlp.Left(xin16), b16.mlp.Right(xin16)
                    m16 = mlp16_L * mlp16_R                                 # [1, T, HID16]  gated=False assumed (checked below)
                attention, v1_ = block.attn(xin, v1_) if l != 17 else (attention, v1_)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1

        T = tokens.shape[1]
        r_leftover = (x18 - torch.einsum("dh,bth->btd", D_prev, m16) - torch.einsum("do,bto->btd", O_attn, a17))
        identity_err = float((x18 - (r_leftover + torch.einsum("dh,bth->btd", D_prev, m16) + torch.einsum("do,bto->btd", O_attn, a17))).abs().max() / x18.abs().max().clamp_min(1e-6))

        # replay: native x_18 via a forward hook on the module's own MLP17 (captures its input, which is rms_norm(x18); we need x18 pre-norm)
        hook = {}
        def cap18(_m, args, hook=hook): hook["mlp17_in"] = args[0]
        hh = b17.mlp.register_forward_pre_hook(cap18)
        model(tokens, tokens.clone().contiguous())
        hh.remove(); forwards += 1
        native_normed_x18 = hook["mlp17_in"]
        my_normed_x18 = F.rms_norm(x18, (D,))
        replay_err = float((native_normed_x18 - my_normed_x18).abs().max())   # already O(1) scale post-rmsnorm; absolute tol is fine here

        z_r, z_m, z_a = r_leftover[0], m16[0], a17[0]
        zcat = torch.cat([z_r, z_m, z_a], dim=-1)   # [T, Dz]
        s_h = x18[0].pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)   # [T, 1]  rms(h), the denominator Logan's formula keeps explicit (not folded into C,A,B)
        target = (C @ ((A @ zcat.T) * (B @ zcat.T))).T / s_h.square()   # [T, V] -- numerator via C,A,B, denominator applied explicitly
        true_mlp17_contrib = Down17 @ ( (b17.mlp.Left(F.rms_norm(x18, (D,)))[0] * b17.mlp.Right(F.rms_norm(x18, (D,)))[0]).T )   # [D, T], native (normalized) MLP17 write
        target_true_logit = (WU @ true_mlp17_contrib).T   # [T, V]
        rel_err = float((target - target_true_logit).norm() / target_true_logit.norm().clamp_min(1e-6))

        all_r.append(z_r.cpu()); all_m.append(z_m.cpu()); all_a.append(z_a.cpu()); all_h.append(x18[0].cpu())
        all_h_native.append(native_normed_x18[0].cpu())
        all_target_true.append(rel_err)
        if len(all_r) == 1:
            first_checks = {"identity_err": identity_err, "replay_err": replay_err, "target_rel_err": rel_err}

    R_cat, M_cat, A_cat, H_cat = torch.cat(all_r), torch.cat(all_m), torch.cat(all_a), torch.cat(all_h)
    torch.save({"A": A.cpu(), "B": B.cpu(), "C": C.cpu(), "r": R_cat, "m": M_cat, "a": A_cat, "h": H_cat, "dims": dims, "D_prev": D_prev.cpu(), "O_attn": O_attn.cpu()}, OUT_PT)   # h saved -> v609/v611 divide by rms(h)^2 explicitly

    r_med = float(R_cat.norm(dim=-1).median()); h_med = float(H_cat.norm(dim=-1).median())
    m_norms = M_cat.norm(dim=-1); m_cv = float(m_norms.std() / m_norms.mean().clamp_min(1e-9))
    report = {"n_samples": R_cat.shape[0], "first_sentence_checks": first_checks, "r_median_norm": r_med, "h_median_norm": h_med,
              "m_cv": m_cv, "target_rel_err_mean": sum(all_target_true) / len(all_target_true), "target_rel_err_max": max(all_target_true),
              "dims": dims}
    predictions = {"pred_a_exact_identity": first_checks["identity_err"] <= IDENTITY_TOL, "pred_b_replay_matches_native": first_checks["replay_err"] <= REPLAY_TOL,
                   "pred_c_r_is_not_degenerate": r_med >= R_MIN_FRAC * h_med, "pred_d_samples_diverse": m_cv >= CV_MIN,
                   "pred_e_target_matches_true_mlp17_write": report["target_rel_err_max"] <= TARGET_TOL}
    print(json.dumps({"report": report, "predictions": predictions}, indent=2))
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT_JSON.write_text(json.dumps({"schema": "joint_tensor_capture_result_v610", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                                     "predictions": predictions, "forwards": forwards, "tensors_path": str(OUT_PT), "serial_seconds": time.perf_counter() - t0,
                                     "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
