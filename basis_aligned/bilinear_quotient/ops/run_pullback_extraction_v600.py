#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_lstsq_closure pred_b_shapes_match pred_c_attention17_explains_some_reads pred_d_M1_finite pred_e_M0_M1_frobenius_ratio_sane
# Weights-only extraction feeding the decomposition-comparison work (Logan, 19 Sep 22:12 UTC). See memory logan-decomposition-pullback-direction.md.
"""Pullback objects, unembedding -> last MLP -> last attention (v600). Builds, weights only:
  M0 = W_U                              [V, D]     depth 0: the unembedding itself.
  M1 = W_U @ Down_17                    [V, H]     depth 1: "unembedding -> last MLP" (Logan's literal next step). Column j is the logit direction
                                                     hidden unit j of MLP 17 writes when active.
  C_L, C_R                              [H, 9]     depth 2: "pulled back through the last attention block" as well. MLP 17's j-th bilinear factor reads a
                                                     direction ell_j = L_17[j] (or r_j = R_17[j]) from rmsnorm(x_18); x_18 = live_17 + attn17_out(rmsnorm(live_17)).
                                                     For each of attention 17's 9 heads, pull ell_j back through the head's OV path the same way every writer
                                                     fold today pulled a task direction back through a head (v406 lineage): pull_h(v) = (1-lamb17) * (v @ c_proj17[:, h]) @ c_v17[h].
                                                     Stack the 9 heads' pulled vectors per unit as a [9, D] matrix A_j and solve the batched least-squares
                                                     fit A_j^T c = ell_j; c (9-dim) is unit j's head-coupling profile, and ||A_j^T c - ell_j|| / ||ell_j|| is the
                                                     fraction of the unit's read direction NOT explained by any last-attention-head's OV path (must come from
                                                     deeper layers, via the skip, or via the token-only v1 branch that lamb17 mixes in). C_L stacks the L-branch
                                                     coefficients (4608 x 9), C_R the R-branch; RESID_L, RESID_R the unexplained fractions (4608,).
  Caveat (stated once, applies everywhere below): directions are pulled back treating rmsnorm as direction-preserving, the convention used by every writer
  fold and template contraction run today (v406, v420-v527, v553-v576) -- not exact for the nonlinearity, exact for the linear weight composition.
  The lambda-chain's TOKEN branch (lamb17 * v1, v1 = block 0's value of the same position) is a separate, deeper pullback and is not included in C_L/C_R
  here (it would land in embedding space, not pre-attn17 space, so it cannot join this least-squares fit); it is the natural next depth if this one is useful.
PREDICTIONS (scored as written; failures preserved; these are sanity/structural checks, not circuit claims -- this script only extracts objects for later decomposition)
    pred_a_lstsq_closure                     the batched least-squares residual fractions RESID_L/RESID_R are finite and in [0, 1.5] for every unit (a solvable, well-posed fit)
    pred_b_shapes_match                      M1 is [V, HID], C_L/C_R are [HID, 9], matching the model config read at runtime (instrument)
    pred_c_attention17_explains_some_reads   the median RESID_L and RESID_R are both < 0.95 (attention 17's OV paths explain at least a little of the median unit's read direction; not all of it must come from deeper layers or the token branch)
    pred_d_M1_finite                         every entry of M1 and M0 is finite (no NaN/Inf from the matmul chain)
    pred_e_M0_M1_frobenius_ratio_sane        ||M1||_F / ||M0||_F is in [0.1, 10] (Down_17 does not blow up or annihilate the unembedding's scale). Prior: unsure
PRICE (registered maximum): 0 forwards; weight matmuls + one batched least-squares solve (4608 x [9,1152]) only; 0 backwards; 0 fits. Bar <= 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT_JSON = ROOT / "circuits/followups/pullback_extraction_v600_result.json"
OUT_PT = ROOT / "circuits/followups/pullback_extraction_v600_tensors.pt"
CANDIDATE_ID = "decomposition.pullback_extraction_v600"
LAYER = 17
FORWARDS_MAX = 0
PREDICTIONS = {"pred_a_lstsq_closure": "finite, in [0, 1.5]", "pred_b_shapes_match": "instrument", "pred_c_attention17_explains_some_reads": "median < 0.95 x 2",
               "pred_d_M1_finite": "all finite", "pred_e_M0_M1_frobenius_ratio_sane": "[0.1, 10]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    D = model.config.n_embd; H_HEAD = model.config.n_head; hd = D // H_HEAD; V = model.config.vocab_size
    with torch.no_grad():
        WU = model.lm_head.weight.detach().float()                       # [V, D]
        block = model.transformer.h[LAYER]
        Down = block.mlp.Down.weight.detach().float()                    # [D, HID]
        L17 = block.mlp.Left.weight.detach().float()                     # [HID, D]
        R17 = block.mlp.Right.weight.detach().float()                    # [HID, D]
        HID = Down.shape[1]
        assert L17.shape == (HID, D) and R17.shape == (HID, D)
        cproj = block.attn.c_proj.weight.detach().float()                # [D, D] (out, in); in-columns are the concatenated per-head outputs
        cv = block.attn.c_v.weight.detach().float()                      # [D, D] (out, in); out-rows are the concatenated per-head values
        lamb17 = float(block.attn.lamb.detach())

        M0 = WU.clone()                                                  # [V, D]
        M1 = WU @ Down                                                   # [V, HID]

        # per-head pullback of every MLP-17 input direction (rows of L17, R17) through attention 17's OV path
        A = torch.zeros(HID, H_HEAD, D, device=WU.device)                # A[j, h, :] = pull_h applied to a generic direction; filled per branch below
        def pulled(rows):                                                # rows: [HID, D] -> returns [HID, H_HEAD, D]
            out = torch.zeros(HID, H_HEAD, D, device=rows.device)
            for h in range(H_HEAD):
                cproj_h = cproj[:, h * hd:(h + 1) * hd]                  # [D, hd]
                cv_h = cv[h * hd:(h + 1) * hd, :]                        # [hd, D]
                out[:, h, :] = (1 - lamb17) * ((rows @ cproj_h) @ cv_h)  # [HID, D]
            return out
        AL = pulled(L17); AR = pulled(R17)                               # [HID, 9, D] each

        def fit(rows, A_):
            # batched least squares: for each unit j, solve A_[j]^T (D x 9) @ c = rows[j] (D,) in the least-squares sense
            At = A_.transpose(1, 2)                                      # [HID, D, 9]
            b = rows.unsqueeze(-1)                                       # [HID, D, 1]
            sol = torch.linalg.lstsq(At, b).solution.squeeze(-1)         # [HID, 9]
            recon = torch.einsum("hnd,hn->hd", A_, sol)                  # [HID, D]  (A_[j].T @ sol[j], batched)
            resid = (recon - rows).norm(dim=-1) / rows.norm(dim=-1).clamp_min(1e-8)
            return sol, resid
        C_L, RESID_L = fit(L17, AL)
        C_R, RESID_R = fit(R17, AR)

        out = {"M0": M0.cpu(), "M1": M1.cpu(), "C_L": C_L.cpu(), "C_R": C_R.cpu(), "RESID_L": RESID_L.cpu(), "RESID_R": RESID_R.cpu(),
               "lamb17": lamb17, "V": V, "D": D, "HID": HID, "H_HEAD": H_HEAD, "hd": hd, "layer": LAYER}
        torch.save(out, OUT_PT)
    forwards = 0
    report = {"shapes": {k: list(v.shape) for k, v in out.items() if hasattr(v, "shape")}, "lamb17": lamb17,
              "resid_L_mean": float(RESID_L.mean()), "resid_L_median": float(RESID_L.median()), "resid_R_mean": float(RESID_R.mean()), "resid_R_median": float(RESID_R.median()),
              "resid_L_min": float(RESID_L.min()), "resid_L_max": float(RESID_L.max()), "resid_R_min": float(RESID_R.min()), "resid_R_max": float(RESID_R.max()),
              "M1_frobenius": float(M1.norm()), "M0_frobenius": float(M0.norm())}
    finite_ok = bool(torch.isfinite(M0).all() and torch.isfinite(M1).all())
    lstsq_ok = bool(torch.isfinite(RESID_L).all() and torch.isfinite(RESID_R).all() and RESID_L.min() >= 0 and RESID_L.max() <= 1.5 and RESID_R.min() >= 0 and RESID_R.max() <= 1.5)
    shapes_ok = list(M1.shape) == [V, HID] and list(C_L.shape) == [HID, H_HEAD] and list(C_R.shape) == [HID, H_HEAD]
    ratio = report["M1_frobenius"] / max(report["M0_frobenius"], 1e-9)
    predictions = {"pred_a_lstsq_closure": lstsq_ok, "pred_b_shapes_match": shapes_ok, "pred_c_attention17_explains_some_reads": report["resid_L_median"] < 0.95 and report["resid_R_median"] < 0.95,
                   "pred_d_M1_finite": finite_ok, "pred_e_M0_M1_frobenius_ratio_sane": 0.1 <= ratio <= 10}
    report["M1_M0_frobenius_ratio"] = ratio
    print(json.dumps({"report": report, "predictions": predictions}, indent=2))
    OUT_JSON.write_text(json.dumps({"schema": "pullback_extraction_result_v600", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions,
                                     "forwards": forwards, "tensors_path": str(OUT_PT), "serial_seconds": time.perf_counter() - t0,
                                     "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
