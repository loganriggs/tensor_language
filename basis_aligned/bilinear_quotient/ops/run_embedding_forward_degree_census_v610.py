#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_mean_dominates_mlp1_not_mlp0 pred_b_centered_fold_still_shrinks pred_c_degree_split_by_layer pred_d_diagonal_affine_residual pred_e_exactness
"""Embedding-forward folding, rung 2 (v610): DEGREE CENSUS of the token-folded MLP tensor in centred token coordinates.

v609 found MLP-1's uniform-vocabulary fold collapsing the input 90%-rank 937 -> 116, but 81% of MLP-1's input second moment sits in ONE direction:
a shared near-constant offset (MLP-0's Down_bias + common-mode write). A constant is "the constant coordinate" of Logan's framework, not vocabulary
structure, so this rung splits it off exactly. With x^_t = mu + delta_t (mu = sum_t p_t x^_t, so sum_t p_t delta_t = 0) the token-folded tensor is
    T'[o,t,t'] = c_o + (A_L delta_t)_o + (A_R delta_t')_o + Q_o(delta_t, delta_t'),
    c = Down((L mu) o (R mu)) + bias,  A_L = Down diag(R mu) L,  A_R = Down diag(L mu) R,  Q_o(d,d') = sum_k Down[o,k] (L d)_k (R d')_k.
Under the PRODUCT token measure p_t p_t' the four parts are exactly orthogonal (every cross term carries a lone E[delta] = 0), so the energy split
constant : linear-left : linear-right : quadratic is a clean degree census — computed through DxD / 4608x4608 Grams only. On the DIAGONAL t = t'
(the actual single-token writes) the parts are not orthogonal; that split is computed directly per token (weights x tables, no model forward) and
reported as a 3x3 weighted Gram (const / linear / quadratic), plus the exactness of c + lin + quad against the true write. The centred fold ranks
(Sigma -> Cov) are the fair version of v609's pred_c.
PREDICTIONS (scored as written; failures preserved; uniform weighting unless stated; frequency weighting reported alongside)
    pred_a_mean_dominates_mlp1_not_mlp0   ||mu||^2 / tr(Sigma) >= 0.70 at MLP-1 and <= 0.10 at MLP-0 (v609's caveat mechanism; the embedding table
                                          itself is near-isotropic after rmsnorm, the MLP-1 input is not)
    pred_b_centered_fold_still_shrinks    MLP-1's covariance-folded 90% input-mode rank <= 750 (= 0.80 x the coefficient rank 937): the vocabulary
                                          still exercises materially fewer directions once the constant is removed. Prior: unsure
    pred_c_degree_split_by_layer          product-measure quadratic energy fraction <= 0.25 at MLP-1 and >= 0.50 at MLP-0 (where mu is small the
                                          bilinear term must carry the write; where mu dominates, the linear terms ~ ||mu||^2 Cov beat Cov^2)
    pred_d_diagonal_affine_residual       on the diagonal (actual single-token writes), the quadratic part's share of MLP-1's write energy <= 0.30
                                          (the exact degree-split residual; it upper-bounds the least-squares affine residual). Prior: unsure
    pred_e_exactness                      c + A_L delta + A_R delta + Q(delta, delta) reproduces the true per-token write, rel-L2 <= 1e-6 in fp64,
                                          both MLPs (instrument)
PRICE (registered maximum): 13 batched T=1 manual table forwards (blocks 0-1, the v609 tables; replay already verified there); the diagonal pass is
weights x tables only. 0 real model forwards; 0 backwards; 0 fits. Bar <= 14.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_degree_census_v610_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_degree_census_v610_tensors.pt"
FREQ_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.degree_census_v610"
FORWARDS_MAX = 14
BATCH = 4096
MEAN_HI, MEAN_LO = 0.70, 0.10
CENTERED_RANK_MAX = 750
QUAD_MLP1_MAX, QUAD_MLP0_MIN = 0.25, 0.50
DIAG_QUAD_MAX = 0.30
EXACT_TOL = 1e-6
PREDICTIONS = {"pred_a_mean_dominates_mlp1_not_mlp0": ">= 0.70 / <= 0.10", "pred_b_centered_fold_still_shrinks": "<= 750",
               "pred_c_degree_split_by_layer": "<= 0.25 / >= 0.50", "pred_d_diagonal_affine_residual": "<= 0.30", "pred_e_exactness": "<= 1e-6"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": [0, 1], "batch": BATCH,
            "bars": {"mean_hi": MEAN_HI, "mean_lo": MEAN_LO, "centered_rank_max": CENTERED_RANK_MAX, "quad_mlp1_max": QUAD_MLP1_MAX,
                     "quad_mlp0_min": QUAD_MLP0_MIN, "diag_quad_max": DIAG_QUAD_MAX, "exact_tol": EXACT_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; V = model.config.vocab_size; dev = "cuda"
    with torch.no_grad():
        tabs = EF.single_token_tables(model, BATCH, dev); forwards = tabs["forwards"]
        p_freq, n_tok, n_seen = EF.unigram_weights(FREQ_ROWS, V, dev=dev)
        p_unif = torch.full((V,), 1.0 / V, device=dev, dtype=torch.float64)
        weights = {"uniform": p_unif, "frequency": p_freq}
        report = {"lambdas": {"block0": tabs["lambdas"][0].tolist(), "block1": tabs["lambdas"][1].tolist()}, "unigram": {"tokens": n_tok, "vocab_seen": n_seen},
                  "layers": {}}
        saved = {}
        for l, Xin in ((0, tabs["X0"]), (1, tabs["X1"])):
            block = blocks[l]; L, R, Dw = EF.mlp_weights(block); bias = block.mlp.Down_bias.detach().double(); Gd = Dw.T @ Dw
            xh = EF.rms(Xin).double()
            rep = {}
            for wname, p in weights.items():
                mu = (xh * p[:, None]).sum(0)                                                 # [D]
                delta = xh - mu
                Sigma = EF.second_moment(xh, xh, p); Cov = EF.second_moment(delta, delta, p)
                mean_frac = float(mu.dot(mu) / Sigma.trace())
                # --- product-measure degree census (exactly orthogonal parts) ---
                Lmu, Rmu = L @ mu, R @ mu
                c = Dw @ (Lmu * Rmu) + bias
                A_L = (Dw * Rmu[None, :]) @ L; A_R = (Dw * Lmu[None, :]) @ R                # Down diag(R mu) L, Down diag(L mu) R
                e_const = float(c.dot(c)); e_linL = float((A_L @ Cov * A_L).sum()); e_linR = float((A_R @ Cov * A_R).sum())
                e_quad = float((Gd * (L @ Cov @ L.T) * (R @ Cov @ R.T)).sum())
                e_tot = e_const + e_linL + e_linR + e_quad
                prod = {"const": e_const / e_tot, "lin_left": e_linL / e_tot, "lin_right": e_linR / e_tot, "quad": e_quad / e_tot, "total": e_tot}
                # --- diagonal (actual single-token writes): 3x3 weighted Gram of const / linear / quadratic, plus exactness ---
                A_lin = A_L + A_R
                G = torch.zeros(3, 3, dtype=torch.float64, device=dev); err_num = 0.0; err_den = 0.0
                for s in range(0, V, BATCH):
                    d = delta[s:s + BATCH]; pw = p[s:s + BATCH]
                    lin = d @ A_lin.T; quad = ((d @ L.T) * (d @ R.T)) @ Dw.T
                    x = xh[s:s + BATCH]; true = ((x @ L.T) * (x @ R.T)) @ Dw.T + bias
                    parts = (c[None, :].expand_as(lin), lin, quad)
                    for i in range(3):
                        for j in range(3):
                            G[i, j] += ((parts[i] * parts[j]).sum(-1) * pw).sum()
                    recon = c + lin + quad
                    err_num += float(((recon - true).square().sum(-1) * pw).sum()); err_den += float((true.square().sum(-1) * pw).sum())
                e_diag_tot = float(G.sum())
                diag = {"gram": (G / e_diag_tot).tolist(), "const": float(G[0, 0] / e_diag_tot), "lin": float(G[1, 1] / e_diag_tot),
                        "quad": float(G[2, 2] / e_diag_tot), "cross_const_lin": float(2 * G[0, 1] / e_diag_tot), "cross_const_quad": float(2 * G[0, 2] / e_diag_tot),
                        "cross_lin_quad": float(2 * G[1, 2] / e_diag_tot), "exact_rel_l2": (err_num / err_den) ** 0.5, "total": e_diag_tot}
                cent = EF.fold_ranks(block, Cov)
                rep[wname] = {"mean_fraction_of_second_moment": mean_frac, "mu_norm": float(mu.norm()), "product_measure": prod, "diagonal": diag,
                              "centered_fold": {"input_ranks": cent["input_ranks"], "output_ranks": cent["output_ranks"], "energy_sym": cent["energy_sym"]},
                              "cov_own_ranks": EF.energy_ranks(torch.linalg.eigvalsh(Cov)), "sigma_own_ranks": EF.energy_ranks(torch.linalg.eigvalsh(Sigma))}
                saved[f"mu_mlp{l}_{wname}"] = mu.float().cpu(); saved[f"cov_mlp{l}_{wname}"] = Cov.float().cpu()
                print(f"MLP-{l} {wname:9s}: mean_frac {mean_frac:.3f} | product const/linL/linR/quad = {prod['const']:.3f}/{prod['lin_left']:.3f}/{prod['lin_right']:.3f}/{prod['quad']:.3f}"
                      f" | diagonal const/lin/quad = {diag['const']:.3f}/{diag['lin']:.3f}/{diag['quad']:.3f} (cross c-l {diag['cross_const_lin']:.3f}, c-q {diag['cross_const_quad']:.3f}, l-q {diag['cross_lin_quad']:.3f}; exact {diag['exact_rel_l2']:.1e})"
                      f" | centred fold in/out 90% = {cent['input_ranks']['0.9']}/{cent['output_ranks']['0.9']} (Cov own {rep[wname]['cov_own_ranks']['0.9']})")
            report["layers"][str(l)] = rep
        disk_guard.guard_torch_save(saved, str(OUT_PT), "v610 means and covariances")

    u0, u1 = report["layers"]["0"]["uniform"], report["layers"]["1"]["uniform"]
    predictions = {"pred_a_mean_dominates_mlp1_not_mlp0": u1["mean_fraction_of_second_moment"] >= MEAN_HI and u0["mean_fraction_of_second_moment"] <= MEAN_LO,
                   "pred_b_centered_fold_still_shrinks": u1["centered_fold"]["input_ranks"]["0.9"] <= CENTERED_RANK_MAX,
                   "pred_c_degree_split_by_layer": u1["product_measure"]["quad"] <= QUAD_MLP1_MAX and u0["product_measure"]["quad"] >= QUAD_MLP0_MIN,
                   "pred_d_diagonal_affine_residual": u1["diagonal"]["quad"] <= DIAG_QUAD_MAX,
                   "pred_e_exactness": all(report["layers"][l][w]["diagonal"]["exact_rel_l2"] <= EXACT_TOL for l in ("0", "1") for w in ("uniform", "frequency"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_degree_census_result_v610", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
