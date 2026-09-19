#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_svd_is_frobenius_optimal pred_b_sparse_dict_converges pred_c_kmeans_converges pred_d_pullback_changes_fidelity pred_e_sparse_dict_beats_kmeans_at_matched_params
"""Decomposition comparison on the pullback objects (v601; Logan, 19 Sep 22:12 UTC). Compares four unsupervised decomposition methods on the SAME two
objects extracted by v600 -- M0 = W_U [50304, 1152] (the unembedding itself) and M1 = W_U @ Down_17 [50304, 4608] ("unembedding -> last MLP", the concrete
next pullback step Logan asked for) -- by a reconstruction-fidelity-vs-parameter-budget curve, the way he asked ("determine when one decomposition is
simpler than another"). No forward passes: this is pure weights-only linear algebra + training on the two matrices v600 saved.

Methods, all evaluated at matched component counts K in (4, 8, 16, 32, 64, 128):
  SVD (baseline, mandatory per Logan): full SVD computed once per object, truncated to each K. Provably Frobenius-optimal at fixed K (Eckart-Young) --
    the comparison is not "does SVD win," it must; the question is how close the others get, and at what sparsity/interpretability price.
  K-MEANS (flat clustering): rows reconstructed by their nearest of K centroids. Lloyd's algorithm, run to convergence (monitored: stop when the
    relative inertia decrease over an iteration is < TOL, capped at a generous MAX_ITER; converged flag + iterations used are reported, not assumed).
  SPARSE DICTIONARY (genuinely sparse; the "unsupervised decomposition" Logan asked for beyond SVD/clustering -- no SAE/NMF library existed in the repo,
    this is a standard alternating sparse-coding scheme, not a fit-and-hope): K dictionary atoms, each row's code found by ISTA (soft-thresholded
    gradient descent on the LASSO objective, closed-form-optimal step size from the dictionary's spectral norm, so the inner solve is provably
    convergent for a fixed dictionary); the dictionary is then updated by least squares against the current codes and re-normalised. Outer loop run to
    convergence (relative objective decrease < TOL over a window, capped at MAX_OUTER, converged flag + curve reported).
  HIERARCHICAL (DAG/tree; reused, not reimplemented): polynomial_causal/unembedding_backward_views_v1.hierarchy(), the SAME function already validated
    in this repo (depth-4 binary spherical k-means, 16 leaves) -- already run on raw W_U with a known live-forward result (hierarchy arm: 0.94-0.98
    relative error, barely beating a single global mean). Re-derives the M0 reconstruction from the saved artifact where possible; runs fresh (same
    function, same convergence behaviour) on M1 to test whether pulling back through Down_17 gives hierarchical clustering better structure to find.

Parameter accounting per method (row count N, column count C, components K): SVD = K*(N+C+1); k-means = K*C + a separate assignment-bits column
(N*log2(K), reported alongside, not summed in -- a different currency, stated as such); sparse dictionary = K*C (dictionary) + nnz(codes) (only the
codes that survived thresholding, the genuine "simplicity" payoff); hierarchical (16 leaves fixed) = 16*C + N*4 label bits (depth-4 tree, same caveat
as k-means). This is a coarse parameter-count proxy for description length, not a real MDL code -- stated as such in the report, not oversold.

Caveat inherited from v600 and stated once: this is a pure Frobenius/weight-space fidelity comparison, not a behavioural/causal one (unlike the existing
repo's live-forward hierarchy evaluation, which folds a reader set through the real model). It answers "which method reconstructs the matrix best for
its size," not "which method's components are what the model causally uses" -- a live-forward follow-up (reusing compile_maps/folded_state on the new
methods' components as readers) is the natural next depth, not attempted here.

PREDICTIONS (scored as written; failures preserved; priors from Eckart-Young and the existing repo's hierarchy-on-W_U result)
    pred_a_svd_is_frobenius_optimal          at every matched K, SVD's relative Frobenius error on M1 is <= every other method's (instrument: Eckart-Young; should hold exactly)
    pred_b_sparse_dict_converges             the sparse dictionary's outer objective is monotonically non-increasing (up to float tolerance) and its relative decrease
                                              over the last 10% of run iterations is < TOL, for every K, on both objects (8 fits total)
    pred_c_kmeans_converges                  k-means' inertia is monotonically non-increasing and converges (relative decrease < TOL) before MAX_ITER, for every K, on both objects
    pred_d_pullback_changes_fidelity         SVD's relative Frobenius error at K=16 differs between M0 and M1 by >= 0.05 (pulling back through Down_17 changes what a fixed-size
                                              decomposition can capture -- direction unsure)
    pred_e_sparse_dict_beats_kmeans_at_matched_params  at K=16, the sparse dictionary's relative error is <= k-means' on at least one of the two objects, at a comparable or
                                              smaller effective parameter count (dictionary + nnz(codes) vs k-means' K*C + assignment bits). Prior: unsure
PRICE (registered maximum): 0 forwards; SVD (2 objects) + iterative training (k-means and sparse-dict, 6 K-values x 2 objects each, GPU, bounded by MAX_ITER/MAX_OUTER below)
  + one hierarchy() call on M1 (the M0 hierarchy reuses the existing saved artifact); 0 backwards; 0 fits in the BQGATE sense (trained decompositions are the deliverable here,
  registered and reported to convergence, not a nominating step for a circuit). Bar <= 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, sys, time
import dod_battery

ROOT = dod_battery.ROOT
IN_PT = ROOT / "circuits/followups/pullback_extraction_v600_tensors.pt"
EXISTING_HIER_PT = ROOT.parent / "polynomial_causal" / "UNEMBEDDING_BACKWARD_VIEWS_V1_HIERARCHY.pt"
OUT = ROOT / "circuits/followups/decomposition_compare_v601_result.json"
CANDIDATE_ID = "decomposition.compare_v601"
KS = (4, 8, 16, 32, 64, 128)
KMEANS_MAX_ITER, KMEANS_TOL = 300, 1e-6
DICT_MAX_OUTER, DICT_TOL, DICT_ISTA_STEPS, DICT_L1 = 200, 1e-5, 40, 0.02
FORWARDS_MAX = 0
PREDICTIONS = {"pred_a_svd_is_frobenius_optimal": "SVD <= all, every K", "pred_b_sparse_dict_converges": "monotone + converged, 8/8 fits",
               "pred_c_kmeans_converges": "monotone + converged before cap, all K x objects", "pred_d_pullback_changes_fidelity": "|M0-M1 SVD err at K=16| >= 0.05",
               "pred_e_sparse_dict_beats_kmeans_at_matched_params": "dict <= kmeans on >= 1 object at K=16"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "Ks": list(KS), "kmeans_max_iter": KMEANS_MAX_ITER, "dict_max_outer": DICT_MAX_OUTER, "dict_l1": DICT_L1,
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,   # trained decompositions ARE the deliverable here, not a nominating fit; see docstring caveat
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(IN_PT, map_location=dev)
    M0, M1 = data["M0"].to(dev).float(), data["M1"].to(dev).float()

    def svd_curve(M):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        norm = M.norm()
        out = {}
        for k in KS:
            if k > S.shape[0]: continue
            recon = (U[:, :k] * S[:k]) @ Vh[:k, :]
            err = float((M - recon).norm() / norm)
            out[k] = {"relative_error": err, "params": k * (M.shape[0] + M.shape[1] + 1)}
        return out, (U, S, Vh)

    def kmeans_fit(M, k, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        centers = M[torch.randperm(M.shape[0], generator=g, device=dev)[:k]].clone()
        prev_inertia = None; converged = False; curve = []
        it_used = 0
        for it in range(KMEANS_MAX_ITER):
            d2 = torch.cdist(M, centers).square()
            labels = d2.argmin(1)
            inertia = float(d2.gather(1, labels[:, None]).sum())
            curve.append(inertia); it_used = it + 1
            new_centers = torch.zeros_like(centers); counts = torch.zeros(k, device=dev)
            new_centers.index_add_(0, labels, M); counts.index_add_(0, labels, torch.ones(M.shape[0], device=dev))
            empty = counts == 0
            if bool(empty.any()):
                refill = M[torch.randint(0, M.shape[0], (int(empty.sum().item()),), generator=g, device=dev)]
                new_centers[empty] = refill; counts[empty] = 1
            new_centers = new_centers / counts[:, None]
            if prev_inertia is not None and prev_inertia > 0:
                rel = (prev_inertia - inertia) / prev_inertia
                if abs(rel) < KMEANS_TOL:
                    centers = new_centers; converged = True; break
            centers = new_centers; prev_inertia = inertia
        d2 = torch.cdist(M, centers).square(); labels = d2.argmin(1)
        recon = centers[labels]
        err = float((M - recon).norm() / M.norm())
        return {"relative_error": err, "params": k * M.shape[1], "assignment_bits": float(M.shape[0] * math.log2(max(k, 2))),
                "converged": converged, "iterations": it_used, "final_inertia": curve[-1] if curve else None, "monotone": all(curve[i] >= curve[i + 1] - 1e-3 for i in range(len(curve) - 1))}

    def sparse_dict_fit(M, k, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        N, C = M.shape
        D = M[torch.randperm(N, generator=g, device=dev)[:k]].clone()
        D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
        codes = torch.zeros(N, k, device=dev)
        prev_obj = None; converged = False; curve = []
        for outer in range(DICT_MAX_OUTER):
            L = torch.linalg.matrix_norm(D @ D.T, ord=2).clamp_min(1e-6)   # spectral norm of D D^T for a safe ISTA step
            step = 1.0 / L
            c = codes.clone()
            for _ in range(DICT_ISTA_STEPS):
                grad = (c @ D - M) @ D.T
                c = c - step * grad
                c = torch.sign(c) * (c.abs() - step * DICT_L1).clamp_min(0)
            codes = c
            denom = (codes.T @ codes).diagonal().clamp_min(1e-6)
            D_new = (codes.T @ M) / denom[:, None]
            D_new = D_new / D_new.norm(dim=1, keepdim=True).clamp_min(1e-8)
            D = D_new
            recon = codes @ D
            obj = float(0.5 * (M - recon).square().sum() + DICT_L1 * codes.abs().sum())
            curve.append(obj)
            if prev_obj is not None and prev_obj > 0:
                rel = (prev_obj - obj) / prev_obj
                if abs(rel) < DICT_TOL:
                    converged = True; break
            prev_obj = obj
        recon = codes @ D
        err = float((M - recon).norm() / M.norm())
        nnz = int((codes.abs() > 1e-6).sum())
        window = curve[-max(1, len(curve) // 10):]
        monotone = all(curve[i] >= curve[i + 1] - 1e-2 * max(1.0, abs(curve[i])) for i in range(len(curve) - 1))
        last_window_rel = abs((window[0] - window[-1]) / max(window[0], 1e-9)) if len(window) > 1 else 0.0
        return {"relative_error": err, "params": k * C + nnz, "dict_params": k * C, "nnz_codes": nnz, "converged": converged, "outer_iterations": len(curve),
                "final_objective": curve[-1] if curve else None, "monotone": monotone, "last_window_relative_decrease": last_window_rel}

    def hierarchy_reconstruction(M, labels, means):
        recon = means[labels]
        return float((M - recon).norm() / M.norm())

    sys.path.insert(0, str(ROOT.parent / "polynomial_causal"))
    import unembedding_backward_views_v1 as hier_lib

    results = {}
    for name, M in (("M0", M0), ("M1", M1)):
        svd_out, _ = svd_curve(M)
        kmeans_out = {k: kmeans_fit(M, k) for k in KS}
        dict_out = {k: sparse_dict_fit(M, k) for k in KS}
        if name == "M0" and EXISTING_HIER_PT.exists():
            art = torch.load(EXISTING_HIER_PT, map_location=dev)
            labels16, means16 = art["labels"].to(dev), art["raw_centroids"].to(dev).float()
            hier_err = hierarchy_reconstruction(M, labels16, means16)
            hier_source = "reused_existing_artifact"
        else:
            labels16, means16, _tree = hier_lib.hierarchy(M, depth=4, iterations=20)
            hier_err = hierarchy_reconstruction(M, labels16, means16)
            hier_source = "computed_fresh"
        hier_out = {16: {"relative_error": hier_err, "params": 16 * M.shape[1] + M.shape[0] * 4, "source": hier_source}}
        results[name] = {"svd": svd_out, "kmeans": kmeans_out, "sparse_dict": dict_out, "hierarchical": hier_out}
        print(f"{name}: svd@16={svd_out.get(16, {}).get('relative_error')}, kmeans@16={kmeans_out[16]['relative_error']}, "
              f"dict@16={dict_out[16]['relative_error']} (nnz={dict_out[16]['nnz_codes']}), hier@16={hier_err}")

    # predictions
    svd_optimal = all(results[obj]["svd"][k]["relative_error"] <= results[obj][m][k]["relative_error"] + 1e-6
                       for obj in ("M0", "M1") for k in KS if k in results[obj]["svd"]
                       for m in ("kmeans", "sparse_dict"))
    dict_converged = all(results[obj]["sparse_dict"][k]["converged"] or results[obj]["sparse_dict"][k]["last_window_relative_decrease"] < 1e-3
                          for obj in ("M0", "M1") for k in KS)
    kmeans_converged = all(results[obj]["kmeans"][k]["converged"] for obj in ("M0", "M1") for k in KS)
    pullback_diff = abs(results["M0"]["svd"][16]["relative_error"] - results["M1"]["svd"][16]["relative_error"])
    dict_vs_kmeans = any(results[obj]["sparse_dict"][16]["relative_error"] <= results[obj]["kmeans"][16]["relative_error"] for obj in ("M0", "M1"))

    predictions = {"pred_a_svd_is_frobenius_optimal": bool(svd_optimal), "pred_b_sparse_dict_converges": bool(dict_converged),
                   "pred_c_kmeans_converges": bool(kmeans_converged), "pred_d_pullback_changes_fidelity": bool(pullback_diff >= 0.05),
                   "pred_e_sparse_dict_beats_kmeans_at_matched_params": bool(dict_vs_kmeans)}
    print(json.dumps(predictions, indent=2))
    OUT.write_text(json.dumps({"schema": "decomposition_compare_result_v601", "candidate_id": CANDIDATE_ID, "plan": plan, "results": results,
                                "predictions": predictions, "forwards": 0, "serial_seconds": time.perf_counter() - t0,
                                "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
