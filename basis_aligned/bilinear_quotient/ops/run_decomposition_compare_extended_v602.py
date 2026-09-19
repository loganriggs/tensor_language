#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_error_keeps_dropping pred_b_no_knee_by_1024 pred_c_svd_still_optimal_vs_dict_at_256 pred_d_M0_M1_still_close pred_e_ambient_rank_bound_respected
"""SVD fidelity curve extended to larger K (v602). v601 found high relative error (0.82-0.96) at K<=128 on both M0=W_U and M1=W_U@Down_17 -- expected,
since K<=128 is tiny against the ambient rank (M0's rows live in a <=1152-dim space, M1's in a <=4608-dim space), but Logan needs the SHAPE of the
curve (is there a knee, i.e. does a moderate K start explaining most of the variance) to judge "how simple is simple enough," not just the low-K end.
Here SVD (cheap, computed once per object) at K in (128, 256, 512, 1024, 2048, 4096 [M1 only, near its ambient rank]), plus the sparse dictionary at
K=256 as one additional fidelity-vs-sparsity check beyond v601's range, reusing the SAME saved M0/M1 tensors (no forwards).
PREDICTIONS (scored as written; failures preserved; priors from v601's monotone-decreasing curve)
    pred_a_error_keeps_dropping     SVD's relative error is strictly decreasing in K, every step, both objects (instrument: Eckart-Young monotonicity)
    pred_b_no_knee_by_1024          SVD's relative error at K=1024 is still >= 0.30 on both objects (no small dense basis explains most of either matrix). Prior: unsure
    pred_c_svd_still_optimal_vs_dict_at_256  SVD's error at K=256 is <= the sparse dictionary's at K=256, both objects (Eckart-Young, instrument)
    pred_d_M0_M1_still_close        |M0 - M1 SVD error| at K=256 is < 0.05, replaying v601's pred_d finding at a higher K
    pred_e_ambient_rank_bound_respected  SVD's relative error at K = min(ambient_rank, 4096) is <= 0.05 (a decomposition using (nearly) full rank must reconstruct almost exactly; instrument)
PRICE (registered maximum): 0 forwards; two full-rank SVDs (already amortised, re-derived once each) + one sparse-dict fit at K=256 x 2 objects; 0 backwards; 0 fits (reported to convergence). Bar <= 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import dod_battery

ROOT = dod_battery.ROOT
IN_PT = ROOT / "circuits/followups/pullback_extraction_v600_tensors.pt"
OUT = ROOT / "circuits/followups/decomposition_compare_extended_v602_result.json"
CANDIDATE_ID = "decomposition.compare_extended_v602"
KS_COMMON = (128, 256, 512, 1024)
DICT_K = 256
DICT_MAX_OUTER, DICT_TOL, DICT_ISTA_STEPS, DICT_L1 = 200, 1e-5, 40, 0.02
FORWARDS_MAX = 0
PREDICTIONS = {"pred_a_error_keeps_dropping": "strictly decreasing", "pred_b_no_knee_by_1024": ">= 0.30 x 2", "pred_c_svd_still_optimal_vs_dict_at_256": "svd <= dict x 2",
               "pred_d_M0_M1_still_close": "< 0.05", "pred_e_ambient_rank_bound_respected": "<= 0.05"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "Ks_common": list(KS_COMMON), "dict_k": DICT_K, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(IN_PT, map_location=dev)
    M0, M1 = data["M0"].to(dev).float(), data["M1"].to(dev).float()

    def sparse_dict_fit(M, k, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        N, C = M.shape
        D = M[torch.randperm(N, generator=g, device=dev)[:k]].clone()
        D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
        codes = torch.zeros(N, k, device=dev)
        prev_obj = None; converged = False; curve = []
        for outer in range(DICT_MAX_OUTER):
            L = torch.linalg.matrix_norm(D @ D.T, ord=2).clamp_min(1e-6); step = 1.0 / L
            c = codes.clone()
            for _ in range(DICT_ISTA_STEPS):
                grad = (c @ D - M) @ D.T
                c = c - step * grad
                c = torch.sign(c) * (c.abs() - step * DICT_L1).clamp_min(0)
            codes = c
            denom = (codes.T @ codes).diagonal().clamp_min(1e-6)
            D_new = (codes.T @ M) / denom[:, None]; D_new = D_new / D_new.norm(dim=1, keepdim=True).clamp_min(1e-8); D = D_new
            recon = codes @ D
            obj = float(0.5 * (M - recon).square().sum() + DICT_L1 * codes.abs().sum())
            curve.append(obj)
            if prev_obj is not None and prev_obj > 0 and abs((prev_obj - obj) / prev_obj) < DICT_TOL:
                converged = True; break
            prev_obj = obj
        recon = codes @ D; err = float((M - recon).norm() / M.norm()); nnz = int((codes.abs() > 1e-6).sum())
        return {"relative_error": err, "nnz_codes": nnz, "converged": converged, "outer_iterations": len(curve)}

    results = {}
    for name, M in (("M0", M0), ("M1", M1)):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        norm = M.norm(); ambient_rank = int(S.shape[0])
        ks = list(KS_COMMON) + ([ambient_rank] if ambient_rank not in KS_COMMON and ambient_rank <= 4608 else [])
        svd_out = {}
        for k in ks:
            k = min(k, ambient_rank)
            recon = (U[:, :k] * S[:k]) @ Vh[:k, :]
            svd_out[k] = {"relative_error": float((M - recon).norm() / norm)}
        dict_out = {DICT_K: sparse_dict_fit(M, DICT_K)}
        results[name] = {"svd": svd_out, "sparse_dict": dict_out, "ambient_rank": ambient_rank}
        print(name, "ambient_rank", ambient_rank, {k: round(v["relative_error"], 4) for k, v in svd_out.items()}, "dict@256", round(dict_out[DICT_K]["relative_error"], 4))

    svd_M0 = sorted(results["M0"]["svd"].items()); svd_M1 = sorted(results["M1"]["svd"].items())
    mono_M0 = all(svd_M0[i][1]["relative_error"] > svd_M0[i + 1][1]["relative_error"] for i in range(len(svd_M0) - 1))
    mono_M1 = all(svd_M1[i][1]["relative_error"] > svd_M1[i + 1][1]["relative_error"] for i in range(len(svd_M1) - 1))
    e1024_M0 = results["M0"]["svd"].get(1024, {}).get("relative_error", 1.0); e1024_M1 = results["M1"]["svd"].get(1024, {}).get("relative_error", 1.0)
    e256_M0 = results["M0"]["svd"][256]["relative_error"]; e256_M1 = results["M1"]["svd"][256]["relative_error"]
    d256_M0 = results["M0"]["sparse_dict"][256]["relative_error"]; d256_M1 = results["M1"]["sparse_dict"][256]["relative_error"]
    ar_M0, ar_M1 = results["M0"]["ambient_rank"], results["M1"]["ambient_rank"]
    e_full_M0 = results["M0"]["svd"][ar_M0]["relative_error"]; e_full_M1 = results["M1"]["svd"][min(ar_M1, 4608)]["relative_error"]
    predictions = {"pred_a_error_keeps_dropping": bool(mono_M0 and mono_M1), "pred_b_no_knee_by_1024": bool(e1024_M0 >= 0.30 and e1024_M1 >= 0.30),
                   "pred_c_svd_still_optimal_vs_dict_at_256": bool(e256_M0 <= d256_M0 + 1e-6 and e256_M1 <= d256_M1 + 1e-6),
                   "pred_d_M0_M1_still_close": bool(abs(e256_M0 - e256_M1) < 0.05), "pred_e_ambient_rank_bound_respected": bool(e_full_M0 <= 0.05 and e_full_M1 <= 0.05)}
    print(json.dumps(predictions, indent=2))
    OUT.write_text(json.dumps({"schema": "decomposition_compare_extended_result_v602", "candidate_id": CANDIDATE_ID, "plan": plan, "results": results,
                                "predictions": predictions, "forwards": 0, "serial_seconds": time.perf_counter() - t0,
                                "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
