#!/usr/bin/env python3
"""Is a low-dim TERM SUBSPACE a circuit-specific MLP0 unit where single terms are not? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_exact_reproduction_of_rung519_bundle
# pred_b_target_term_subspace_generalizes_as_circuit_specific
# pred_c_subspace_localization_is_a_general_reusable_property

Parallel-lane CPU red-team of the rung519 (§2654) single-term null and a scout for the finer-grain / DAS /
reusable-decomposition direction. Zero model forwards, zero deployed parameters. Fits a minimum-norm term
combination on document half 0 to produce a pure per-circuit response and tests out-of-sample selectivity on
half 1, with the single-term baseline reported for contrast. Preregistration:
polynomial_causal/MLP0_TERM_SUBSPACE_CIRCUIT_SPECIFICITY_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP0_TERM_SUBSPACE_CIRCUIT_SPECIFICITY_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp0_one_circuit_interaction_atlas_rung519_bundle.pt"
R519_RESULT = ROOT / "mlp0_one_circuit_interaction_atlas_rung519_results.json"
OUT = ROOT / "mlp0_term_subspace_circuit_specificity_probe_results.json"
HASHES = {
    PREREG: "37bde55a9fc5c5a2321d3d5f867cc35bb9357df85e10947a289894650da923c4",
    BUNDLE: "54a4ce1c465b6b953b54d2fa4e104c055f5446f39f3ac5167f7aae12b320bd8a",
    R519_RESULT: "3eb5188fa65a746a987d4bee851aaed46b08d7ba905b596dd091d01bd29386f6",
}
TARGET_IDX = 8
TARGET_TAG = "r.2.0.2"
TARGET_W0 = 0.003909140586171755
TARGET_W1 = 0.004190039411971824
SELECT_BAR = 2.0          # rung519's own single-term bar -> apples-to-apples
MAJORITY_BAR = 17         # strict majority of 32 circuits


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_hashes():
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def selectivity(profile, k):
    p = np.abs(np.asarray(profile, dtype=np.float64))
    off = np.delete(p, k)
    med = float(np.median(off))
    return float(p[k] / med) if med > 0 else float("inf")


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp0_term_subspace_circuit_specificity_probe",
                          "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")

    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    tags = list(b["discovery_tags"])
    A = np.asarray(b["discovery_effects"]["circuit"], dtype=np.float64)   # (49,2,32)
    W = np.asarray(b["discovery_effects"]["whole_circuit"], dtype=np.float64)  # (2,32)

    shape_ok = (A.shape == (49, 2, 32)) and (W.shape == (2, 32))
    target_ok = (tags[TARGET_IDX] == TARGET_TAG
                 and abs(W[0, TARGET_IDX] - TARGET_W0) < 1e-6
                 and abs(W[1, TARGET_IDX] - TARGET_W1) < 1e-6)

    A0 = A[:, 0, :]   # (49,32)
    A1 = A[:, 1, :]
    # minimum-norm w_k solving A0^T w_k = e_k, all k at once: W_all = pinv(A0^T) I = pinv(A0^T)
    A0T = A0.T                     # (32,49)
    pinvA0T = np.linalg.pinv(A0T)  # (49,32); column k is w_k
    fit_resid = float(np.max(np.abs(A0T @ pinvA0T - np.eye(32))))  # ||A0^T w_k - e_k||_inf over k
    Q1 = A1.T @ pinvA0T            # (32,32): column k is q_k = A1^T w_k

    per_circuit = []
    n_localize = 0
    for k in range(32):
        q = Q1[:, k]
        s1 = selectivity(q, k)
        am = int(np.argmax(np.abs(q)))
        loc = bool(s1 >= SELECT_BAR and am == k)
        n_localize += int(loc)
        per_circuit.append({"idx": k, "tag": tags[k], "S1": s1,
                            "argmax_idx": am, "localizes": loc})

    tgt = per_circuit[TARGET_IDX]
    # single-term baseline on half1 for the target (the object rung519 tested)
    single_S1 = [selectivity(A1[t, :], TARGET_IDX) for t in range(49)]
    single_best = float(np.max(single_S1))
    stability = float(np.corrcoef(A0[:, TARGET_IDX], A1[:, TARGET_IDX])[0, 1])

    pred_a = bool(shape_ok and target_ok and fit_resid < 1e-6
                  and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(tgt["S1"] >= SELECT_BAR and tgt["argmax_idx"] == TARGET_IDX)
    pred_c = bool(n_localize >= MAJORITY_BAR)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "term_subspace_fails_to_localize_target_route_to_activation_DAS"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "target_localizes_but_subspaces_are_bespoke_not_a_general_decomposition"
    elif not strong_null:
        verdict = "low_dim_term_subspaces_are_general_circuit_specific_units_seed_gpu_reuse_rung"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp0_term_subspace_circuit_specificity_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "out_of_sample_selectivity_of_min_norm_term_combinations_not_compression_or_physical_substitution",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "target": {"idx": TARGET_IDX, "tag": TARGET_TAG,
                   "whole_effect_h0": W[0, TARGET_IDX], "whole_effect_h1": W[1, TARGET_IDX],
                   "subspace_S1": tgt["S1"], "subspace_argmax_idx": tgt["argmax_idx"],
                   "single_term_best_S1": single_best,
                   "cross_half_term_stability": stability},
        "fit_residual_inf": fit_resid,
        "n_circuits_localize": n_localize,
        "per_circuit_selectivity": per_circuit,
        "bars": {"select_bar": SELECT_BAR, "majority_bar": MAJORITY_BAR},
        'pred_a_instrument_exact_reproduction_of_rung519_bundle': pred_a,
        'pred_b_target_term_subspace_generalizes_as_circuit_specific': pred_b,
        'pred_c_subspace_localization_is_a_general_reusable_property': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "n_circuits_localize",
                               "target", "fit_residual_inf", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
