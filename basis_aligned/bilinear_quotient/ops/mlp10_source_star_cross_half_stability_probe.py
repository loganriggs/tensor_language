#!/usr/bin/env python3
"""At what granularity is the MLP10 causal fingerprint cross-half STABLE? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_rung520_material_count
# pred_b_source_star_fingerprint_is_cross_half_stable
# pred_c_stability_beats_circuit_label_permutation_null

Parallel-lane CPU probe resolving the §2655/§2656 route fork: single MLP0 terms are cross-half unstable
(corr 0.106) while the whole-source aggregate is stable. Source stars (22 terms each) are the intermediate
granularity. Measures per-node cross-half Pearson correlation of the 32-circuit member-minus-control fingerprint
on the frozen rung520 discovery bundle. Zero model forwards, zero deployed parameters. Preregistration:
polynomial_causal/MLP10_SOURCE_STAR_CROSS_HALF_STABILITY_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_SOURCE_STAR_CROSS_HALF_STABILITY_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R520_RESULT = ROOT / "mlp10_source_star_causal_quotient_rung520_results.json"
OUT = ROOT / "mlp10_source_star_cross_half_stability_probe_results.json"
HASHES = {
    PREREG: "bb7e5f9e08258bb15241b1ec08de5552844e4135c0d8defbd1eea34967a1a035",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R520_RESULT: "1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b",
}
MATERIAL_NODES = 83
STABLE_BAR = 0.50
TERM_REFERENCE = 0.106     # §2655 single-term cross-half correlation
N_PERM = 200
SEED0 = 5090


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


def corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_source_star_cross_half_stability_probe",
                          "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")

    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    disc = b["collections"]["discovery"]
    cs = np.asarray(disc["circuit_sums"], dtype=np.float64)   # (4,23,2,2,32)
    cc = np.asarray(disc["circuit_counts"], dtype=np.float64)  # (2,2,32)
    task = np.asarray(disc["task"], dtype=np.float64)          # (4,23,248,6)
    tcnt = np.asarray(disc["task_counts"], dtype=np.float64)   # (248,6)
    counts_positive = bool(np.all(cc > 0) and np.all(tcnt >= 0))

    # per-half circuit effect (member-minus-control removal), arm0=intact, member=mc0 control=mc1
    memMean = cs[:, :, :, 0, :] / cc[None, None, :, 0, :]      # (4,23,2,32)
    ctrlMean = cs[:, :, :, 1, :] / cc[None, None, :, 1, :]
    eff_h = (memMean - memMean[:, 0:1]) - (ctrlMean - ctrlMean[:, 0:1])  # (4,23,2,32)
    eff_h = eff_h[:, 1:, :, :]                                 # (4,22,2,32) sources

    # pooled-both-halves effect for material rule (reproduces published count)
    memP = cs[:, :, :, 0, :].sum(2) / cc[:, 0, :].sum(0)[None, None]     # (4,23,32)
    ctrlP = cs[:, :, :, 1, :].sum(2) / cc[:, 1, :].sum(0)[None, None]
    effP = ((memP - memP[:, 0:1]) - (ctrlP - ctrlP[:, 0:1]))[:, 1:, :]   # (4,22,32)
    circ_rms = np.sqrt((effP ** 2).mean(-1))                             # (4,22)
    tMean = task.sum(2) / tcnt.sum(0)[None, None]                        # (4,23,6)
    teffP = (tMean - tMean[:, 0:1])[:, 1:, :]                            # (4,22,6)
    task_norm = np.sqrt((teffP[:, :, 1:5] ** 2).sum(-1))                 # (4,22)
    material = (circ_rms >= .0005) & (task_norm >= .00025)               # (4,22)
    n_material = int(material.sum())

    # per-node cross-half correlation over 32 circuits (material nodes only)
    A = eff_h.reshape(4 * 22, 2, 32)
    mat_flat = material.reshape(4 * 22)
    rhos = np.array([corr(A[i, 0], A[i, 1]) for i in range(A.shape[0])])
    rho_mat = rhos[mat_flat]
    median_rho = float(np.median(rho_mat))

    # circuit-label permutation null on half-1 vector
    rng_meds = []
    for k in range(N_PERM):
        rng = np.random.default_rng(SEED0 + k)
        perm = rng.permutation(32)
        r = np.array([corr(A[i, 0], A[i, 1][perm]) for i in range(A.shape[0])])
        rng_meds.append(float(np.median(r[mat_flat])))
    null_q95 = float(np.quantile(rng_meds, 0.95))

    pred_a = bool(n_material == MATERIAL_NODES and counts_positive
                  and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(median_rho >= STABLE_BAR)
    pred_c = bool(median_rho > null_q95)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "source_star_fingerprint_is_also_cross_half_unstable_instrument_underpowered_raise_N"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "apparent_stability_is_support_geometry_artifact"
    elif not strong_null:
        verdict = "source_level_is_the_stable_granularity_finer_grain_must_pool_within_source_activation_DAS"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_source_star_cross_half_stability_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "cross_half_correlation_of_source_star_circuit_fingerprints_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "median_cross_half_rho_material": median_rho,
        "term_reference_rho_2655": TERM_REFERENCE,
        "permutation_null_q95": null_q95,
        "rho_quartiles_material": [float(np.quantile(rho_mat, q)) for q in (0.25, 0.5, 0.75)],
        "rho_min_max_material": [float(rho_mat.min()), float(rho_mat.max())],
        "bars": {"stable_bar": STABLE_BAR, "material_target": MATERIAL_NODES, "n_perm": N_PERM},
        'pred_a_instrument_reproduces_rung520_material_count': pred_a,
        'pred_b_source_star_fingerprint_is_cross_half_stable': pred_b,
        'pred_c_stability_beats_circuit_label_permutation_null': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "n_material_nodes",
                               "median_cross_half_rho_material", "permutation_null_q95",
                               "term_reference_rho_2655", "rho_quartiles_material", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
