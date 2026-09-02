#!/usr/bin/env python3
"""Source-invariant response directions of the rung-510 tensor (CPU companion v2).

# BQGATE: EXPERIMENT
# pred_a_exact_mapped_input_and_sanity
# pred_b_nonmacro_material_invariant_direction_identified
# pred_c_identified_direction_term_stability

Parallel-lane CPU analysis (Claude), preregistration frozen 22:58 BEFORE the
rung-510 outcome existed. Computes, by generalized eigenanalysis (GSVD sense),
the term-combinations of MLP10's 253 exact terms whose finite downstream
response is maximal under the native score while source-variation across the
validated gauge (P/Z7/Z8 vs N) stays below a ceiling DERIVED from the bundle's
own cross-half fluctuation. Zero model forwards. Preregistration:
polynomial_causal/MLP10_SOURCE_INVARIANT_SUBSPACE_COMPANION_V2_PREREGISTRATION.md
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_SOURCE_INVARIANT_SUBSPACE_COMPANION_V2_PREREGISTRATION.md"
R510_RESULT = ROOT / "mlp10_observable_predictive_state_quotient_rung510_results.json"
R510_BUNDLE = ROOT / "mlp10_observable_predictive_state_quotient_rung510_bundle.pt"
LEDGER = ROOT / "BILIN18_CONNECTION.md"
OUT = ROOT / "mlp10_source_invariant_subspace_companion_results.json"
HASHES = {
    PREREG: "7884a7927c1f1750468bbcc20dac3cef413ca0f90b0f56399a0ee1f99aa4ac6d",
    R510_RESULT: "16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177",
}
SOURCES = ("N", "P", "Z7", "Z8")
TERMS = 253
COORDS = 36  # 4 context cells + 32 circuit tags
CONTEXT_INDICES = (1, 2, 3, 4)  # CELLS order: all, near, far, one, multiple, off
PERM_SEEDS = tuple(20260905 + i for i in range(16))
MATCH_COSINE = .70
JACCARD_MIN = .5
EIG_REG = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(R510_RESULT.read_text())
    if receipt.get("pred_a_exact_live_singleton_and_substitution_instrument") is not True:
        raise RuntimeError("rung510 pred_a not true")
    bundle_sha = receipt["sufficient_statistics"]["sha256"]
    if sha256(R510_BUNDLE) != bundle_sha:
        raise RuntimeError("rung510 bundle sha mismatch vs receipt")
    ledger = LEDGER.read_text()
    if "## \u00a72642" not in ledger and "## §2642" not in ledger:
        raise RuntimeError("conditioning unmet: section 2642 not yet in ledger")
    return receipt, bundle_sha


def build_tensor():
    """Map the bundle to R[a,p,h,c] (4,253,2,36); return R and mapping notes."""
    b = torch.load(R510_BUNDLE, map_location="cpu", weights_only=False)
    e = b["collections"]["exact_discovery"]
    arms = e["arms"]
    if arms[0] != "intact" or len(arms) != 254:
        raise RuntimeError("unexpected arms layout")
    lo, hi, split = e["bounds"]
    local_split = split - lo
    docs = hi - lo
    halves = ((0, local_split), (local_split, docs))
    task = e["task"].double()            # (4,254,docs,6) per-doc CE sums
    counts = e["task_counts"].double()   # (docs,6)
    cs = e["circuit_sums"].double()      # (4,254,2,2,32) member/control sums
    cc = e["circuit_counts"].double()    # (2,2,32)
    R = torch.zeros(len(SOURCES), TERMS, 2, COORDS, dtype=torch.float64)
    for h, (dlo, dhi) in enumerate(halves):
        cell_counts = counts[dlo:dhi].sum(0)                  # (6,)
        intact = task[:, 0, dlo:dhi, :].sum(1)                # (4,6)
        removed = task[:, 1:, dlo:dhi, :].sum(2)              # (4,253,6)
        effect = (removed - intact[:, None, :]) / cell_counts.clamp_min(1)
        R[:, :, h, :4] = effect[:, :, list(CONTEXT_INDICES)]
        mem = (cs[:, 1:, h, 0, :] - cs[:, 0:1, h, 0, :]) / cc[h, 0, :].clamp_min(1)
        ctl = (cs[:, 1:, h, 1, :] - cs[:, 0:1, h, 1, :]) / cc[h, 1, :].clamp_min(1)
        R[:, :, h, 4:] = mem - ctl
    return R, {"bounds": list(e["bounds"]), "arm0": arms[0],
               "member_index_assumed": 0, "control_index_assumed": 1,
               "context_cell_indices": list(CONTEXT_INDICES)}


def reproduce_materiality(R, receipt):
    """Sanity: recompute rung510's material-node count from R (pooled halves)."""
    pooled = R.mean(2)  # (4,253,36)
    circuit_rms = pooled[:, :, 4:].square().mean(-1).sqrt()
    task_norm = pooled[:, :, :4].square().sum(-1).sqrt()
    material = int(((circuit_rms >= .0005) & (task_norm >= .00025)).sum())
    expected = receipt["analysis"]["discovery_summary"]["material_nodes"]
    return material, expected


def half_analysis(R, h, rng_perms):
    RN = R[0, :, h, :].numpy()                       # (253,36)
    scale = np.sqrt((RN ** 2).mean(0)).clip(1e-30)   # per-coordinate N RMS
    RNs = RN / scale
    deltas = [ (R[a, :, h, :].numpy() / scale) - RNs for a in range(1, 4) ]
    A = np.concatenate(deltas, axis=1)               # (253,108)
    eps = np.sqrt((((R[0, :, 0, :].numpy() - R[0, :, 1, :].numpy())
                    / scale) ** 2).mean())
    ceiling = 2 * eps
    mu = .05 * np.sqrt((RNs ** 2).mean())
    B = RNs @ RNs.T
    C = A @ A.T
    reg = EIG_REG * np.trace(C) / TERMS + 1e-30
    from scipy.linalg import eigh
    vals, vecs = eigh(B, C + reg * np.eye(TERMS))
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    rows = []
    for i in range(TERMS):
        v = vecs[:, i]
        v = v / np.linalg.norm(v)
        resp = np.sqrt(((v @ RNs) ** 2).mean())
        var = np.sqrt(((v @ A) ** 2).mean())
        rows.append((resp, var, v))
    qualifying = [(r, w, v) for r, w, v in rows if r >= mu and w <= ceiling]
    # permutation control: permute term labels of A only
    perm_counts = []
    for seed in rng_perms:
        rng = np.random.default_rng(seed)
        Ap = A[rng.permutation(TERMS)]
        Cp = Ap @ Ap.T
        valsp, vecsp = eigh(B, Cp + reg * np.eye(TERMS))
        orderp = np.argsort(valsp)[::-1]
        cnt = 0
        for i in range(TERMS):
            v = vecsp[:, orderp[i]]
            v = v / np.linalg.norm(v)
            if (np.sqrt(((v @ RNs) ** 2).mean()) >= mu
                    and np.sqrt(((v @ Ap) ** 2).mean()) <= ceiling):
                cnt += 1
        perm_counts.append(cnt)
    macro = np.ones(TERMS) / np.sqrt(TERMS)
    macro_var = float(np.sqrt(((macro @ A) ** 2).mean()))
    macro_resp = float(np.sqrt(((macro @ RNs) ** 2).mean()))
    return {
        "epsilon": float(eps), "ceiling": float(ceiling), "mu": float(mu),
        "k": len(qualifying),
        "qualifying": [(float(r), float(w), v) for r, w, v in qualifying[:24]],
        "perm_counts": perm_counts,
        "perm_q95": float(np.quantile(perm_counts, .95)),
        "macro": {"response": macro_resp, "variation": macro_var,
                  "passes_ceiling": bool(macro_var <= ceiling)},
    }


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_source_invariant_subspace_companion",
                          "model_loaded": False}, indent=2))
        return
    receipt, bundle_sha = validate_inputs()
    if OUT.exists():
        raise RuntimeError("companion output namespace already exists")
    R, mapping = build_tensor()
    material, expected_material = reproduce_materiality(R, receipt)
    halves = [half_analysis(R, h, PERM_SEEDS) for h in (0, 1)]
    # cross-half identification among non-macro qualifying directions
    identified = []
    used = set()
    for i, (r0, w0, v0) in enumerate(halves[0]["qualifying"]):
        best, bj = 0.0, None
        for j, (r1, w1, v1) in enumerate(halves[1]["qualifying"]):
            if j in used:
                continue
            c = abs(float(np.dot(v0, v1)))
            if c > best:
                best, bj = c, j
        if bj is not None and best >= MATCH_COSINE:
            used.add(bj)
            v1 = halves[1]["qualifying"][bj][2]
            t0 = set(np.where(np.abs(v0) >= .5 * np.abs(v0).max())[0].tolist())
            t1 = set(np.where(np.abs(v1) >= .5 * np.abs(v1).max())[0].tolist())
            jac = len(t0 & t1) / max(len(t0 | t1), 1)
            macro_cos = abs(float(v0.sum()) / np.sqrt(TERMS))
            identified.append({
                "half0_index": i, "half1_index": bj, "match_cosine": best,
                "top_term_jaccard": jac, "macro_cosine": macro_cos,
                "is_macro_like": bool(macro_cos >= .9),
                "response": [halves[0]["qualifying"][i][0],
                             halves[1]["qualifying"][bj][0]],
            })
    nonmacro_identified = [d for d in identified if not d["is_macro_like"]]
    pred_a = bool(
        material == expected_material
        and halves[0]["macro"]["passes_ceiling"]
        and halves[1]["macro"]["passes_ceiling"])
    pred_b = bool(
        halves[0]["k"] >= 2 and halves[1]["k"] >= 2
        and halves[0]["k"] > halves[0]["perm_q95"]
        and halves[1]["k"] > halves[1]["perm_q95"]
        and len(nonmacro_identified) >= 1)
    pred_c = bool(pred_b and all(d["top_term_jaccard"] >= JACCARD_MIN
                                 for d in nonmacro_identified))
    strong_null = bool(not pred_a or not pred_b)
    for h in halves:
        h["qualifying"] = [{"response": r, "variation": w}
                           for r, w, _v in h["qualifying"]]
    result = {
        "status": "complete", "rung": "mlp10_source_invariant_subspace_companion",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "exact_linear_invariance_analysis_of_published_statistics",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "rung510_bundle_sha256": bundle_sha,
        "mapping": mapping,
        "material_nodes_reproduced": material,
        "material_nodes_expected": expected_material,
        "halves": halves,
        "identified_directions": identified,
        "nonmacro_identified_count": len(nonmacro_identified),
        'pred_a_exact_mapped_input_and_sanity': pred_a,
        'pred_b_nonmacro_material_invariant_direction_identified': pred_b,
        'pred_c_identified_direction_term_stability': pred_c,
        "strong_null": strong_null,
        "execution_price": {"full_model_forwards": 0, "cpu_only": True},
        "next_step": (
            "instrument_mapping_repair_only" if not pred_a else
            ("invariant_directions_reported_beside_510_route" if not strong_null
             else "closure_only_macro_direction_is_gauge_covariant")),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "material_nodes_reproduced",
                               "material_nodes_expected",
                               "nonmacro_identified_count", "strong_null",
                               "next_step", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))
    print(json.dumps({"k_half0": result["halves"][0]["k"],
                      "k_half1": result["halves"][1]["k"],
                      "perm_q95": [result["halves"][0]["perm_q95"],
                                   result["halves"][1]["perm_q95"]],
                      "macro": [result["halves"][0]["macro"],
                                result["halves"][1]["macro"]]}, indent=2))


if __name__ == "__main__":
    main()
