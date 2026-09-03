#!/usr/bin/env python3
"""Is the exact linear separator of R536's context target corpus-specific? (CPU probe, zero forwards)

# BQGATE: EXPERIMENT
# pred_a_instrument_uniform_map_reproduces_frozen_result
# pred_b_cross_corpus_map_disagreement_exceeds_split_half_noise
# pred_c_cross_corpus_transfer_penalty_is_material

Wiener maps (rho=1) under natural vs code token unigram distributions (frozen copy-induction v2 row caches, token ids
only) vs within-corpus split halves. Codex's response distance + transfer penalties. Lower residual = more faithful.
Preregistration: polynomial_causal/MLP0_CONTEXT_TARGET_CROSS_CORPUS_SEPARABILITY_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import torch
from receipt import dump
import mlp0_hybrid_separability_lib as LIB
import mlp0_hybrid_separability_corpus_lib as CL

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP0_CONTEXT_TARGET_CROSS_CORPUS_SEPARABILITY_PROBE_PREREGISTRATION.md"
LIB1 = ROOT / "ops" / "mlp0_hybrid_separability_lib.py"
LIB2 = ROOT / "ops" / "mlp0_hybrid_separability_corpus_lib.py"
FROZEN = ROOT / "mlp0_context_target_linear_separability_probe_results.json"
NAT = Path(CL.NAT); CODE = Path(CL.CODE)
OUT = ROOT / "mlp0_context_target_cross_corpus_separability_probe_results.json"
HASHES = {
    PREREG: "639eaf2788242b9b8e2f86abc79a092b96e6dc0a970f02bcc18b782f9c7f7382", LIB1: "0168d99083c003846b4f14e62dec645e3c801c725201a2bd0c681b0147ae28c8", LIB2: "c1f190905fab4955e26da98e5b026e98faa3f9e3c08988ae0709a69faf62b268", FROZEN: "fd0caa09150e0183d6bb582f1a080e05b452dcf6820b991d8cfb7f4e1ac4bc39", NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1", CODE: "6cf514e75dfd03399f223a9ba5f6ebe5f4b1315bcb839a515e1c19e7b5474bd9",
}
RUNG = "mlp0_context_target_cross_corpus_separability_probe"
RHO = 1.0


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


def stats_for(L, R, p, w):
    Mp, _, _ = CL.weighted_token_moments(p, w)
    St = LIB.cov_gI_given_token_moment(L, R, Mp, RHO)
    Sn = LIB.cov_gC_gaussian(L, R, RHO)
    return St, Sn


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    L, R, D, p = LIB.load_mlp0()
    W = CL.corpus_weights()
    uniform = torch.full((p.shape[0],), 1.0 / p.shape[0], dtype=torch.float64)
    arms = {"uniform": uniform, **{k: W[k] for k in ("natural", "code", "natural_h0", "natural_h1", "code_h0", "code_h1")}}
    S, P, own, psd = {}, {}, {}, {}
    for k, w in arms.items():
        St, Sn = stats_for(L, R, p, w)
        S[k] = (St, Sn)
        P[k] = CL.wiener_map(St, Sn)
        own[k] = CL.residual_under(P[k], St, Sn, D)
        e = torch.linalg.eigvalsh(St); psd[k] = float(e.min() / e.max())
    frozen = json.load(open(FROZEN))["per_rho"]["1.0"]["residual_fraction_output_metric"]
    distinct = {k: int((W[k] > 0).sum()) for k in ("natural", "code")}
    pred_a = bool(abs(own["uniform"] - frozen) <= 1e-4 and min(distinct.values()) >= 2000
                  and min(psd.values()) >= -1e-8)
    Sig = 0.5 * (S["natural"][0] + S["natural"][1] + S["code"][0] + S["code"][1])
    d = {"nat_code": CL.response_distance(P["natural"], P["code"], Sig, D),
         "nat_h0_h1": CL.response_distance(P["natural_h0"], P["natural_h1"], Sig, D),
         "code_h0_h1": CL.response_distance(P["code_h0"], P["code_h1"], Sig, D),
         "uniform_nat": CL.response_distance(P["uniform"], P["natural"], Sig, D),
         "uniform_code": CL.response_distance(P["uniform"], P["code"], Sig, D)}
    noise = max(d["nat_h0_h1"], d["code_h0_h1"])
    pred_b = bool(pred_a and d["nat_code"] >= 2.0 * noise)
    def pen(a, b):
        return CL.residual_under(P[a], *S[b], D) - own[b]
    pens = {"nat_to_code": pen("natural", "code"), "code_to_nat": pen("code", "natural"),
            "nat_h0_to_h1": pen("natural_h0", "natural_h1"), "nat_h1_to_h0": pen("natural_h1", "natural_h0"),
            "code_h0_to_h1": pen("code_h0", "code_h1"), "code_h1_to_h0": pen("code_h1", "code_h0"),
            "uniform_to_nat": pen("uniform", "natural"), "uniform_to_code": pen("uniform", "code")}
    floor_code = max(pens["code_h0_to_h1"], pens["code_h1_to_h0"])
    floor_nat = max(pens["nat_h0_to_h1"], pens["nat_h1_to_h0"])
    pred_c = bool(pred_a and pens["nat_to_code"] >= 0.05 and pens["code_to_nat"] >= 0.05
                  and pens["nat_to_code"] >= 3 * floor_code and pens["code_to_nat"] >= 3 * floor_nat)
    strong_null = bool(not (pred_a and pred_b and pred_c))
    if not pred_a:
        verdict = "instrument_invalid"
    elif pred_b and pred_c:
        verdict = "linear_separator_is_corpus_specific_clause_B_structurally_binding"
    elif pred_b and not pred_c:
        verdict = "maps_differ_beyond_noise_but_transfer_penalty_small"
    elif not pred_b and pred_c:
        verdict = "transfer_penalty_without_map_disagreement_check_metric"
    else:
        verdict = "linear_regime_corpus_robust_clause_B_diagnostic_not_structural"
    result = {
        "status": "complete", "rung": RUNG, "owner_lane": "claude_parallel_probe",
        "claim_level": "exact_weight_space_linear_regime_cross_corpus_under_stated_context_model_no_circuit_claim",
        "source_hashes": {str(k): v for k, v in HASHES.items()},
        "target": "context", "rho": RHO, "distinct_tokens": distinct, "n_tokens": W["n_tokens"],
        "own_residual_output_metric": own, "frozen_uniform_reference": frozen, "psd_min_over_max": psd,
        "response_distance": d, "split_half_noise_d": noise, "transfer_penalty": pens,
        "within_corpus_penalty_floor": {"natural": floor_nat, "code": floor_code},
        "bars": {"d_ratio_min": 2.0, "pen_min": 0.05, "pen_floor_multiple": 3.0, "uniform_repro_tol": 1e-4},
        'pred_a_instrument_uniform_map_reproduces_frozen_result': pred_a,
        'pred_b_cross_corpus_map_disagreement_exceeds_split_half_noise': pred_b,
        'pred_c_cross_corpus_transfer_penalty_is_material': pred_c,
        "strong_null": strong_null, "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0, "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "verdict", "strong_null", "own_residual_output_metric",
                      "response_distance", "transfer_penalty", "runtime_s") or k.startswith("pred_")}, indent=2))


if __name__ == "__main__":
    main()
