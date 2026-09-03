#!/usr/bin/env python3
"""Is MLP0's diffuse source grammar LOW-RANK? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_exact_reproduction_of_517_profiles
# pred_b_diffuse_grammar_is_low_rank
# pred_c_low_rank_structure_stable_group_space

Parallel-lane CPU analysis (Claude). Effective rank of rung517's five
source-relation CE-effect profiles, explaining the redundancy. Zero model
forwards. Preregistration:
polynomial_causal/MLP0_SOURCE_GRAMMAR_RANK_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP0_SOURCE_GRAMMAR_RANK_PROBE_PREREGISTRATION.md"
R517_RESULT = ROOT / "mlp0_source_relation_factorial_rung517_results.json"
OUT = ROOT / "mlp0_source_grammar_rank_probe_results.json"
HASHES = {PREREG: "b2436dba811cb1c7d282ed9d71df4dafff7a324bde8b2b7a303c59619c4b7d7e", R517_RESULT: "c8405a36cab0e8b50d91e3f525bf5a5106a95d2c42447ce9b83ab29378fd8307"}
CORPORA = ("PROSE", "STRUCTURED")
RANK_BAR = 2.5
STAB_BAR = .90

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def analyze(M):
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    p = S ** 2 / (S ** 2).sum()
    eff = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return {"effective_rank": eff, "top1_energy": float(p[0]),
            "top2_energy": float(p[0] + p[1]),
            "singular_values": [float(x) for x in S]}, U[:, 0]

def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for p, e in HASHES.items():
            if not p.is_file() or sha256(p) != e:
                raise RuntimeError(f"frozen hash mismatch: {p}")
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp0_source_grammar_rank_probe",
                          "model_loaded": False}, indent=2))
        return
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    d = json.loads(R517_RESULT.read_text())
    reports, left = {}, {}
    ok = True
    for cn in CORPORA:
        reports[cn] = {}
        for role in ("FIT", "SELECT"):
            M = d["corpora"][cn]["roles"][role]["endpoint_position_ce_profiles"]
            if np.asarray(M).shape != (5, 192):
                ok = False
            rep, l0 = analyze(M)
            reports[cn][role] = rep
            left[(cn, role)] = l0 / (np.linalg.norm(l0) + 1e-30)
    def cos(a, b): return float(abs(np.dot(a, b)))
    stab = {
        "PROSE_fit_select": cos(left[("PROSE", "FIT")], left[("PROSE", "SELECT")]),
        "STRUCTURED_fit_select": cos(left[("STRUCTURED", "FIT")], left[("STRUCTURED", "SELECT")]),
        "prose_vs_structured_select": cos(left[("PROSE", "SELECT")], left[("STRUCTURED", "SELECT")]),
    }
    pred_a = bool(ok and sha256(R517_RESULT) == list(HASHES.values())[1])
    pred_b = bool(reports["PROSE"]["SELECT"]["effective_rank"] <= RANK_BAR
                  and reports["STRUCTURED"]["SELECT"]["effective_rank"] <= RANK_BAR)
    pred_c = bool(all(v >= STAB_BAR for v in stab.values()))
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    result = {
        "status": "complete", "rung": "mlp0_source_grammar_rank_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "effective_rank_of_mlp0_source_relation_effect_profiles",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "reports": reports, "group_loading_stability": stab,
        "group_order": ["SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER"],
        "top_group_loading_prose_select": [float(x) for x in left[("PROSE", "SELECT")]],
        'pred_a_exact_reproduction_of_517_profiles': pred_a,
        'pred_b_diffuse_grammar_is_low_rank': pred_b,
        'pred_c_low_rank_structure_stable_group_space': pred_c,
        "strong_null": strong_null,
        "verdict": ("diffuse_grammar_is_a_stable_low_rank_context_summary"
                    if not strong_null else
                    "diffuse_grammar_is_genuinely_multidimensional_or_unstable"),
        "execution_price": {"full_model_forwards": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null",
                               "group_loading_stability", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))
    for cn in CORPORA:
        r = reports[cn]["SELECT"]
        print(f"{cn} SELECT: eff_rank={r['effective_rank']:.3f} top1={r['top1_energy']:.3f} top2={r['top2_energy']:.3f}")

if __name__ == "__main__":
    main()
