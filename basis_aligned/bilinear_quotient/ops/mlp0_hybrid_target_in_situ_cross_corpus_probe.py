#!/usr/bin/env python3
"""Does the §2688/§2689 token-vs-context transfer contrast replicate IN SITU (real block-0 context)? (CPU, 0 full forwards)

# BQGATE: EXPERIMENT
# pred_a_instrument
# pred_b_token_target_separator_corpus_specific_in_situ
# pred_c_context_target_separator_transportable_in_situ

Reuses mlp0_hybrid_target_in_situ_separability_probe (same seed, same construction) with per-half covariance
accumulation; general Wiener maps per arm; transfer penalties relative to the destination's own residual.
LOWER residual = more faithful. Preregistration: polynomial_causal/MLP0_HYBRID_TARGET_IN_SITU_CROSS_CORPUS_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from receipt import dump
import mlp0_hybrid_target_in_situ_separability_probe as M

ROOT = M.ROOT
PREREG = M.POLY / "MLP0_HYBRID_TARGET_IN_SITU_CROSS_CORPUS_PROBE_PREREGISTRATION.md"
MODULE = ROOT / "ops" / "mlp0_hybrid_target_in_situ_separability_probe.py"
FROZEN = ROOT / "mlp0_hybrid_target_in_situ_separability_probe_results.json"
OUT = ROOT / "mlp0_hybrid_target_in_situ_cross_corpus_probe_results.json"
HASHES = {
    PREREG: "a25ba0dbf5866d7fc850b98519018077d4d400ffe778d8f380a1d6a091d1ecbc",
    MODULE: "71cf276ea113b0cfe4dd4f8c49b278dee378432ae5a1f02bd5a88c13d6174e39",
    M.BLOB: M.HASHES[M.BLOB], M.NAT: M.HASHES[M.NAT], M.CODE: M.HASHES[M.CODE],
}
RUNG = "mlp0_hybrid_target_in_situ_cross_corpus_probe"
HALF_DOCS = 96
_BaseCov = M.Cov                                   # bound before M.Cov is swapped for SplitCov
PER_DOC = len(M.POS)


def check_hashes():
    for p, e in HASHES.items():
        if not p.is_file() or M.sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")


class SplitCov:
    """Same interface as M.Cov; accumulates two half-corpus Cov objects by doc-major row offset."""
    def __init__(self, n):
        self.h = [_BaseCov(n), _BaseCov(n)]; self.off = 0
    def add(self, t, o):
        n = t.shape[0]
        half = (torch.arange(self.off, self.off + n) // PER_DOC) >= HALF_DOCS
        for k in (0, 1):
            m = half == bool(k)
            if m.any():
                self.h[k].add(t[m], o[m])
        self.off += n
    def finish(self):
        return self                       # blocks are pulled by blocks()

def blocks(sc):
    out = {}
    for k in (0, 1):
        c = sc.h[k]; out[f"h{k}"] = (c.Stt.clone(), c.Cto.clone(), c.Coo.clone(), c.mt.clone(), c.mo.clone(), c.cnt)
    S = tuple(a + b for a, b in zip(out["h0"][:5], out["h1"][:5])) + (out["h0"][5] + out["h1"][5],)
    out["all"] = S
    fin = {}
    for name, (Stt, Cto, Coo, mt, mo, cnt) in out.items():
        mt = mt / cnt; mo = mo / cnt
        fin[name] = {"Stt": Stt / cnt - torch.outer(mt, mt), "Cto": Cto / cnt - torch.outer(mt, mo),
                     "Coo": Coo / cnt - torch.outer(mo, mo), "n": cnt}
    return fin


def wiener_map(b):
    Coo = b["Coo"]; ridge = 1e-8 * float(torch.trace(Coo)) / Coo.shape[0]
    return torch.linalg.solve(Coo + ridge * torch.eye(Coo.shape[0], dtype=torch.float64), b["Cto"].T).T   # C_to Coo^{-1}


def residual_under(P, b, Dn):
    Dn = Dn.double()
    tr = lambda X: float(torch.trace(Dn @ X @ Dn.T))
    e_t = tr(b["Stt"])
    return (e_t - 2 * tr(P @ b["Cto"].T) + tr(P @ b["Coo"] @ P.T)) / e_t


def sqrt_psd(S):
    ev, U = torch.linalg.eigh(S)
    return (U * ev.clamp_min(0).sqrt()) @ U.T


def d_response(Pa, Pb, Sh, Dn):
    Dn = Dn.double()
    na, nb = (Dn @ Pa @ Sh).norm(), (Dn @ Pb @ Sh).norm()
    return float((Dn @ (Pa - Pb) @ Sh).norm() / ((na + nb) / 2))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    check_hashes()
    W, sd = M.load_weights()
    nat = torch.load(M.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat)[:, :M.T].long()
    code = torch.load(M.CODE, map_location="cpu"); code = (code["rows"] if isinstance(code, dict) else code)[:, :M.T].long()
    gen = torch.Generator().manual_seed(M.SEED)
    M.Cov = SplitCov                                   # same draws, same order, split accumulation
    B = {}
    for name, rows in (("natural", nat), ("code", code)):
        p, q = M.block0_states(W, rows)
        st = M.hybrid_stats(W, p, q, rows, gen, name)
        B[name] = {"token": blocks(st["cov_tok"]), "context": blocks(st["cov_ctx"])}
        del p, q, st
    Dn = W["Dn"]
    res, own_all = {}, {}
    frozen = json.load(open(FROZEN)) if FROZEN.is_file() else None
    frozen_hash = M.sha256(FROZEN) if FROZEN.is_file() else None
    repro_err, psd_ok, halves_ok = 0.0, True, True
    for tgt in ("token", "context"):
        arms = {"nat": B["natural"][tgt]["all"], "code": B["code"][tgt]["all"],
                "nat_h0": B["natural"][tgt]["h0"], "nat_h1": B["natural"][tgt]["h1"],
                "code_h0": B["code"][tgt]["h0"], "code_h1": B["code"][tgt]["h1"]}
        P = {a: wiener_map(b) for a, b in arms.items()}
        own = {a: residual_under(P[a], arms[a], Dn) for a in arms}
        for a, b in arms.items():
            for m in ("Stt", "Coo"):
                ev = torch.linalg.eigvalsh(b[m]); psd_ok &= bool(ev.min() >= -1e-8 * ev.max())
        halves_ok &= all(arms[a]["n"] >= 10000 for a in ("nat_h0", "nat_h1", "code_h0", "code_h1"))
        if frozen is not None:
            for a, c in (("nat", "natural"), ("code", "code")):
                repro_err = max(repro_err, abs(own[a] - frozen["per_corpus"][c][f"{tgt}_target"]["residual_any_rank"]))
        pen = lambda a, b: residual_under(P[a], arms[b], Dn) - own[b]
        pens = {"nat_to_code": pen("nat", "code"), "code_to_nat": pen("code", "nat"),
                "nat_h0_to_h1": pen("nat_h0", "nat_h1"), "nat_h1_to_h0": pen("nat_h1", "nat_h0"),
                "code_h0_to_h1": pen("code_h0", "code_h1"), "code_h1_to_h0": pen("code_h1", "code_h0")}
        floors = {"natural": max(pens["nat_h0_to_h1"], pens["nat_h1_to_h0"]), "code": max(pens["code_h0_to_h1"], pens["code_h1_to_h0"])}
        Sh = sqrt_psd((arms["nat"]["Coo"] + arms["code"]["Coo"]) / 2)
        d = {"nat_code": d_response(P["nat"], P["code"], Sh, Dn), "nat_h0_h1": d_response(P["nat_h0"], P["nat_h1"], Sh, Dn),
             "code_h0_h1": d_response(P["code_h0"], P["code_h1"], Sh, Dn)}
        res[tgt] = {"own_residual": own, "transfer_penalty": pens, "within_corpus_floor": floors,
                    "penalty_relative_to_destination_own": {"nat_to_code": pens["nat_to_code"] / own["code"], "code_to_nat": pens["code_to_nat"] / own["nat"]},
                    "response_distance": d, "n": {a: arms[a]["n"] for a in arms}}
    pred_a = bool(psd_ok and halves_ok and (frozen is None or repro_err <= 1e-6))
    tk, cx = res["token"], res["context"]
    pred_b = bool(pred_a and tk["transfer_penalty"]["nat_to_code"] >= 0.5 * tk["own_residual"]["code"]
                  and tk["transfer_penalty"]["code_to_nat"] >= 0.5 * tk["own_residual"]["nat"]
                  and tk["transfer_penalty"]["nat_to_code"] >= 3 * tk["within_corpus_floor"]["code"]
                  and tk["transfer_penalty"]["code_to_nat"] >= 3 * tk["within_corpus_floor"]["natural"])
    pred_c = bool(pred_a and cx["transfer_penalty"]["nat_to_code"] <= 0.1 * cx["own_residual"]["code"]
                  and cx["transfer_penalty"]["code_to_nat"] <= 0.1 * cx["own_residual"]["nat"])
    strong_null = bool(not (pred_a and pred_b and pred_c))
    if not pred_a:
        verdict = "instrument_invalid"
    elif pred_b and pred_c:
        verdict = "contrast_replicates_in_situ_token_corpus_specific_context_transportable"
    elif pred_b:
        verdict = "token_corpus_specific_but_context_not_transportable_in_situ"
    elif pred_c:
        verdict = "context_transportable_but_token_not_corpus_specific_in_situ"
    else:
        verdict = "contrast_does_not_replicate_in_situ"
    result = {
        "status": "complete", "rung": RUNG, "owner_lane": "claude_parallel_probe",
        "claim_level": "exact_block0_in_situ_linear_regime_cross_corpus_no_circuit_claim",
        "source_hashes": {str(k): v for k, v in HASHES.items()}, "frozen_in_situ_results_hash": frozen_hash,
        "reproduction_error_vs_frozen": repro_err if frozen is not None else None, "results": res,
        "bars": {"rel_pen_min_token": 0.5, "floor_multiple": 3.0, "rel_pen_max_context": 0.1, "repro_tol": 1e-6,
                 "nulls": {"token_rel_pen": 0.1, "context_rel_pen": 0.5}},
        'pred_a_instrument': pred_a,
        'pred_b_token_target_separator_corpus_specific_in_situ': pred_b,
        'pred_c_context_target_separator_transportable_in_situ': pred_c,
        "strong_null": strong_null, "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "block0_attention_passes_docs": int(nat.shape[0] + code.shape[0]),
                            "backwards": 0, "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({"verdict": verdict, "strong_null": strong_null, "repro_err": result["reproduction_error_vs_frozen"],
                      "token": {k: tk[k] for k in ("own_residual", "transfer_penalty", "response_distance")},
                      "context": {k: cx[k] for k in ("own_residual", "transfer_penalty", "response_distance")},
                      **{k: v for k, v in result.items() if k.startswith("pred_")}, "runtime_s": result["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
