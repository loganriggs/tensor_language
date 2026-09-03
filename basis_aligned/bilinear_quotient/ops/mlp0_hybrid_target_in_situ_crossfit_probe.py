#!/usr/bin/env python
"""mlp0_hybrid_target_in_situ_crossfit_probe -- honest (out-of-sample) in-situ separability of the MLP0 hybrid targets.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_token_oos_within_bar pred_c_context_oos_outside_bar
#                     pred_d_token_rank32_oos_capture pred_e_token_cross_corpus_penalty pred_f_context_cross_corpus_transports

Fix for the instrument problem diagnosed in ledger §2691: the sample general Wiener map (4608-dim, 21M entries) overfits
at 12-25k samples, so §2690's in-sample residuals are lower bounds. Here every reported residual is CROSS-FITTED:
ridge chosen by nested validation (fit quarter q0, choose lambda on q1, refit on the half q0+q1, evaluate on the other
half), both directions. Cross-corpus transfer is measured against the cross-fitted own residual of the destination.
Preregistration: polynomial_causal/MLP0_HYBRID_TARGET_IN_SITU_CROSSFIT_PROBE_PREREGISTRATION.md
Reuses the in-situ instrument (ops/mlp0_hybrid_target_in_situ_separability_probe.py, same draws/seed/order).
"""
import hashlib, json, os, sys, time
from pathlib import Path
import torch

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/mlp0_hybrid_target_in_situ_crossfit_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp0_hybrid_target_in_situ_separability_probe as M

ROOT = M.ROOT
PREREG = M.POLY / "MLP0_HYBRID_TARGET_IN_SITU_CROSSFIT_PROBE_PREREGISTRATION.md"
MODULE = ROOT / "ops" / "mlp0_hybrid_target_in_situ_separability_probe.py"
FROZEN = ROOT / "mlp0_hybrid_target_in_situ_separability_probe_results.json"
OUT = ROOT / "mlp0_hybrid_target_in_situ_crossfit_probe_results.json"
HASHES = {
    PREREG: "c3f20d5cbcc5d4796bb2e4bbb8c362a7447a215aba1312e1c7659eef8d3fe106",
    MODULE: "71cf276ea113b0cfe4dd4f8c49b278dee378432ae5a1f02bd5a88c13d6174e39",
    FROZEN: "bacf00a8c574e7c20286d0cf5fece9d438f5b73fc15973052c61a51dc4aa385b",
}
RUNG = "mlp0_hybrid_target_in_situ_crossfit_probe"
NQ = 4; DOCS_PER_Q = 48; PER_DOC = len(M.POS)
LAMBDAS = [1e-8, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]     # x tr(Coo)/dim
KS = [3, 8, 32, 128, 512]
BARS = {"repro_tol": 1e-6, "min_quarter": 5000, "token_oos_max": 0.15, "context_oos_min": 0.15,
        "rank32_capture_min": 0.60, "token_pen_min": 0.05, "context_pen_max": 0.05}
_BaseCov = M.Cov


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or M.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


class QuarterCov:
    def __init__(self, n):
        self.q = [_BaseCov(n) for _ in range(NQ)]; self.off = 0
    def add(self, t, o):
        n = t.shape[0]
        qi = ((torch.arange(self.off, self.off + n) // PER_DOC) // DOCS_PER_Q).clamp(max=NQ - 1)
        for k in range(NQ):
            m = qi == k
            if m.any():
                self.q[k].add(t[m], o[m])
        self.off += n
    def finish(self):
        return self


def raw(c):
    return [c.Stt.clone(), c.Cto.clone(), c.Coo.clone(), c.mt.clone(), c.mo.clone(), c.cnt]


def combine(parts):
    S = [sum(p[i] for p in parts) for i in range(5)] + [sum(p[5] for p in parts)]
    Stt, Cto, Coo, mt, mo, cnt = S; mt = mt / cnt; mo = mo / cnt
    return {"Stt": Stt / cnt - torch.outer(mt, mt), "Cto": Cto / cnt - torch.outer(mt, mo),
            "Coo": Coo / cnt - torch.outer(mo, mo), "n": int(cnt)}


def blocks(qc):
    r = [raw(c) for c in qc.q]
    return {"q0": combine([r[0]]), "q1": combine([r[1]]), "q2": combine([r[2]]), "q3": combine([r[3]]),
            "h0": combine(r[:2]), "h1": combine(r[2:]), "all": combine(r)}


class Fit:
    """Eigendecomposition of a training block's Coo; ridge maps for any lambda are then cheap."""
    def __init__(self, b):
        self.b = b; self.ev, self.U = torch.linalg.eigh(b["Coo"]); self.scale = float(self.ev.sum()) / b["Coo"].shape[0]
        self.CU = b["Cto"] @ self.U                                   # target dim x 4608
    def P(self, lam_rel):
        lam = lam_rel * self.scale
        return (self.CU / (self.ev + lam).clamp_min(1e-300)) @ self.U.T   # C_to (Coo + lam I)^{-1}
    def sqrt_pair(self, lam_rel):
        lam = lam_rel * self.scale; s = (self.ev + lam).clamp_min(1e-300).sqrt()
        return (self.U * s) @ self.U.T, (self.U / s) @ self.U.T


def residual_under(P, b, Dn):
    tr = lambda X: float(torch.trace(Dn @ X @ Dn.T))
    e_t = tr(b["Stt"])
    return (e_t - 2 * tr(P @ b["Cto"].T) + tr(P @ b["Coo"] @ P.T)) / e_t


def residual_under_map(Mk, b, Dn):
    """Mk = Dn P_k already in output coordinates (1152 x 4608)."""
    e_t = float(torch.trace(Dn @ b["Stt"] @ Dn.T))
    return (e_t - 2 * float(torch.trace(Mk @ b["Cto"].T @ Dn.T)) + float(torch.trace(Mk @ b["Coo"] @ Mk.T))) / e_t


def crossfit(B, Dn, fit_q, sel_q, refit_h, eval_h):
    F0 = Fit(B[fit_q])
    grid = {str(l): residual_under(F0.P(l), B[sel_q], Dn) for l in LAMBDAS}
    lam = min(LAMBDAS, key=lambda l: grid[str(l)])
    F1 = Fit(B[refit_h]); P = F1.P(lam)
    oos = residual_under(P, B[eval_h], Dn); ins = residual_under(P, B[refit_h], Dn)
    Bsq, Binv = F1.sqrt_pair(lam); A = Dn @ P @ Bsq
    Ua, S, Vh = torch.linalg.svd(A, full_matrices=False)
    ladder = {}
    for k in KS:
        Ak = (Ua[:, :k] * S[:k]) @ Vh[:k]; Mk = Ak @ Binv
        ladder[str(k)] = {"oos": residual_under_map(Mk, B[eval_h], Dn), "in": residual_under_map(Mk, B[refit_h], Dn)}
    return {"lambda_rel": lam, "selection_grid": grid, "oos": oos, "in_sample_refit": ins, "ladder": ladder,
            "fit": fit_q, "select": sel_q, "refit": refit_h, "eval": eval_h}, P, lam


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
    M.Cov = QuarterCov
    B = {}
    for name, rows in (("natural", nat), ("code", code)):
        p, q = M.block0_states(W, rows)
        st = M.hybrid_stats(W, p, q, rows, gen, name)
        B[name] = {"token": blocks(st["cov_tok"]), "context": blocks(st["cov_ctx"])}
        del p, q, st
    Dn = W["Dn"].double()
    frozen = json.load(open(FROZEN))
    repro_err, min_quarter = 0.0, 10 ** 9
    res = {}
    for corpus in ("natural", "code"):
        res[corpus] = {}
        for tgt in ("token", "context"):
            Bc = B[corpus][tgt]
            min_quarter = min(min_quarter, *[Bc[f"q{i}"]["n"] for i in range(NQ)])
            own_all_unridged = residual_under(Fit(Bc["all"]).P(1e-8), Bc["all"], Dn)
            fz = frozen["per_corpus"][corpus][f"{tgt}_target"]["residual_any_rank"]
            repro_err = max(repro_err, abs(own_all_unridged - fz))
            d0, P0, l0 = crossfit(Bc, Dn, "q0", "q1", "h0", "h1")
            d1, P1, l1 = crossfit(Bc, Dn, "q2", "q3", "h1", "h0")
            Pall = Fit(Bc["all"]).P(l0)
            res[corpus][tgt] = {"in_sample_all_unridged": own_all_unridged, "frozen_2690": fz,
                                "dir0": d0, "dir1": d1, "oos_mean": 0.5 * (d0["oos"] + d1["oos"]),
                                "rank32_capture_oos_mean": 1 - 0.5 * (d0["ladder"]["32"]["oos"] + d1["ladder"]["32"]["oos"]),
                                "ladder_oos_mean": {k: 0.5 * (d0["ladder"][k]["oos"] + d1["ladder"][k]["oos"]) for k in d0["ladder"]},
                                "_Pall": Pall, "_lambda_all": l0}
    # cross-corpus transfer, honest: source map (whole source corpus, its nested lambda) evaluated on each destination half,
    # minus the destination's own cross-fitted residual on that half
    xfer = {}
    for tgt in ("token", "context"):
        xfer[tgt] = {}
        for src, dst in (("natural", "code"), ("code", "natural")):
            Ps = res[src][tgt]["_Pall"]; Bd = B[dst][tgt]
            t_h1 = residual_under(Ps, Bd["h1"], Dn); t_h0 = residual_under(Ps, Bd["h0"], Dn)
            own_h1 = res[dst][tgt]["dir0"]["oos"]; own_h0 = res[dst][tgt]["dir1"]["oos"]
            pen = 0.5 * ((t_h1 - own_h1) + (t_h0 - own_h0))
            xfer[tgt][f"{src}->{dst}"] = {"transfer_oos": 0.5 * (t_h0 + t_h1), "own_oos": 0.5 * (own_h0 + own_h1), "penalty": pen}
    for c in res:
        for t in res[c]:
            res[c][t].pop("_Pall"); res[c][t]["lambda_all"] = res[c][t].pop("_lambda_all")
    tok_nat = res["natural"]["token"]["oos_mean"]; ctx_nat = res["natural"]["context"]["oos_mean"]
    cap32 = res["natural"]["token"]["rank32_capture_oos_mean"]
    tok_pen = [xfer["token"][k]["penalty"] for k in xfer["token"]]; ctx_pen = [xfer["context"][k]["penalty"] for k in xfer["context"]]
    preds = {
        'pred_a_instrument': bool(repro_err <= BARS["repro_tol"] and min_quarter >= BARS["min_quarter"]),
        'pred_b_token_oos_within_bar': bool(tok_nat <= BARS["token_oos_max"]),
        'pred_c_context_oos_outside_bar': bool(ctx_nat > BARS["context_oos_min"]),
        'pred_d_token_rank32_oos_capture': bool(cap32 >= BARS["rank32_capture_min"]),
        'pred_e_token_cross_corpus_penalty': bool(min(tok_pen) >= BARS["token_pen_min"]),
        'pred_f_context_cross_corpus_transports': bool(max(ctx_pen) <= BARS["context_pen_max"]),
    }
    nulls = {"b_null_token_oos_ge_.25": bool(tok_nat >= 0.25), "c_null_context_oos_le_.15": bool(ctx_nat <= 0.15),
             "d_null_capture_le_.40": bool(cap32 <= 0.40), "e_null_token_pen_le_.02_either": bool(min(tok_pen) <= 0.02),
             "f_null_context_pen_ge_.10_either": bool(max(ctx_pen) >= 0.10)}
    out = {"rung": RUNG, "status": "complete", "preds": preds, "nulls": nulls, "bars": BARS, "lambdas_rel": LAMBDAS, "ks": KS,
           "instrument": {"repro_err_vs_2690": repro_err, "min_quarter_samples": min_quarter},
           "headline": {"token_natural_oos": tok_nat, "context_natural_oos": ctx_nat, "token_natural_rank32_capture_oos": cap32,
                        "token_code_oos": res["code"]["token"]["oos_mean"], "context_code_oos": res["code"]["context"]["oos_mean"],
                        "token_penalties": tok_pen, "context_penalties": ctx_pen},
           "per_corpus": res, "transfer": xfer, "price": {"gpu_forwards": 0, "cpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v else M.sha256(k)) for k, v in HASHES.items()}, "script_sha256": M.sha256(SELF)}
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "headline": out["headline"], "instrument": out["instrument"]}, indent=1))



if __name__ == "__main__":
    main()
