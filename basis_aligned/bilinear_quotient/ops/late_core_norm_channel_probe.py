#!/usr/bin/env python
"""late_core_norm_channel_probe -- is the shared late-MLP write core (§2710/§2713: the residual stream's dominant geometry) a GAIN channel?
Remove the core component of every late MLP write (mlp11-17) and either leave the norm (PLAIN), restore the token residual norm (NORMFIX),
or keep the direction but drop the norm effect (KEEPDIR); k in {16, 128}; same arms on mlp0-6 with their own core as the control stack.
CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_CORE_NORM_CHANNEL_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_core_norm_channel_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_core_norm_channel_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_CORE_NORM_CHANNEL_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_core_readout_alignment_probe_results.json"   # §2713
OUT = ROOT / "late_core_norm_channel_probe_results.json"
HASHES = {PREREG: "c07c1f5834db6239c6c02e169d1223874e673217fda20a3e663d6e2abef9e10a", PRIOR: "20ff21d0bef18132772b47f243f273ab0ebb01f807b72f8de0b6c71aa830c82d",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_core_norm_channel_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KS = [16, 128]
LATE7 = [("mlp", l) for l in range(11, 18)]; EARLY7 = [("mlp", l) for l in range(0, 7)]
PRIOR_BASE = 3.0322401; PRIOR_TW_EFF = 10.004
BARS = {"ce_tol": 1e-4, "eff_tol": 0.5, "a_plain_min": 0.10, "b_ratio": 0.30, "c_ratio": 0.70, "d_ratio": 0.50, "e_mult": 2.0}
NULLS = {"b_ratio": 0.80, "c_ratio": 0.30, "d_ratio": 0.90, "e_mult": 1.0}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


class GAcc:
    """float64 second-moment accumulator on DEV."""
    def __init__(self):
        self.S = torch.zeros(D, D, dtype=torch.float64, device=DEV); self.mu = torch.zeros(D, dtype=torch.float64, device=DEV); self.cnt = 0
    def add(self, w):
        X = w.reshape(-1, D).double(); self.S += X.T @ X; self.mu += X.sum(0); self.cnt += X.shape[0]
    def cov(self):
        mu = self.mu / self.cnt; return mu, (self.S / self.cnt - torch.outer(mu, mu)).cpu()
    def finish(self):
        mu, C = self.cov()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        return {"mu": mu.float(), "U": U.float().to(DEV), "spec": R.spectrum(C), "C": C}


def pooled_basis(bs):
    """One basis from the equal-weight mean of the per-site centred covariances (= PCA of the pooled per-site-centred writes)."""
    C = sum(b["C"] for b in bs) / len(bs)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    return {"U": U.float().to(DEV), "spec": R.spectrum(C), "C": C}


def captured(b, U, k):
    """Fraction of site b's write variance inside span(U[:, :k])."""
    Uk = U[:, :k].cpu().double(); return float(torch.trace(Uk.T @ b["C"] @ Uk) / torch.trace(b["C"]))


@torch.no_grad()
def forward(m, idx, patch=None, collect=None):
    """tt_model semantics; patch: dict site -> fn(w) applied to that site's write; collect(site, w)."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = (t.to(DEV) for t in R.rope(Tn))
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
    def apply(s, w, x):
        if collect is not None:
            collect(s, w)
        return patch[s](w, x) if patch and s in patch else w
    for l, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; h = F.rms_norm(x, (D,))
        def pr(lin):
            return R.rot(F.rms_norm(lin(h).view(B, Tn, NH, HD), (HD,)), cos, sin)
        v = a.c_v(h).view(B, Tn, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        pat = (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q), pr(a.c_k)) / HD) * (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q2), pr(a.c_k2)) / HD)
        pat = pat.masked_fill(~mask, 0.0)
        aw = a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
        x = x + apply(("attn", l), aw, x)
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        x = x + apply(("mlp", l), mw, x) + mlp.Down_bias
    if collect is not None:
        collect(("final", -1), x)
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, patch=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg = forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


def fit(m, rows, sites):
    acc = {s: GAcc() for s in sites}
    for i in range(0, rows.shape[0], CH):
        forward(m, rows[i:i + CH, :TI], collect=lambda s, w: acc[s].add(w) if s in acc else None)
    return {s: acc[s].finish() for s in sites}


def pooled(bs, normalise):
    C = sum((b["C"] / torch.trace(b["C"]) if normalise else b["C"]) for b in bs) / len(bs)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    return {"U": U.double(), "spec": R.spectrum(C)}


def arm(U, k, mode):
    Uk = U[:, :k].to(DEV).float()
    def fn(w, x):
        wp = w - (w @ Uk) @ Uk.T
        if mode == "PLAIN":
            return wp
        xo = x + w; xn = x + wp
        no = xo.norm(dim=-1, keepdim=True) + 1e-30; nn = xn.norm(dim=-1, keepdim=True) + 1e-30
        if mode == "NORMFIX":
            return xn * (no / nn) - x
        return xo * (nn / no) - x          # KEEPDIR
    return fn


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model().to(DEV)
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (8, 257), generator=g); fit_rows, ev = nat[:4].to(DEV), nat[4:].to(DEV)
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV)
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    bases = fit(m, fit_rows, LATE7 + EARLY7); log(stage="fit")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce4 = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ident = abs(float(F.cross_entropy(forward(m, idx, {("mlp", 17): (lambda w, x: w)}).reshape(-1, V), tgt.reshape(-1))) - ce4)
    cores = {"late": pooled([bases[s] for s in LATE7], False), "early": pooled([bases[s] for s in EARLY7], False)}
    stacks = {"late": LATE7, "early": EARLY7}
    res = {st: {k: {} for k in KS} for st in stacks}
    for st in stacks:
        for k in KS:
            for mode in ("PLAIN", "NORMFIX", "KEEPDIR"):
                res[st][k][mode] = ce_of(m, ev, {s: arm(cores[st]["U"], k, mode) for s in stacks[st]}) - ce0
            log(stage="arms", stack=st, k=k, **{m_: round(v, 4) for m_, v in res[st][k].items()})
    L = res["late"]; E = res["early"]
    ratio = lambda d: (d["NORMFIX"] / d["PLAIN"]) if d["PLAIN"] > 0 else float("inf")
    r_late16, r_early16 = ratio(L[16]), ratio(E[16]); r_late128 = ratio(L[128])
    kd16 = (L[16]["KEEPDIR"] / L[16]["PLAIN"]) if L[16]["PLAIN"] > 0 else float("inf")
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(cores["late"]["spec"]["eff_rank"] - PRIOR_TW_EFF) <= BARS["eff_tol"] and L[16]["PLAIN"] >= BARS["a_plain_min"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok and ident <= BARS["ce_tol"]),
        'pred_b_norm_channel': bool(r_late16 <= BARS["b_ratio"]),
        'pred_c_direction_alone_is_not_the_message': bool(kd16 >= BARS["c_ratio"]),
        'pred_d_holds_at_128': bool(r_late128 <= BARS["d_ratio"]),
        'pred_e_early_control': bool(r_early16 >= BARS["e_mult"] * r_late16),
    }
    nulls = {"b_null_normfix_ge_.8_plain": bool(r_late16 >= NULLS["b_ratio"]), "c_null_keepdir_le_.3_plain": bool(kd16 <= NULLS["c_ratio"]),
             "d_null_normfix128_ge_.9_plain": bool(r_late128 >= NULLS["d_ratio"]), "e_null_early_ratio_le_late": bool(r_early16 <= NULLS["e_mult"] * r_late16)}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split, bases from docs 96-191); LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "ks": KS,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "identity_abs_diff": ident, "core_tw_eff_rank": cores["late"]["spec"]["eff_rank"], "prior_core_tw_eff_rank": PRIOR_TW_EFF,
                          "early_core_eff_rank": cores["early"]["spec"]["eff_rank"], "late_plain_16": L[16]["PLAIN"]},
           "baseline_ce_eval": ce0, "ce_added": res, "ratios": {"late_normfix_over_plain_16": r_late16, "late_keepdir_over_plain_16": kd16, "late_normfix_over_plain_128": r_late128,
                                                                 "early_normfix_over_plain_16": r_early16, "early_keepdir_over_plain_16": (E[16]["KEEPDIR"] / E[16]["PLAIN"]) if E[16]["PLAIN"] > 0 else None},
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + 12) + 8, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "ratios", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ratios": out["ratios"], "ce_added": res}, indent=1))


if __name__ == "__main__":
    main()
