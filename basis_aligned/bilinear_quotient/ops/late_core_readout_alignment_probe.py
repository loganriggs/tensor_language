#!/usr/bin/env python
"""late_core_readout_alignment_probe -- WHAT is the shared late-MLP write dictionary of §2710? Weight-space overlap of the pooled late core
(mlp11-17; trace-weighted as in §2710 and trace-normalised) with the lm_head top right-singular subspace LM_k, with the final residual
stream PCA XPCA_k, and with the early-stack control core (mlp0-6); plus the read-energy ratio of late Left/Right on the core. Only the
96-doc fit and the 64-doc baseline check touch the GPU. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_CORE_READOUT_ALIGNMENT_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_core_readout_alignment_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_core_readout_alignment_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_CORE_READOUT_ALIGNMENT_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_mlp_shared_write_dictionary_probe_results.json"   # §2710
OUT = ROOT / "late_core_readout_alignment_probe_results.json"
HASHES = {PREREG: "f02761f31f546fc3909b601f92bfc03c5c800c9ee8bb0428eee8e36764e06f32", PRIOR: "b932d545c7c16113d85f48ab794798d2d8ad4e0c92976ac327e9c6a54e4d5f98",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_core_readout_alignment_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KS = [16, 128]
LATE7 = [("mlp", l) for l in range(11, 18)]; EARLY7 = [("mlp", l) for l in range(0, 7)]; READERS = list(range(12, 18))
PRIOR_BASE = 3.0322401; PRIOR_TW_EFF = 10.004
BARS = {"ce_tol": 1e-4, "eff_tol": 0.5, "orth_tol": 1e-4, "chance_lo": 0.08, "chance_hi": 0.15, "b_min": 0.60, "c_mult": 1.5, "d_er_min": 1.5, "d_n_min": 5, "e_slack": 0.10}
NULLS = {"b_max": 0.25, "c_mult": 1.1, "d_er_max": 1.1, "d_n_min": 5, "e_slack": 0.10}


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
    def apply(s, w):
        if collect is not None:
            collect(s, w)
        return patch[s](w) if patch and s in patch else w
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
        x = x + apply(("attn", l), aw)
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        x = x + apply(("mlp", l), mw) + mlp.Down_bias
    if collect is not None:
        collect(("final", -1), x)
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg = forward(m, idx)
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


def top_right(W, k=None):
    """Top right-singular directions of W [n, D] via eigh(W^T W)."""
    G = W.double().T @ W.double(); ev, U = torch.linalg.eigh(G); return U.flip(1), R.spectrum(G)


def ov(U, j, Vb, k):
    """Fraction of span(U[:, :j]) inside span(V[:, :k]); chance k/D."""
    A = U[:, :j].cpu().double().T @ Vb[:, :k].cpu().double(); return float((A * A).sum() / j)


def orth_err(U, k):
    Uk = U[:, :k].cpu().double(); return float((Uk.T @ Uk - torch.eye(k, dtype=torch.float64)).abs().max())


def read_energy_ratio(m, l, U, k):
    Uk = U[:, :k].to(DEV).float(); L = m.transformer.h[l].mlp.Left.weight; Rw = m.transformer.h[l].mlp.Right.weight
    on = ((L @ Uk) ** 2).sum() + ((Rw @ Uk) ** 2).sum(); tot = (L ** 2).sum() + (Rw ** 2).sum()
    return float((on / k) / (tot / D))


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
    sites = LATE7 + EARLY7 + [("final", -1)]
    bases = fit(m, fit_rows, sites); log(stage="fit")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    late = {s: bases[s] for s in LATE7}; early = {s: bases[s] for s in EARLY7}
    core_tw = pooled(late.values(), False); core_tn = pooled(late.values(), True); early_tn = pooled(early.values(), True)
    xpca = {"U": bases[("final", -1)]["U"].double(), "spec": bases[("final", -1)]["spec"]}
    W = m.lm_head.weight.detach()
    lm_U, lm_spec = top_right(W)
    g = torch.Generator().manual_seed(0); rnd, _ = torch.linalg.qr(torch.randn(D, 128, generator=g, dtype=torch.float64))
    subs = {"CORE_TW": core_tw["U"], "CORE_TN": core_tn["U"], "EARLY_TN": early_tn["U"], "XPCA": xpca["U"], "LM": lm_U, "RANDOM": rnd}
    ovs = {}
    for a in ("CORE_TW", "CORE_TN", "EARLY_TN", "XPCA", "RANDOM"):
        for j in (16, 128):
            for b in ("LM", "XPCA", "CORE_TW"):
                if a == b: continue
                ovs[f"ov({a}_{j},{b}_128)"] = ov(subs[a], j, subs[b], 128)
    log(stage="overlaps", **{k: round(v, 4) for k, v in ovs.items() if "LM_128" in k})
    er = {l: read_energy_ratio(m, l, core_tn["U"], 128) for l in READERS}
    er_loo = {}
    for l in READERS:
        prior = [late[("mlp", q)] for q in range(11, l)]
        er_loo[l] = read_energy_ratio(m, l, pooled(prior, True)["U"], 128)
    er_early = {l: read_energy_ratio(m, l, early_tn["U"], 128) for l in READERS}
    er_rnd = {l: read_energy_ratio(m, l, rnd, 128) for l in READERS}
    log(stage="read_energy", er=er, er_leave_own_out=er_loo)
    chance = ovs["ov(RANDOM_128,LM_128)"]
    orth = max(orth_err(U, 128) for U in subs.values())
    b_val = ovs["ov(CORE_TW_16,LM_128)"]; c_late = ovs["ov(CORE_TN_128,LM_128)"]; c_early = ovs["ov(EARLY_TN_128,LM_128)"]
    e_ref = ovs["ov(XPCA_16,LM_128)"]
    d_n = sum(v >= BARS["d_er_min"] for v in er.values()); d_n_loo = sum(v >= BARS["d_er_min"] for v in er_loo.values())
    d_null_n = sum(v <= NULLS["d_er_max"] for v in er.values())
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(core_tw["spec"]["eff_rank"] - PRIOR_TW_EFF) <= BARS["eff_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok and orth <= BARS["orth_tol"] and BARS["chance_lo"] <= chance <= BARS["chance_hi"]),
        'pred_b_core_faces_readout': bool(b_val >= BARS["b_min"]),
        'pred_c_late_specific': bool(c_late >= BARS["c_mult"] * c_early),
        'pred_d_core_is_read': bool(d_n >= BARS["d_n_min"]),
        'pred_e_more_readout_facing_than_the_stream': bool(b_val >= e_ref + BARS["e_slack"]),
    }
    nulls = {"b_null_core16_in_lm_le_.25": bool(b_val <= NULLS["b_max"]), "c_null_late_le_1.1x_early": bool(c_late <= NULLS["c_mult"] * c_early),
             "d_null_er_le_1.1_for_5of6": bool(d_null_n >= NULLS["d_n_min"]), "e_null_core_le_stream_minus_.10": bool(b_val <= e_ref - NULLS["e_slack"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "baseline CE is CE on held-out docs 0-63 (instrument only); all other numbers are subspace-overlap fractions (chance k/1152 = .111 at k=128) or read-energy ratios (1.0 = isotropic) -- HIGHER = more aligned",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "core_tw_eff_rank": core_tw["spec"]["eff_rank"], "prior_core_tw_eff_rank": PRIOR_TW_EFF,
                          "max_orth_err_128": orth, "random128_in_lm128": chance},
           "overlaps": ovs, "spectra": {"CORE_TW": core_tw["spec"], "CORE_TN": core_tn["spec"], "EARLY_TN": early_tn["spec"], "XPCA": xpca["spec"], "LM_gram": lm_spec},
           "read_energy_ratio_core_tn_128": er, "read_energy_ratio_leave_own_out": er_loo, "read_energy_ratio_early_tn_128": er_early, "read_energy_ratio_random_128": er_rnd,
           "d_n_of_6": d_n, "d_n_of_6_leave_own_out": d_n_loo, "b_value": b_val, "c_late": c_late, "c_early": c_early, "e_ref_xpca16_in_lm128": e_ref,
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "overlaps", "read_energy_ratio_core_tn_128", "read_energy_ratio_leave_own_out", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "b": b_val, "c": [c_late, c_early], "d_n": d_n, "d_n_loo": d_n_loo, "e_ref": e_ref}, indent=1))


if __name__ == "__main__":
    main()
