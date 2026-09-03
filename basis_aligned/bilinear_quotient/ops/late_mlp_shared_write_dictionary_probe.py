#!/usr/bin/env python
"""late_mlp_shared_write_dictionary_probe -- do the seven late MLP writes (mlp11..17) share a residual-stream subspace? SEP_k: each site
truncated in its OWN in-situ PCA frame (k in {32,128,512}, all seven at once); SHARED_K: ONE basis from the pooled per-site-centred writes,
applied at all seven (K in {32,128,512}); adjacent-pair versions at 128. FRESH split (bases docs 96-191, CE docs 0-63). CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_MLP_SHARED_WRITE_DICTIONARY_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_mlp_shared_write_dictionary_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_mlp_shared_write_dictionary_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_MLP_SHARED_WRITE_DICTIONARY_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_joint_k_ladder_probe_results.json"   # §2709
OUT = ROOT / "late_mlp_shared_write_dictionary_probe_results.json"
HASHES = {PREREG: "ded1b551dc36da18c3f59a60b57103b8e454b10e3995f82157bb8782d7d55d82", PRIOR: "6b2708a34e0eaf1cf226217995b37c6a8c6398da99dc0e0688d8c960a716b2fe",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_mlp_shared_write_dictionary_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KS = [32, 128, 512]
MLP7 = [("mlp", l) for l in range(11, 18)]
PAIRS = [(("mlp", l), ("mlp", l + 1)) for l in range(11, 17)]
PRIOR_BASE = 3.0322401; PRIOR_SEP128 = 0.38495
BARS = {"ce_tol": 1e-4, "repro_tol": 0.01, "b_mult": 1.5, "c_slack": 0.03, "d_frac": 0.7, "d_n_min": 5, "e_mult": 1.3, "e_n_min": 4}
NULLS = {"b_mult": 2.5, "c_slack": 0.10, "d_n_max": 3, "e_n_max": 1}


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


def trunc(b, k, U=None):
    Uk = (b["U"] if U is None else U)[:, :k]; mu = b["mu"]
    return lambda w: mu + ((w - mu) @ Uk) @ Uk.T


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
        sites = [("mlp", 15), ("mlp", 16), ("mlp", 17)]; ks = [32, 512]; pairs = [(("mlp", 15), ("mlp", 16)), (("mlp", 16), ("mlp", 17))]
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV); sites = MLP7; ks = KS; pairs = PAIRS
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce4 = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ident = abs(float(F.cross_entropy(forward(m, idx, {("mlp", 17): (lambda w: w)}).reshape(-1, V), tgt.reshape(-1))) - ce4)
    bases = fit(m, fit_rows, sites); shared = pooled_basis([bases[s] for s in sites])
    pair_shared = {p: pooled_basis([bases[p[0]], bases[p[1]]]) for p in pairs}
    log(stage="fit", shared_eff_rank=shared["spec"]["eff_rank"], mlp17_eff_rank=bases[("mlp", 17)]["spec"]["eff_rank"])
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    sep = {}; shr = {}
    for k in ks:
        sep[k] = ce_of(m, ev, {s: trunc(bases[s], k) for s in sites}) - ce0
        shr[k] = ce_of(m, ev, {s: trunc(bases[s], k, shared["U"]) for s in sites}) - ce0
        log(stage="k", k=k, sep=sep[k], shared=shr[k], ratio=(shr[k] / sep[k] if sep[k] > 0 else float("nan")))
    kp = 128 if 128 in ks else ks[0]
    pair_sep = {}; pair_shr = {}
    for p in pairs:
        n = f"{p[0][0]}{p[0][1]}+{p[1][0]}{p[1][1]}"
        pair_sep[n] = ce_of(m, ev, {s: trunc(bases[s], kp) for s in p}) - ce0
        pair_shr[n] = ce_of(m, ev, {s: trunc(bases[s], kp, pair_shared[p]["U"]) for s in p}) - ce0
        log(stage="pair", pair=n, sep=pair_sep[n], shared=pair_shr[n])
    energy = {f"{s[0]}{s[1]}": {"own_128": captured(bases[s], bases[s]["U"], 128), "shared_128": captured(bases[s], shared["U"], 128)} for s in sites}
    d_n = sum(v["shared_128"] >= BARS["d_frac"] * v["own_128"] for v in energy.values())
    e_n = sum(pair_shr[n] <= BARS["e_mult"] * pair_sep[n] for n in pair_sep)
    mono = lambda d: all(d[ks[i + 1]] <= d[ks[i]] + 1e-9 for i in range(len(ks) - 1))
    full = (128 in ks and 512 in ks)
    preds = {
        'pred_a_instrument': bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and ident <= BARS["ce_tol"] and full and abs(sep[128] - PRIOR_SEP128) <= BARS["repro_tol"] and mono(sep) and mono(shr)) if not smoke else bool(ident <= BARS["ce_tol"] and mono(sep) and mono(shr)),
        'pred_b_shared_dictionary_cheap': bool(full and shr[128] <= BARS["b_mult"] * sep[128]),
        'pred_c_shared_512_converges': bool(512 in ks and shr[512] <= sep[512] + BARS["c_slack"]),
        'pred_d_energy_overlap': bool(len(sites) == 7 and d_n >= BARS["d_n_min"]),
        'pred_e_adjacent_pairs_share': bool(len(pairs) == 6 and e_n >= BARS["e_n_min"]),
    }
    nulls = {"b_null_shared128_ge_2.5x": bool(full and shr[128] >= NULLS["b_mult"] * sep[128]),
             "c_null_shared512_ge_sep+.10": bool(512 in ks and shr[512] >= sep[512] + NULLS["c_slack"]),
             "d_null_le_3_of_7": bool(len(sites) == 7 and d_n <= NULLS["d_n_max"]),
             "e_null_le_1_of_6": bool(len(pairs) == 6 and e_n <= NULLS["e_n_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split, bases from docs 96-191); LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "ks": ks,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "identity_abs_diff": ident, "sep_128": sep.get(128), "prior_sep_128": PRIOR_SEP128},
           "baseline_ce_eval": ce0, "sep_ce_added": sep, "shared_ce_added": shr, "shared_over_sep": {k: (shr[k] / sep[k] if sep[k] > 0 else None) for k in ks},
           "pair_sep_128": pair_sep, "pair_shared_128": pair_shr, "energy_captured_128": energy, "d_n_of_7": int(d_n), "e_n_of_6": int(e_n),
           "shared_spectrum_fit": shared["spec"], "write_spectrum_fit": {f"{s[0]}{s[1]}": bases[s]["spec"] for s in sites},
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + 2 * len(ks) + 2 * len(pairs)) + 8, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "sep_ce_added", "shared_ce_added", "energy_captured_128", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "sep": sep, "shared": shr, "pairs": {n: (pair_sep[n], pair_shr[n]) for n in pair_sep}}, indent=1))


if __name__ == "__main__":
    main()
