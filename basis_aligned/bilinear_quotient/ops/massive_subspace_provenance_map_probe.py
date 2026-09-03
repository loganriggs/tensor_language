#!/usr/bin/env python
"""massive_subspace_provenance_map_probe -- where does the late shared write core (§2710/§2713 = the final residual stream's dominant
geometry) come from? Per block: overlap of the core with the stream's top-128 PCA and the stream's energy fraction inside the core; per
site (36 writes): the write's energy fraction inside the core. One 96-doc collection pass + 64-doc baseline check. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/MASSIVE_SUBSPACE_PROVENANCE_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/massive_subspace_provenance_map_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("massive_subspace_provenance_map_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "MASSIVE_SUBSPACE_PROVENANCE_MAP_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_core_readout_alignment_probe_results.json"   # §2713
OUT = ROOT / "massive_subspace_provenance_map_probe_results.json"
HASHES = {PREREG: "eca43f4186aaf68aa1af9b68411475310ddc395ff4bcb9eb1b38a56255854915", PRIOR: "20ff21d0bef18132772b47f243f273ab0ebb01f807b72f8de0b6c71aa830c82d",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "massive_subspace_provenance_map_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; ALL36 = [(kind, l) for l in range(NL) for kind in ("attn", "mlp")]
PRIOR_BASE = 3.0322401; PRIOR_TW_EFF = 10.004
BARS = {"ce_tol": 1e-4, "eff_tol": 0.5, "a_ov_min": 0.70, "b_min": 0.70, "c_e17_min": 0.50, "c_rho_min": 0.80, "d_frac": 0.50, "e_med_max": 0.15}
NULLS = {"b_max": 0.30, "c_e17_max": 0.20, "d_early_max": 0.20, "e_med_min": 0.35}


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
        return {"mu": mu.float(), "U": U.float().to(DEV), "spec": R.spectrum(C), "C": C, "S": (self.S / self.cnt).cpu()}


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
            collect(("resid", l), x)
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


def ov(U, j, Vb, k):
    """Fraction of span(U[:, :j]) inside span(V[:, :k]); chance k/D."""
    A = U[:, :j].cpu().double().T @ Vb[:, :k].cpu().double(); return float((A * A).sum() / j)


def frac(M, C):
    """Energy fraction of the (co)variance C inside span(M)."""
    Mc = M.cpu().double(); return float(torch.trace(Mc.T @ C @ Mc) / torch.trace(C))


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
    sites = ALL36 + [("resid", l) for l in range(NL)]
    bases = fit(m, fit_rows, sites); log(stage="fit")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    core = pooled([bases[s] for s in LATE7], False); M = core["U"][:, :KM]
    X = bases[("resid", NL - 1)]["U"].double()
    a_ov = ov(core["U"], KM, X, 128)
    blocks = {}
    for l in range(NL):
        r = bases[("resid", l)]
        blocks[l] = {"ov_core16_in_resid128": ov(core["U"], KM, r["U"].double(), 128), "e_uncentred": frac(M, r["S"]), "e_centred": frac(M, r["C"]),
                     "resid_eff_rank": r["spec"]["eff_rank"], "ov_x16_in_resid128": ov(X, KM, r["U"].double(), 128)}
        log(stage="block", l=l, **{k: round(v, 4) for k, v in blocks[l].items()})
    site_f = {f"{s[0]}{s[1]}": {"f_core16": frac(M, bases[s]["C"]), "f_x16": frac(X[:, :KM], bases[s]["C"]), "eff_rank": bases[s]["spec"]["eff_rank"]} for s in ALL36}
    log(stage="sites", f={k: round(v["f_core16"], 3) for k, v in site_f.items()})
    ls = torch.arange(NL, dtype=torch.float64); es = torch.tensor([blocks[l]["e_uncentred"] for l in range(NL)], dtype=torch.float64)
    rho = R.spearman(ls, es)
    early_f = {k: v["f_core16"] for k, v in site_f.items() if int(k[4:] if k.startswith("attn") else k[3:]) <= 3}
    mid_f = [v["f_core16"] for k, v in site_f.items() if 4 <= int(k[4:] if k.startswith("attn") else k[3:]) <= 13]
    mid_med = float(torch.tensor(mid_f).median()) if mid_f else float("nan")
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(core["spec"]["eff_rank"] - PRIOR_TW_EFF) <= BARS["eff_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok and a_ov >= BARS["a_ov_min"]) if not smoke else bool(inst_ok),
        'pred_b_established_by_block_3': bool(blocks[3]["ov_core16_in_resid128"] >= BARS["b_min"]),
        'pred_c_carries_the_stream': bool(blocks[NL - 1]["e_uncentred"] >= BARS["c_e17_min"] and rho >= BARS["c_rho_min"]),
        'pred_d_writers_early_and_late': bool(site_f["mlp16"]["f_core16"] >= BARS["d_frac"] and site_f["mlp17"]["f_core16"] >= BARS["d_frac"] and max(early_f.values()) >= BARS["d_frac"]),
        'pred_e_middle_avoids_it': bool(mid_med <= BARS["e_med_max"]),
    }
    nulls = {"b_null_ov3_le_.30": bool(blocks[3]["ov_core16_in_resid128"] <= NULLS["b_max"]), "c_null_e17_le_.20": bool(blocks[NL - 1]["e_uncentred"] <= NULLS["c_e17_max"]),
             "d_null_no_early_site_ge_.20": bool(max(early_f.values()) < NULLS["d_early_max"]), "e_null_mid_median_ge_.35": bool(mid_med >= NULLS["e_med_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "baseline CE is CE on held-out docs 0-63 (instrument only); all other numbers are overlap / energy fractions in [0,1] -- HIGHER = more of the object lies in the core",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "k_core": KM,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "core_tw_eff_rank": core["spec"]["eff_rank"], "prior_core_tw_eff_rank": PRIOR_TW_EFF, "ov_core16_in_x128": a_ov},
           "blocks": blocks, "spearman_depth_vs_e_uncentred": rho, "sites": site_f, "early_site_f": early_f, "mid_median_f": mid_med, "core_spectrum": core["spec"],
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "mid_median_f", "early_site_f", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ov3": blocks[3]["ov_core16_in_resid128"], "e17": blocks[NL - 1]["e_uncentred"], "rho": rho, "early_f": early_f, "mid_med": mid_med}, indent=1))


if __name__ == "__main__":
    main()
