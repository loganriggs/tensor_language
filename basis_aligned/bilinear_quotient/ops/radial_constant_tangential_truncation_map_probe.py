#!/usr/bin/env python
"""radial_constant_tangential_truncation_map_probe -- with the radial axis of each write replaced by its per-site constant gain
(§2705: RADIAL_MEAN <= .036 at all 36 sites), truncate the TANGENTIAL remainder to its top-k in-situ principal directions (fitted on
docs 0-95) and score CE on docs 96-159: RM_TAN_k = rbar x̂ + mu + U_k U_kᵀ (w_perp - mu), k in {8, 32, 128}, all 36 sites, one at a
time; RM_TAN_FULL at attn1 / mlp4 as the instrument. Comparators: §2696 PLAIN_32, §2705 RADIAL_MEAN. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_radial_out_helps_per_site pred_c_attn1_compact pred_d_total_price_drops
#                     pred_e_k128_ladder

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-159 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/RADIAL_CONSTANT_TANGENTIAL_TRUNCATION_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/radial_constant_tangential_truncation_map_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import radial_gauge_map_probe_gpu as G

ROOT = R.ROOT
PREREG = R.POLY / "RADIAL_CONSTANT_TANGENTIAL_TRUNCATION_MAP_PROBE_PREREGISTRATION.md"
PRIOR_RC = ROOT / "attention_radial_channel_probe_results.json"             # §2705
PRIOR_MAP = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"    # §2696
OUT = ROOT / "radial_constant_tangential_truncation_map_probe_results.json"
HASHES = {PREREG: "47d3673818839d38e93cdcab997b2ba0c1c8f4cc8b6870dda24a8b2adb7b6d5a",
          PRIOR_RC: "db47a079d9969ee50e96901da03ca7e852e66c56205f0eb867ad7e749ffd8518",
          PRIOR_MAP: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "radial_constant_tangential_truncation_map_probe"
DEV = G.DEV
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (0, 96); EVAL = (96, 160); CH = 8
ALL_SITES = G.ALL_SITES
KS = [8, 32, 128]
FULL_SITES = [("attn", 1), ("mlp", 4)]
PRIOR_SUM_PLAIN32 = 2.3712
BARS = {"ce_tol": 1e-4, "full_tol": 1e-3, "mono_n_min": 34, "b_n_min": 28, "attn1_k8_max": 0.05, "d_sum_max": 0.8 * PRIOR_SUM_PLAIN32, "e_le": 0.02, "e_n_min": 24}
NULLS = {"b_n_max": 18, "attn1_k8_min": 0.10, "d_sum_min": 0.95 * PRIOR_SUM_PLAIN32, "e_n_max": 14}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


class GAcc:
    """float64 second-moment accumulator on DEV."""
    def __init__(self):
        self.S = torch.zeros(D, D, dtype=torch.float64, device=DEV); self.mu = torch.zeros(D, dtype=torch.float64, device=DEV)
        self.r = 0.0; self.cnt = 0
    def add(self, wp, r):
        X = wp.reshape(-1, D).double(); self.S += X.T @ X; self.mu += X.sum(0); self.cnt += X.shape[0]; self.r += float(r.sum())
    def finish(self):
        mu = self.mu / self.cnt; C = (self.S / self.cnt - torch.outer(mu, mu)).cpu()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        return {"rbar": self.r / self.cnt, "mu": mu.float(), "U": U.float().to(DEV), "spec": R.spectrum(C)}


@torch.no_grad()
def forward_collect(m, idx, acc):
    """One forward that feeds (w_perp, r) at every site to acc[site]."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = (t.to(DEV) for t in R.rope(Tn))
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
    def rec(s, w, xpre):
        r, xh, wp = G.split(w, xpre); acc[s].add(wp, r)
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
        rec(("attn", l), aw, x); x = x + aw
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        rec(("mlp", l), mw, x); x = x + mw + mlp.Down_bias


def fit(m, rows):
    acc = {s: GAcc() for s in ALL_SITES}
    for i in range(0, rows.shape[0], CH):
        forward_collect(m, rows[i:i + CH, :TI], acc)
    return {s: acc[s].finish() for s in ALL_SITES}


def rm_tan(b, k):
    Uk = b["U"][:, :k]; mu = b["mu"]; rbar = b["rbar"]
    def fn(w, x):
        r, xh, wp = G.split(w, x)
        return rbar * xh + mu + ((wp - mu) @ Uk) @ Uk.T
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
        nat = torch.randint(0, 50000, (8, 257), generator=g).to(DEV); fit_rows, ev = nat[:4], nat[4:]
        sites = [("attn", 1), ("mlp", 4), ("mlp", 17)]; prior_rc = prior_map = None
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV); sites = ALL_SITES
        prior_rc = json.load(open(PRIOR_RC)); prior_map = {r["site"]: r for r in json.load(open(PRIOR_MAP))["sites"]}
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    bases = fit(m, fit_rows); log(stage="fit", attn1_rbar=bases[("attn", 1)]["rbar"], attn1_tan_eff_rank=bases[("attn", 1)]["spec"]["eff_rank"])
    ce0 = G.ce_of(m, ev); log(stage="baseline", ce=ce0)
    arms = {}; spec = {}
    for s in sites:
        name = f"{s[0]}{s[1]}"; arms[name] = {}; spec[name] = bases[s]["spec"]
        for k in KS + ([D] if s in FULL_SITES else []):
            arms[name][f"RM_TAN_{k if k < D else 'FULL'}"] = G.ce_of(m, ev, s, rm_tan(bases[s], k)) - ce0
        log(stage="site", site=name, arms=arms[name])
    names = [f"{s[0]}{s[1]}" for s in sites]
    g = lambda n, k: arms[n][f"RM_TAN_{k}"]
    prior_base = prior_rc["baseline_ce_eval"] if prior_rc else float("nan")
    full_ok = all(abs(arms[f"{s[0]}{s[1]}"]["RM_TAN_FULL"] - prior_rc["arms_ce_added"][f"{s[0]}{s[1]}"]["RADIAL_MEAN"]) <= BARS["full_tol"] for s in FULL_SITES) if prior_rc else True
    mono = sum((g(n, 128) <= g(n, 32) + 1e-9) and (g(n, 32) <= g(n, 8) + 1e-9) for n in names)
    better = sum(g(n, 32) < prior_map[n]["ce_added_k32"] for n in names) if prior_map else 0
    total32 = sum(g(n, 32) for n in names)
    e_n = sum(g(n, 128) <= BARS["e_le"] for n in names)
    a1k8 = arms.get("attn1", {}).get("RM_TAN_8", float("nan"))
    preds = {
        'pred_a_instrument': bool(abs(ce0 - prior_base) <= BARS["ce_tol"] and full_ok and mono >= BARS["mono_n_min"]) if not smoke else bool(mono == len(names)),
        'pred_b_radial_out_helps_per_site': bool(len(names) == 36 and better >= BARS["b_n_min"]),
        'pred_c_attn1_compact': bool(a1k8 <= BARS["attn1_k8_max"]),
        'pred_d_total_price_drops': bool(len(names) == 36 and total32 <= BARS["d_sum_max"]),
        'pred_e_k128_ladder': bool(len(names) == 36 and e_n >= BARS["e_n_min"]),
    }
    nulls = {"b_null_le_18_of_36": bool(len(names) == 36 and better <= NULLS["b_n_max"]),
             "c_null_attn1_k8_ge_.10": bool(a1k8 >= NULLS["attn1_k8_min"]),
             "d_null_sum_ge_.95x": bool(len(names) == 36 and total32 >= NULLS["d_sum_min"]),
             "e_null_le_14_of_36": bool(len(names) == 36 and e_n <= NULLS["e_n_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "device": "cuda",
           "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
           "instrument": {"baseline_ce": ce0, "prior_baseline": prior_base, "full_vs_radial_mean": {f"{s[0]}{s[1]}": arms[f"{s[0]}{s[1]}"]["RM_TAN_FULL"] for s in FULL_SITES}, "n_monotone": int(mono)},
           "baseline_ce_eval": ce0, "n_eval_docs": int(ev.shape[0]), "n_fit_docs": int(fit_rows.shape[0]), "arms_ce_added": arms,
           "tangential_spectrum_fit": spec, "rbar_fit": {f"{s[0]}{s[1]}": bases[s]["rbar"] for s in sites},
           "n_sites_rm_tan_32_below_plain_32": int(better), "sum_rm_tan_32": total32, "prior_sum_plain_32": PRIOR_SUM_PLAIN32, "n_sites_rm_tan_128_le_.02": int(e_n),
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + sum(len(v) for v in arms.values())), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "arms_ce_added", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "better": better, "sum32": total32, "e_n": e_n, "attn1": arms.get("attn1")}, indent=1))


if __name__ == "__main__":
    main()
