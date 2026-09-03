#!/usr/bin/env python
"""plain_truncation_radial_fix_map_probe -- truncate each write in its OWN frame (plain in-situ PCA, docs 0-95) and then reset the radial
scalar of the reconstruction to the per-site constant rbar (§2705): PLAIN_k = mu + U_k U_kᵀ (w - mu); PLAINFIX_k = P_k(w) + (rbar - P_k(w)·x̂) x̂,
k in {8, 32, 128}, all 36 sites one at a time, CE on docs 96-159. Tests the rule extracted in §2706. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_fix_helps_where_the_frame_failed pred_c_fix_never_hurts pred_d_attn1_compact_corrected
#                     pred_e_best_of_three_total

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-159 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/PLAIN_TRUNCATION_RADIAL_FIX_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/plain_truncation_radial_fix_map_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import radial_gauge_map_probe_gpu as G

ROOT = R.ROOT
PREREG = R.POLY / "PLAIN_TRUNCATION_RADIAL_FIX_MAP_PROBE_PREREGISTRATION.md"
PRIOR_RC = ROOT / "attention_radial_channel_probe_results.json"             # §2705
PRIOR_RT = ROOT / "radial_constant_tangential_truncation_map_probe_results.json"  # §2706
PRIOR_MAP = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"    # §2696
OUT = ROOT / "plain_truncation_radial_fix_map_probe_results.json"
HASHES = {PREREG: "e768f58ee39a35b1b662e39c51d5fc751fefda61b9f7df332ab41e0273997fad", PRIOR_RT: "828d86b31346dc59ee27f105a30994f9fe2ce46dbdfe279502c59f7b9c1a142d",
          PRIOR_RC: "db47a079d9969ee50e96901da03ca7e852e66c56205f0eb867ad7e749ffd8518",
          PRIOR_MAP: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "plain_truncation_radial_fix_map_probe"
DEV = G.DEV
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (0, 96); EVAL = (96, 160); CH = 8
ALL_SITES = G.ALL_SITES
KS = [8, 32, 128]
FIVE = ["attn1", "attn5", "attn6", "mlp16", "mlp17"]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.01, "repro_n_min": 34, "mono_n_min": 34, "b_n_min": 5, "c_slack": 0.005, "c_n_min": 33, "attn1_k8_max": 0.03, "e_sum_max": 1.60}
NULLS = {"b_n_max": 2, "c_n_max": 25, "attn1_k8_min": 0.06, "e_sum_min": 1.65}


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
    def add(self, w, r):
        X = w.reshape(-1, D).double(); self.S += X.T @ X; self.mu += X.sum(0); self.cnt += X.shape[0]; self.r += float(r.sum())
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
        r, xh, wp = G.split(w, xpre); acc[s].add(w, r)
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


def plain(b, k, fix):
    Uk = b["U"][:, :k]; mu = b["mu"]; rbar = b["rbar"]
    def fn(w, x):
        p = mu + ((w - mu) @ Uk) @ Uk.T
        if not fix:
            return p
        xh = x / (x.norm(dim=-1, keepdim=True) + 1e-30)
        return p + (rbar - (p * xh).sum(-1, keepdim=True)) * xh
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
        sites = [("attn", 1), ("mlp", 4), ("mlp", 17)]; prior_rc = prior_map = prior_rt = None
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV); sites = ALL_SITES
        prior_rc = json.load(open(PRIOR_RC)); prior_map = {r["site"]: r for r in json.load(open(PRIOR_MAP))["sites"]}; prior_rt = json.load(open(PRIOR_RT))["arms_ce_added"]
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    bases = fit(m, fit_rows); log(stage="fit", attn1_rbar=bases[("attn", 1)]["rbar"], attn1_eff_rank=bases[("attn", 1)]["spec"]["eff_rank"])
    ce0 = G.ce_of(m, ev); log(stage="baseline", ce=ce0)
    arms = {}; spec = {}
    for s in sites:
        name = f"{s[0]}{s[1]}"; arms[name] = {}; spec[name] = bases[s]["spec"]
        for k in KS:
            arms[name][f"PLAIN_{k}"] = G.ce_of(m, ev, s, plain(bases[s], k, False)) - ce0
            arms[name][f"PLAINFIX_{k}"] = G.ce_of(m, ev, s, plain(bases[s], k, True)) - ce0
        log(stage="site", site=name, arms=arms[name])
    names = [f"{s[0]}{s[1]}" for s in sites]
    g = lambda n, arm, k: arms[n][f"{arm}_{k}"]
    prior_base = prior_rc["baseline_ce_eval"] if prior_rc else float("nan")
    repro_n = sum(abs(g(n, "PLAIN", 32) - prior_map[n]["ce_added_k32"]) <= BARS["repro_tol"] for n in names) if prior_map else 0
    mono = lambda arm: sum((g(n, arm, 128) <= g(n, arm, 32) + 1e-9) and (g(n, arm, 32) <= g(n, arm, 8) + 1e-9) for n in names)
    mono_p, mono_f = mono("PLAIN"), mono("PLAINFIX")
    five = [n for n in FIVE if n in arms]
    b_n = sum(g(n, "PLAINFIX", 8) < g(n, "PLAIN", 8) for n in five)
    c_n = sum(g(n, "PLAINFIX", 32) <= g(n, "PLAIN", 32) + BARS["c_slack"] for n in names)
    a1k8 = arms.get("attn1", {}).get("PLAINFIX_8", float("nan"))
    best3 = sum(min(g(n, "PLAIN", 32), g(n, "PLAINFIX", 32), prior_rt[n]["RM_TAN_32"]) for n in names) if prior_rt else float("nan")
    preds = {
        'pred_a_instrument': bool(abs(ce0 - prior_base) <= BARS["ce_tol"] and repro_n >= BARS["repro_n_min"] and min(mono_p, mono_f) >= BARS["mono_n_min"]) if not smoke else bool(min(mono_p, mono_f) == len(names)),
        'pred_b_fix_helps_where_the_frame_failed': bool(len(five) == 5 and b_n >= BARS["b_n_min"]),
        'pred_c_fix_never_hurts': bool(len(names) == 36 and c_n >= BARS["c_n_min"]),
        'pred_d_attn1_compact_corrected': bool(a1k8 <= BARS["attn1_k8_max"]),
        'pred_e_best_of_three_total': bool(len(names) == 36 and best3 <= BARS["e_sum_max"]),
    }
    nulls = {"b_null_le_2_of_5": bool(len(five) == 5 and b_n <= NULLS["b_n_max"]),
             "c_null_le_25_of_36": bool(len(names) == 36 and c_n <= NULLS["c_n_max"]),
             "d_null_attn1_fix8_ge_.06": bool(a1k8 >= NULLS["attn1_k8_min"]),
             "e_null_best3_ge_1.65": bool(len(names) == 36 and best3 >= NULLS["e_sum_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "device": "cuda",
           "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
           "instrument": {"baseline_ce": ce0, "prior_baseline": prior_base, "n_plain32_repro_within_.01": int(repro_n), "n_monotone_plain": int(mono_p), "n_monotone_plainfix": int(mono_f)},
           "baseline_ce_eval": ce0, "n_eval_docs": int(ev.shape[0]), "n_fit_docs": int(fit_rows.shape[0]), "arms_ce_added": arms,
           "write_spectrum_fit": spec, "rbar_fit": {f"{s[0]}{s[1]}": bases[s]["rbar"] for s in sites},
           "b_n_of_5": int(b_n), "c_n_of_36": int(c_n), "attn1_plainfix_8": a1k8, "sum_best_of_three_k32": best3,
           "sum_plain_32": sum(g(n, "PLAIN", 32) for n in names), "sum_plainfix_32": sum(g(n, "PLAINFIX", 32) for n in names),
           "sum_plain_128": sum(g(n, "PLAIN", 128) for n in names), "sum_plainfix_128": sum(g(n, "PLAINFIX", 128) for n in names),
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + sum(len(v) for v in arms.values())), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "arms_ce_added", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "b_n": b_n, "c_n": c_n, "best3": best3, "attn1": arms.get("attn1")}, indent=1))


if __name__ == "__main__":
    main()
