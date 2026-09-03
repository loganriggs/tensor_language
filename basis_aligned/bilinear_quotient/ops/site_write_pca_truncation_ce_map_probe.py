#!/usr/bin/env python
"""site_write_pca_truncation_ce_map_probe -- for all 36 write sites (attention and MLP write of each block), the held-out CE cost of
projecting the write onto its top-k in-situ principal directions (fitted on docs 0-95, scored on docs 96-159). Companion to
§2692 (usage ranks) and to mlp_final_blocks_low_rank_surrogate_probe (blocks 16/17 only).

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_low_usage_attention_sites_cheap pred_c_usage_rank_orders_truncation_cost
#                     pred_d_high_usage_sites_expensive

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out documents -- LOWER IS BETTER.
This is a descriptive map of activation-PCA write truncation; it does not install anything into the §312 frontier and does not
reopen the closed metric-constructed-basis items (§2118): the bases here are data covariances of the writes, scored by CE only.
Preregistration: polynomial_causal/SITE_WRITE_PCA_TRUNCATION_CE_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/site_write_pca_truncation_ce_map_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

ROOT = R.ROOT
PREREG = R.POLY / "SITE_WRITE_PCA_TRUNCATION_CE_MAP_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "mlp_in_situ_usage_rank_map_probe_results.json"
OUT = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"
HASHES = {PREREG: "d37d2c337375a8106a3828be16dde2bda04eb39367fe1e4cca9c99b14ebafc43", PRIOR: "63483cec6f68964235c1033ee0ebd0aba04c15b0dcd0f2f42e9b3d2b9b1a90b2",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "site_write_pca_truncation_ce_map_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (0, 96); EVAL = (96, 160); CH = 8
K_MAP = 32                       # the map rank
K_LOW = 8                        # extra rank for the low-usage sites
LOW_ATTN = [1, 6, 17]            # §2692 natural attention-write eff ranks 21 / 31 / 20
BARS = {"ce_tol": 1e-4, "low_attn_k32_max": 0.02, "spearman_min": 0.6, "high_site_eff_rank": 500.0, "high_site_k32_min": 0.30}
NULLS = {"low_attn_k32_min": 0.10, "spearman_max": 0.2, "high_all_k32_max": 0.10}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


@torch.no_grad()
def forward(m, idx, patch=None, collect=None):
    """tt_model semantics; patch(site, write[B,T,D]) -> write for site in {('attn',l),('mlp',l)}; collect(site, write)."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = R.rope(Tn)
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
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
        if patch is not None:
            aw = patch(("attn", l), aw)
        if collect is not None:
            collect(("attn", l), aw)
        x = x + aw
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        if patch is not None:
            mw = patch(("mlp", l), mw)
        if collect is not None:
            collect(("mlp", l), mw)
        x = x + mw + mlp.Down_bias
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, patch=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg = forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


def fit_bases(m, rows):
    sites = [(kind, l) for l in range(NL) for kind in ("attn", "mlp")]
    acc = {s: R.Acc(D) for s in sites}
    def collect(s, w):
        acc[s].add(w.reshape(-1, D))
    for i in range(0, rows.shape[0], CH):
        forward(m, rows[i:i + CH, :TI], collect=collect)
    out = {}
    for s in sites:
        C, e = acc[s].cov(); mu = (acc[s].mu / acc[s].cnt).float()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        out[s] = {"mu": mu, "U": U.float(), "spec": R.spectrum(C), "mean_energy": e}
    return out


def make_patch(bases, site, k):
    b = bases[site]; Uk = b["U"][:, :k]
    def patch(s, w):
        if s != site:
            return w
        return b["mu"] + ((w - b["mu"]) @ Uk) @ Uk.T
    return patch


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    check_hashes()
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model()
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=g); fit, ev = nat[:3], nat[3:]
        sites_map = [("attn", 1), ("mlp", 17), ("mlp", 3)]
    else:
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, ev = nat[FIT[0]:FIT[1]], nat[EVAL[0]:EVAL[1]]
        sites_map = [(kind, l) for l in range(NL) for kind in ("attn", "mlp")]
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce_manual = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ce_module = float(m(idx.contiguous(), tgt.contiguous()))
    bases = fit_bases(m, fit)
    ce0 = ce_of(m, ev)
    rows_out = []
    for s in sites_map:
        r = {"site": f"{s[0]}{s[1]}", "kind": s[0], "block": s[1], "eff_rank_fit": bases[s]["spec"]["eff_rank"], "rank_90_fit": bases[s]["spec"]["rank_90"],
             "mean_energy_fit": bases[s]["mean_energy"], "ce_added_k32": ce_of(m, ev, make_patch(bases, s, K_MAP)) - ce0}
        if s[0] == "attn" and s[1] in LOW_ATTN or (s[0] == "mlp" and s[1] in (16, 17)):
            r["ce_added_k8"] = ce_of(m, ev, make_patch(bases, s, K_LOW)) - ce0
        rows_out.append(r)
        print(json.dumps(r), flush=True)
    effs = torch.tensor([r["eff_rank_fit"] for r in rows_out]); costs = torch.tensor([r["ce_added_k32"] for r in rows_out])
    rho = R.spearman(effs, costs)
    low = [r["ce_added_k32"] for r in rows_out if r["kind"] == "attn" and r["block"] in LOW_ATTN]
    high = [r["ce_added_k32"] for r in rows_out if r["eff_rank_fit"] >= BARS["high_site_eff_rank"]]
    preds = {
        'pred_a_instrument': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
        'pred_b_low_usage_attention_sites_cheap': bool(len(low) == len(LOW_ATTN) and max(low) <= BARS["low_attn_k32_max"]),
        'pred_c_usage_rank_orders_truncation_cost': bool(rho >= BARS["spearman_min"]),
        'pred_d_high_usage_sites_expensive': bool(len(high) > 0 and max(high) >= BARS["high_site_k32_min"]),
    }
    nulls = {"b_null_low_attn_k32_ge_.10": bool(len(low) > 0 and max(low) >= NULLS["low_attn_k32_min"]),
             "c_null_spearman_le_.2": bool(rho <= NULLS["spearman_max"]),
             "d_null_all_high_sites_le_.10": bool(len(high) > 0 and max(high) <= NULLS["high_all_k32_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "k_map": K_MAP, "k_low": K_LOW,
           "instrument": {"ce_manual": ce_manual, "ce_module": ce_module, "abs_diff": abs(ce_manual - ce_module)},
           "baseline_ce_eval": ce0, "n_fit_docs": int(fit.shape[0]), "n_eval_docs": int(ev.shape[0]),
           "spearman_eff_rank_vs_ce_added_k32": rho, "sites": rows_out,
           "price": {"gpu_forwards": 0, "cpu_full_forwards_docs": int(fit.shape[0]) + int(ev.shape[0]) * (2 + len(sites_map) + 5),
                     "cpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "spearman_eff_rank_vs_ce_added_k32", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "rho": rho}, indent=1))


if __name__ == "__main__":
    main()
