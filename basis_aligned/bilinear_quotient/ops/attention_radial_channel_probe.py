#!/usr/bin/env python
"""attention_radial_channel_probe -- is the attn1 / attn5 write (DROP_RADIAL +5.28 / +3.29 nat in §2704) functionally a per-token
RADIAL SCALAR (norm gate) on the residual? Arms at every write site, one site at a time, held-out docs 96-159: RADIAL_ONLY
(w' = r x̂), RADIAL_MEAN (w' = r̄_site x̂ + w_perp with r̄_site the mean radial scalar over FIT docs 0-95), plus DROP_RADIAL at
attn1/attn5 as the §2704 reproduction. Signed radial statistics per site. CUDA lane-1 script (no silent CPU fallback).

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_radial_scalar_carries_the_cliff pred_c_per_token_gate_not_constant
#                     pred_d_norm_shrinking pred_e_specific_to_attn1_attn5

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-159 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier.
Preregistration: polynomial_causal/ATTENTION_RADIAL_CHANNEL_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/attention_radial_channel_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import radial_gauge_map_probe_gpu as G

ROOT = R.ROOT
PREREG = R.POLY / "ATTENTION_RADIAL_CHANNEL_PROBE_PREREGISTRATION.md"
PRIOR_GPU = ROOT / "radial_gauge_map_probe_gpu_results.json"                # §2704
PRIOR_MAP = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"    # §2696
OUT = ROOT / "attention_radial_channel_probe_results.json"
HASHES = {PREREG: "f9423d6b476c7020a641c9293c56843ddaa9537b7fd5f13af6dcb7fc0d26c317",
          PRIOR_GPU: "248a5af49328eb21f69acb5d8a8de3c7c9eef380ea7ae4ebf9b61e3e7c0fa063",
          PRIOR_MAP: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "attention_radial_channel_probe"
DEV = G.DEV
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (0, 96); EVAL = (96, 160); CH = 8
ALL_SITES = G.ALL_SITES
TWO = [("attn", 1), ("attn", 5)]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "radial_only_max": 1.0, "radial_only_frac": 0.3, "radial_mean_min": 0.5, "mean_rx_max": -0.10, "spec_n_min": 14}
NULLS = {"radial_mean_max": 0.1, "mean_rx_null": 0.0, "spec_n_max": 10}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def arm_fn(arm, rbar=None):
    if arm == "RADIAL_ONLY":
        return lambda w, x: (lambda r, xh, wp: r * xh)(*G.split(w, x))
    if arm == "RADIAL_MEAN":
        return lambda w, x: (lambda r, xh, wp: rbar * xh + wp)(*G.split(w, x))
    return G.arm_fn(arm)


@torch.no_grad()
def forward_all_stats(m, idx, acc):
    """One forward that records (r, r/|x|) at every site; equivalent to G.forward with an identity write at every site."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = (t.to(DEV) for t in R.rope(Tn))
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
    def rec(s, w, xpre):
        r, xh, wp = G.split(w, xpre)
        acc[s].append(torch.stack([r.reshape(-1), (r / (xpre.norm(dim=-1, keepdim=True) + 1e-30)).reshape(-1)], 1).cpu())
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
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def stats_pass(m, rows):
    acc = {s: [] for s in ALL_SITES}
    for i in range(0, rows.shape[0], CH):
        forward_all_stats(m, rows[i:i + CH, :TI], acc)
    out = {}
    for s in ALL_SITES:
        t = torch.cat(acc[s], 0); r, rx = t[:, 0], t[:, 1]
        q = torch.quantile(rx, torch.tensor([0.1, 0.5, 0.9]))
        out[s] = {"rbar": float(r.mean()), "mean_r_over_x": float(rx.mean()), "frac_r_negative": float((r < 0).float().mean()),
                  "rx_q10": float(q[0]), "rx_q50": float(q[1]), "rx_q90": float(q[2])}
    return out


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model().to(DEV)
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (8, 257), generator=g).to(DEV); fit, ev = nat[:4], nat[4:]
        sites = [("attn", 1), ("attn", 5), ("mlp", 17)]; prior = None
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV); sites = ALL_SITES
        prior = json.load(open(PRIOR_GPU))
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    fit_stats = stats_pass(m, fit); log(stage="fit_stats", attn1=fit_stats[("attn", 1)], attn5=fit_stats[("attn", 5)])
    ev_stats = stats_pass(m, ev)
    ce0 = G.ce_of(m, ev); log(stage="baseline", ce=ce0)
    arms = {}
    for s in sites:
        name = f"{s[0]}{s[1]}"; arms[name] = {}
        for arm in ["RADIAL_ONLY", "RADIAL_MEAN"] + (["DROP_RADIAL"] if s in TWO else []):
            arms[name][arm] = G.ce_of(m, ev, s, arm_fn(arm, rbar=fit_stats[s]["rbar"])) - ce0
            log(stage="arm", site=name, arm=arm, ce_added=arms[name][arm])
    g = lambda s, arm: arms.get(f"{s[0]}{s[1]}", {}).get(arm, float("nan"))
    prior_base = prior["baseline_ce_eval"] if prior else float("nan")
    prior_drop = {f"{s[0]}{s[1]}": (prior["arms_ce_added"][f"{s[0]}{s[1]}"]["DROP_RADIAL"] if prior else float("nan")) for s in TWO}
    repro_ok = all(abs(g(s, "DROP_RADIAL") - prior_drop[f"{s[0]}{s[1]}"]) <= BARS["repro_tol"] for s in TWO) if prior else True
    ro = {f"{s[0]}{s[1]}": g(s, "RADIAL_ONLY") for s in TWO}; rm = {f"{s[0]}{s[1]}": g(s, "RADIAL_MEAN") for s in TWO}
    dr = {f"{s[0]}{s[1]}": g(s, "DROP_RADIAL") for s in TWO}
    mrx = {f"{s[0]}{s[1]}": ev_stats[s]["mean_r_over_x"] for s in TWO}
    others = [("attn", l) for l in range(NL) if ("attn", l) not in TWO]
    if prior:
        spec = [g(s, "RADIAL_ONLY") >= prior["arms_ce_added"][f"{s[0]}{s[1]}"]["DROP_RADIAL"] for s in others if f"{s[0]}{s[1]}" in arms]
    else:
        spec = []
    n_spec = int(sum(spec))
    preds = {
        'pred_a_instrument': bool(abs(ce0 - prior_base) <= BARS["ce_tol"] and repro_ok) if not smoke else True,
        'pred_b_radial_scalar_carries_the_cliff': bool(all(ro[k] <= BARS["radial_only_max"] and ro[k] <= BARS["radial_only_frac"] * dr[k] for k in ro)),
        'pred_c_per_token_gate_not_constant': bool(all(rm[k] >= BARS["radial_mean_min"] for k in rm)),
        'pred_d_norm_shrinking': bool(all(mrx[k] < BARS["mean_rx_max"] for k in mrx)),
        'pred_e_specific_to_attn1_attn5': bool(len(spec) == 16 and n_spec >= BARS["spec_n_min"]),
    }
    nulls = {"b_null_radial_only_ge_drop_radial_at_either": bool(any(ro[k] >= dr[k] for k in ro)),
             "c_null_both_radial_mean_le_.1": bool(all(rm[k] <= NULLS["radial_mean_max"] for k in rm)),
             "d_null_mean_rx_positive_at_either": bool(any(mrx[k] > NULLS["mean_rx_null"] for k in mrx)),
             "e_null_spec_le_10_of_16": bool(len(spec) == 16 and n_spec <= NULLS["spec_n_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "device": "cuda",
           "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
           "instrument": {"baseline_ce": ce0, "prior_baseline": prior_base, "drop_radial_repro": dr, "prior_drop_radial": prior_drop},
           "baseline_ce_eval": ce0, "n_eval_docs": int(ev.shape[0]), "n_fit_docs": int(fit.shape[0]), "arms_ce_added": arms,
           "radial_stats_fit": {f"{s[0]}{s[1]}": fit_stats[s] for s in ALL_SITES}, "radial_stats_eval": {f"{s[0]}{s[1]}": ev_stats[s] for s in ALL_SITES},
           "specificity_n_of_16": n_spec,
           "price": {"gpu_doc_forwards": int(fit.shape[0]) + int(ev.shape[0]) * (2 + sum(len(v) for v in arms.values())), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "arms_ce_added", "specificity_n_of_16", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "arms": {k: arms[k] for k in ("attn1", "attn5")}, "mean_rx": mrx}, indent=1))


if __name__ == "__main__":
    main()
