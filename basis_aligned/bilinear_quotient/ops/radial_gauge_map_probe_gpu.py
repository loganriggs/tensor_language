#!/usr/bin/env python
"""radial_gauge_map_probe_gpu -- CUDA (lane 1) copy of radial_gauge_map_probe: identical arms, sites, split, bars and nulls; the
only change is the device (see RADIAL_GAUGE_MAP_PROBE_PREREGISTRATION_GPU_ADDENDUM.md: cross-device baseline tolerance). Original docstring:
at every one of the 36 write sites, split the write in the PRE-write frame (x̂ = unit residual the
write is added to; r = w·x̂; w_perp = w - r x̂) and measure DROP_RADIAL (w' = w_perp) and SCALE_RADIAL_2 (w' = 2r x̂ + w_perp),
one site at a time, on held-out docs 96-159. A map of where the rms_norm scale-gauge can be taken. Successor to §2702.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_mid_mlp_radial_inert pred_c_final_mlp_radial_functional
#                     pred_d_early_radial_soft_both_ways

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-159 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; no bases are fitted.
Preregistration: polynomial_causal/RADIAL_GAUGE_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/radial_gauge_map_probe_gpu.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

ROOT = R.ROOT
PREREG = R.POLY / "RADIAL_GAUGE_MAP_PROBE_PREREGISTRATION.md"
PRIOR_RT = ROOT / "early_mlp_radial_tangential_probe_results.json"          # §2702
PRIOR_MAP = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"    # §2696 (baseline)
OUT = ROOT / "radial_gauge_map_probe_gpu_results.json"
ADDENDUM = R.POLY / "RADIAL_GAUGE_MAP_PROBE_PREREGISTRATION_GPU_ADDENDUM.md"
HASHES = {PREREG: "7a5bfdc9a17d504c109d47d30c5925e6a118169191b92b7c438e0ba2fd3eaae9", ADDENDUM: "59ddb7dbf600041772956737f6e628adba21889d5c116257e3849e30fbda6c31",
          PRIOR_RT: "237b90fe1b020cfbe1f8b1cea11f0b10e62a24b7cc1d669022ef5a704eb67e6c",
          PRIOR_MAP: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "radial_gauge_map_probe_gpu"
if not torch.cuda.is_available():
    raise RuntimeError("radial_gauge_map_probe_gpu is a lane-1 CUDA script; no silent CPU fallback")
DEV = torch.device("cuda")
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; EVAL = (96, 160); CH = 8
ALL_SITES = [(kind, l) for l in range(NL) for kind in ("attn", "mlp")]
ARMS = ["DROP_RADIAL", "SCALE_RADIAL_2"]
MID_MLP = [("mlp", l) for l in range(2, 16)]
EARLY3 = [("mlp", 1), ("mlp", 2), ("mlp", 3)]
BARS = {"ce_tol": 1e-4, "xdev_tol": 0.015, "repro_tol": 0.003, "mid_drop_max": 0.03, "final_drop_min": 0.30, "early_scale_max": 0.05}
NULLS = {"mid_n_bad": 3, "mid_drop_bad": 0.10, "final_drop_max": 0.05, "early_scale_min": 0.20}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def split(w, x):
    xh = x / (x.norm(dim=-1, keepdim=True) + 1e-30)
    r = (w * xh).sum(-1, keepdim=True)
    return r, xh, w - r * xh


def arm_fn(arm):
    if arm == "IDENTITY":
        return lambda w, x: (lambda r, xh, wp: r * xh + wp)(*split(w, x))
    if arm == "DROP_RADIAL":
        return lambda w, x: split(w, x)[2]
    if arm == "SCALE_RADIAL_2":
        return lambda w, x: (lambda r, xh, wp: 2.0 * r * xh + wp)(*split(w, x))
    raise ValueError(arm)


@torch.no_grad()
def forward(m, idx, site=None, fn=None, stats=None):
    """tt_model semantics; at `site` the write w is replaced by fn(w, x_pre). stats[site] accumulates radial energy."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = (t.to(DEV) for t in R.rope(Tn))
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
    def apply(s, w, xpre):
        if stats is not None:
            r, _, wp = split(w, xpre); stats[s][0] += float((r ** 2).sum()); stats[s][1] += float((w ** 2).sum())
        return fn(w, xpre) if s == site else w
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
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, site=None, fn=None, stats=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg = forward(m, idx, site, fn, stats)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model().to(DEV)
    if smoke:
        g = torch.Generator().manual_seed(0)
        ev = torch.randint(0, 50000, (4, 257), generator=g).to(DEV); sites = [("attn", 0), ("mlp", 1), ("mlp", 17)]
        prior_base = float("nan"); prior_drop1 = float("nan")
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        ev = nat[EVAL[0]:EVAL[1]].to(DEV); sites = ALL_SITES
        prior_base = json.load(open(PRIOR_MAP))["baseline_ce_eval"]
        prior_drop1 = json.load(open(PRIOR_RT))["arms_ce_added"]["1"]["DROP_RADIAL"]
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce4 = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ident = {}
    for s in [("mlp", 17), ("attn", 0)]:
        ident[f"{s[0]}{s[1]}"] = abs(float(F.cross_entropy(forward(m, idx, s, arm_fn("IDENTITY")).reshape(-1, V), tgt.reshape(-1))) - ce4)
    stats = {s: [0.0, 0.0] for s in ALL_SITES}
    ce0 = ce_of(m, ev, stats=stats); log(stage="baseline", ce=ce0)
    arms = {}
    for s in sites:
        name = f"{s[0]}{s[1]}"; arms[name] = {}
        for arm in ARMS:
            arms[name][arm] = ce_of(m, ev, s, arm_fn(arm)) - ce0
            log(stage="arm", site=name, arm=arm, ce_added=arms[name][arm])
    rad_frac = {f"{s[0]}{s[1]}": (stats[s][0] / stats[s][1] if stats[s][1] > 0 else float("nan")) for s in ALL_SITES}
    g = lambda s, arm: arms.get(f"{s[0]}{s[1]}", {}).get(arm, float("nan"))
    mid = [g(s, "DROP_RADIAL") for s in MID_MLP if f"{s[0]}{s[1]}" in arms]
    early = [g(s, "SCALE_RADIAL_2") for s in EARLY3 if f"{s[0]}{s[1]}" in arms]
    d17 = g(("mlp", 17), "DROP_RADIAL"); d1 = g(("mlp", 1), "DROP_RADIAL")
    preds = {
        'pred_a_instrument': bool(abs(ce0 - prior_base) <= BARS["xdev_tol"] and max(ident.values()) <= BARS["ce_tol"] and abs(d1 - prior_drop1) <= BARS["repro_tol"]) if not smoke else bool(max(ident.values()) <= BARS["ce_tol"]),
        'pred_b_mid_mlp_radial_inert': bool(len(mid) == len(MID_MLP) and max(mid) <= BARS["mid_drop_max"]),
        'pred_c_final_mlp_radial_functional': bool(d17 >= BARS["final_drop_min"]),
        'pred_d_early_radial_soft_both_ways': bool(len(early) == 3 and max(early) <= BARS["early_scale_max"]),
    }
    nulls = {"b_null_ge_3_mid_mlp_drop_ge_.10": bool(sum(v >= NULLS["mid_drop_bad"] for v in mid) >= NULLS["mid_n_bad"]),
             "c_null_mlp17_drop_le_.05": bool(d17 <= NULLS["final_drop_max"]),
             "d_null_any_early_scale_ge_.20": bool(len(early) > 0 and max(early) >= NULLS["early_scale_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "same_device_baseline_abs_diff_le_1e-4": bool(abs(ce0 - prior_base) <= BARS["ce_tol"]),
           "instrument": {"baseline_ce": ce0, "prior_baseline": prior_base, "identity_abs_diff": ident, "drop_radial_mlp1": d1, "prior_drop_radial_mlp1": prior_drop1},
           "baseline_ce_eval": ce0, "n_eval_docs": int(ev.shape[0]), "arms_ce_added": arms, "radial_energy_fraction_eval": rad_frac,
           "price": {"gpu_doc_forwards": int(ev.shape[0]) * (1 + len(sites) * len(ARMS)) + 12, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "arms_ce_added", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "drop_radial": {k: v["DROP_RADIAL"] for k, v in arms.items()}}, indent=1))


if __name__ == "__main__":
    main()
