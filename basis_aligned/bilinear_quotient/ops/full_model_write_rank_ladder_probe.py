#!/usr/bin/env python
"""full_model_write_rank_ladder_probe -- ALL 36 writes truncated to their top-k in-situ write-PCA directions at once (bases fitted on docs
96-191, CE on docs 0-63), k in {64, 128, 256, 512, 768}, with the three stacks EARLY8 (blocks 0-3), MID14 (4-10), LATE14 (11-17) at the
same k and the stack cross term X_stack = ALL36 - EARLY8 - MID14 - LATE14. The price curve of a rank-k write program. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_all36_256 pred_c_all36_512 pred_d_stack_cross_term pred_e_768_near_free

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/FULL_MODEL_WRITE_RANK_LADDER_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/full_model_write_rank_ladder_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("full_model_write_rank_ladder_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "FULL_MODEL_WRITE_RANK_LADDER_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_joint_k_ladder_probe_results.json"   # §2709
PRIOR_E = ROOT / "early_joint_k_ladder_probe_results.json"   # §2711
OUT = ROOT / "full_model_write_rank_ladder_probe_results.json"
HASHES = {PREREG: "cda51b398a12beee40cf412458438c15d8422aab9f2febb8f1ab86d5a194e9a9", PRIOR: "6b2708a34e0eaf1cf226217995b37c6a8c6398da99dc0e0688d8c960a716b2fe", PRIOR_E: "1e0c96c50d63366a43a500716718752656ab8e1d8e93d09989757ed4d04235e7",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "full_model_write_rank_ladder_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KS = [64, 128, 256, 512, 768]
ALL36 = [(kind, l) for l in range(NL) for kind in ("attn", "mlp")]
STACKS = {"EARLY8": [(k, l) for l in range(0, 4) for k in ("attn", "mlp")], "MID14": [(k, l) for l in range(4, 11) for k in ("attn", "mlp")],
          "LATE14": [(k, l) for l in range(11, 18) for k in ("attn", "mlp")]}
PRIOR_BASE = 3.0322401; PRIOR_EARLY128 = 0.69191; PRIOR_LATE128 = 0.48584
BARS = {"ce_tol": 1e-4, "repro_tol": 0.01, "b_max": 0.80, "c_max": 0.25, "d_mult": 0.5, "e_max": 0.05}
NULLS = {"b_min": 1.5, "c_min": 0.50, "d_mult": 1.0, "e_min": 0.15}


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
    def finish(self):
        mu = self.mu / self.cnt; C = (self.S / self.cnt - torch.outer(mu, mu)).cpu()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        return {"mu": mu.float(), "U": U.float().to(DEV), "spec": R.spectrum(C)}


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


def trunc(b, k):
    Uk = b["U"][:, :k]; mu = b["mu"]
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
        sites = [("attn", 0), ("mlp", 1), ("mlp", 7), ("attn", 16), ("mlp", 17)]; ks = [64, 768]
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV); sites = ALL36; ks = KS
    stacks = {n: [s for s in ss if s in sites] for n, ss in STACKS.items()}
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce4 = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ident = abs(float(F.cross_entropy(forward(m, idx, {("mlp", 17): (lambda w: w)}).reshape(-1, V), tgt.reshape(-1))) - ce4)
    bases = fit(m, fit_rows, sites); log(stage="fit", n_sites=len(sites))
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    allk = {}; st = {n: {} for n in stacks}; xs = {}
    for k in ks:
        allk[k] = ce_of(m, ev, {s: trunc(bases[s], k) for s in sites}) - ce0
        for n in stacks:
            st[n][k] = ce_of(m, ev, {s: trunc(bases[s], k) for s in stacks[n]}) - ce0
        xs[k] = allk[k] - sum(st[n][k] for n in stacks)
        log(stage="k", k=k, all36=allk[k], stacks={n: st[n][k] for n in stacks}, x_stack=xs[k])
    mono = lambda d: all(d[ks[i + 1]] <= d[ks[i]] + 1e-9 for i in range(len(ks) - 1))
    all_mono = mono(allk) and all(mono(st[n]) for n in stacks)
    ssum = {k: sum(st[n][k] for n in stacks) for k in ks}
    full = all(k in allk for k in (128, 256, 512, 768))
    preds = {
        'pred_a_instrument': bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and ident <= BARS["ce_tol"] and full and abs(st["EARLY8"][128] - PRIOR_EARLY128) <= BARS["repro_tol"]
                                  and abs(st["LATE14"][128] - PRIOR_LATE128) <= BARS["repro_tol"] and all_mono) if not smoke else bool(ident <= BARS["ce_tol"] and all_mono),
        'pred_b_all36_256': bool(full and allk[256] <= BARS["b_max"]),
        'pred_c_all36_512': bool(full and allk[512] <= BARS["c_max"]),
        'pred_d_stack_cross_term': bool(full and xs[256] <= BARS["d_mult"] * ssum[256]),
        'pred_e_768_near_free': bool(full and allk[768] <= BARS["e_max"]),
    }
    nulls = {"b_null_all36_256_ge_1.5": bool(full and allk[256] >= NULLS["b_min"]), "c_null_all36_512_ge_.50": bool(full and allk[512] >= NULLS["c_min"]),
             "d_null_x_stack_ge_sum": bool(full and xs[256] >= NULLS["d_mult"] * ssum[256]), "e_null_all36_768_ge_.15": bool(full and allk[768] >= NULLS["e_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split, bases from docs 96-191); LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "ks": ks,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "identity_abs_diff": ident, "early8_128": st["EARLY8"].get(128), "prior_early8_128": PRIOR_EARLY128,
                          "late14_128": st["LATE14"].get(128), "prior_late14_128": PRIOR_LATE128, "all_monotone": bool(all_mono)},
           "baseline_ce_eval": ce0, "all36_ce_added": allk, "stack_ce_added": st, "stack_sum_ce_added": ssum, "x_stack_ce_added": xs,
           "write_spectrum_fit": {f"{s[0]}{s[1]}": bases[s]["spec"] for s in sites},
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + 4 * len(ks)) + 8, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "all36_ce_added", "stack_ce_added", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "all36": allk, "stacks": st, "x_stack": xs}, indent=1))


if __name__ == "__main__":
    main()
