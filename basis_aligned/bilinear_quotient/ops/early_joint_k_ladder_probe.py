#!/usr/bin/env python
"""early_joint_k_ladder_probe -- the 8 early writes (attn_l, mlp_l, l = 0..3) truncated to their top-k in-situ write-PCA directions (bases
fitted on docs 96-191) one at a time (SINGLE_s(k)) and all together (JOINT(k)), k in {32, 64, 128, 256, 512}, CE on docs 0-63 (FRESH split);
JOINT_MLP4(128) for mlp0-3 only. Superadditivity factor F(k) = JOINT(k) / sum_s SINGLE_s(k). Companion to §2709 (late stack). CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_early_superadditive pred_c_factor_decays pred_d_early_stack_price_512
#                     pred_e_early_costs_more_than_late

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/EARLY_JOINT_K_LADDER_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/early_joint_k_ladder_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("early_joint_k_ladder_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "EARLY_JOINT_K_LADDER_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_joint_k_ladder_probe_results.json"   # §2709
OUT = ROOT / "early_joint_k_ladder_probe_results.json"
HASHES = {PREREG: "821edbd04b7e24335467a03265b8b7a4d19b2558eb00817e7c06d6f112e7b900", PRIOR: "6b2708a34e0eaf1cf226217995b37c6a8c6398da99dc0e0688d8c960a716b2fe",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "early_joint_k_ladder_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KS = [32, 64, 128, 256, 512]
LATE14 = [(kind, l) for l in range(0, 4) for kind in ("attn", "mlp")]     # EARLY8 (name kept so the shared code path is unchanged)
MLP7 = [("mlp", l) for l in range(0, 4)]                                    # MLP4
PRIOR_BASE = 3.0322401; PRIOR_LATE_JOINT128 = 0.48584
BARS = {"ce_tol": 1e-4, "mono_n_min": 9, "b_f32_min": 1.5, "c_decay": 0.8, "d_max": 0.10, "e_min": PRIOR_LATE_JOINT128}
NULLS = {"b_f32_max": 1.1, "d_min": 0.30, "e_max": 0.30}


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
        nat = torch.randint(0, 50000, (8, 257), generator=g).to(DEV); fit_rows, ev = nat[:4], nat[4:]
        sites = [("attn", 0), ("mlp", 1), ("mlp", 3)]; ks = [32, 512]
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV); sites = LATE14; ks = KS
    mlp7 = [s for s in MLP7 if s in sites]
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce4 = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ident = abs(float(F.cross_entropy(forward(m, idx, {("mlp", 1): (lambda w: w)}).reshape(-1, V), tgt.reshape(-1))) - ce4)
    bases = fit(m, fit_rows, sites); log(stage="fit", mlp1_eff_rank=bases[("mlp", 1)]["spec"]["eff_rank"])
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    single = {f"{s[0]}{s[1]}": {} for s in sites}; joint = {}; joint_mlp7 = {}; sum_single = {}; factor = {}
    for k in ks:
        for s in sites:
            single[f"{s[0]}{s[1]}"][k] = ce_of(m, ev, {s: trunc(bases[s], k)}) - ce0
        joint[k] = ce_of(m, ev, {s: trunc(bases[s], k) for s in sites}) - ce0
        sum_single[k] = sum(single[f"{s[0]}{s[1]}"][k] for s in sites)
        factor[k] = joint[k] / sum_single[k] if sum_single[k] > 0 else float("nan")
        log(stage="k", k=k, joint=joint[k], sum_single=sum_single[k], factor=factor[k])
    k_e = 128 if 128 in ks else ks[-1]
    joint_mlp7[k_e] = ce_of(m, ev, {s: trunc(bases[s], k_e) for s in mlp7}) - ce0
    chains = [[joint[k] for k in ks]] + [[single[n][k] for k in ks] for n in single]
    mono = sum(all(c[i + 1] <= c[i] + 1e-9 for i in range(len(c) - 1)) for c in chains)
    e_frac = joint_mlp7[k_e] / joint[k_e] if joint[k_e] > 0 else float("nan")
    f32 = factor.get(32, float("nan")); f512 = factor.get(512, float("nan"))
    preds = {
        'pred_a_instrument': bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and ident <= BARS["ce_tol"] and mono >= BARS["mono_n_min"]) if not smoke else bool(ident <= BARS["ce_tol"] and mono == len(chains)),
        'pred_b_early_superadditive': bool(f32 >= BARS["b_f32_min"]),
        'pred_c_factor_decays': bool(f512 <= BARS["c_decay"] * f32),
        'pred_d_early_stack_price_512': bool(512 in joint and joint[512] <= BARS["d_max"]),
        'pred_e_early_costs_more_than_late': bool(128 in joint and joint[128] >= BARS["e_min"]),
    }
    nulls = {"b_null_f32_le_1.1": bool(f32 <= NULLS["b_f32_max"]), "c_null_f512_ge_f32": bool(f512 >= f32),
             "d_null_joint512_ge_.30": bool(512 in joint and joint[512] >= NULLS["d_min"]),
             "e_null_joint128_le_.30": bool(128 in joint and joint[128] <= NULLS["e_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split, bases from docs 96-191); LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "ks": ks,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "identity_abs_diff": ident, "prior_late_joint128": PRIOR_LATE_JOINT128, "n_monotone_chains": int(mono), "n_chains": len(chains)},
           "baseline_ce_eval": ce0, "joint_ce_added": joint, "sum_single_ce_added": sum_single, "superadditivity_factor": factor,
           "joint_mlp4_ce_added": joint_mlp7, "mlp4_fraction_of_joint": {k_e: e_frac}, "single_ce_added": single,
           "write_spectrum_fit": {f"{s[0]}{s[1]}": bases[s]["spec"] for s in sites},
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (2 + len(ks) * (len(sites) + 1)) + 8, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "joint_ce_added", "superadditivity_factor", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "joint": joint, "factor": factor}, indent=1))


if __name__ == "__main__":
    main()
