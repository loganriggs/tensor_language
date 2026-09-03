#!/usr/bin/env python
"""mlp_final_blocks_low_rank_surrogate_probe -- can MLP16/MLP17's writes be replaced by a rank-k projection (fitted on one
document half) at negligible CE cost on the other half?  Follow-up to ledger §2692 (in-situ MLP-write eff rank 9 / 6).

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_low_rank_replicates_held_out pred_c_mlp17_rank8_cheap pred_d_mlp16_rank8_cheap
#                     pred_e_both_rank8_cheap pred_f_top_direction_tracks_entropy

SIGN CONVENTION (§2135): every CE number reported here is CE ADDED ABOVE THE REAL MODEL on the held-out half -- LOWER IS BETTER.
Preregistration: polynomial_causal/MLP_FINAL_BLOCKS_LOW_RANK_SURROGATE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/mlp_final_blocks_low_rank_surrogate_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

ROOT = R.ROOT
PREREG = R.POLY / "MLP_FINAL_BLOCKS_LOW_RANK_SURROGATE_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "mlp_in_situ_usage_rank_map_probe_results.json"
OUT = ROOT / "mlp_final_blocks_low_rank_surrogate_probe_results.json"
HASHES = {PREREG: "b86719fbeec9ea942bf5ec8b386b12d9a867d96bc7c0c0dcce9d682b484ec786", PRIOR: "63483cec6f68964235c1033ee0ebd0aba04c15b0dcd0f2f42e9b3d2b9b1a90b2",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1",
          R.CODE: "6cf514e75dfd03399f223a9ba5f6ebe5f4b1315bcb839a515e1c19e7b5474bd9"}
RUNG = "mlp_final_blocks_low_rank_surrogate_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256                       # inputs 0..255 predict 1..256
BLOCKS = [16, 17]
KS = [0, 1, 2, 4, 8, 16, 32, 64]
HALF = 96; CH = 8
BARS = {"ce_tol": 1e-4, "eff17_max": 15.0, "eff16_max": 25.0, "mlp17_k8_max": 0.02, "mlp16_k8_max": 0.05, "both_k8_max": 0.08,
        "spearman_min": 0.5}
NULLS = {"eff_either_min": 50.0, "mlp17_k8_min": 0.10, "mlp16_k8_min": 0.20, "both_k8_min": 0.30, "spearman_max": 0.2}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


@torch.no_grad()
def forward(m, idx, patch=None, collect=None):
    """R.manual_forward with an MLP-write patch: patch(l, mw[B,T,D]) -> mw. collect(l, mw, x_pre_final) on all positions."""
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
        x = x + a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        if patch is not None:
            mw = patch(l, mw)
        if collect is not None:
            collect(l, mw, x)
        x = x + mw + mlp.Down_bias
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, patch=None, want_entropy=False):
    tot, n, ents, coef = 0.0, 0, [], None
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg = forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
        if want_entropy:
            lp = F.log_softmax(lg.float(), -1); ents.append(-(lp.exp() * lp).sum(-1).reshape(-1))
    return tot / n, (torch.cat(ents) if want_entropy else None)


def fit_bases(m, rows):
    acc = {l: R.Acc(D) for l in BLOCKS}
    def collect(l, mw, x):
        if l in acc:
            acc[l].add(mw.reshape(-1, D))
    for i in range(0, rows.shape[0], CH):
        forward(m, rows[i:i + CH, :TI], collect=collect)
    out = {}
    for l in BLOCKS:
        C, e = acc[l].cov(); mu = (acc[l].mu / acc[l].cnt).float()
        ev, U = torch.linalg.eigh(C); ev = ev.flip(0); U = U.flip(1)
        out[l] = {"mu": mu, "U": U.float(), "ev": ev, "spec": R.spectrum(C), "mean_energy": e}
    return out


def make_patch(bases, cfg):
    """cfg: {block: k}. k=0 -> replace the write by its fitted mean; k>0 -> mean + rank-k projection of the deviation."""
    def patch(l, mw):
        if l not in cfg:
            return mw
        b = bases[l]; k = cfg[l]; dev = mw - b["mu"]
        if k == 0:
            return b["mu"].expand_as(mw).clone()
        Uk = b["U"][:, :k]
        return b["mu"] + (dev @ Uk) @ Uk.T
    return patch


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    check_hashes()
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model()
    if smoke:   # random tokens only; never the registered rows
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=g); code = torch.randint(0, 50000, (4, 257), generator=g)
        h0, h1, code_eval = nat[:3], nat[3:], code
    else:
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        code = torch.load(R.CODE, map_location="cpu"); code = (code["rows"] if isinstance(code, dict) else code).long()
        h0, h1, code_eval = nat[:HALF], nat[HALF:2 * HALF], code[HALF:2 * HALF]
    # instrument
    idx = h1[:4, :TI]; tgt = h1[:4, 1:TI + 1]
    ce_manual = float(F.cross_entropy(forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ce_module = float(m(idx.contiguous(), tgt.contiguous()))
    # fit on h0, replicate spectra on h1
    bases = fit_bases(m, h0)
    held = fit_bases(m, h1)
    spectra = {str(l): {"fit_half": bases[l]["spec"], "held_half": held[l]["spec"],
                        "cos_u1_mean": float(torch.dot(bases[l]["U"][:, 0], bases[l]["mu"]) / bases[l]["mu"].norm()),
                        "cos_u1_fit_vs_held": abs(float(torch.dot(bases[l]["U"][:, 0], held[l]["U"][:, 0]))),
                        "mean_norm": float(bases[l]["mu"].norm()), "mean_energy_fit": bases[l]["mean_energy"]} for l in BLOCKS}
    # baseline + ladder on natural h1
    ce0, ent0 = ce_of(m, h1, None, want_entropy=True)
    ladder = {}
    for l in BLOCKS:
        ladder[str(l)] = {str(k): ce_of(m, h1, make_patch(bases, {l: k}))[0] - ce0 for k in KS}
    ladder["both"] = {str(k): ce_of(m, h1, make_patch(bases, {16: k, 17: k}))[0] - ce0 for k in (8, 32)}
    # code transfer of the natural-fitted rank-8 bases
    ce0_code = ce_of(m, code_eval)[0]
    code_added = {str(l): ce_of(m, code_eval, make_patch(bases, {l: 8}))[0] - ce0_code for l in BLOCKS}
    # top-direction coefficient of the block-17 write vs baseline next-token entropy (same tokens)
    coefs = []
    def collect(l, mw, x):
        if l == 17:
            coefs.append(((mw - bases[17]["mu"]).reshape(-1, D) @ bases[17]["U"][:, 0]))
    for i in range(0, h1.shape[0], CH):
        forward(m, h1[i:i + CH, :TI], collect=collect)
    c1 = torch.cat(coefs); rho = spearman(c1, ent0)
    rho_norm = spearman(c1.abs(), ent0)
    e17, e16 = held[17]["spec"]["eff_rank"], held[16]["spec"]["eff_rank"]
    preds = {
        'pred_a_instrument': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
        'pred_b_low_rank_replicates_held_out': bool(e17 <= BARS["eff17_max"] and e16 <= BARS["eff16_max"]),
        'pred_c_mlp17_rank8_cheap': bool(ladder["17"]["8"] <= BARS["mlp17_k8_max"]),
        'pred_d_mlp16_rank8_cheap': bool(ladder["16"]["8"] <= BARS["mlp16_k8_max"]),
        'pred_e_both_rank8_cheap': bool(ladder["both"]["8"] <= BARS["both_k8_max"]),
        'pred_f_top_direction_tracks_entropy': bool(abs(rho) >= BARS["spearman_min"]),
    }
    nulls = {"b_null_eff_either_ge_50": bool(e17 >= NULLS["eff_either_min"] or e16 >= NULLS["eff_either_min"]),
             "c_null_mlp17_k8_ge_.10": bool(ladder["17"]["8"] >= NULLS["mlp17_k8_min"]),
             "d_null_mlp16_k8_ge_.20": bool(ladder["16"]["8"] >= NULLS["mlp16_k8_min"]),
             "e_null_both_k8_ge_.30": bool(ladder["both"]["8"] >= NULLS["both_k8_min"]),
             "f_null_abs_spearman_le_.2": bool(abs(rho) <= NULLS["spearman_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
           "instrument": {"ce_manual": ce_manual, "ce_module": ce_module, "abs_diff": abs(ce_manual - ce_module)},
           "baseline_ce": {"natural_h1": ce0, "code_h1": ce0_code, "n_fit_docs": int(h0.shape[0]), "n_eval_docs": int(h1.shape[0])},
           "spectra": spectra, "ce_added_ladder": ladder, "code_transfer_rank8_ce_added": code_added,
           "entropy": {"spearman_c1_vs_entropy": rho, "spearman_abs_c1_vs_entropy": rho_norm, "mean_entropy_h1": float(ent0.mean()),
                       "c1_mean": float(c1.mean()), "c1_std": float(c1.std())},
           "price": {"gpu_forwards": 0, "cpu_full_forwards_docs": int(h0.shape[0]) * 2 + int(h1.shape[0]) * (3 + 2 * len(KS) + 2 + 1) + int(code_eval.shape[0]) * 3,
                     "cpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "spectra", "ce_added_ladder", "entropy", "instrument", "price")}, indent=1, default=str)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ladder": ladder, "spectra": {l: (spectra[l]["held_half"]["eff_rank"], spectra[l]["cos_u1_mean"]) for l in spectra},
                      "entropy": out["entropy"], "code": code_added}, indent=1))


if __name__ == "__main__":
    main()
