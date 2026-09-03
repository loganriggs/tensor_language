#!/usr/bin/env python
"""mlp_final_blocks_quadratic_form_rank_probe -- the §2694 rank-8 write surrogate of MLP16/MLP17 is exactly 8 quadratic forms
in the block's rms-normed input.  What is their exact weight rank, and how many eigen-directions does the model USE in situ?

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_forms_high_rank pred_c_mlp16_r64_cheap pred_d_mlp16_r256_near_exact pred_e_mlp17_r64

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-191 -- LOWER IS BETTER.
Preregistration: polynomial_causal/MLP_FINAL_BLOCKS_QUADRATIC_FORM_RANK_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/mlp_final_blocks_quadratic_form_rank_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import mlp_final_blocks_low_rank_surrogate_probe as S

ROOT = R.ROOT
PREREG = R.POLY / "MLP_FINAL_BLOCKS_QUADRATIC_FORM_RANK_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "mlp_final_blocks_low_rank_surrogate_probe_results.json"
OUT = ROOT / "mlp_final_blocks_quadratic_form_rank_probe_results.json"
HASHES = {PREREG: "47cda64016d6ccc02df4b18a91fe7078c7c0654656fd534ffadc53057a500f8e", PRIOR: "8a88b71455c8087ca5de84cfa83e71df35080236932b4ccbd466ce0bc089326e",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "mlp_final_blocks_quadratic_form_rank_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI, HALF, CH = S.TI, S.HALF, S.CH
BLOCKS = [16, 17]
K = 8
RS = [16, 64, 256, D]
BARS = {"ce_tol": 1e-4, "exact_tol": 0.002, "eff_min": 200.0, "mlp16_r64_max": 0.06, "mlp16_r256_max": 0.045, "mlp17_r64_max": 0.12}
NULLS = {"eff_max": 64.0, "mlp16_r64_min": 0.15, "mlp16_r256_min": 0.10, "mlp17_r64_min": 0.25}


def check_hashes():
    for p, h in HASHES.items():
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


@torch.no_grad()
def forward(m, idx, patch=None):
    """S.forward with the MLP-write patch also receiving the rms-normed block input: patch(l, mw, xhat) -> mw."""
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
            mw = patch(l, mw, xhat)
        x = x + mw + mlp.Down_bias
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


@torch.no_grad()
def ce_of(m, rows, patch=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg = forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


@torch.no_grad()
def forms(m, l, U):
    """Qs_j = sym(Left^T diag(M_j) Right), M = U^T Down (k x 4608). Returns eigenvalues [k, D] (desc by |lambda|) and vectors [k, D, D]."""
    mlp = m.transformer.h[l].mlp
    L, Rw, Dw = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()   # [4608, D], [4608, D], [D, 4608]
    M = U.T @ Dw                                                                             # [k, 4608]
    lam, vec, spec = [], [], []
    for j in range(U.shape[1]):
        Q = (L * M[j][:, None]).T @ Rw; Q = 0.5 * (Q + Q.T)
        e, v = torch.linalg.eigh(Q)
        o = e.abs().argsort(descending=True); e, v = e[o], v[:, o]
        p2 = e ** 2; p2 = p2 / p2.sum()
        eff = float(torch.exp(-(p2[p2 > 0] * p2[p2 > 0].log()).sum()))
        spec.append({"energy_eff_rank": eff, "top1_abs_share": float(e[0].abs() / e.abs().sum()),
                     "positive_mass_fraction": float(e.clamp_min(0).sum() / e.abs().sum()), "top4_lambda": [float(t) for t in e[:4]]})
        lam.append(e); vec.append(v)
    return torch.stack(lam), torch.stack(vec), spec


def make_patch(bases, F_, cfg):
    """cfg: {block: r}. Write' = mu + U (c_r(xhat) - U^T mu), c_r,j = sum_{i<=r} lambda_ji (v_ji . xhat)^2."""
    def patch(l, mw, xhat):
        if l not in cfg:
            return mw
        b = bases[l]; U = b["U"][:, :K]; lam, vec = F_[l]; r = cfg[l]
        Vr = vec[:, :, :r].permute(1, 0, 2).reshape(D, K * r)                # [D, k*r]
        proj = (xhat.reshape(-1, D) @ Vr).reshape(-1, K, r)                    # [N, k, r]
        c = (proj ** 2 * lam[:, :r]).sum(-1)                                   # [N, k]
        return (b["mu"] + (c - U.T @ b["mu"]) @ U.T).reshape(mw.shape)
    return patch


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    torch.manual_seed(0)
    if smoke:   # random tokens only; never the registered rows
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=g); h0, h1 = nat[:3], nat[3:]
        prior_k8 = {"16": 0.0, "17": 0.0}
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        h0, h1 = nat[:HALF], nat[HALF:2 * HALF]
        prior_k8 = {l: json.load(open(PRIOR))["ce_added_ladder"][l]["8"] for l in ("16", "17")}
    m = R.load_model()
    # instrument (i): this forward == S's forward with no patch
    ce_a = ce_of(m, h1[:4]); ce_b = S.ce_of(m, h1[:4])[0]
    inst = {"ce_manual": ce_a, "ce_module": ce_b, "abs_diff": abs(ce_a - ce_b)}
    bases = S.fit_bases(m, h0)
    F_ = {}; spectra = {}
    for l in BLOCKS:
        lam, vec, spec = forms(m, l, bases[l]["U"][:, :K]); F_[l] = (lam, vec); spectra[str(l)] = spec
        print(json.dumps({"block": l, "energy_eff_ranks": [round(s["energy_eff_rank"], 1) for s in spec]}), flush=True)
    ce0 = ce_of(m, h1)
    ladder = {str(l): {} for l in BLOCKS}
    for l in BLOCKS:
        for r in RS:
            ladder[str(l)][str(r)] = ce_of(m, h1, make_patch(bases, F_, {l: r})) - ce0
            print(json.dumps({"block": l, "r": r, "ce_added": ladder[str(l)][str(r)]}), flush=True)
    # coefficient RMS per form on EVAL (exact forms), collected from a plain pass
    rms = {str(l): None for l in BLOCKS}
    acc = {l: torch.zeros(K, dtype=torch.float64) for l in BLOCKS}; cnt = [0]
    def collect(l, mw, xhat):
        if l in acc:
            c = (mw - bases[l]["mu"]).reshape(-1, D) @ bases[l]["U"][:, :K]
            acc[l] += (c.double() ** 2).sum(0)
            if l == BLOCKS[0]:
                cnt[0] += c.shape[0]
        return mw
    for i in range(0, h1.shape[0], CH):
        forward(m, h1[i:i + CH, :TI], collect)
    rms = {str(l): [float(t) for t in (acc[l] / max(cnt[0], 1)).sqrt()] for l in BLOCKS}
    mean_eff = {str(l): sum(s["energy_eff_rank"] for s in spectra[str(l)]) / K for l in BLOCKS}
    exact_diff = {l: abs(ladder[l][str(D)] - prior_k8[l]) for l in ("16", "17")}
    preds = {"pred_a_instrument": bool(inst["abs_diff"] <= BARS["ce_tol"] and (smoke or all(v <= BARS["exact_tol"] for v in exact_diff.values()))),
             "pred_b_forms_high_rank": bool(all(v >= BARS["eff_min"] for v in mean_eff.values())),
             "pred_c_mlp16_r64_cheap": bool(ladder["16"]["64"] <= BARS["mlp16_r64_max"]),
             "pred_d_mlp16_r256_near_exact": bool(ladder["16"]["256"] <= BARS["mlp16_r256_max"]),
             "pred_e_mlp17_r64": bool(ladder["17"]["64"] <= BARS["mlp17_r64_max"])}
    nulls = {"b_null_eff_either_le_64": bool(any(v <= NULLS["eff_max"] for v in mean_eff.values())),
             "c_null_mlp16_r64_ge_.15": bool(ladder["16"]["64"] >= NULLS["mlp16_r64_min"]),
             "d_null_mlp16_r256_ge_.10": bool(ladder["16"]["256"] >= NULLS["mlp16_r256_min"]),
             "e_null_mlp17_r64_ge_.25": bool(ladder["17"]["64"] >= NULLS["mlp17_r64_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 96-191; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "instrument": inst, "exact_vs_prior_k8": {"prior": prior_k8, "abs_diff": exact_diff},
           "baseline_ce": ce0, "k": K, "rs": RS, "ce_added_ladder": ladder, "form_spectra": spectra, "mean_energy_eff_rank": mean_eff,
           "coef_rms_eval": rms, "r64_excess_over_exact": {l: ladder[l]["64"] - ladder[l][str(D)] for l in ladder},
           "price": {"gpu_forwards": 0, "cpu_full_forwards_docs": 4 * 2 + int(h0.shape[0]) + int(h1.shape[0]) * (2 + len(BLOCKS) * len(RS)),
                     "cpu_seconds": time.time() - started},
           "hashes": {str(p): h for p, h in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added_ladder", "mean_energy_eff_rank", "price")}, indent=1, default=str)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ladder": ladder, "mean_eff": mean_eff, "exact_diff": exact_diff, "price": out["price"]}, indent=1))


if __name__ == "__main__":
    main()
