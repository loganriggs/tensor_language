#!/usr/bin/env python
# BQLANE: cpu
"""early_mlp_radial_tangential_probe -- for MLP blocks 0-3, split the write into its RADIAL part (a rescale of the pre-write
residual, one scalar per position) and its TANGENTIAL part, and price each in situ: DROP_RADIAL, RADIAL_ONLY, and radial-exact +
tangential-PCA-k arms. Follows §2700 (early price = fat head, ~256-d suffices) and §2699 (radial fractions).

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_radial_is_functional_mlp1 pred_c_tangential_cheaper_given_radial_mlp1 pred_d_radial_only_insufficient_mlp1

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-159 -- LOWER IS BETTER.
Preregistration: polynomial_causal/EARLY_MLP_RADIAL_TANGENTIAL_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/early_mlp_radial_tangential_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import mlp_final_blocks_fisher_certificate_probe as FC   # forward(m, idx, hook) with hook(l, mw, x_pre_write); ce_of(m, rows, hook)

ROOT = R.ROOT
PREREG = R.POLY / "EARLY_MLP_RADIAL_TANGENTIAL_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "early_mlp_isolated_token_program_probe_results.json"
OUT = ROOT / "early_mlp_radial_tangential_probe_results.json"
HASHES = {PREREG: "fb1869a056a2c80ccbd7cb2e2c5b1446efe13fb49df3704ef537622e9a0359f3",
          PRIOR: "4c07581ffb1ce2115d13277efabfbc91cc0caae40e51f8bca8e1fa4e2aa802b0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "early_mlp_radial_tangential_probe"
D, NL, V = R.D, R.NL, R.V
TI, CH = FC.TI, FC.CH
FIT = (0, 96); EVAL = (96, 160)
SITES = [0, 1, 2, 3]; KS = [64, 128]; KEY = 1
BARS = {"ce_tol": 1e-4, "drop_radial_min": 0.30, "rad_tan64_max": 0.15, "radial_only_min": 0.50}
NULLS = {"drop_radial_max": 0.05, "rad_tan64_min": 0.30, "radial_only_max": 0.10}


def check_hashes():
    for p, h in HASHES.items():
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def split(mw, x):
    """w = r xh + w_perp with xh the unit pre-write residual, per position."""
    xh = x / (x.norm(dim=-1, keepdim=True) + 1e-30)
    r = (mw * xh).sum(-1, keepdim=True)
    return r, xh, mw - r * xh


def fit_tangential(m, rows):
    acc = {l: R.Acc(D) for l in SITES}; rad_e = {l: 0.0 for l in SITES}; tot_e = {l: 0.0 for l in SITES}
    def hook(l, mw, x):
        if l in SITES:
            r, xh, wp = split(mw, x)
            acc[l].add(wp.reshape(-1, D)); rad_e[l] += float((r ** 2).sum()); tot_e[l] += float((mw ** 2).sum())
        return mw
    with torch.no_grad():
        for i in range(0, rows.shape[0], CH):
            FC.forward(m, rows[i:i + CH, :TI], hook)
    out = {}
    for l in SITES:
        C, _ = acc[l].cov(); mu = (acc[l].mu / acc[l].cnt).float()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        out[l] = {"mu": mu, "U": U.float(), "spec": R.spectrum(C), "radial_energy_fraction_fit": rad_e[l] / tot_e[l]}
    return out


def make_hook(l, arm, bases=None, k=None):
    def hook(ll, mw, x):
        if ll != l:
            return mw
        r, xh, wp = split(mw, x)
        if arm == "IDENTITY":
            return r * xh + wp
        if arm == "DROP_RADIAL":
            return wp
        if arm == "RADIAL_ONLY":
            return r * xh
        if arm == "RAD_EXACT_TAN":
            b = bases[l]; Uk = b["U"][:, :k]
            return r * xh + b["mu"] + ((wp - b["mu"]) @ Uk) @ Uk.T
        raise ValueError(arm)
    return hook


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=g); fit, ev = nat[:3], nat[3:]
        prior_base = float("nan"); prior_plain = {}
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, ev = nat[FIT[0]:FIT[1]], nat[EVAL[0]:EVAL[1]]
        pj = json.load(open(PRIOR)); prior_base = pj["baseline_ce_eval"]; prior_plain = pj["ladder_ce_added"]
    m = R.load_model()
    bases = fit_tangential(m, fit)
    print(json.dumps({"stage": "bases", "t": round(time.time() - started, 1)}), flush=True)
    ce0 = FC.ce_of(m, ev)
    ident = FC.ce_of(m, ev[:4], make_hook(KEY, "IDENTITY")) - FC.ce_of(m, ev[:4])
    arms = {}
    for l in SITES:
        arms[str(l)] = {}
        for arm in ("DROP_RADIAL", "RADIAL_ONLY"):
            arms[str(l)][arm] = FC.ce_of(m, ev, make_hook(l, arm)) - ce0
            print(json.dumps({"stage": "arm", "block": l, "arm": arm, "ce_added": arms[str(l)][arm], "t": round(time.time() - started, 1)}), flush=True)
        for k in KS:
            arms[str(l)][f"RAD_EXACT_TAN_{k}"] = FC.ce_of(m, ev, make_hook(l, "RAD_EXACT_TAN", bases, k)) - ce0
            print(json.dumps({"stage": "arm", "block": l, "arm": f"RAD_EXACT_TAN_{k}", "ce_added": arms[str(l)][f"RAD_EXACT_TAN_{k}"], "t": round(time.time() - started, 1)}), flush=True)
    a = arms[str(KEY)]
    inst = {"baseline_ce": ce0, "prior_baseline": prior_base, "abs_diff": abs(ce0 - prior_base) if not smoke else 0.0, "identity_arm_abs": abs(ident)}
    preds = {"pred_a_instrument": bool(inst["abs_diff"] <= BARS["ce_tol"] and inst["identity_arm_abs"] <= BARS["ce_tol"]),
             "pred_b_radial_is_functional_mlp1": bool(a["DROP_RADIAL"] >= BARS["drop_radial_min"]),
             "pred_c_tangential_cheaper_given_radial_mlp1": bool(a["RAD_EXACT_TAN_64"] <= BARS["rad_tan64_max"]),
             "pred_d_radial_only_insufficient_mlp1": bool(a["RADIAL_ONLY"] >= BARS["radial_only_min"])}
    nulls = {"b_null_drop_radial_le_.05": bool(a["DROP_RADIAL"] <= NULLS["drop_radial_max"]),
             "c_null_rad_tan64_ge_.30": bool(a["RAD_EXACT_TAN_64"] >= NULLS["rad_tan64_min"]),
             "d_null_radial_only_le_.10": bool(a["RADIAL_ONLY"] <= NULLS["radial_only_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "sites": SITES, "ks": KS, "key_site": KEY,
           "instrument": inst, "arms_ce_added": arms,
           "prior_plain_pca_ce_added": {str(l): {str(k): prior_plain.get(str(l), {}).get(str(k)) for k in KS} for l in SITES},
           "radial_energy_fraction_fit": {str(l): bases[l]["radial_energy_fraction_fit"] for l in SITES},
           "tangential_spectrum_fit": {str(l): bases[l]["spec"] for l in SITES},
           "n_fit_docs": int(fit.shape[0]), "n_eval_docs": int(ev.shape[0]),
           "price": {"gpu_forwards": 0, "cpu_doc_forwards": int(fit.shape[0]) + int(ev.shape[0]) * (1 + len(SITES) * (2 + len(KS))) + 8, "cpu_seconds": time.time() - started},
           "hashes": {str(p): h for p, h in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "arms_ce_added", "radial_energy_fraction_fit", "price")}, indent=1, default=str)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "arms": arms, "radial_energy_fraction_fit": out["radial_energy_fraction_fit"], "price": out["price"]}, indent=1))


if __name__ == "__main__":
    main()
