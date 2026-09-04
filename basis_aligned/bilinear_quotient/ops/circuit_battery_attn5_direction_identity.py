#!/usr/bin/env python
"""circuit_battery_attn5_direction_identity -- can the price cliff be COMPILED to one vector and one scalar?

SS2834: attention 5 has the most one-dimensional write in bilin18 (top-direction energy .981, effective rank 1.0, 1st of 36).
SS2833: that direction is universal -- |cos| 1.000 across disjoint natural document sets and .997 against code.
SS2832: a constant write costs .119 nats against a 2.200-nat ablation, so most of the component's value is direction-and-scale, not
context.

If those hold together, attention 5 is compilable: write(pos) ~ alpha(pos) * u, with u ONE FIXED VECTOR estimated off-line and alpha ONE
SCALAR PER POSITION. This rung measures exactly that reduction and asks what u and alpha are: whether u is simply the mean-write
direction, and how much alpha varies at all. Both the reconstruction and its ingredients are fitted on documents DISJOINT from those
they are scored on.

# BQGATE: EXPERIMENT  pred_a_rank_one_reconstruction_is_cheap pred_b_the_scalar_carries_value
#                     pred_c_the_direction_is_the_mean_write pred_d_the_gain_is_stable
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in nats, POSITIVE = the arm HURTS. NOT the SS312 frontier's L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, SS2135); nothing installs here; metric-constructed
bases/spans remain CLOSED (SS2118 lineage) and the reconstruction below is a DIAGNOSTIC of compilability, not a proposed interface.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ATTN5_DIRECTION_IDENTITY_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_attn5_direction_identity.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ATTN5_DIRECTION_IDENTITY_PREREGISTRATION.md"
CENSUS = ROOT / "circuit_battery_write_rank_census_results.json"
RUNG = "circuit_battery_attn5_direction_identity"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "4c9be5221a8e5bebbdbbb7c382af4d373ec1a863df35f972f7e0253bbf8d545d",
          CENSUS: "269689cc0586ef591c8395c338d4c8b526244f47c90540d26aa3db272bcbca41",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, V, T = R.D, R.V, R.T
LAYER = 5
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
NFIT = 8 if SMOKE else 24
NEVAL = 8 if SMOKE else 24
CHUNK = 8
BARS = {"rank1_nats": 0.05, "scalar_value": 0.03, "cos_mean": 0.90, "gain_cv": 0.50, "ce_tol": 0.01}
NULLS = {"rank1_ge": 0.50, "scalar_value_le": 0.0, "cos_mean_le": 0.50, "gain_cv_ge": 1.50}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def doc_forward(m, idx, mode=None, u=None, alpha_const=None, collect=None):
    """mode: None | 'zero' | 'rank1' (alpha(pos)*u) | 'const' (alpha_const*u)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x; v1 = None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if site == LAYER:
            if collect is not None:
                collect.append(write.reshape(-1, D).float())
            if mode == "zero":
                write = torch.zeros_like(write)
            elif mode == "rank1":
                w = write.float()
                write = ((w @ u).unsqueeze(-1) * u).to(write.dtype)
            elif mode == "const":
                write = (alpha_const * u).to(write.dtype).expand_as(write)
        x = x + write
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def main():
    t0 = time.time()
    check_hashes()
    m = R.load_model().to(DEV).eval()
    t = torch.load(NAT, map_location="cpu")
    t = (t["rows"] if isinstance(t, dict) else t).long()
    fit_docs, eval_docs = t[:NFIT], t[NFIT:NFIT + NEVAL]
    fwd = [0]

    def ce_of(docs, **kw):
        s_, n_ = 0.0, 0
        for i in range(0, docs.shape[0], CHUNK):
            idx = docs[i:i + CHUNK, :T - 1].to(DEV)
            tgt = docs[i:i + CHUNK, 1:T].to(DEV)
            lg = doc_forward(m, idx, **kw); fwd[0] += 1
            s_ += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum"))
            n_ += tgt.numel()
        return s_ / n_

    # fit u and the constant gain on the FIT documents only
    coll = []
    ce_of(fit_docs, collect=coll)
    Wf = torch.cat(coll, 0)
    u = torch.linalg.svd(Wf, full_matrices=False)[2][0].contiguous()
    u = u / u.norm()
    mean_w = Wf.mean(0)
    cos_mean = float(abs(torch.dot(u, mean_w / mean_w.norm().clamp_min(1e-9))))
    alpha_fit = (Wf @ u)
    alpha_const = float(alpha_fit.mean())
    del Wf, coll

    # alpha statistics measured on the EVAL documents (never used for fitting)
    coll = []
    ce_eval = ce_of(eval_docs, collect=coll)
    We = torch.cat(coll, 0)
    alpha_eval = (We @ u)
    gain_mean = float(alpha_eval.mean())
    gain_cv = float(alpha_eval.std() / max(abs(gain_mean), 1e-9))
    resid = float(((We - (alpha_eval.unsqueeze(-1) * u)) ** 2).sum() / (We ** 2).sum())
    del We, coll

    idx0 = eval_docs[:CHUNK, :T - 1].to(DEV); tgt0 = eval_docs[:CHUNK, 1:T].to(DEV)
    lg0 = doc_forward(m, idx0); fwd[0] += 1
    ce_manual = float(F.cross_entropy(lg0.reshape(-1, V), tgt0.reshape(-1)))
    ce_module = float(m(idx0.contiguous(), tgt0.contiguous()))

    d_zero = ce_of(eval_docs, mode="zero") - ce_eval
    d_rank1 = ce_of(eval_docs, mode="rank1", u=u) - ce_eval
    d_const = ce_of(eval_docs, mode="const", u=u, alpha_const=alpha_const) - ce_eval

    preds = {
        'pred_a_rank_one_reconstruction_is_cheap': bool(d_rank1 <= BARS["rank1_nats"]),
        'pred_b_the_scalar_carries_value': bool(d_const - d_rank1 >= BARS["scalar_value"]),
        'pred_c_the_direction_is_the_mean_write': bool(cos_mean >= BARS["cos_mean"]),
        'pred_d_the_gain_is_stable': bool(gain_cv <= BARS["gain_cv"]),
        'pred_e_instrument_reproduces_native_ce_matched': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_rank1_fails": bool(d_rank1 >= NULLS["rank1_ge"]),
        "b_null_scalar_worthless": bool(d_const - d_rank1 <= NULLS["scalar_value_le"]),
        "c_null_direction_not_mean": bool(cos_mean <= NULLS["cos_mean_le"]),
        "d_null_gain_wild": bool(gain_cv >= NULLS["gain_cv_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_eval_native": ce_eval, "ce_manual_chunk": ce_manual, "ce_module_chunk": ce_module,
              "summary": {"zero_damage": d_zero, "rank1_damage": d_rank1, "const_damage": d_const,
                          "scalar_value_nats": d_const - d_rank1,
                          "cos_u_vs_mean_write": cos_mean, "gain_mean": gain_mean,
                          "gain_cv": gain_cv, "residual_energy_off_u": resid,
                          "alpha_const_fit": alpha_const},
              "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": D + 1,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: attention {LAYER} write reduced to alpha(pos)*u, u and the constant gain fitted on "
              f"{NFIT} documents and scored on {NEVAL} DISJOINT ones; no model loaded")
        sys.exit(0)
    main()
