#!/usr/bin/env python
"""circuit_battery_reader_response_certificate -- the reader response of a bilinear block is an EXACT (2,2)-rational function.

bilin18's MLP is Bilinear with gated=False: mlp(u) = Down(Left(u) * Right(u)) + b = Q(u) + b with Q homogeneous of degree 2, and
RMSNorm is u -> sqrt(D) u / ||u||.  Therefore, for a writer's final-position write W and a downstream reader whose input residual is x,
the path-patched arm is EXACTLY

    mlp(rms_norm(x - tW)) - b = D * [ Q(x) - t B(x,W) + t^2 Q(W) ] / [ ||x||^2 - 2t <x,W> + t^2 ||W||^2 ],
    B(x,W) = Down(Left(x)*Right(W) + Left(W)*Right(x))   (the polarization of Q)

-- a vector of quadratics over a scalar quadratic.  Three vectors (Q(x), B(x,W), Q(W)) and three scalars determine the whole removal
curve with no forward passes, and they SPLIT the read of W inside the block into a CROSS term linear in W (the reader reading W against
its context) and a SELF term quadratic in W (the reader squaring W on its own).  That is a decomposition finer than an MLP block, in
closed form.  This rung tests the identity, measures which term carries the read, and prices the RMSNorm gain channel separately.
Writers/readers come from SS2809's battery receipt; the battery modules are imported, not edited.

# BQGATE: EXPERIMENT  pred_a_rational_identity_is_exact pred_b_cross_term_carries_the_read pred_c_gain_channel_is_small
#                     pred_d_linear_attribution_is_materially_wrong pred_e_half_removal_is_not_half_damage

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS the behaviour. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_READER_RESPONSE_CERTIFICATE_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_reader_response_certificate.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_READER_RESPONSE_CERTIFICATE_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_results.json"
RUNG = "circuit_battery_reader_response_certificate"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "99a3f59c12cd9849f77608ecd64fe0309beb1efb72f4ffe8a49ef693e79cb2d9",
          BATTERY: "6d1eda1cc05adf72c525375a0602bbafbf9b4335653be0e410de3d69da03265c",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL = R.D, R.NL
PER_CELL = 4 if SMOKE else 16
TOP_READERS = 3
TS = (0.25, 0.5, 0.75, 1.0)
BARS = {"exact_rel": 1e-5, "cross_frac": 0.70, "gain_share": 0.25, "linear_err": 0.25,
        "half_gap": 0.10, "floor": 0.5}
NULLS = {"cross_frac_le": 0.50, "gain_share_ge": 0.50, "linear_err_le": 0.10, "half_gap_le": 0.05}


def quad(blk, u):
    """Q(u) = Down(Left(u) * Right(u)), the homogeneous degree-2 part of the reader."""
    return blk.mlp.Down(blk.mlp.Left(u) * blk.mlp.Right(u))


def polar(blk, x, w):
    """B(x,W) = Down(Left(x)*Right(W) + Left(W)*Right(x)), the polarization of Q."""
    lx, rx = blk.mlp.Left(x), blk.mlp.Right(x)
    lw, rw = blk.mlp.Left(w), blk.mlp.Right(w)
    return blk.mlp.Down(lx * rw + lw * rx)


@torch.no_grad()
def probe(m, tokens, finals, writer, readers):
    """One forward that, at each chosen reader, records the exact rational ingredients and the
    actual mlp(rms_norm(x - tW)) at every t, plus the margin-level t-sweep of the FULL arm."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; W = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    rec = {}
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if W is not None:
            W = blk.lambdas[0] * W
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if writer == ("attn", site):
            W = torch.zeros_like(x); W[ar, finals] = write[ar, finals]
        x = x + write
        if ("mlp", site) in readers:
            xr = x[ar, finals].float()
            w = W[ar, finals].float()
            b = blk.mlp.Down_bias.float()
            qx, qw, bxw = quad(blk, xr), quad(blk, w), polar(blk, xr, w)
            nx = xr.pow(2).sum(-1); nw = w.pow(2).sum(-1); xw = (xr * w).sum(-1)
            actual = {}
            for t in TS:
                u = (x - t * W)[ar, finals]
                actual[t] = blk.mlp(F.rms_norm(u, (D,))).float()
            rec[("mlp", site)] = dict(qx=qx, qw=qw, bxw=bxw, nx=nx, nw=nw, xw=xw, b=b,
                                      actual=actual, base=blk.mlp(F.rms_norm(x[ar, finals], (D,))).float())
        out = blk.mlp(F.rms_norm(x, (D,)))
        if writer == ("mlp", site):
            W = torch.zeros_like(x); W[ar, finals] = out[ar, finals]
        x = x + out
    return rec


@torch.no_grad()
def sweep(m, tokens, finals, writer, readers, t):
    """Actual logits with a FRACTION t of the writer's write removed from every reader edge."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; W = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if W is not None:
            W = blk.lambdas[0] * W
        xr = x - t * W if (W is not None and ("attn", site) in readers) else x
        write, v1 = blk.attn(F.rms_norm(xr, (D,)), v1)
        if writer == ("attn", site):
            W = torch.zeros_like(x); W[ar, finals] = write[ar, finals]
        x = x + write
        xm = x - t * W if (W is not None and ("mlp", site) in readers) else x
        out = blk.mlp(F.rms_norm(xm, (D,)))
        if writer == ("mlp", site):
            W = torch.zeros_like(x); W[ar, finals] = out[ar, finals]
        x = x + out
    xf = x - t * W if W is not None else x
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(xf, (D,))) / 30.0))[ar, finals].float()


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    battery = json.load(open(BATTERY))
    tasks = [t for t in battery["summary"]["capable"]
             if battery["tasks"][t]["writer_recovery_select"] >= battery["bars"]["localise_rec"]]
    m = R.load_model().to(DEV).eval()
    fwd = 0
    results = {}
    for tid in tasks:
        tb = battery["tasks"][tid]
        wname = tb["writer"]
        kind = "attn" if wname.startswith("attn") else "mlp"
        writer = (kind, int(wname[len(kind):]))
        top = [k for k, _ in tb["reader_ladder"] if k.startswith("COMP_mlp")][:TOP_READERS]
        readers = {("mlp", int(k[len("COMP_mlp"):])) for k in top}
        rows = [r for r in BANK.build_rows(tid, per_cell=PER_CELL)
                if r["family"] == "A1" and r["split"] == "SELECT"]
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        rel, cross, gain, lin, half = [], [], [], [], []
        for b in CB.batches(rows):
            ids, fin, ans = CB.pack(b, "base")
            rec = probe(m, ids, fin, writer, readers); fwd += 1
            for comp, r in rec.items():
                for t in TS:
                    num = r["qx"] - t * r["bxw"] + t * t * r["qw"]
                    den = (r["nx"] - 2 * t * r["xw"] + t * t * r["nw"]).clamp_min(1e-12)
                    pred = D * num / den[:, None] + r["b"]
                    err = (pred - r["actual"][t]).abs().max(-1).values
                    rel.append((err / r["actual"][t].abs().max(-1).values.clamp_min(1e-6)).cpu().numpy())
                cross.append((r["bxw"].norm(dim=-1) /
                              (r["bxw"].norm(dim=-1) + r["qw"].norm(dim=-1)).clamp_min(1e-9)).cpu().numpy())
                # gain channel: numerator frozen at t=0, denominator at t=1 (pure RMSNorm rescaling)
                den1 = (r["nx"] - 2 * r["xw"] + r["nw"]).clamp_min(1e-12)
                only_gain = D * r["qx"] / den1[:, None] + r["b"]
                full = r["actual"][1.0]
                gain.append((( only_gain - r["base"]).norm(dim=-1) /
                             (full - r["base"]).norm(dim=-1).clamp_min(1e-9)).cpu().numpy())
            lg = {t: sweep(m, ids, fin, writer, readers, t) for t in (0.0,) + TS}
            fwd += len(lg)
            mm = {t: CB.margins(v, ans, cand) for t, v in lg.items()}
            dfull = (mm[0.0] - mm[1.0])
            dhalf = (mm[0.0] - mm[0.5])
            dq = (mm[0.0] - mm[0.25])
            floor = dfull.abs().clamp_min(BARS["floor"])
            lin.append(((4 * dq - dfull).abs() / floor).cpu().numpy())     # linear extrapolation from t=.25
            half.append(((dhalf - 0.5 * dfull).abs() / floor).cpu().numpy())
        results[tid] = {
            "writer": wname, "readers": sorted(f"mlp{l}" for _, l in readers),
            "max_rel_identity_error": float(np.concatenate(rel).max()),
            "median_rel_identity_error": float(np.median(np.concatenate(rel))),
            "cross_fraction": float(np.median(np.concatenate(cross))),
            "gain_share": float(np.median(np.concatenate(gain))),
            "linear_extrapolation_error": float(np.median(np.concatenate(lin))),
            "half_removal_gap": float(np.median(np.concatenate(half))),
            "rows": len(rows),
        }
        print(f"[cert] {tid:32s} rel={results[tid]['max_rel_identity_error']:.2e} "
              f"cross={results[tid]['cross_fraction']:.3f} gain={results[tid]['gain_share']:.3f} "
              f"lin={results[tid]['linear_extrapolation_error']:.3f} "
              f"half={results[tid]['half_removal_gap']:.3f}", flush=True)

    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    preds = {
        'pred_a_rational_identity_is_exact':
            bool(results and max(results[t]["max_rel_identity_error"] for t in results) <= BARS["exact_rel"]),
        'pred_b_cross_term_carries_the_read': bool(med("cross_fraction") >= BARS["cross_frac"]),
        'pred_c_gain_channel_is_small': bool(med("gain_share") <= BARS["gain_share"]),
        'pred_d_linear_attribution_is_materially_wrong': bool(med("linear_extrapolation_error") >= BARS["linear_err"]),
        'pred_e_half_removal_is_not_half_damage': bool(med("half_removal_gap") >= BARS["half_gap"]),
    }
    nulls = {
        "b_null_cross_le_.5": bool(med("cross_fraction") <= NULLS["cross_frac_le"]),
        "c_null_gain_ge_.5": bool(med("gain_share") >= NULLS["gain_share_ge"]),
        "d_null_linear_le_.1": bool(med("linear_extrapolation_error") <= NULLS["linear_err_le"]),
        "e_null_half_le_.05": bool(med("half_removal_gap") <= NULLS["half_gap_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "summary": {"tasks": sorted(results), "medians": {k: med(k) for k in
                          ("max_rel_identity_error", "cross_fraction", "gain_share",
                           "linear_extrapolation_error", "half_removal_gap")}},
              "tasks": results, "ts": list(TS), "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2809 capable+localised tasks x {TOP_READERS} readers x {len(TS)} removal fractions; "
              f"per_cell={PER_CELL}; no model loaded")
        sys.exit(0)
    main()
