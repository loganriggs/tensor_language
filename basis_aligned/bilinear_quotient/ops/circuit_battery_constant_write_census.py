#!/usr/bin/env python
"""circuit_battery_constant_write_census -- how much of bilin18 is literally a CONSTANT?

SS2835: attention 5's write is one fixed vector whose top direction IS its mean write (|cos| .9999996), whose gain varies by 8% and
never changes sign, and which a single constant vector reproduces to 94.3% of its 2.211-nat value -- the per-position gain is worth
.0022 nats. That is a compilation-relevant fact about ONE component. SS2834's census covered top-direction ENERGY for all 36 but not the
two properties that make a write a constant: whether its dominant direction is its mean, and whether its gain is stable.

This rung measures those for every component, and the CE cost of replacing each write outright with a single fixed vector fitted on
disjoint documents. The output is an enumeration of the parts of this model that are, to measurable tolerance, biases -- the cheapest
possible entries in a compiled tensor program.

# BQGATE: EXPERIMENT  pred_a_attn5_is_not_alone pred_b_constant_arms_recover_their_components
#                     pred_c_attn5_has_the_steadiest_gain pred_d_constant_writes_are_early
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in nats, POSITIVE = the arm HURTS. NOT the SS312 frontier's L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, SS2135); nothing installs; these are DIAGNOSTICS and
metric-constructed bases/spans remain CLOSED (SS2118 lineage).
Preregistration: polynomial_causal/CIRCUIT_BATTERY_CONSTANT_WRITE_CENSUS_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_constant_write_census.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_CONSTANT_WRITE_CENSUS_PREREGISTRATION.md"
IDENT = ROOT / "circuit_battery_attn5_direction_identity_results.json"
RUNG = "circuit_battery_constant_write_census"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "74bd5684e6f9619979ff6dfe5d384fe0f9b6f4e7e2467134563c5d460a9cfad2",
          IDENT: "1424ab0c93f5560009b5c64b206e6f734e6a79bbbca179db79eb4f2373b4e4ed",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL, V, T = R.D, R.NL, R.V, R.T
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
COMPONENTS = [(kd, l) for l in range(NL) for kd in ("attn", "mlp")]
NFIT = 8 if SMOKE else 24
NEVAL = 8 if SMOKE else 24
CHUNK = 8
COS_T, CV_T = 0.90, 0.50            # a write is "constant-like" if cos(top dir, mean) >= COS_T and gain CV <= CV_T
BARS = {"n_constant": 4, "recover": 0.80, "attn5_cv_top": 3, "early_frac": 0.60, "ce_tol": 0.01}
NULLS = {"n_constant_le": 1, "recover_le": 0.40, "attn5_cv_rank_ge": 12, "early_frac_le": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def doc_forward(m, idx, target=None, mode=None, const=None, collect=None):
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x; v1 = None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if target == ("attn", site):
            if collect is not None:
                collect.append(write.reshape(-1, D).float())
            if mode == "zero":
                write = torch.zeros_like(write)
            elif mode == "const":
                write = const.to(write.dtype).expand_as(write)
        x = x + write
        out = blk.mlp(F.rms_norm(x, (D,)))
        if target == ("mlp", site):
            if collect is not None:
                collect.append(out.reshape(-1, D).float())
            if mode == "zero":
                out = torch.zeros_like(out)
            elif mode == "const":
                out = const.to(out.dtype).expand_as(out)
        x = x + out
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

    ce_eval = ce_of(eval_docs)
    idx0 = eval_docs[:CHUNK, :T - 1].to(DEV); tgt0 = eval_docs[:CHUNK, 1:T].to(DEV)
    lg0 = doc_forward(m, idx0); fwd[0] += 1
    ce_manual = float(F.cross_entropy(lg0.reshape(-1, V), tgt0.reshape(-1)))
    ce_module = float(m(idx0.contiguous(), tgt0.contiguous()))

    rows = {}
    for comp in COMPONENTS:
        name = f"{comp[0]}{comp[1]}"
        coll = []
        for i in range(0, fit_docs.shape[0], CHUNK):
            doc_forward(m, fit_docs[i:i + CHUNK, :T - 1].to(DEV), target=comp, collect=coll); fwd[0] += 1
        W = torch.cat(coll, 0)
        mean_w = W.mean(0)
        u = torch.linalg.svd(W, full_matrices=False)[2][0].contiguous()
        u = u / u.norm()
        cos_mean = float(abs(torch.dot(u, mean_w / mean_w.norm().clamp_min(1e-9))))
        a = W @ u
        cv = float(a.std() / max(abs(float(a.mean())), 1e-9))
        del W, coll
        z = ce_of(eval_docs, target=comp, mode="zero") - ce_eval
        c = ce_of(eval_docs, target=comp, mode="const", const=mean_w) - ce_eval
        rows[name] = {"cos_top_vs_mean": cos_mean, "gain_cv": cv, "layer": comp[1], "kind": comp[0],
                      "zero_damage": z, "const_damage": c,
                      "recovered": (1.0 - c / z) if abs(z) > 1e-9 else float("nan"),
                      "constant_like": bool(cos_mean >= COS_T and cv <= CV_T)}
        print(f"[const] {name:8s} cos={cos_mean:.4f} cv={cv:6.3f} zero={z:+.4f} const={c:+.4f} "
              f"rec={rows[name]['recovered']:+.3f} {'CONSTANT-LIKE' if rows[name]['constant_like'] else ''}",
              flush=True)

    names = sorted(rows)
    const_like = [n for n in names if rows[n]["constant_like"]]
    rec = [rows[n]["recovered"] for n in const_like if not np.isnan(rows[n]["recovered"])]
    cv_order = sorted(names, key=lambda n: rows[n]["gain_cv"])
    attn5_cv_rank = cv_order.index("attn5") + 1
    early = [n for n in const_like if rows[n]["layer"] <= 8]
    early_frac = (len(early) / len(const_like)) if const_like else float("nan")
    preds = {
        'pred_a_attn5_is_not_alone': bool(len(const_like) >= BARS["n_constant"]),
        'pred_b_constant_arms_recover_their_components':
            bool(rec and float(np.median(rec)) >= BARS["recover"]),
        'pred_c_attn5_has_the_steadiest_gain': bool(attn5_cv_rank <= BARS["attn5_cv_top"]),
        'pred_d_constant_writes_are_early': bool(const_like and early_frac >= BARS["early_frac"]),
        'pred_e_instrument_reproduces_native_ce_matched': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_attn5_alone": bool(len(const_like) <= NULLS["n_constant_le"]),
        "b_null_const_arms_fail": bool(rec and float(np.median(rec)) <= NULLS["recover_le"]),
        "c_null_attn5_not_steady": bool(attn5_cv_rank >= NULLS["attn5_cv_rank_ge"]),
        "d_null_constants_are_late": bool(const_like and early_frac <= NULLS["early_frac_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "thresholds": {"cos": COS_T, "gain_cv": CV_T},
              "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_eval_native": ce_eval, "ce_manual_chunk": ce_manual, "ce_module_chunk": ce_module,
              "summary": {"constant_like": const_like, "n_constant_like": len(const_like),
                          "median_recovered_of_constant_like": float(np.median(rec)) if rec else None,
                          "attn5_gain_cv": rows["attn5"]["gain_cv"], "attn5_cv_rank": attn5_cv_rank,
                          "steadiest_gains_top8": cv_order[:8],
                          "early_fraction_of_constant_like": early_frac,
                          "total_zero_damage_of_constant_like": float(sum(rows[n]["zero_damage"] for n in const_like)),
                          "total_const_damage_of_constant_like": float(sum(rows[n]["const_damage"] for n in const_like))},
              "components": rows, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": len(COMPONENTS) * D,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(COMPONENTS)} components x (mean write fitted on {NFIT} docs, zero and const arms scored on "
              f"{NEVAL} DISJOINT docs); no model loaded")
        sys.exit(0)
    main()
