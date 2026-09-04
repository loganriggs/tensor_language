#!/usr/bin/env python
"""circuit_battery_write_rank_census -- is the rank-1 write a fact about ATTENTION 5, or about bilin18?

SS2832/SS2833: attention 5's write is 98.1% one direction, that direction is universal across corpora (|cos| .997 natural vs code), and
a held-out rank-32 basis reproduces 97.8% of its 2.20-nat value. Those sections read as statements about the price cliff -- but they
have no across-component control, so they cannot yet distinguish "attention 5's write is remarkably low-rank" from "every write in this
model is low-rank and attention 5 is merely expensive". This rung supplies that control: the same measurement on all 36 components,
with the basis fitted on one document set and scored on a DISJOINT one.

# BQGATE: EXPERIMENT  pred_a_low_rank_writes_are_architectural pred_b_attn5_is_extreme_not_unique
#                     pred_c_cheap_surrogates_are_general pred_d_expensive_is_not_hard_to_approximate
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in nats on natural documents, POSITIVE = the arm HURTS. NOT the SS312 frontier's L2 (CE added
above the real model by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, SS2135); nothing installs; the rank
arms are DIAGNOSTICS and metric-constructed bases/spans remain CLOSED (SS2118 lineage).
Preregistration: polynomial_causal/CIRCUIT_BATTERY_WRITE_RANK_CENSUS_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_write_rank_census.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_WRITE_RANK_CENSUS_PREREGISTRATION.md"
HELDOUT = ROOT / "circuit_battery_attn5_heldout_surrogate_results.json"
RUNG = "circuit_battery_write_rank_census"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "3190144d5b0610a384aa4268ed7121326366a3398fd72f603b9918ca6f6ff961",
          HELDOUT: "a474688fba25cdfdcac1e4b87518efc949a51ccaf2284ae4437d095a122706c0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL, V, T = R.D, R.NL, R.V, R.T
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
COMPONENTS = [(kd, l) for l in range(NL) for kd in ("attn", "mlp")]
RANK = 32
NFIT = 8 if SMOKE else 24
NEVAL = 8 if SMOKE else 24
CHUNK = 8
BARS = {"median_e1": 0.50, "attn5_rank_top": 6, "median_rank32": 0.10, "rho": 0.50, "ce_tol": 0.01}
NULLS = {"median_e1_le": 0.20, "attn5_rank_ge": 20, "median_rank32_ge": 0.50, "rho_ge": 0.80}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def doc_forward(m, idx, target=None, mode=None, basis=None, collect=None):
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
            elif mode == "rank":
                w = write.float(); write = ((w @ basis) @ basis.T).to(write.dtype)
        x = x + write
        out = blk.mlp(F.rms_norm(x, (D,)))
        if target == ("mlp", site):
            if collect is not None:
                collect.append(out.reshape(-1, D).float())
            if mode == "zero":
                out = torch.zeros_like(out)
            elif mode == "rank":
                w = out.float(); out = ((w @ basis) @ basis.T).to(out.dtype)
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
    fitted = 0
    for comp in COMPONENTS:
        name = f"{comp[0]}{comp[1]}"
        coll = []
        for i in range(0, fit_docs.shape[0], CHUNK):
            idx = fit_docs[i:i + CHUNK, :T - 1].to(DEV)
            doc_forward(m, idx, target=comp, collect=coll); fwd[0] += 1
        W = torch.cat(coll, 0)
        sv = torch.linalg.svdvals(W)
        p = (sv ** 2 / (sv ** 2).sum())
        e1 = float(p[0])
        eff = float(1.0 / (p ** 2).sum())
        B = torch.linalg.svd(W, full_matrices=False)[2][:RANK].T.contiguous()
        fitted += RANK * D
        del W, coll
        z = ce_of(eval_docs, target=comp, mode="zero") - ce_eval
        r = ce_of(eval_docs, target=comp, mode="rank", basis=B) - ce_eval
        rows[name] = {"top_direction_energy": e1, "effective_rank": eff,
                      "zero_damage": z, f"rank{RANK}_damage": r,
                      "retained_fraction": 1.0 - (r / z if abs(z) > 1e-9 else float("nan"))}
        print(f"[census] {name:8s} e1={e1:.3f} effrank={eff:5.1f} zero={z:+.4f} r{RANK}={r:+.4f}", flush=True)

    names = sorted(rows)
    e1s = [rows[n]["top_direction_energy"] for n in names]
    r32 = [rows[n][f"rank{RANK}_damage"] for n in names]
    zs = [rows[n]["zero_damage"] for n in names]
    rank_order = sorted(names, key=lambda n: -rows[n]["top_direction_energy"])
    attn5_rank = rank_order.index("attn5") + 1
    def spearman(a, b):
        rk = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
        ra, rb = rk(a), rk(b)
        return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else float("nan")
    rho = spearman(zs, r32)
    preds = {
        'pred_a_low_rank_writes_are_architectural': bool(float(np.median(e1s)) >= BARS["median_e1"]),
        'pred_b_attn5_is_extreme_not_unique': bool(attn5_rank <= BARS["attn5_rank_top"]),
        'pred_c_cheap_surrogates_are_general': bool(float(np.median(r32)) <= BARS["median_rank32"]),
        'pred_d_expensive_is_not_hard_to_approximate': bool(rho <= BARS["rho"]),
        'pred_e_instrument_reproduces_native_ce_matched': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_writes_are_high_rank": bool(float(np.median(e1s)) <= NULLS["median_e1_le"]),
        "b_null_attn5_ordinary": bool(attn5_rank >= NULLS["attn5_rank_ge"]),
        "c_null_surrogates_expensive": bool(float(np.median(r32)) >= NULLS["median_rank32_ge"]),
        "d_null_expensive_is_hard": bool(rho >= NULLS["rho_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "rank": RANK, "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_eval_native": ce_eval, "ce_manual_chunk": ce_manual, "ce_module_chunk": ce_module,
              "summary": {"median_top_direction_energy": float(np.median(e1s)),
                          "attn5_energy_rank": attn5_rank, "attn5_top_direction_energy": rows["attn5"]["top_direction_energy"],
                          "median_rank32_damage": float(np.median(r32)),
                          "median_zero_damage": float(np.median(zs)),
                          "rho_zero_vs_rank32": rho,
                          "energy_rank_order_top8": rank_order[:8],
                          "most_expensive_top8": sorted(names, key=lambda n: -rows[n]["zero_damage"])[:8]},
              "components": rows, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": fitted,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(COMPONENTS)} components x (fit basis on {NFIT} docs, score rank-{RANK} and zero on "
              f"{NEVAL} DISJOINT docs); no model loaded")
        sys.exit(0)
    main()
