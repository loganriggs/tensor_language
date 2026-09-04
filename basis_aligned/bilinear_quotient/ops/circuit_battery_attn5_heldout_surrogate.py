#!/usr/bin/env python
"""circuit_battery_attn5_heldout_surrogate -- does SS2832's cheap surrogate survive being fitted OUT OF SAMPLE?

SS2830: attention 5 costs 2.200 nats of document CE when ablated (3rd of 36) and is 20.4x more expensive per unit of its own write norm
than the median component. SS2831: its class gate lives in heads {5, 7}, which carry .542 of its class damage, and its class and margin
head maps are identical. SS2825/SS2826: in this model, ENERGY-ranked structure is not causal structure -- an in-sample rank-4 subspace
holds .700 of a removal effect's energy and delivers .139 of its damage.

Those three put a falsifiable, useful prediction on the table: a surrogate for attention 5 chosen CAUSALLY (keep the two heads that
gate the class) should cost less document CE than one chosen by ENERGY (project the write onto its own top-k singular directions), even
when the energy surrogate is given far more freedom. This rung measures that directly.

# BQGATE: EXPERIMENT  pred_a_heldout_rank_transports pred_b_cross_corpus_rank_transports
#                     pred_c_parameter_free_heads_transport pred_d_the_top_direction_is_stable
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in NATS on natural documents, POSITIVE = the arm HURTS. This is a LOCAL SURROGATE
measurement; it is NOT the SS312 frontier's L2 (CE added above the real model by an installed approximation, LOWER IS BETTER, frontier
norm-2304 at 2.6735), NOTHING here installs into that frontier, and no number below may be quoted as an L2. The energy-basis arm is a
NEGATIVE CONTROL, not a proposed interface -- metric-constructed bases/spans are CLOSED (SS2118 lineage) and this rung does not reopen
them.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ATTN5_HELDOUT_SURROGATE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_attn5_heldout_surrogate.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ATTN5_HELDOUT_SURROGATE_PREREGISTRATION.md"
PRICE = ROOT / "circuit_battery_attn5_class_gate_price_results.json"
HEADSPLIT = ROOT / "circuit_battery_attn5_head_class_split_results.json"
RUNG = "circuit_battery_attn5_heldout_surrogate"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "13e1a6725a2bc4f057e36117702970a6522b9e9dd441f953baeede2bdf7a098a",
          PRICE: "870f1c4065ddfea418a545db5536f93776cd445ff8e1b97cbb9f33f88b592e56",
          HEADSPLIT: "42c1d8adcd9c7d241d32b1383f9a8ca17a06d0d10c95ad99000c32b072bbd7fc",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD, NL, V, T = R.D, R.NH, R.HD, R.NL, R.V, R.T
LAYER = 5
KEEP_HEADS = (5, 7)          # SS2831's class-gate pair, fixed before this run
RANKS = (8, 32, 128)
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
CODE = ROOT / ".rowcache_terminal_copy_induction_v2/ood_code.pt"
NFIT = 8 if SMOKE else 48        # documents the basis is fitted on
NEVAL = 8 if SMOKE else 48       # DISJOINT documents it is scored on
CHUNK = 8
BARS = {"heldout_rank32": 0.10, "cross_rank32": 0.25, "heads_gap": 0.05, "cos_within": 0.90,
        "cos_cross": 0.70, "ce_tol": 0.01}
NULLS = {"heldout_rank32_ge": 0.50, "cross_rank32_ge": 1.00, "heads_gap_ge": 0.20,
         "cos_within_le": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def doc_forward(m, idx, mode=None, basis=None, mean=None, collect=None):
    """mode: None native | 'zero' | 'heads' | 'rank' | 'mean'.  collect: list to receive attn5 writes."""
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x; v1 = None
    stats = {}
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if site == LAYER:
            cap = {}
            h = blk.attn.c_proj.register_forward_pre_hook(lambda _m, a: cap.__setitem__("y", a[0]))
            try:
                write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
            finally:
                h.remove()
            ycat = cap["y"]
            if collect is not None:
                collect.append(write.reshape(-1, D).float().cpu())
            if mode == "zero":
                write = torch.zeros_like(write)
            elif mode == "heads":
                mask = torch.zeros_like(ycat)
                for hd in KEEP_HEADS:
                    mask[..., hd * HD:(hd + 1) * HD] = ycat[..., hd * HD:(hd + 1) * HD]
                kept = blk.attn.c_proj(mask)
                stats["kept_energy_frac"] = float((kept.float() ** 2).sum() / (write.float() ** 2).sum())
                write = kept
            elif mode == "rank":
                w = write.float()
                write = ((w @ basis) @ basis.T).to(write.dtype)
            elif mode == "mean":
                write = mean.to(write.dtype).expand_as(write)
        else:
            write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        x = x + write
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0), stats


def main():
    t0 = time.time()
    check_hashes()
    m = R.load_model().to(DEV).eval()

    def load(path, lo, hi):
        t = torch.load(path, map_location="cpu")
        t = (t["rows"] if isinstance(t, dict) else t).long()
        return t[lo:hi]

    fit_docs = load(NAT, 0, NFIT)                       # basis fitted here
    eval_docs = load(NAT, NFIT, NFIT + NEVAL)           # scored here -- DISJOINT
    code_docs = load(CODE, 0, NEVAL)                    # a different corpus entirely
    fwd = [0]

    def ce_of(docs, **kw):
        s_, n_, st = 0.0, 0, {}
        for i in range(0, docs.shape[0], CHUNK):
            idx = docs[i:i + CHUNK, :T - 1].to(DEV)
            tgt = docs[i:i + CHUNK, 1:T].to(DEV)
            lg, stats = doc_forward(m, idx, **kw); fwd[0] += 1
            s_ += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum"))
            n_ += tgt.numel()
            st.update(stats)
        return s_ / n_, st

    def basis_from(docs):
        coll = []
        ce_of(docs, collect=coll)
        W = torch.cat(coll, 0).to(DEV)
        Vh = torch.linalg.svd(W, full_matrices=False)[2]
        return Vh, W.mean(0, keepdim=True)

    # instrument check on MATCHED data (SS2832's pred_e compared different sample sets and could not be read)
    idx0 = eval_docs[:CHUNK, :T - 1].to(DEV); tgt0 = eval_docs[:CHUNK, 1:T].to(DEV)
    lg0, _ = doc_forward(m, idx0); fwd[0] += 1
    ce_manual_chunk = float(F.cross_entropy(lg0.reshape(-1, V), tgt0.reshape(-1)))
    ce_module_chunk = float(m(idx0.contiguous(), tgt0.contiguous()))

    Vf, meanf = basis_from(fit_docs)                    # fitted on fit_docs only
    Ve, _meane = basis_from(eval_docs)                  # only to measure direction stability
    Vc, _meanc = basis_from(code_docs)
    fitted = sum(k * D for k in RANKS) + D

    cos_within = float(abs(torch.dot(Vf[0], Ve[0])))
    cos_cross = float(abs(torch.dot(Vf[0], Vc[0])))

    ce_eval, _ = ce_of(eval_docs)
    ce_code, _ = ce_of(code_docs)
    res = {"natural_heldout": {}, "code": {}}
    for label, docs, base in (("natural_heldout", eval_docs, ce_eval), ("code", code_docs, ce_code)):
        z, _ = ce_of(docs, mode="zero")
        h, hs = ce_of(docs, mode="heads")
        mn, _ = ce_of(docs, mode="mean", mean=meanf)
        row = {"native": base, "ZERO": z - base, "HEADS_57": h - base, "MEAN_fit": mn - base,
               "heads57_energy_fraction": hs.get("kept_energy_frac", float("nan"))}
        for k in RANKS:
            c, _ = ce_of(docs, mode="rank", basis=Vf[:k].T.contiguous())
            row[f"RANK_{k}_fitbasis"] = c - base
        res[label] = row
        print(f"[heldout] {label}: " + " ".join(f"{k}={v:+.4f}" for k, v in row.items()
                                                if k not in ("native", "heads57_energy_fraction")), flush=True)

    heads_gap = abs(res["natural_heldout"]["HEADS_57"] - 0.0883)   # SS2832's in-sample value
    preds = {
        'pred_a_heldout_rank_transports': bool(res["natural_heldout"]["RANK_32_fitbasis"] <= BARS["heldout_rank32"]),
        'pred_b_cross_corpus_rank_transports': bool(res["code"]["RANK_32_fitbasis"] <= BARS["cross_rank32"]),
        'pred_c_parameter_free_heads_transport': bool(heads_gap <= BARS["heads_gap"]),
        'pred_d_the_top_direction_is_stable': bool(cos_within >= BARS["cos_within"] and cos_cross >= BARS["cos_cross"]),
        'pred_e_instrument_reproduces_native_ce_matched':
            bool(abs(ce_manual_chunk - ce_module_chunk) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_heldout_fails": bool(res["natural_heldout"]["RANK_32_fitbasis"] >= NULLS["heldout_rank32_ge"]),
        "b_null_cross_fails": bool(res["code"]["RANK_32_fitbasis"] >= NULLS["cross_rank32_ge"]),
        "c_null_heads_do_not_transport": bool(heads_gap >= NULLS["heads_gap_ge"]),
        "d_null_direction_unstable": bool(cos_within <= NULLS["cos_within_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "keep_heads": list(KEEP_HEADS), "ranks": list(RANKS),
              "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_manual_chunk": ce_manual_chunk, "ce_module_chunk": ce_module_chunk,
              "summary": {"arms": res, "cos_top_direction_within_corpus": cos_within,
                          "cos_top_direction_cross_corpus": cos_cross,
                          "heads_gap_vs_in_sample": heads_gap,
                          "in_sample_reference_heads57": 0.0883},
              "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": fitted,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1400])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: attention {LAYER} surrogates fitted on {NFIT} documents, scored on {NEVAL} DISJOINT "
              f"natural documents and on {NEVAL} code documents; no model loaded")
        sys.exit(0)
    main()
