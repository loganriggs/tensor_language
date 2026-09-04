#!/usr/bin/env python
"""circuit_battery_causal_basis_rank -- how MANY unfitted causal directions does the read use?

SS2826: one parameter-free direction, u = W_U[answer] - W_U[best competing candidate], carries .199 of the reader block's damage and is
2.4x more task-specific than the block, while holding .0021 of the removal effect's energy. That is partial: four fifths of the block's
effect on the margin travels elsewhere. The obvious question is whether the rest is also causally structured -- the margin is a
comparison against ONE competitor, but the model is choosing among many candidates, so the read may act on several answer-versus-
competitor axes at once. This rung builds the span of u_i = W_U[answer] - W_U[competitor_i] for the top k competitors of each row
(k = 1, 2, 4, 8), orthonormalises it per row, and removes only the component of the removal effect inside it. Still ZERO fitted
parameters: every direction comes from the unembedding and that row's own native logits.

# BQGATE: EXPERIMENT  pred_a_four_causal_directions_carry_more pred_b_share_grows_with_k
#                     pred_c_random_basis_is_inert pred_d_specificity_survives_the_wider_basis
#                     pred_e_the_causal_basis_stays_low_energy

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's answer; ratio = max(|d_P|,|d_C|)/max(d_A1,.5),
LOWER IS MORE SPECIFIC. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_CAUSAL_BASIS_RANK_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_causal_basis_rank.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_CAUSAL_BASIS_RANK_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
CAUSAL1 = ROOT / "circuit_battery_causal_direction_read_results.json"
RUNG = "circuit_battery_causal_basis_rank"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "00238533400e5ff68ec8303a8f0cbebeb68a68e8ffcf18d209928ad9d2008ed8",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D = R.D
WRITER = ("attn", 8)
READERS = (10, 11)
KS = (1, 2, 4, 8)
SEED = 2826
PER_CELL = 4 if SMOKE else 16
BARS = {"k4_share": 0.40, "growth": 0.10, "random_share": 0.05, "specific_gain": 0.10,
        "basis_energy": 0.05, "admit_block": 0.25, "floor": 0.5}
NULLS = {"k4_share_le": 0.20, "growth_le": 0.0, "random_share_ge": 0.30, "basis_energy_ge": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def causal_basis(m, logits, ans, cand, k):
    """Per-row orthonormal span of W_U[answer] - W_U[competitor_i] for the top k competitors."""
    sub = logits[:, cand]
    pos = (cand.unsqueeze(0) == ans.unsqueeze(1))
    order = (sub - 1e4 * pos.float()).argsort(dim=1, descending=True)[:, :k]
    comps = cand[order]                                     # (B, k)
    W = m.lm_head.weight.float()
    V = W[ans].unsqueeze(1) - W[comps]                      # (B, k, D)
    Q, _ = torch.linalg.qr(V.transpose(1, 2))               # (B, D, k)
    return Q


@torch.no_grad()
def run_basis(m, tokens, finals, layer, B):
    """Remove only the component of the reader's removal effect inside the per-row subspace B (B, D, k)."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; W = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    delta = None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if W is not None:
            W = blk.lambdas[0] * W
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if WRITER == ("attn", site):
            W = torch.zeros_like(x); W[ar, finals] = write[ar, finals]
        x = x + write
        if site == layer and W is not None and B is not None:
            nat = blk.mlp(F.rms_norm(x, (D,)))
            rem = blk.mlp(F.rms_norm(x - W, (D,)))
            d = (nat - rem)[ar, finals].float()
            delta = d
            if isinstance(B, str):
                proj = d
            else:
                coef = torch.einsum("bd,bdk->bk", d, B)
                proj = torch.einsum("bk,bdk->bd", coef, B)
            out = nat.clone()
            out[ar, finals] = nat[ar, finals] - proj.to(nat.dtype)
        else:
            out = blk.mlp(F.rms_norm(x, (D,)))
        x = x + out
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return logits, delta


def arm(m, rows, cand, layer, kind, k, g, fwd):
    dm, en = [], []
    for b in CB.batches(rows):
        ids, fin, ans = CB.pack(b, "base")
        lg, _ = run_basis(m, ids, fin, layer, None); fwd[0] += 1
        mn = CB.margins(lg, ans, cand)
        if kind == "all":
            B = "all"
        elif kind == "causal":
            B = causal_basis(m, lg, ans, cand, k)
        else:
            v = torch.randn(len(b), D, k, generator=g).to(DEV)
            B = torch.linalg.qr(v)[0]
        lg2, d = run_basis(m, ids, fin, layer, B); fwd[0] += 1
        dm.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
        if not isinstance(B, str) and d is not None:
            coef = torch.einsum("bd,bdk->bk", d, B)
            frac = (coef ** 2).sum(-1) / (d ** 2).sum(-1).clamp_min(1e-12)
            en.append(frac.cpu().numpy())
    return (float(np.concatenate(dm).mean()) if dm else float("nan"),
            float(np.concatenate(en).mean()) if en else float("nan"))


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fwd = [0]
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        per_layer = {}
        for layer in READERS:
            cells = {f: [r for r in rows if r["family"] == f and r["split"] == "OOD"]
                     for f in ("A1", "P", "C") if f in fams}
            blockd = {f: arm(m, cells[f], cand, layer, "all", 0, g, fwd)[0] for f in cells}
            shares, energies, k4 = {}, {}, {}
            for k in KS:
                d, e = arm(m, cells["A1"], cand, layer, "causal", k, g, fwd)
                shares[f"k{k}"] = d / max(blockd["A1"], BARS["floor"])
                energies[f"k{k}"] = e
                if k == 4:
                    k4["A1"] = d
                    for f in cells:
                        if f != "A1":
                            k4[f] = arm(m, cells[f], cand, layer, "causal", k, g, fwd)[0]
            rnd, ren = arm(m, cells["A1"], cand, layer, "random", 4, g, fwd)
            ratio = lambda dd: max(abs(dd.get("P", 0.0)), abs(dd.get("C", 0.0))) / max(dd["A1"], BARS["floor"])
            per_layer[f"mlp{layer}"] = {
                "block_damage": blockd, "causal_share": shares, "causal_energy": energies,
                "k4_damage": k4, "random_share": rnd / max(blockd["A1"], BARS["floor"]),
                "random_energy": ren,
                "growth": shares["k8"] - shares["k1"],
                "block_ratio": ratio(blockd), "k4_ratio": ratio(k4),
                "specific_gain": ratio(blockd) - ratio(k4),
                "admissible": bool(k4["A1"] >= BARS["admit_block"] * max(blockd["A1"], BARS["floor"])),
            }
            p = per_layer[f"mlp{layer}"]
            print(f"[basis] {tid:28s} mlp{layer} k1={shares['k1']:.2f} k2={shares['k2']:.2f} "
                  f"k4={shares['k4']:.2f} k8={shares['k8']:.2f} rnd={p['random_share']:.3f} "
                  f"E4={energies['k4']:.4f} gain={p['specific_gain']:.2f} adm={p['admissible']}", flush=True)
        results[tid] = per_layer

    flat = [results[t][l] for t in results for l in results[t]]
    med = lambda k: float(np.median([f[k] for f in flat])) if flat else float("nan")
    meds = lambda k: float(np.median([f["causal_share"][k] for f in flat])) if flat else float("nan")
    mede = lambda k: float(np.median([f["causal_energy"][k] for f in flat])) if flat else float("nan")
    adm = [f for f in flat if f["admissible"]]
    preds = {
        'pred_a_four_causal_directions_carry_more': bool(meds("k4") >= BARS["k4_share"]),
        'pred_b_share_grows_with_k': bool(med("growth") >= BARS["growth"]),
        'pred_c_random_basis_is_inert': bool(med("random_share") <= BARS["random_share"]),
        'pred_d_specificity_survives_the_wider_basis':
            bool(adm and float(np.median([f["specific_gain"] for f in adm])) >= BARS["specific_gain"]),
        'pred_e_the_causal_basis_stays_low_energy': bool(mede("k4") <= BARS["basis_energy"]),
    }
    nulls = {
        "a_null_k4_le_.2": bool(meds("k4") <= NULLS["k4_share_le"]),
        "b_null_no_growth": bool(med("growth") <= NULLS["growth_le"]),
        "c_null_random_ge_.3": bool(med("random_share") >= NULLS["random_share_ge"]),
        "e_null_high_energy": bool(mede("k4") >= NULLS["basis_energy_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in READERS], "ks": list(KS), "seed": SEED,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "n_cells": len(flat), "n_admissible": len(adm),
                          "median_share_by_k": {f"k{k}": meds(f"k{k}") for k in KS},
                          "median_energy_by_k": {f"k{k}": mede(f"k{k}") for k in KS},
                          "medians": {k: med(k) for k in ("growth", "random_share", "random_energy",
                                                          "block_ratio", "k4_ratio", "specific_gain")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls,
                      "share_by_k": result["summary"]["median_share_by_k"],
                      "energy_by_k": result["summary"]["median_energy_by_k"],
                      "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x readers {READERS} x causal bases k={KS}; "
              f"zero fitted parameters; no model loaded")
        sys.exit(0)
    main()
