#!/usr/bin/env python
"""circuit_battery_causal_direction_read -- rank by CAUSALITY instead of energy: one unembedding-defined direction, zero fitted parameters.

SS2825: the in-sample rank-4 subspace of the reader's removal effect holds .700 of its ENERGY but delivers .139 of its DAMAGE. Every
localisation this campaign has tried (unit magnitude SS2822, exact lens magnitude SS2823, SVD energy SS2824/SS2825) ranked directions by
size, and size is not causality here. This rung ranks by causality instead, and does it WITHOUT FITTING ANYTHING: for each row the
direction that matters is u = W_U[answer] - W_U[best competing candidate], read straight off the unembedding, normalised. The arm removes
only the component of the removal effect along u. If the block's read of attention 8's write acts on the answer-versus-competitor axis,
this rank-1 parameter-free direction should beat a fitted rank-4 energy subspace outright.

# BQGATE: EXPERIMENT  pred_a_causal_direction_carries_the_damage pred_b_causal_beats_fitted_energy
#                     pred_c_random_direction_is_inert pred_d_the_causal_direction_is_low_energy
#                     pred_e_causal_arm_is_more_specific

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's answer; ratio = max(|d_P|,|d_C|)/max(d_A1,.5),
LOWER IS MORE SPECIFIC. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_CAUSAL_DIRECTION_READ_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_causal_direction_read.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import circuit_battery_reader_rank_decomposition as RD
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_CAUSAL_DIRECTION_READ_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_causal_direction_read"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "e364719426a9a7556be429df9f38580bbac6c8a64ae5a1768a6b8a358c36d731",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D = R.D
WRITER = ("attn", 8)
READERS = (10, 11)
SEED = 2825
PER_CELL = 4 if SMOKE else 16
BARS = {"causal_share": 0.50, "beat_energy": 0.20, "random_share": 0.05, "causal_energy": 0.25,
        "specific_gain": 0.10, "admit_block": 0.25, "floor": 0.5}
NULLS = {"causal_share_le": 0.15, "beat_energy_le": 0.0, "random_share_ge": 0.30,
         "causal_energy_ge": 0.60}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def directions(m, logits, ans, cand):
    """u = W_U[answer] - W_U[best competing candidate], per row, normalised. No fitting."""
    sub = logits[:, cand]
    pos = (cand.unsqueeze(0) == ans.unsqueeze(1))
    comp = cand[(sub - 1e4 * pos.float()).argmax(1)]
    u = (m.lm_head.weight[ans] - m.lm_head.weight[comp]).float()
    return u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9), comp


@torch.no_grad()
def run_dir(m, tokens, finals, layer, U):
    """Forward removing only the component of the reader's removal effect along the per-row direction U.

    U=None -> native; U="all" -> the whole removal effect; else U is (B, D) unit rows.
    """
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; W = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if W is not None:
            W = blk.lambdas[0] * W
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if WRITER == ("attn", site):
            W = torch.zeros_like(x); W[ar, finals] = write[ar, finals]
        x = x + write
        if site == layer and W is not None and U is not None:
            nat = blk.mlp(F.rms_norm(x, (D,)))
            rem = blk.mlp(F.rms_norm(x - W, (D,)))
            d = (nat - rem)[ar, finals].float()
            proj = d if isinstance(U, str) else (d * U).sum(-1, keepdim=True) * U
            out = nat.clone()
            out[ar, finals] = nat[ar, finals] - proj.to(nat.dtype)
        else:
            out = blk.mlp(F.rms_norm(x, (D,)))
        x = x + out
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()


def arm_damage(m, rows, cand, layer, kind, g, fwd):
    """kind in {'all','causal','random'}; returns mean damage and (for causal) mean energy fraction."""
    dm, en = [], []
    for b in CB.batches(rows):
        ids, fin, ans = CB.pack(b, "base")
        lg = run_dir(m, ids, fin, layer, None); fwd[0] += 1
        mn = CB.margins(lg, ans, cand)
        if kind == "all":
            U = "all"
        elif kind == "causal":
            U, _c = directions(m, lg, ans, cand)
        else:
            v = torch.randn(len(b), D, generator=g).to(DEV)
            U = v / v.norm(dim=-1, keepdim=True)
        lg2 = run_dir(m, ids, fin, layer, U); fwd[0] += 1
        dm.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
        if not isinstance(U, str):
            _lg, d = RD.run_rank(m, ids, fin, layer, None); fwd[0] += 1
            frac = ((d * U).sum(-1) ** 2) / (d ** 2).sum(-1).clamp_min(1e-12)
            en.append(frac.cpu().numpy())
    return (float(np.concatenate(dm).mean()) if dm else float("nan"),
            float(np.concatenate(en).mean()) if en else float("nan"))


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    geo = json.load(open(ROOT / "circuit_battery_removal_effect_geometry_results.json"))
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
            blockd = {f: arm_damage(m, cells[f], cand, layer, "all", g, fwd)[0] for f in cells}
            caus = {}
            cen = None
            for f in cells:
                d, e = arm_damage(m, cells[f], cand, layer, "causal", g, fwd)
                caus[f] = d
                if f == "A1":
                    cen = e
            rnd, ren = arm_damage(m, cells["A1"], cand, layer, "random", g, fwd)
            energy4 = geo["tasks"][tid][f"mlp{layer}"]["in_sample_share"]
            ratio = lambda d: max(abs(d.get("P", 0.0)), abs(d.get("C", 0.0))) / max(d["A1"], BARS["floor"])
            share = lambda v: v / max(blockd["A1"], BARS["floor"])
            per_layer[f"mlp{layer}"] = {
                "block_damage": blockd, "causal_damage": caus,
                "causal_share": share(caus["A1"]), "random_share": share(rnd),
                "energy_rank4_in_sample_share": energy4,
                "beat_energy": share(caus["A1"]) - energy4,
                "causal_direction_energy_fraction": cen,
                "random_direction_energy_fraction": ren,
                "block_ratio": ratio(blockd), "causal_ratio": ratio(caus),
                "specific_gain": ratio(blockd) - ratio(caus),
                "admissible": bool(caus["A1"] >= BARS["admit_block"] * max(blockd["A1"], BARS["floor"])),
            }
            p = per_layer[f"mlp{layer}"]
            print(f"[causal] {tid:28s} mlp{layer} causal={p['causal_share']:.2f} "
                  f"energy_r4={energy4:.2f} rnd={p['random_share']:.3f} "
                  f"E_frac={cen:.4f} block_r={p['block_ratio']:.2f} caus_r={p['causal_ratio']:.2f} "
                  f"adm={p['admissible']}", flush=True)
        results[tid] = per_layer

    flat = [results[t][l] for t in results for l in results[t]]
    med = lambda k: float(np.median([f[k] for f in flat])) if flat else float("nan")
    adm = [f for f in flat if f["admissible"]]
    preds = {
        'pred_a_causal_direction_carries_the_damage': bool(med("causal_share") >= BARS["causal_share"]),
        'pred_b_causal_beats_fitted_energy': bool(med("beat_energy") >= BARS["beat_energy"]),
        'pred_c_random_direction_is_inert': bool(med("random_share") <= BARS["random_share"]),
        'pred_d_the_causal_direction_is_low_energy':
            bool(med("causal_direction_energy_fraction") <= BARS["causal_energy"]),
        'pred_e_causal_arm_is_more_specific':
            bool(adm and float(np.median([f["specific_gain"] for f in adm])) >= BARS["specific_gain"]),
    }
    nulls = {
        "a_null_causal_le_.15": bool(med("causal_share") <= NULLS["causal_share_le"]),
        "b_null_no_beat": bool(med("beat_energy") <= NULLS["beat_energy_le"]),
        "c_null_random_ge_.3": bool(med("random_share") >= NULLS["random_share_ge"]),
        "d_null_causal_high_energy": bool(med("causal_direction_energy_fraction") >= NULLS["causal_energy_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in READERS], "seed": SEED,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "n_cells": len(flat), "n_admissible": len(adm),
                          "medians": {k: med(k) for k in
                                      ("causal_share", "energy_rank4_in_sample_share", "beat_energy",
                                       "random_share", "causal_direction_energy_fraction",
                                       "random_direction_energy_fraction", "block_ratio",
                                       "causal_ratio", "specific_gain")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x readers {READERS}; "
              f"parameter-free causal direction vs SS2825's fitted rank-4 energy subspace; no model loaded")
        sys.exit(0)
    main()
