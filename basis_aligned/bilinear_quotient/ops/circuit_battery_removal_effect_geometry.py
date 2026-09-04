#!/usr/bin/env python
"""circuit_battery_removal_effect_geometry -- WHY no fitted subspace transports: is the removal effect row-specific, or was the arm wrong?

SS2824 found that a rank-1..8 subspace of the reader's removal effect, fitted on FIT rows, carries -.01 of the block's damage on OOD rows
-- no better than a random subspace. Two explanations survive that result and SS2824 explicitly declined to choose between them: either
the effect delta = mlp(rms_norm(x)) - mlp(rms_norm(x - W)) points a different way for every input (so nothing transports), or the
low-rank ARM is mis-specified (so nothing would work). This rung separates them with an IN-SAMPLE control: the same rank-4 arm, with
the subspace fitted on the very rows it is scored on. If the in-sample arm works and the transported one does not, the effect is
row-specific and SS2824's negative is about the model. If neither works, the arm is wrong and SS2824's negative is about my instrument.
Everything else -- writer, readers, behaviours, gate, sign convention -- is unchanged.

# BQGATE: EXPERIMENT  pred_a_fit_subspace_misses_ood_energy pred_b_in_sample_arm_works
#                     pred_c_effect_has_high_effective_rank pred_d_rows_point_different_ways
#                     pred_e_random_subspace_captures_chance_energy

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's answer. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_REMOVAL_EFFECT_GEOMETRY_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_removal_effect_geometry.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_REMOVAL_EFFECT_GEOMETRY_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_removal_effect_geometry"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "f90a8bce06b84309177afade8ae8c4132ec489c0d352af85e63ab21277458ab0",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D = R.D
READERS = (10, 11)
RANK = 4
SEED = 2824
PER_CELL = 4 if SMOKE else 16
BARS = {"transport_energy": 0.25, "in_sample_share": 0.50, "eff_rank": 8.0, "row_cos": 0.30,
        "random_energy": 0.02, "floor": 0.5}
NULLS = {"transport_energy_ge": 0.60, "in_sample_share_le": 0.10, "eff_rank_le": 3.0,
         "row_cos_ge": 0.70}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def collect(m, rows, layer, fwd):
    acc = []
    for b in CB.batches(rows):
        ids, fin, _ = CB.pack(b, "base")
        _lg, d = RD.run_rank(m, ids, fin, layer, None); fwd[0] += 1
        acc.append(d)
    return torch.cat(acc, 0) if acc else None


def energy_in(Dm, B):
    """Fraction of Dm's squared Frobenius energy lying inside the subspace spanned by B."""
    tot = (Dm ** 2).sum()
    if tot <= 0:
        return float("nan")
    return float(((Dm @ B) ** 2).sum() / tot)


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fwd = [0]
    results = {}
    fitted = 0
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        per_layer = {}
        for layer in READERS:
            fit_rows = [r for r in rows if r["family"] == "A1" and r["split"] == "FIT"]
            ood_rows = [r for r in rows if r["family"] == "A1" and r["split"] == "OOD"]
            Df = collect(m, fit_rows, layer, fwd)
            Do = collect(m, ood_rows, layer, fwd)
            Bf = torch.linalg.svd(Df, full_matrices=False)[2][:RANK].T.contiguous()
            Bo = torch.linalg.svd(Do, full_matrices=False)[2][:RANK].T.contiguous()
            fitted += 2 * RANK * D
            Br = torch.linalg.qr(torch.randn(D, RANK, generator=g).to(DEV))[0]
            s = torch.linalg.svdvals(Do)
            p = (s ** 2) / (s ** 2).sum()
            eff_rank = float(1.0 / (p ** 2).sum())            # participation ratio
            cum = torch.cumsum(p, 0)
            rank90 = int((cum < 0.90).sum().item()) + 1
            n = Do / Do.norm(dim=1, keepdim=True).clamp_min(1e-9)
            cs = (n @ n.T)
            iu = torch.triu_indices(cs.size(0), cs.size(1), offset=1)
            row_cos = float(cs[iu[0], iu[1]].median())
            blockd = RD.dmg(m, ood_rows, cand, layer, "all", fwd)
            trans = RD.dmg(m, ood_rows, cand, layer, Bf, fwd)
            insamp = RD.dmg(m, ood_rows, cand, layer, Bo, fwd)
            per_layer[f"mlp{layer}"] = {
                "transport_energy": energy_in(Do, Bf),
                "in_sample_energy": energy_in(Do, Bo),
                "random_energy": energy_in(Do, Br),
                "effective_rank": eff_rank, "rank_for_90pct_energy": rank90,
                "median_row_cosine": row_cos,
                "block_damage": blockd,
                "transported_share": trans / max(blockd, BARS["floor"]),
                "in_sample_share": insamp / max(blockd, BARS["floor"]),
                "n_fit": int(Df.shape[0]), "n_ood": int(Do.shape[0]),
            }
            p_ = per_layer[f"mlp{layer}"]
            print(f"[geom] {tid:28s} mlp{layer} E_fit={p_['transport_energy']:.3f} "
                  f"E_in={p_['in_sample_energy']:.3f} E_rnd={p_['random_energy']:.4f} "
                  f"effrank={eff_rank:.1f} r90={rank90} cos={row_cos:.2f} "
                  f"trans={p_['transported_share']:.2f} in={p_['in_sample_share']:.2f}", flush=True)
        results[tid] = per_layer

    flat = [results[t][l] for t in results for l in results[t]]
    med = lambda k: float(np.median([f[k] for f in flat])) if flat else float("nan")
    preds = {
        'pred_a_fit_subspace_misses_ood_energy': bool(med("transport_energy") <= BARS["transport_energy"]),
        'pred_b_in_sample_arm_works': bool(med("in_sample_share") >= BARS["in_sample_share"]),
        'pred_c_effect_has_high_effective_rank': bool(med("effective_rank") >= BARS["eff_rank"]),
        'pred_d_rows_point_different_ways': bool(med("median_row_cosine") <= BARS["row_cos"]),
        'pred_e_random_subspace_captures_chance_energy': bool(med("random_energy") <= BARS["random_energy"]),
    }
    nulls = {
        "a_null_transport_ge_.6": bool(med("transport_energy") >= NULLS["transport_energy_ge"]),
        "b_null_in_sample_le_.1": bool(med("in_sample_share") <= NULLS["in_sample_share_le"]),
        "c_null_low_rank": bool(med("effective_rank") <= NULLS["eff_rank_le"]),
        "d_null_rows_aligned": bool(med("median_row_cosine") >= NULLS["row_cos_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in READERS], "rank": RANK, "seed": SEED,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "n_cells": len(flat),
                          "medians": {k: med(k) for k in
                                      ("transport_energy", "in_sample_energy", "random_energy",
                                       "effective_rank", "rank_for_90pct_energy",
                                       "median_row_cosine", "transported_share", "in_sample_share")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": fitted,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x readers {READERS}; "
              f"FIT-subspace vs IN-SAMPLE vs RANDOM at rank {RANK}; no model loaded")
        sys.exit(0)
    main()
