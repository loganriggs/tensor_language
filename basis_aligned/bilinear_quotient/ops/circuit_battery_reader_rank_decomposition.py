#!/usr/bin/env python
"""circuit_battery_reader_rank_decomposition -- the read is not unit-sparse (SS2823); is it LOW-RANK?

SS2822 and SS2823 agree that the causal read of attention 8's write is dense in the hidden-unit basis: the top 1.4% of units by the
exactly right statistic are indistinguishable from a random 1.4%. Unit coordinates are basis-dependent, so the honest follow-up is a
BASIS-FREE question about the same object. For a reader block, the removal effect is the vector
    delta(row) = mlp(rms_norm(x)) - mlp(rms_norm(x - W))   in R^1152,
one vector per row. This rung takes its SVD over FIT rows, then intervenes with only the rank-r part -- the arm subtracts P_r delta
instead of delta -- and scores on OOD. Unlike every previous battery rung this one HAS fitted parameters (r x 1152 per behaviour and
reader, fitted on FIT only); they are declared in the receipt and the evaluation split was never opened for the fit.

# BQGATE: EXPERIMENT  pred_a_rank_one_carries_half pred_b_rank_four_carries_most
#                     pred_c_the_subspace_is_shared_across_behaviours pred_d_low_rank_arm_is_more_specific
#                     pred_e_random_subspace_does_not_carry_the_read

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's answer; ratio = max(|d_P|,|d_C|)/max(d_A1,.5),
LOWER IS MORE SPECIFIC. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_READER_RANK_DECOMPOSITION_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_reader_rank_decomposition.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_READER_RANK_DECOMPOSITION_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_reader_rank_decomposition"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "55f67b8ec55b047ecf154c3ec00e60b795eccdcbd2a490c5998884278385cba7",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL = R.D, R.NL
WRITER = ("attn", 8)
READERS = (10, 11)
RANKS = (1, 2, 4, 8)
SEED = 2823
PER_CELL = 4 if SMOKE else 16
BARS = {"rank1": 0.50, "rank4": 0.80, "overlap": 0.50, "specific_gain": 0.20,
        "random_share": 0.15, "admit_block": 0.25, "floor": 0.5}
NULLS = {"rank1_le": 0.20, "rank4_le": 0.40, "overlap_le": 0.10, "specific_gain_le": 0.0,
         "random_share_ge": 0.40}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_rank(m, tokens, finals, layer, basis):
    """Forward where the reader's removal effect is applied only inside `basis` (D x r, orthonormal).

    basis=None -> native; basis="all" -> the whole removal (the block's read removed outright).
    """
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
        if site == layer and W is not None:
            nat = blk.mlp(F.rms_norm(x, (D,)))
            rem = blk.mlp(F.rms_norm(x - W, (D,)))
            d = (nat - rem)[ar, finals].float()
            delta = d
            out = nat
            if basis is not None:
                proj = d if isinstance(basis, str) else (d @ basis) @ basis.T
                out = nat.clone()
                out[ar, finals] = nat[ar, finals] - proj.to(nat.dtype)
        else:
            out = blk.mlp(F.rms_norm(x, (D,)))
        x = x + out
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return logits, delta


def dmg(m, rows, cand, layer, basis, fwd):
    out = []
    for b in CB.batches(rows):
        ids, fin, ans = CB.pack(b, "base")
        lg, _ = run_rank(m, ids, fin, layer, None); fwd[0] += 1
        lg2, _ = run_rank(m, ids, fin, layer, basis); fwd[0] += 1
        out.append((CB.margins(lg, ans, cand) - CB.margins(lg2, ans, cand)).cpu().numpy())
    return float(np.concatenate(out).mean()) if out else float("nan")


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fwd = [0]
    results, bases = {}, {}
    fitted = 0
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        per_layer = {}
        for layer in READERS:
            # ---- FIT: SVD of the removal effect over target rows (the only fitted object here) ----
            acc = []
            for b in CB.batches([r for r in rows if r["family"] == "A1" and r["split"] == "FIT"]):
                ids, fin, _ = CB.pack(b, "base")
                _lg, d = run_rank(m, ids, fin, layer, None); fwd[0] += 1
                acc.append(d)
            Dm = torch.cat(acc, 0)
            U, S, Vh = torch.linalg.svd(Dm, full_matrices=False)
            energy = (S ** 2 / (S ** 2).sum()).cpu().numpy()
            cells = {f: [r for r in rows if r["family"] == f and r["split"] == "OOD"]
                     for f in ("A1", "P", "C") if f in fams}
            blockd = {f: dmg(m, cells[f], cand, layer, "all", fwd) for f in cells}
            ranks = {}
            for r in RANKS:
                B = Vh[:r].T.contiguous()
                fitted += r * D
                dm_r = {f: dmg(m, cells[f], cand, layer, B, fwd) for f in cells}
                ranks[f"rank{r}"] = dm_r
            rb = torch.linalg.qr(torch.randn(D, max(RANKS), generator=g).to(DEV))[0]
            rnd = dmg(m, cells["A1"], cand, layer, rb[:, :4].contiguous(), fwd)
            ratio = lambda d: max(abs(d.get("P", 0.0)), abs(d.get("C", 0.0))) / max(d["A1"], BARS["floor"])
            share = lambda d: d["A1"] / max(blockd["A1"], BARS["floor"])
            per_layer[f"mlp{layer}"] = {
                "block_damage": blockd, "block_ratio": ratio(blockd),
                "rank_damage": ranks,
                "rank_share": {k: share(v) for k, v in ranks.items()},
                "rank_ratio": {k: ratio(v) for k, v in ranks.items()},
                "rank4_admissible": bool(ranks["rank4"]["A1"] >= BARS["admit_block"] * max(blockd["A1"], BARS["floor"])),
                "specific_gain_rank4": ratio(blockd) - ratio(ranks["rank4"]),
                "random_rank4_share": rnd / max(blockd["A1"], BARS["floor"]),
                "singular_energy_top8": [float(x) for x in energy[:8]],
                "fit_rows": Dm.shape[0],
            }
            bases.setdefault(f"mlp{layer}", {})[tid] = Vh[:4].T.contiguous()
            p = per_layer[f"mlp{layer}"]
            print(f"[rank] {tid:28s} mlp{layer} r1={p['rank_share']['rank1']:.2f} r4={p['rank_share']['rank4']:.2f} "
                  f"rnd={p['random_rank4_share']:.2f} block_r={p['block_ratio']:.2f} "
                  f"r4_r={p['rank_ratio']['rank4']:.2f} e1={p['singular_energy_top8'][0]:.2f}", flush=True)
        results[tid] = per_layer

    ov = {}
    for lname, bs in bases.items():
        keys = sorted(bs)
        vals = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s = torch.linalg.svdvals(bs[keys[i]].T @ bs[keys[j]])
                vals.append(float((s ** 2).mean()))         # mean squared principal cosine
        ov[lname] = float(np.median(vals)) if vals else float("nan")
    flat = [results[t][l] for t in results for l in results[t]]
    med_share = lambda r: float(np.median([f["rank_share"][r] for f in flat])) if flat else float("nan")
    adm = [f for f in flat if f["rank4_admissible"]]
    preds = {
        'pred_a_rank_one_carries_half': bool(med_share("rank1") >= BARS["rank1"]),
        'pred_b_rank_four_carries_most': bool(med_share("rank4") >= BARS["rank4"]),
        'pred_c_the_subspace_is_shared_across_behaviours':
            bool(ov and float(np.median(list(ov.values()))) >= BARS["overlap"]),
        'pred_d_low_rank_arm_is_more_specific':
            bool(adm and float(np.median([f["specific_gain_rank4"] for f in adm])) >= BARS["specific_gain"]),
        'pred_e_random_subspace_does_not_carry_the_read':
            bool(flat and float(np.median([f["random_rank4_share"] for f in flat])) <= BARS["random_share"]),
    }
    nulls = {
        "a_null_rank1_le_.2": bool(med_share("rank1") <= NULLS["rank1_le"]),
        "b_null_rank4_le_.4": bool(med_share("rank4") <= NULLS["rank4_le"]),
        "c_null_overlap_le_.1": bool(ov and float(np.median(list(ov.values()))) <= NULLS["overlap_le"]),
        "d_null_no_gain": bool(adm and float(np.median([f["specific_gain_rank4"] for f in adm])) <= NULLS["specific_gain_le"]),
        "e_null_random_ge_.4": bool(flat and float(np.median([f["random_rank4_share"] for f in flat])) >= NULLS["random_share_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in READERS], "ranks": list(RANKS),
              "seed": SEED, "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "subspace_overlap_by_layer": ov,
                          "n_admissible_rank4": len(adm), "n_cells": len(flat),
                          "median_rank_share": {f"rank{r}": med_share(f"rank{r}") for r in RANKS},
                          "median_random_rank4_share":
                              float(np.median([f["random_rank4_share"] for f in flat])) if flat else None},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": fitted,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls,
                      "median_rank_share": result["summary"]["median_rank_share"],
                      "overlap": ov}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x readers {READERS} x ranks {RANKS}; "
              f"FIT fits the subspace, OOD scores; no model loaded")
        sys.exit(0)
    main()
