#!/usr/bin/env python
"""circuit_battery_reader_unit_lens_ranked -- the same sub-block decomposition, ranked by LENS CONTRIBUTION instead of magnitude.

SS2819/SS2821: the causally live read of attention 8's write is confined to blocks 8-11 and task specificity rises with depth, peaking at
mlp10/mlp11. SS2812: the block is an exact bilinear form, mlp(u) = Down(Left(u) * Right(u)) + b, so the reader's response to removing the
write decomposes ADDITIVELY over the 4,608 hidden units -- removing unit u's read of W means taking that one coordinate of the hidden
vector from the removed-input forward and the rest from the native one. That is a decomposition strictly finer than an MLP block, and it
is exact, so "smaller than an MLP block" becomes a measurement rather than an aspiration.

Two phases with separate authorities: FIT ranks units by how much more they engage with the write on the TARGET family than on the COPY
control (a selection, so FIT only); OOD scores the predeclared top-64 set. The admissibility gate is on the BLOCK's own damage, not on
READS -- the correction forced by SS2821, where a gate at .10 x READS admitted almost nothing.

# BQGATE: EXPERIMENT  pred_a_unit_decomposition_is_exact pred_b_lens_ranked_units_carry_the_read
#                     pred_c_the_unit_set_is_more_specific_than_its_block pred_d_the_lens_unit_set_is_shared
#                     pred_e_random_units_do_not_carry_the_read

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's answer; ratio = max(|d_P|,|d_C|)/max(d_A1,.5),
LOWER IS MORE SPECIFIC. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_READER_UNIT_LENS_RANKED_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_reader_unit_lens_ranked.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_READER_UNIT_LENS_RANKED_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_reader_unit_lens_ranked"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "ed75ff2d3ba1580fff037a405e00d3b877d128274cde0cd9d070a42ae9559d15",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL = R.D, R.NL
WRITER = ("attn", 8)
READERS = (10, 11)          # predeclared from SS2819/SS2821: the specificity peak
TOPK = 64                   # of 4,608 hidden units = 1.4%
SEED = 2821
PER_CELL = 4 if SMOKE else 16
BARS = {"exact_tol": 1e-3, "topk_share": 0.50, "specific_gain": 0.20, "jaccard": 0.15,
        "random_share": 0.15, "admit_block": 0.25, "floor": 0.5}
NULLS = {"topk_share_le": 0.20, "specific_gain_le": 0.0, "jaccard_le": 0.02, "random_share_ge": 0.40}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_units(m, tokens, finals, layer, units, *, contrib=None):
    """Forward with the write removed from ONLY the listed hidden units of `layer`'s MLP.

    units=None -> native; units="all" -> the whole block's read removed (the instrument check).
    contrib=True -> also return each unit's |delta| x ||Down column|| at the final position.
    """
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; W = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    info = None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if W is not None:
            W = blk.lambdas[0] * W
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if WRITER == ("attn", site):
            W = torch.zeros_like(x); W[ar, finals] = write[ar, finals]
        x = x + write
        if site == layer and W is not None and (units is not None or contrib is not None):
            nat = F.rms_norm(x, (D,))
            rem = F.rms_norm(x - W, (D,))
            hn = blk.mlp.Left(nat) * blk.mlp.Right(nat)
            hr = blk.mlp.Left(rem) * blk.mlp.Right(rem)
            if contrib is not None:
                # LENS-WEIGHTED score: each unit's signed push on the answer logit, not its raw size.
                # delta_u * (Down[:, u] . W_U[answer]) is the unit's exact contribution to the answer
                # direction, so cancelling units score ~0 instead of dominating a magnitude ranking.
                dh = (hn - hr)[ar, finals].float()                  # (B, hidden)
                wu = m.lm_head.weight[contrib].float()              # (B, D) unembed row of the answer
                proj = wu @ blk.mlp.Down.weight.float()             # (B, hidden)
                info = dh * proj
            if units is not None:
                h = hn.clone()
                if isinstance(units, str):
                    h[ar, finals] = hr[ar, finals]
                else:                       # row-then-unit indexing: advanced indices do not broadcast
                    sel = h[ar, finals]
                    sel[:, units] = hr[ar, finals][:, units]
                    h[ar, finals] = sel
                out = blk.mlp.Down(h) + blk.mlp.Down_bias
            else:
                out = blk.mlp(nat)
        else:
            out = blk.mlp(F.rms_norm(x, (D,)))
        x = x + out
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return (logits, info) if contrib is not None else logits


def dmg(m, rows, cand, layer, units, fwd):
    out = []
    for b in CB.batches(rows):
        ids, fin, ans = CB.pack(b, "base")
        lg = run_units(m, ids, fin, layer, None); fwd[0] += 1
        lg2 = run_units(m, ids, fin, layer, units); fwd[0] += 1
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
    results, chosen = {}, {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        per_layer = {}
        for layer in READERS:
            # ---- FIT: rank units by target engagement minus copy-control engagement ----
            score = None
            for fam, sign in (("A1", 1.0), ("C", -1.0)):
                if fam not in fams:
                    continue
                acc = []
                for b in CB.batches([r for r in rows if r["family"] == fam and r["split"] == "FIT"]):
                    ids, fin, ansv = CB.pack(b, "base")
                    _lg, info = run_units(m, ids, fin, layer, None, contrib=ansv); fwd[0] += 1
                    acc.append(info.mean(0).cpu().numpy())
                if acc:
                    v = sign * np.mean(acc, axis=0)
                    score = v if score is None else score + v
            top = np.argsort(-score)[:TOPK]
            idx = torch.tensor(np.sort(top).copy(), device=DEV)
            rnd = torch.tensor(np.sort(torch.randperm(len(score), generator=g)[:TOPK].numpy()).copy(), device=DEV)
            # ---- OOD: score the predeclared set, a random set, and the whole block ----
            cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
            cells = {f: [r for r in rows if r["family"] == f and r["split"] == "OOD"] for f in ("A1", "P", "C") if f in fams}
            blockd = {f: dmg(m, cells[f], cand, layer, "all", fwd) for f in cells}
            topd = {f: dmg(m, cells[f], cand, layer, idx, fwd) for f in cells}
            rndd = dmg(m, cells["A1"], cand, layer, rnd, fwd)
            exact = []
            for b in CB.batches(cells["A1"][:16]):
                ids, fin, _ = CB.pack(b, "base")
                a = run_units(m, ids, fin, layer, "all"); fwd[0] += 1
                bl = run_units(m, ids, fin, layer, torch.arange(len(score), device=DEV)); fwd[0] += 1
                exact.append((a - bl).abs().max().item())
            ratio = lambda d: max(abs(d.get("P", 0.0)), abs(d.get("C", 0.0))) / max(d["A1"], BARS["floor"])
            per_layer[f"mlp{layer}"] = {
                "block_damage": blockd, "top_damage": topd, "random_a1_damage": rndd,
                "block_ratio": ratio(blockd), "top_ratio": ratio(topd),
                "specific_gain": ratio(blockd) - ratio(topd),
                "top_share": topd["A1"] / max(blockd["A1"], BARS["floor"]),
                "random_share": rndd / max(blockd["A1"], BARS["floor"]),
                "admissible": bool(topd["A1"] >= BARS["admit_block"] * max(blockd["A1"], BARS["floor"])),
                "max_exact_dev": float(max(exact)) if exact else float("nan"),
                "top_units": sorted(int(u) for u in top),
            }
            chosen.setdefault(f"mlp{layer}", {})[tid] = set(int(u) for u in top)
            p = per_layer[f"mlp{layer}"]
            print(f"[units] {tid:28s} mlp{layer} share={p['top_share']:.2f} rnd={p['random_share']:.2f} "
                  f"block_r={p['block_ratio']:.2f} top_r={p['top_ratio']:.2f} adm={p['admissible']} "
                  f"dev={p['max_exact_dev']:.1e}", flush=True)
        results[tid] = per_layer

    jac = {}
    for lname, sets in chosen.items():
        vals = []
        keys = sorted(sets)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = sets[keys[i]], sets[keys[j]]
                vals.append(len(a & b) / max(len(a | b), 1))
        jac[lname] = float(np.median(vals)) if vals else float("nan")
    flat = [results[t][l] for t in results for l in results[t]]
    med = lambda k: float(np.median([f[k] for f in flat if not np.isnan(f[k])])) if flat else float("nan")
    adm = [f for f in flat if f["admissible"]]
    preds = {
        'pred_a_unit_decomposition_is_exact': bool(flat and max(f["max_exact_dev"] for f in flat) <= BARS["exact_tol"]),
        'pred_b_lens_ranked_units_carry_the_read': bool(med("top_share") >= BARS["topk_share"]),
        'pred_c_the_unit_set_is_more_specific_than_its_block':
            bool(adm and float(np.median([f["specific_gain"] for f in adm])) >= BARS["specific_gain"]),
        'pred_d_the_lens_unit_set_is_shared': bool(jac and float(np.median(list(jac.values()))) >= BARS["jaccard"]),
        'pred_e_random_units_do_not_carry_the_read': bool(med("random_share") <= BARS["random_share"]),
    }
    nulls = {
        "b_null_share_le_.2": bool(med("top_share") <= NULLS["topk_share_le"]),
        "c_null_no_gain": bool(adm and float(np.median([f["specific_gain"] for f in adm])) <= NULLS["specific_gain_le"]),
        "d_null_jaccard_le_.02": bool(jac and float(np.median(list(jac.values()))) <= NULLS["jaccard_le"]),
        "e_null_random_ge_.4": bool(med("random_share") >= NULLS["random_share_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in READERS], "topk": TOPK, "seed": SEED,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "median_jaccard_by_layer": jac,
                          "n_admissible": len(adm), "n_cells": len(flat),
                          "medians": {k: med(k) for k in ("top_share", "random_share", "block_ratio",
                                                          "top_ratio", "specific_gain", "max_exact_dev")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"],
                      "jaccard": jac}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x readers {READERS} x top-{TOPK} of 4608 units; "
              f"FIT ranks, OOD scores; no model loaded")
        sys.exit(0)
    main()
