#!/usr/bin/env python
"""circuit_battery_common_reader_interaction -- ONE predeclared reader set, evaluated on OOD only.

Codex SS2815 asked for exactly this: the SS2813 interaction claims restated "for a prospective run with one predeclared common reader
set and valid FIT->SELECT->TEST->OOD authorities". The reader set {mlp8, mlp9, mlp10, mlp11} is fixed in the preregistration BEFORE
this run for EVERY behaviour -- it is not re-chosen per behaviour and nothing is selected here. The damage set function over its 16
subsets is evaluated ONLY on OOD rows, which no run has opened for any selection; behaviour eligibility comes from SS2817's capability
and FIT writer choice on the repaired bank, never from the SS2809 screen. This also repairs MY defect disclosed in SS2813: the
profile-sharing clause was under-powered because each behaviour had a different top-4 set, so the correlations ran over as few as three
aligned keys; with a common set all six pair coefficients align by construction.

# BQGATE: EXPERIMENT  pred_a_common_set_is_super_additive pred_b_order_two_interactions_are_positive
#                     pred_c_redundancy_order_exceeds_one pred_d_interaction_profile_is_shared
#                     pred_e_common_set_carries_the_reads

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS the behaviour. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_COMMON_READER_INTERACTION_PREREGISTRATION.md
"""
import json, os, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_common_reader_interaction.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_COMMON_READER_INTERACTION_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_common_reader_interaction"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "af506ea922ddf85131617045a8bf89bae49736b50386a609e764720f3a6e761f", BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
COMMON = (("mlp", 8), ("mlp", 9), ("mlp", 10), ("mlp", 11))   # PREDECLARED, identical for every behaviour
NAMES = tuple(f"{k}{l}" for k, l in COMMON)
EVAL_SPLIT = "OOD"                                            # never opened for any selection
PER_CELL = 4 if SMOKE else 16
BARS = {"super_additive": 0.90, "order2_positive_frac": 0.75, "redundancy_order": 2,
        "profile_corr": 0.50, "reads_share": 0.50, "floor": 0.5}
NULLS = {"super_additive_ge": 1.0, "order2_positive_frac_le": 0.50, "profile_corr_le": 0.0,
         "reads_share_le": 0.25}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def subsets(items):
    for r in range(len(items) + 1):
        for c in combinations(items, r):
            yield c


def moebius(v):
    return {S: sum((-1) ** (len(S) - len(T)) * v[T] for T in subsets(S)) for S in v}


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    fwd = 0
    results = {}
    idx = tuple(range(len(COMMON)))
    for tid in tasks:
        writer = ("attn", 8)
        allr = [("mlp", 8)] + [(kd, l) for l in range(9, R.NL) for kd in ("attn", "mlp")]
        rows = [r for r in BANK.build_rows(tid, per_cell=PER_CELL)
                if r["family"] == "A1" and r["split"] == EVAL_SPLIT]
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        acc = {S: [] for S in subsets(idx)}
        reads = []
        for b in CB.batches(rows):
            ids, fin, ans = CB.pack(b, "base")
            lg = CB.run(m, ids, fin); fwd += 1
            mn = CB.margins(lg, ans, cand)
            for S in acc:
                if not S:
                    acc[S].append(np.zeros(len(b))); continue
                lg2 = CB.run(m, ids, fin, writer=writer, removed=tuple(COMMON[i] for i in S)); fwd += 1
                acc[S].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            lgr = CB.run(m, ids, fin, writer=writer, removed=tuple(allr)); fwd += 1
            reads.append((mn - CB.margins(lgr, ans, cand)).cpu().numpy())
        v = {S: float(np.concatenate(x).mean()) for S, x in acc.items()}
        mob = moebius(v)
        joint = v[idx]
        full = max(joint, BARS["floor"])
        pairs = {S: mob[S] for S in mob if len(S) == 2}
        order = next((k for k in range(1, len(COMMON) + 1)
                      if max(v[S] for S in v if len(S) == k) >= 0.5 * full), len(COMMON) + 1)
        rd = float(np.concatenate(reads).mean())
        results[tid] = {
            "readers": list(NAMES), "eval_split": EVAL_SPLIT, "rows": len(rows),
            "v": {"+".join(NAMES[i] for i in S) or "none": val for S, val in v.items()},
            "moebius_order2": {"+".join(NAMES[i] for i in S): val for S, val in pairs.items()},
            "singles_sum": sum(v[(i,)] for i in idx), "joint": joint,
            "super_additive_ratio": sum(v[(i,)] for i in idx) / full,
            "order2_positive_frac": float(np.mean([p > 0 for p in pairs.values()])),
            "redundancy_order": order, "reads_damage": rd,
            "common_share_of_reads": joint / max(rd, BARS["floor"]),
        }
        print(f"[common] {tid:30s} singles={results[tid]['singles_sum']:.3f} joint={joint:.3f} "
              f"ratio={results[tid]['super_additive_ratio']:.3f} order={order} "
              f"share={results[tid]['common_share_of_reads']:.3f}", flush=True)

    keys = ["+".join(NAMES[i] for i in S) for S in combinations(idx, 2)]
    profs = {}
    for t in results:
        vec = np.array([results[t]["moebius_order2"][k] for k in keys], dtype=float)
        profs[t] = vec / (np.max(np.abs(vec)) or 1.0)
    corrs = [float(np.corrcoef(profs[a], profs[b])[0, 1]) for a, b in combinations(sorted(profs), 2)
             if np.std(profs[a]) > 0 and np.std(profs[b]) > 0]
    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    pc = float(np.median(corrs)) if corrs else float("nan")
    preds = {
        'pred_a_common_set_is_super_additive': bool(med("super_additive_ratio") <= BARS["super_additive"]),
        'pred_b_order_two_interactions_are_positive': bool(med("order2_positive_frac") >= BARS["order2_positive_frac"]),
        'pred_c_redundancy_order_exceeds_one': bool(med("redundancy_order") >= BARS["redundancy_order"]),
        'pred_d_interaction_profile_is_shared': bool(corrs and pc >= BARS["profile_corr"]),
        'pred_e_common_set_carries_the_reads': bool(med("common_share_of_reads") >= BARS["reads_share"]),
    }
    nulls = {
        "a_null_ratio_ge_1": bool(med("super_additive_ratio") >= NULLS["super_additive_ge"]),
        "b_null_order2_le_.5": bool(med("order2_positive_frac") <= NULLS["order2_positive_frac_le"]),
        "d_null_corr_le_0": bool(corrs and pc <= NULLS["profile_corr_le"]),
        "e_null_share_le_.25": bool(med("common_share_of_reads") <= NULLS["reads_share_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "common_reader_set": list(NAMES), "eval_split": EVAL_SPLIT,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "pair_keys": keys,
                          "profile_pair_correlations": corrs, "median_profile_corr": pc,
                          "medians": {k: med(k) for k in ("super_additive_ratio", "order2_positive_frac",
                                                          "redundancy_order", "common_share_of_reads")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"],
                      "profile_corr": pc}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x 16 subsets of {NAMES} on {EVAL_SPLIT}; no model loaded")
        sys.exit(0)
    main()
