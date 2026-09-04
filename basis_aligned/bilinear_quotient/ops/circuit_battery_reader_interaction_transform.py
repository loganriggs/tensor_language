#!/usr/bin/env python
"""circuit_battery_reader_interaction_transform -- the Moebius/Harsanyi transform of the reader damage set function.

SS2808/SS2809 measured that the readers of a writer's write are super-additive: single-reader removals sum to about half the damage of
removing them jointly (median top-3 share .49 across the bank).  "Distributed" is an admission; this rung turns it into a statement.
For the top-4 MLP readers of each SS2809 capable+localised behaviour it evaluates the damage set function v(S) on ALL 16 subsets and
takes the Moebius (Harsanyi) transform m(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T), whose order-1 coefficients are the single-reader
damages and whose higher orders are exactly the interaction that single-component ablation misses (the hydra/self-repair effect,
arXiv:2307.15771, arXiv:2607.01940).  It also reports the REDUNDANCY ORDER -- the smallest k such that some k readers carry half the
joint damage -- which is a statement about executable cost, and asks whether the interaction PROFILE is shared across behaviours (the
re-use question SS2809 opened at the level of the reader ladder).

# BQGATE: EXPERIMENT  pred_a_readers_are_super_additive pred_b_order_two_interactions_are_positive
#                     pred_c_redundancy_order_exceeds_one pred_d_interaction_profile_is_shared pred_e_top4_carries_the_reads

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS the behaviour. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_READER_INTERACTION_TRANSFORM_PREREGISTRATION.md
"""
import json, os, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_reader_interaction_transform.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_READER_INTERACTION_TRANSFORM_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_results.json"
RUNG = "circuit_battery_reader_interaction_transform"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "553fff9847d38b1ad53409f57998f1d3c77e8007056570ee0e73e1ed2afb330a",
          BATTERY: "6d1eda1cc05adf72c525375a0602bbafbf9b4335653be0e410de3d69da03265c",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
K = 4
PER_CELL = 4 if SMOKE else 16
BARS = {"super_additive": 0.70, "order2_positive_frac": 0.75, "redundancy_order": 2,
        "profile_corr": 0.50, "top4_share": 0.50, "floor": 0.5}
NULLS = {"super_additive_ge": 1.0, "order2_positive_frac_le": 0.50, "profile_corr_le": 0.0,
         "top4_share_le": 0.25}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def subsets(items):
    for r in range(len(items) + 1):
        for c in combinations(items, r):
            yield c


def moebius(v):
    """m(S) = sum_{T subset S} (-1)^(|S|-|T|) v(T), over a dict keyed by sorted tuples."""
    out = {}
    for S in v:
        out[S] = sum((-1) ** (len(S) - len(T)) * v[T] for T in subsets(S))
    return out


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
        top = [k[len("COMP_"):] for k, _ in tb["reader_ladder"] if k.startswith("COMP_mlp")][:K]
        readers = [("mlp", int(n[3:])) for n in top]
        allr = ([("mlp", writer[1])] if kind == "attn" else []) + \
               [(kd, l) for l in range(writer[1] + 1, R.NL) for kd in ("attn", "mlp")]
        rows = [r for r in BANK.build_rows(tid, per_cell=PER_CELL)
                if r["family"] == "A1" and r["split"] == "SELECT"]
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        acc = {S: [] for S in subsets(tuple(range(len(readers))))}
        reads = []
        for b in CB.batches(rows):
            ids, fin, ans = CB.pack(b, "base")
            lg = CB.run(m, ids, fin); fwd += 1
            mn = CB.margins(lg, ans, cand)
            for S in acc:
                rem = tuple(readers[i] for i in S)
                if not rem:
                    acc[S].append(np.zeros(len(b)))
                    continue
                lg2 = CB.run(m, ids, fin, writer=writer, removed=rem); fwd += 1
                acc[S].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            lgr = CB.run(m, ids, fin, writer=writer, removed=tuple(allr)); fwd += 1
            reads.append((mn - CB.margins(lgr, ans, cand)).cpu().numpy())
        v = {S: float(np.concatenate(x).mean()) for S, x in acc.items()}
        mob = moebius(v)
        full = max(v[tuple(range(len(readers)))], BARS["floor"])
        singles = sum(v[(i,)] for i in range(len(readers)))
        pairs = {S: mob[S] for S in mob if len(S) == 2}
        order = next((k for k in range(1, len(readers) + 1)
                      if max(v[S] for S in v if len(S) == k) >= 0.5 * full), len(readers) + 1)
        reads_mean = float(np.concatenate(reads).mean())
        results[tid] = {
            "writer": wname, "readers": top,
            "v": {",".join(top[i] for i in S): val for S, val in v.items()},
            "moebius_order2": {",".join(top[i] for i in S): val for S, val in pairs.items()},
            "singles_sum": singles, "joint": v[tuple(range(len(readers)))],
            "super_additive_ratio": singles / full,
            "order2_positive_frac": float(np.mean([p > 0 for p in pairs.values()])),
            "redundancy_order": order,
            "reads_damage": reads_mean,
            "top4_share": v[tuple(range(len(readers)))] / max(reads_mean, BARS["floor"]),
            "rows": len(rows),
        }
        print(f"[moebius] {tid:30s} singles={singles:.3f} joint={v[tuple(range(len(readers)))]:.3f} "
              f"ratio={results[tid]['super_additive_ratio']:.3f} order={order} "
              f"top4/READS={results[tid]['top4_share']:.3f}", flush=True)

    # shared interaction profile: correlate the 6 normalized pair coefficients across behaviours
    keys = sorted({k for t in results for k in results[t]["moebius_order2"]})
    profs = {}
    for t in results:
        vec = np.array([results[t]["moebius_order2"].get(k, np.nan) for k in keys], dtype=float)
        den = np.nanmax(np.abs(vec)) or 1.0
        profs[t] = vec / den
    corrs = []
    for a, b in combinations(sorted(profs), 2):
        u, w = profs[a], profs[b]
        ok = ~(np.isnan(u) | np.isnan(w))
        if ok.sum() >= 3 and np.std(u[ok]) > 0 and np.std(w[ok]) > 0:
            corrs.append(float(np.corrcoef(u[ok], w[ok])[0, 1]))
    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    prof_corr = float(np.median(corrs)) if corrs else float("nan")
    preds = {
        'pred_a_readers_are_super_additive': bool(med("super_additive_ratio") <= BARS["super_additive"]),
        'pred_b_order_two_interactions_are_positive': bool(med("order2_positive_frac") >= BARS["order2_positive_frac"]),
        'pred_c_redundancy_order_exceeds_one': bool(med("redundancy_order") >= BARS["redundancy_order"]),
        'pred_d_interaction_profile_is_shared': bool(corrs and prof_corr >= BARS["profile_corr"]),
        'pred_e_top4_carries_the_reads': bool(med("top4_share") >= BARS["top4_share"]),
    }
    nulls = {
        "a_null_ratio_ge_1": bool(med("super_additive_ratio") >= NULLS["super_additive_ge"]),
        "b_null_order2_le_.5": bool(med("order2_positive_frac") <= NULLS["order2_positive_frac_le"]),
        "d_null_corr_le_0": bool(corrs and prof_corr <= NULLS["profile_corr_le"]),
        "e_null_top4_le_.25": bool(med("top4_share") <= NULLS["top4_share_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "summary": {"tasks": sorted(results), "profile_pair_correlations": corrs,
                          "median_profile_corr": prof_corr, "pair_keys": keys,
                          "medians": {k: med(k) for k in ("super_additive_ratio", "order2_positive_frac",
                                                          "redundancy_order", "top4_share")}},
              "tasks": results, "k": K, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"],
                      "profile_corr": prof_corr}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2809 capable+localised tasks x 2^{K} reader subsets; per_cell={PER_CELL}; no model loaded")
        sys.exit(0)
    main()
