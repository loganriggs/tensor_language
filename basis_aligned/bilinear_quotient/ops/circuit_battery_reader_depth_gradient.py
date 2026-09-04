#!/usr/bin/env python
"""circuit_battery_reader_depth_gradient -- is "the read gets more task-specific with depth" a law, and where does it stop?

SS2819 found a monotone gradient on four readers: mlp8 is as generic as the writer (selectivity ratio 1.00-1.12) and mlp11 is the most
specific (.14-.59) on 6 of 7 behaviours. This rung extends it to ALL ten MLP readers of attention 8's write (mlp8..mlp17) and asks
whether specificity keeps rising past mlp11 or saturates. It also carries the fix forced by SS2820: every selectivity ratio now has a
registered ADMISSIBILITY GATE, because a ratio without one crowns components that do nothing -- SS2820's "most selective head" was an
inert head with +-.001 margin units of target damage. The rung MEASURES that failure mode explicitly (pred_d) instead of only avoiding
it. Readers, writer, split and gate are all fixed before the run; nothing is selected here.

# BQGATE: EXPERIMENT  pred_a_specificity_rises_with_depth pred_b_the_specific_reader_is_deep
#                     pred_c_the_gradient_saturates_after_eleven pred_d_ungated_ratios_crown_inert_readers
#                     pred_e_admissible_readers_carry_the_read

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer. Selectivity ratio
max(|d_P|, |d_C|) / max(d_A1, .5); LOWER IS MORE SELECTIVE. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_READER_DEPTH_GRADIENT_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_reader_depth_gradient.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_READER_DEPTH_GRADIENT_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_reader_depth_gradient"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "bde01acb8794434233313e6a098b7580d34b304579438cfb4a5284403ddccbd2",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
WRITER = ("attn", 8)
LAYERS = tuple(range(8, 18))                      # every MLP that can read attention 8's write
READERS = tuple(("mlp", l) for l in LAYERS)
EVAL_SPLIT = "OOD"
FAMILIES = ("A1", "P", "C")
PER_CELL = 4 if SMOKE else 16
ADMIT = 0.10                                      # a reader is admissible if its A1 damage >= .10 * READS damage
BARS = {"rho": -0.50, "deep_tasks": 5, "saturate": -0.10, "inert_tasks": 4, "coverage": 0.80,
        "floor": 0.5}
NULLS = {"rho_ge": 0.0, "deep_tasks_le": 2, "saturate_ge": 0.20, "inert_tasks_le": 1, "coverage_le": 0.40}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return order.astype(float)
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def damage(m, rows, cand, removed, fwd):
    out = []
    for b in CB.batches(rows):
        ids, fin, ans = CB.pack(b, "base")
        lg = CB.run(m, ids, fin); fwd[0] += 1
        lg2 = CB.run(m, ids, fin, writer=WRITER, removed=removed); fwd[0] += 1
        out.append((CB.margins(lg, ans, cand) - CB.margins(lg2, ans, cand)).cpu().numpy())
    return float(np.concatenate(out).mean()) if out else float("nan")


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    allr = tuple([("mlp", 8)] + [(kd, l) for l in range(9, R.NL) for kd in ("attn", "mlp")])
    fwd = [0]
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cells = {f: [r for r in rows if r["family"] == f and r["split"] == EVAL_SPLIT]
                 for f in FAMILIES if f in fams}
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        reads = damage(m, cells["A1"], cand, allr, fwd)
        per = {}
        for (kd, l) in READERS:
            d = {f: damage(m, cells[f], cand, ((kd, l),), fwd) for f in cells}
            ctrl = max(abs(d.get("P", 0.0)), abs(d.get("C", 0.0)))
            per[f"mlp{l}"] = {"layer": l, "damage": d,
                              "ratio": ctrl / max(d["A1"], BARS["floor"]),
                              "admissible": bool(d["A1"] >= ADMIT * max(reads, BARS["floor"]))}
        adm = {k: v for k, v in per.items() if v["admissible"]}
        inert_low = [k for k, v in per.items() if (not v["admissible"]) and v["ratio"] <= 0.25]
        rho = spearman([v["layer"] for v in adm.values()], [v["ratio"] for v in adm.values()]) if len(adm) >= 3 else float("nan")
        best = min(adm, key=lambda k: adm[k]["ratio"]) if adm else None
        shallow = [v["ratio"] for v in adm.values() if v["layer"] <= 11]
        deep = [v["ratio"] for v in adm.values() if v["layer"] >= 12]
        results[tid] = {
            "reads_damage": reads, "per_reader": per,
            "admissible": sorted(adm), "n_admissible": len(adm),
            "inert_readers_with_low_ratio": inert_low,
            "rho_layer_vs_ratio": rho, "best_admissible": best,
            "best_ratio": adm[best]["ratio"] if best else float("nan"),
            "best_layer": adm[best]["layer"] if best else None,
            "min_ratio_8_11": float(min(shallow)) if shallow else float("nan"),
            "min_ratio_12_17": float(min(deep)) if deep else float("nan"),
            "saturation_gap": (float(min(deep)) - float(min(shallow))) if (deep and shallow) else float("nan"),
            "admissible_coverage": sum(adm[k]["damage"]["A1"] for k in adm) / max(reads, BARS["floor"]),
        }
        print(f"[depth] {tid:30s} adm={len(adm)} rho={rho:.2f} best={best}({results[tid]['best_ratio']:.2f}) "
              f"sat={results[tid]['saturation_gap']:.2f} inert_low={len(inert_low)}", flush=True)

    fin = lambda k: [results[t][k] for t in results if not np.isnan(results[t][k])]
    med = lambda k: float(np.median(fin(k))) if fin(k) else float("nan")
    deep_tasks = [t for t in results if results[t]["best_layer"] is not None and results[t]["best_layer"] >= 10]
    inert_tasks = [t for t in results if results[t]["inert_readers_with_low_ratio"]]
    preds = {
        'pred_a_specificity_rises_with_depth': bool(med("rho_layer_vs_ratio") <= BARS["rho"]),
        'pred_b_the_specific_reader_is_deep': bool(len(deep_tasks) >= BARS["deep_tasks"]),
        'pred_c_the_gradient_saturates_after_eleven': bool(med("saturation_gap") >= BARS["saturate"]),
        'pred_d_ungated_ratios_crown_inert_readers': bool(len(inert_tasks) >= BARS["inert_tasks"]),
        'pred_e_admissible_readers_carry_the_read': bool(med("admissible_coverage") >= BARS["coverage"]),
    }
    nulls = {
        "a_null_no_gradient": bool(med("rho_layer_vs_ratio") >= NULLS["rho_ge"]),
        "b_null_shallow": bool(len(deep_tasks) <= NULLS["deep_tasks_le"]),
        "c_null_keeps_rising": bool(med("saturation_gap") <= -NULLS["saturate_ge"]),
        "d_null_gate_unnecessary": bool(len(inert_tasks) <= NULLS["inert_tasks_le"]),
        "e_null_low_coverage": bool(med("admissible_coverage") <= NULLS["coverage_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in LAYERS], "admissibility_fraction": ADMIT,
              "eval_split": EVAL_SPLIT, "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "deep_best_tasks": deep_tasks,
                          "tasks_with_inert_low_ratio_readers": inert_tasks,
                          "medians": {k: med(k) for k in ("rho_layer_vs_ratio", "best_ratio",
                                                          "saturation_gap", "admissible_coverage")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x {len(READERS)} MLP readers x {FAMILIES} "
              f"on {EVAL_SPLIT}, admissibility {ADMIT}; no model loaded")
        sys.exit(0)
    main()
