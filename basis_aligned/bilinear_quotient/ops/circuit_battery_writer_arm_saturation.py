"""circuit_battery_writer_arm_saturation -- is "ZERO behaviours are writer-selective" the same ceiling artifact?

SS2850 showed that SS2849's selectivity ratios were arithmetic: whole-component ablation removes 1.538x the native margin on the
target family AND 1.538x on the controls, so every ratio pinned at 1.00. The battery's own selectivity stage uses a DIFFERENT
arm -- the writer's final-position write removed from every reader edge -- and its central negative result, "ZERO behaviours are
writer-selective", has now been reported three times (SS2817, SS2840, and SS2819's reader-side version). Nothing has checked whether
THAT arm is saturated too. If it is, the campaign's most-repeated finding is the same artifact at a different granularity.

This rung measures it directly on SS2840's capable behaviours: the battery's FULL arm's damage as a fraction of each family's NATIVE
margin, a half-strength version of the same arm, and the native margin of the copy control itself -- which SS2850 measured at .18 on
the numbered list, small enough that a ratio built on it may be unstable regardless of saturation.

# BQGATE: EXPERIMENT  pred_a_the_writer_arm_is_not_saturated pred_b_controls_are_not_saturated
#                     pred_c_ratios_replicate_the_battery pred_d_half_strength_agrees
#                     pred_e_the_copy_control_has_a_usable_margin

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer; saturation = d_m /
max(m_NATIVE, .5), 1.0 meaning the whole native margin is gone; ratio = max(|d_P|,|d_C|)/max(d_A1,.5), LOWER IS MORE SELECTIVE.
No CE and no SS312 L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_WRITER_ARM_SATURATION_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_writer_arm_saturation.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_WRITER_ARM_SATURATION_PREREGISTRATION.md"
NVE = ROOT / "circuit_battery_node_vs_edge_selectivity_results.json"
BANK21 = ROOT / "circuit_battery_v2_bank21_results.json"
RUNG = "circuit_battery_writer_arm_saturation"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "86a7944e2562ac9d1d7689728c90d11e80ad3c1d52b5b858f8e278407228ae1c",
          NVE: "575ef8071ba51b3097eb76fd84ea574c8abd040233b637249b8037ec1b4f262e",
          BANK21: "7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
SPLIT = "SELECT"          # the split the battery's own selectivity stage scores on
PER_CELL = 4 if SMOKE else 24
BARS = {"sat": 0.60, "ctrl_sat": 0.60, "repro": 0.15, "half_agree": 0.15, "c_margin": 0.50,
        "c_margin_frac": 0.50, "floor": 0.5}
NULLS = {"sat_ge": 0.90, "ctrl_sat_ge": 0.90, "repro_ge": 0.40, "c_margin_frac_le": 0.20}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    b21 = json.load(open(BANK21))
    tasks = [t for t in b21["summary"]["capable"] if b21["tasks"][t]["writer"] == "attn8"]
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    readers = [("mlp", 8)] + [(kd, l) for l in range(9, NLAY) for kd in ("attn", "mlp")]
    FULL = tuple(readers) + ("direct",)
    # a GENUINE partial arm: the write removed from only the first half of the reader edges.
    # (My first draft used ablate=False, which is CB.run's default and reproduces FULL exactly -- a no-op
    # predicate. Caught before the preregistration was written, so no bar was ever set against it.)
    HALF = tuple(readers[:max(1, len(readers) // 2)])
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cells = {f: [r for r in rows if r["family"] == f and r["split"] == SPLIT]
                 for f in ("A1", "P", "C") if f in fams}
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        nat, full_d, half_d = {}, {}, {}
        for fam, rws in cells.items():
            n, f1, f2 = [], [], []
            for b in CB.batches(rws):
                ids, fin, ans = CB.pack(b, "base")
                lg = CB.run(m, ids, fin); fwd[0] += 1
                mn = CB.margins(lg, ans, cand)
                n.append(mn.cpu().numpy())
                lgF = CB.run(m, ids, fin, writer=("attn", 8), removed=FULL); fwd[0] += 1
                f1.append((mn - CB.margins(lgF, ans, cand)).cpu().numpy())
                lgH = CB.run(m, ids, fin, writer=("attn", 8), removed=HALF); fwd[0] += 1
                f2.append((mn - CB.margins(lgH, ans, cand)).cpu().numpy())
            nat[fam] = float(np.concatenate(n).mean())
            full_d[fam] = float(np.concatenate(f1).mean())
            half_d[fam] = float(np.concatenate(f2).mean())
        ratio = lambda d: max(abs(d.get("P", 0.0)), abs(d.get("C", 0.0))) / max(d["A1"], BARS["floor"])
        results[tid] = {
            "native": nat, "full_damage": full_d, "half_damage": half_d,
            "half_ratio": ratio(half_d), "half_saturation_A1": half_d["A1"] / max(nat["A1"], BARS["floor"]),
            "saturation_A1": full_d["A1"] / max(nat["A1"], BARS["floor"]),
            "saturation_ctrl": max(abs(full_d.get("P", 0.0)) / max(nat.get("P", 1.0), BARS["floor"]),
                                   abs(full_d.get("C", 0.0)) / max(nat.get("C", 1.0), BARS["floor"])),
            "ratio": ratio(full_d), "battery_ratio": b21["tasks"][tid]["selectivity_ratio"],
            "c_native_margin": nat.get("C", float("nan")),
            "c_margin_over_a1": nat.get("C", float("nan")) / max(nat["A1"], 1e-9),
        }
        p = results[tid]
        print(f"[sat] {tid:30s} satA1={p['saturation_A1']:.2f} satC={p['saturation_ctrl']:.2f} "
              f"ratio={p['ratio']:.2f} battery={p['battery_ratio']:.2f} "
              f"C_native={p['c_native_margin']:.2f}", flush=True)

    med = lambda k: float(np.median([results[t][k] for t in results if not np.isnan(results[t][k])]))
    repro = max(abs(results[t]["ratio"] - results[t]["battery_ratio"]) for t in results)
    cfrac = [results[t]["c_margin_over_a1"] for t in results if not np.isnan(results[t]["c_margin_over_a1"])]
    usable_c = [t for t in results if results[t]["c_native_margin"] >= BARS["c_margin"]]
    preds = {
        'pred_a_the_writer_arm_is_not_saturated': bool(med("saturation_A1") <= BARS["sat"]),
        'pred_b_controls_are_not_saturated': bool(med("saturation_ctrl") <= BARS["ctrl_sat"]),
        'pred_c_ratios_replicate_the_battery': bool(repro <= BARS["repro"]),
        'pred_d_half_strength_agrees': bool(max(abs(results[t]["ratio"] - results[t]["half_ratio"])
                                                 for t in results) <= BARS["half_agree"]),
        'pred_e_the_copy_control_has_a_usable_margin':
            bool(len(usable_c) >= 0.5 * max(len(results), 1)),
    }
    nulls = {
        "a_null_saturated": bool(med("saturation_A1") >= NULLS["sat_ge"]),
        "b_null_controls_saturated": bool(med("saturation_ctrl") >= NULLS["ctrl_sat_ge"]),
        "c_null_no_replication": bool(repro >= NULLS["repro_ge"]),
        "e_null_copy_margin_tiny": bool(cfrac and float(np.median(cfrac)) <= NULLS["c_margin_frac_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "split": SPLIT, "tasks": sorted(results),
              "summary": {"medians": {k: med(k) for k in
                                      ("saturation_A1", "saturation_ctrl", "ratio", "half_ratio",
                                       "half_saturation_A1", "c_native_margin", "c_margin_over_a1")},
                          "max_ratio_gap_vs_battery": repro,
                          "tasks_with_usable_copy_margin": usable_c},
              "tasks_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1300])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: saturation of the battery's FULL writer arm on SS2840's capable attn8 behaviours, "
              f"split {SPLIT}; no model loaded")
        sys.exit(0)
    main()
