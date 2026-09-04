#!/usr/bin/env python
"""circuit_battery_reader_selectivity -- if NO writer is selective, is any READER?

SS2817's most robust finding is negative: attention 8 is the writer for 7 of 8 capable behaviours and NONE of them is
writer-selective (control damage .55-1.03 of target damage, surviving the paired-situation repair). Task specificity therefore is not
at the writer. This rung asks the next question on the repaired bank: does the specificity live in the READ? For the predeclared common
reader set {mlp8, mlp9, mlp10, mlp11} it measures, on OOD rows only and with the answer-preserving family P and the copy control C
drawn from the SAME generated situation as the target (the grouped bank guarantees this), each reader's damage on the target versus its
damage on the controls -- and compares the best reader's selectivity with the writer's. Nothing here is selected: readers, writer,
behaviours and split are all fixed before the run.

# BQGATE: EXPERIMENT  pred_a_some_reader_is_selective pred_b_readers_beat_the_writer_on_selectivity
#                     pred_c_the_selective_reader_is_shared pred_d_readers_push_away_from_copying
#                     pred_e_ood_target_damage_reproduces_the_battery

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's answer; a NEGATIVE damage on a control means
removing the edge HELPS that control's answer. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_READER_SELECTIVITY_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_reader_selectivity.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_READER_SELECTIVITY_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_reader_selectivity"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "dfa262bf64362d28b14af4225fd0e26666eb674b6fba576f981c936d56d03abd",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
COMMON = (("mlp", 8), ("mlp", 9), ("mlp", 10), ("mlp", 11))   # predeclared, identical for every behaviour
NAMES = tuple(f"{k}{l}" for k, l in COMMON)
WRITER = ("attn", 8)
EVAL_SPLIT = "OOD"
FAMILIES = ("A1", "P", "C")
PER_CELL = 4 if SMOKE else 16
BARS = {"sel_ratio": 0.25, "sel_tasks": 4, "beat_writer": 0.25, "shared_mode": 4,
        "copy_help_tasks": 4, "repro_tol": 0.15, "floor": 0.5}
NULLS = {"sel_tasks_le": 0, "beat_writer_le": 0.0, "shared_mode_le": 2, "copy_help_tasks_le": 1}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def damage(m, rows, cand, removed, fwd):
    """Mean margin damage of removing `removed` from the writer's write, on these rows."""
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
    fwd = [0]
    allr = [("mlp", 8)] + [(kd, l) for l in range(9, R.NL) for kd in ("attn", "mlp")]
    full_arm = tuple(allr) + ("direct",)
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cells = {f: [r for r in rows if r["family"] == f and r["split"] == EVAL_SPLIT]
                 for f in FAMILIES if f in fams}
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        per_reader = {}
        for name, comp in zip(NAMES, COMMON):
            per_reader[name] = {f: damage(m, cells[f], cand, (comp,), fwd) for f in cells}
        joint = {f: damage(m, cells[f], cand, tuple(COMMON), fwd) for f in cells}
        writer_arm = {f: damage(m, cells[f], cand, full_arm, fwd) for f in cells}
        def ratio(d):
            ctrl = max(abs(d.get("P", 0.0)), abs(d.get("C", 0.0)))
            return ctrl / max(d["A1"], BARS["floor"])
        rr = {n: ratio(d) for n, d in per_reader.items()}
        best = min(rr, key=lambda n: rr[n])
        results[tid] = {
            "per_reader": per_reader, "reader_ratios": rr, "joint": joint,
            "writer_full": writer_arm, "writer_ratio": ratio(writer_arm),
            "joint_ratio": ratio(joint), "best_reader": best, "best_reader_ratio": rr[best],
            "beat_writer": ratio(writer_arm) - rr[best],
            "copy_control_joint_damage": joint.get("C", float("nan")),
            "ood_a1_joint": joint["A1"],
            "battery_ood_full": b2["tasks"][tid]["ood_full_d_m"],
            "rows": {f: len(v) for f, v in cells.items()},
        }
        print(f"[readersel] {tid:30s} writer={ratio(writer_arm):.2f} best={best}:{rr[best]:.2f} "
              f"joint={results[tid]['joint_ratio']:.2f} C_dmg={joint.get('C', float('nan')):.3f}", flush=True)

    sel = [t for t in results if results[t]["best_reader_ratio"] <= BARS["sel_ratio"]]
    modes = defaultdict(int)
    for t in results:
        modes[results[t]["best_reader"]] += 1
    mode_n = max(modes.values()) if modes else 0
    copy_help = [t for t in results if results[t]["copy_control_joint_damage"] < 0]
    # instrument consistency: the four common readers are a SUBSET of the battery's FULL arm, so their
    # joint OOD damage must be a proper fraction of it -- between .30 and 1.05 (1.05 allows float wobble)
    repro = [results[t]["ood_a1_joint"] / max(results[t]["battery_ood_full"], BARS["floor"]) for t in results]
    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    preds = {
        'pred_a_some_reader_is_selective': bool(len(sel) >= BARS["sel_tasks"]),
        'pred_b_readers_beat_the_writer_on_selectivity': bool(med("beat_writer") >= BARS["beat_writer"]),
        'pred_c_the_selective_reader_is_shared': bool(mode_n >= BARS["shared_mode"]),
        'pred_d_readers_push_away_from_copying': bool(len(copy_help) >= BARS["copy_help_tasks"]),
        'pred_e_ood_target_damage_reproduces_the_battery':
            bool(repro and all(0.30 <= r <= 1.05 for r in repro)),
    }
    nulls = {
        "a_null_none_selective": bool(len(sel) <= NULLS["sel_tasks_le"]),
        "b_null_no_gain_over_writer": bool(med("beat_writer") <= NULLS["beat_writer_le"]),
        "c_null_not_shared": bool(mode_n <= NULLS["shared_mode_le"]),
        "d_null_no_copy_push": bool(len(copy_help) <= NULLS["copy_help_tasks_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "common_reader_set": list(NAMES), "writer": "attn8", "eval_split": EVAL_SPLIT,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "selective_tasks": sel,
                          "best_reader_modes": dict(modes), "copy_help_tasks": copy_help,
                          "common_over_full_ood": {t: repro[i] for i, t in enumerate(sorted(results))},
                          "medians": {k: med(k) for k in ("best_reader_ratio", "writer_ratio",
                                                          "joint_ratio", "beat_writer")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"],
                      "modes": dict(modes)}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x ({len(NAMES)} single readers + joint + writer) "
              f"x {FAMILIES} on {EVAL_SPLIT}; no model loaded")
        sys.exit(0)
    main()
