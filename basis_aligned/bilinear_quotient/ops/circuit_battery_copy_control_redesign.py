"""circuit_battery_copy_control_redesign -- the battery's copy control is broken on five of eight behaviours. Fix the BANK.

SS2851 measured the copy control's OWN native margin at a median .39 -- .09 on the paren list, .20 on the numbered list, and -.72 on
month, where the model does not natively give the copy answer at all. SS2852's protocol v3 gated those out, so five of eight capable
behaviours are now scored on the answer-preserving family alone. A selectivity stage running on ONE control for most of the bank is
not measuring selectivity, and SS2852 named this as bank work and left it.

This rung measures three copy-control DESIGNS per behaviour and adopts one only where it clears the .50 usability bar SS2852 already
registered:

  v1 CURRENT   the bank's existing repeated-item form ("48. a / 48. b" -> "48"), included as the baseline to beat.
  v2 TRIPLE    the same idea with three identical labels ("48. a / 48. b / 48. c" -> "48"): more repetitions of the token to copy,
               so the copy reading is more strongly cued without changing what is being controlled for.
  v3 ADJACENT  the token to copy immediately precedes the answer position ("48. a / 48. b / 48" -> "."-free continuation), the form
               SS2841's verbatim_repeat.copy behaviour showed the model performs at capability 1.00.

Only the CONTROL family changes; the target family A1 is untouched and is measured here to prove it.

# BQGATE: EXPERIMENT  pred_a_a_design_clears_the_bar_widely pred_b_the_best_design_beats_the_current_one
#                     pred_c_one_design_wins_across_behaviours pred_d_the_target_family_is_untouched
#                     pred_e_the_control_answer_is_not_the_successor

SIGN CONVENTION: native margin m = logit(answer) - max logit(other candidate), HIGHER MEANS THE MODEL NATIVELY GIVES THAT ANSWER; a
control family is USABLE when its own native margin is >= .50 and positive (SS2852's registered bar). No CE and no SS312 L2; nothing
installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_COPY_CONTROL_REDESIGN_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_copy_control_redesign.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_COPY_CONTROL_REDESIGN_PREREGISTRATION.md"
CALIB = ROOT / "circuit_battery_calibrated_selectivity_results.json"
BANK21 = ROOT / "circuit_battery_v2_bank21_results.json"
RUNG = "circuit_battery_copy_control_redesign"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "66f2cb6716b21495853b608ea7d9e510f12df110a0c48c2793ad3a3c6408a20b",
          CALIB: "bb493ffa74ac41381155a8c72a915aa92cc6714200fee0b9e19e590be892ee4f",
          BANK21: "7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
ENC = BANK.ENC
SPLIT = "SELECT"
PER_CELL = 4 if SMOKE else 24
USABLE = 0.50
BARS = {"n_usable": 6, "beat_current": 2.0, "shared_design": 5, "target_drift": 0.10, "floor": 0.5}
NULLS = {"n_usable_le": 3, "beat_current_le": 1.0, "shared_design_le": 2}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def variants(row):
    """Three copy-control designs derived from a bank C-family row. Returns {design: (text, answer)}.

    The bank's C rows already have the copied token as their answer; we re-shape the PROMPT only.
    """
    t, a = row["base_text"], row["base_answer"]
    out = {"v1_current": (t, a)}
    lines = [l for l in t.split("\n") if l]
    if len(lines) >= 2 and t.endswith("\n"):
        out["v2_triple"] = (t + lines[-1] + "\n", a)                    # one more identical-label line
        out["v3_adjacent"] = (t + a.strip(), a)                         # the token to copy sits adjacent
    else:
        parts = t.split()
        if parts:
            out["v2_triple"] = (t + " " + parts[-1], a)
            out["v3_adjacent"] = (t + " " + a.strip(), a)
    return out


def main():
    t0 = time.time()
    check_hashes()
    b21 = json.load(open(BANK21))
    calib = json.load(open(CALIB))
    tasks = [t for t in b21["summary"]["capable"] if b21["tasks"][t]["writer"] == "attn8"]
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    results = {}
    for tid in tasks:
        fams = set(BANK.TASKS[tid].families)
        if "C" not in fams:
            results[tid] = {"skipped": "no copy-control family in this task"}
            print(f"[cc] {tid:30s} SKIPPED (no C family)", flush=True)
            continue
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        crows = [r for r in rows if r["family"] == "C" and r["split"] == SPLIT]
        arows = [r for r in rows if r["family"] == "A1" and r["split"] == SPLIT]
        cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)

        def native(items):
            """items: list of (text, answer). Only joint-tokenizable single-token answers are scored."""
            keep = [(x, y) for x, y in items
                    if len(ENC.encode(y)) == 1 and ENC.encode(x + y) == ENC.encode(x) + ENC.encode(y)]
            if not keep:
                return float("nan"), 0
            by_len = {}
            for x, y in keep:
                by_len.setdefault(len(ENC.encode(x)), []).append((x, y))
            acc = []
            for grp in by_len.values():
                for i in range(0, len(grp), 32):
                    ch = grp[i:i + 32]
                    ids = torch.tensor([ENC.encode(x) for x, _ in ch], device=DEV)
                    fin = torch.full((len(ch),), ids.size(1) - 1, device=DEV, dtype=torch.long)
                    ans = torch.tensor([ENC.encode(y)[0] for _, y in ch], device=DEV)
                    lg = CB.run(m, ids, fin); fwd[0] += 1
                    acc.append(CB.margins(lg, ans, cand).cpu().numpy())
            return float(np.concatenate(acc).mean()), len(keep)

        per = {}
        for design in ("v1_current", "v2_triple", "v3_adjacent"):
            items = []
            for r in crows:
                v = variants(r)
                if design in v:
                    items.append(v[design])
            mg, n = native(items)
            per[design] = {"native_margin": mg, "n_scored": n, "usable": bool(mg >= USABLE)}
        a1_margin, _n = native([(r["base_text"], r["base_answer"]) for r in arows])
        # A copy control must (i) have an answer that is COPYABLE from its own prompt and (ii) differ from
        # the successor answer of the SAME generated situation. The bank groups families by group_id, so
        # the second is an exact check rather than a task-specific heuristic.
        a1_by_group = {r["group_id"]: r["base_answer"] for r in arows}
        clash = sum(1 for r in crows if a1_by_group.get(r["group_id"]) == r["base_answer"])
        noncopy = sum(1 for r in crows if r["base_answer"].strip() not in
                      r["base_text"].translate(str.maketrans("\n.:+=()[]{},%", "             ")).split())
        best = max(per, key=lambda d: (per[d]["native_margin"] if not np.isnan(per[d]["native_margin"]) else -9))
        results[tid] = {"per_design": per, "best_design": best,
                        "rows_where_control_equals_successor": clash,
                        "rows_where_control_is_not_copyable": noncopy, "n_control_rows": len(crows),
                        "best_margin": per[best]["native_margin"],
                        "current_margin": per["v1_current"]["native_margin"],
                        "a1_native_margin": a1_margin,
                        "a1_reference_2852": calib["tasks_detail"].get(tid, {}).get("native", {}).get("A1"),
                        "any_usable": any(v["usable"] for v in per.values())}
        p = results[tid]
        print(f"[cc] {tid:30s} v1={per['v1_current']['native_margin']:+.2f} "
              f"v2={per.get('v2_triple',{}).get('native_margin',float('nan')):+.2f} "
              f"v3={per.get('v3_adjacent',{}).get('native_margin',float('nan')):+.2f} "
              f"best={best} usable={p['any_usable']}", flush=True)

    live = {t: v for t, v in results.items() if "skipped" not in v}
    n_usable = sum(1 for v in live.values() if v["any_usable"])
    ratios = [abs(v["best_margin"]) / max(abs(v["current_margin"]), 1e-6) for v in live.values()
              if not np.isnan(v["best_margin"]) and not np.isnan(v["current_margin"])]
    from collections import Counter
    winners = Counter(v["best_design"] for v in live.values())
    top_design, top_n = (winners.most_common(1)[0] if winners else ("", 0))
    drift = [abs(v["a1_native_margin"] - v["a1_reference_2852"]) / max(abs(v["a1_reference_2852"]), BARS["floor"])
             for v in live.values() if v.get("a1_reference_2852") is not None]
    preds = {
        'pred_a_a_design_clears_the_bar_widely': bool(n_usable >= BARS["n_usable"]),
        'pred_b_the_best_design_beats_the_current_one': bool(ratios and float(np.median(ratios)) >= BARS["beat_current"]),
        'pred_c_one_design_wins_across_behaviours': bool(top_n >= BARS["shared_design"]),
        'pred_d_the_target_family_is_untouched': bool(drift and max(drift) <= BARS["target_drift"]),
        'pred_e_the_control_answer_is_not_the_successor': bool(all(
            v["rows_where_control_equals_successor"] == 0 and v["rows_where_control_is_not_copyable"] == 0
            for v in live.values())),
    }
    nulls = {
        "a_null_still_broken": bool(n_usable <= NULLS["n_usable_le"]),
        "b_null_no_improvement": bool(ratios and float(np.median(ratios)) <= NULLS["beat_current_le"]),
        "c_null_no_shared_design": bool(top_n <= NULLS["shared_design_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "split": SPLIT, "usable_bar": USABLE, "tasks": sorted(results),
              "summary": {"n_usable": n_usable, "n_live": len(live),
                          "median_best_over_current": float(np.median(ratios)) if ratios else None,
                          "winning_designs": dict(winners), "top_design": top_design, "top_n": top_n,
                          "max_target_drift": float(max(drift)) if drift else None,
                          "control_successor_clashes": {t: v["rows_where_control_equals_successor"]
                                                        for t, v in live.items()},
                          "control_noncopyable": {t: v["rows_where_control_is_not_copyable"]
                                                  for t, v in live.items()}},
              "tasks_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: three copy-control designs x SS2840's capable attn8 behaviours, native margin on "
              f"{SPLIT}, usability bar {USABLE}; no model loaded")
        sys.exit(0)
    main()
