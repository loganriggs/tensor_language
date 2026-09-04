"""circuit_battery_v6_precision_replication -- are SS2863's two misses sample-size limits or real?

SS2863 ran the v6 metric over 14 behaviours at PER_CELL=24 and two clauses missed narrowly: the writer opposes the copy answer on
**10 of 14 = 71.4%** against an 80% bar, and the behaviour ordering reproduced at Spearman **.556** against a .60 bar (SS2861 had .714
on the seven attn8 behaviours). Both are close enough to their bars that the honest question is whether they are sampling noise.

This rung is the SAME design at **PER_CELL=64** -- 2.7x the rows per cell, everything else identical, **bars unchanged**. Re-registering
the bars would make a replication meaningless, so they are carried over verbatim from the bank-sweep preregistration. A miss that
survives 2.7x the data is a fact about the model; one that flips was a sample-size limit, and the ledger will say which.

# BQGATE: EXPERIMENT  pred_a_the_two_controls_measure_different_things
#                     pred_b_the_preserving_control_behaves_as_a_positive_control
#                     pred_c_the_writer_opposes_the_copy_answer
#                     pred_d_selectivity_generalises_to_held_out_splits
#                     pred_e_the_behaviour_ordering_is_stable

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer, NEGATIVE = the arm HELPS it. No CE
and no SS312 L2; nothing installs. (Frontier convention, per standing rule though unused here: frontier L2 is CE ADDED ABOVE THE REAL
MODEL, LOWER IS BETTER, SS2135; frontier is norm-2304 at 2.6735, SS2125.)
Preregistration: polynomial_causal/CIRCUIT_BATTERY_V6_PRECISION_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/"
            "circuit_battery_v6_precision_replication.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import circuit_battery_copy_control_redesign as CCR
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_V6_PRECISION_PREREGISTRATION.md"
BANK21 = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_v6_precision_replication"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "227598985cc07cc7643f89981b9741c94ba5f349cd3653b0b32ad9869f2a89d1", BANK21: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
ENC = BANK.ENC
PER_CELL = 4 if SMOKE else 64
SPLITS = ("SELECT", "TEST", "OOD")
BARS = {"differ": 0.30, "positive_ctrl": 0.25, "oppose_frac": 0.80, "generalise": 0.15, "stable": 0.60, "floor": 0.5}
NULLS = {"differ_le": 0.10, "positive_ctrl_ge": 0.50, "oppose_frac_le": 0.50, "generalise_ge": 0.40, "stable_le": 0.20}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    b21 = json.load(open(BANK21))
    tasks = sorted(t for t, v in b21["tasks"].items()
                   if "writer" in v and {"A1", "P", "C"} <= set(BANK.TASKS[t].families))
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    readers = [("mlp", 8)] + [(kd, l) for l in range(9, NLAY) for kd in ("attn", "mlp")]
    LADDER = {"full": tuple(readers) + ("direct",), "half": tuple(readers[:max(1, len(readers) // 2)]),
              "quarter": tuple(readers[:max(1, len(readers) // 4)]),
              "eighth": tuple(readers[:max(1, len(readers) // 8)])}
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        wname = b21["tasks"][tid]["writer"]
        wkind = "".join(c for c in wname if not c.isdigit())
        wlayer = int("".join(c for c in wname if c.isdigit()))
        writer = (wkind, wlayer)
        arm = tuple((kd, l) for l in range(wlayer, NLAY) for kd in ("attn", "mlp")
                    if not (l == wlayer and kd == wkind)) + ("direct",)   # the FULL arm SS2840 published

        def measure(pairs):
            keep = [(x, y) for x, y in pairs
                    if len(ENC.encode(y)) == 1 and ENC.encode(x + y) == ENC.encode(x) + ENC.encode(y)]
            if not keep:
                return float("nan"), float("nan"), 0
            by_len = {}
            for x, y in keep:
                by_len.setdefault(len(ENC.encode(x)), []).append((x, y))
            nat, dm = [], []
            for grp in by_len.values():
                for i in range(0, len(grp), 32):
                    ch = grp[i:i + 32]
                    ids = torch.tensor([ENC.encode(x) for x, _ in ch], device=DEV)
                    fin = torch.full((len(ch),), ids.size(1) - 1, device=DEV, dtype=torch.long)
                    ans = torch.tensor([ENC.encode(y)[0] for _, y in ch], device=DEV)
                    lg = CB.run(m, ids, fin); fwd[0] += 1
                    mn = CB.margins(lg, ans, cand)
                    lg2 = CB.run(m, ids, fin, writer=writer, removed=arm); fwd[0] += 1
                    nat.append(mn.cpu().numpy())
                    dm.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            return (float(np.concatenate(nat).mean()), float(np.concatenate(dm).mean()), len(keep))

        per_split = {}
        for sp in SPLITS:
            a1 = [(r["base_text"], r["base_answer"]) for r in rows if r["family"] == "A1" and r["split"] == sp]
            pd_ = [(r["donor_text"], r["donor_answer"]) for r in rows if r["family"] == "P" and r["split"] == sp]
            c3 = [v for r in rows if r["family"] == "C" and r["split"] == sp
                  for v in [CCR.variants(r).get("v2_triple")] if v]
            d = {}
            for k, pairs in (("A1", a1), ("P_donor", pd_), ("C3", c3)):
                n, dm, kk = measure(pairs)
                if kk:
                    d[k] = {"native": n, "damage": dm, "n": kk}
            if "A1" not in d:
                continue
            den = max(d["A1"]["damage"], BARS["floor"])
            per_split[sp] = {"cond": d,
                             "selectivity": (abs(d["C3"]["damage"]) / den) if "C3" in d else float("nan"),
                             "positive_control": (abs(d["P_donor"]["damage"] - d["A1"]["damage"]) / den)
                                                 if "P_donor" in d else float("nan"),
                             "ratio_old_style": (max(abs(d[k]["damage"]) for k in ("P_donor", "C3") if k in d) / den)}
        results[tid] = {"arm": "full", "writer": wname,
                        "capable": b21["tasks"][tid].get("capable"), "per_split": per_split}
        s = per_split.get("SELECT", {}); t_ = per_split.get("TEST", {})
        print(f"[v6] {tid:30s} arm={results[tid]['arm']:8s} "
              f"sel(SELECT)={s.get('selectivity', float('nan')):.3f} sel(TEST)={t_.get('selectivity', float('nan')):.3f} "
              f"pos_ctrl={s.get('positive_control', float('nan')):.3f} "
              f"old_style={s.get('ratio_old_style', float('nan')):.3f}", flush=True)

    def col(sp, key):
        return {t: v["per_split"][sp][key] for t, v in results.items()
                if sp in v["per_split"] and not np.isnan(v["per_split"][sp][key])}
    sel_s, sel_t, pos_s = col("SELECT", "selectivity"), col("TEST", "selectivity"), col("SELECT", "positive_control")
    old_s = col("SELECT", "ratio_old_style")
    differ = [abs(old_s[t] - sel_s[t]) for t in sel_s if t in old_s]
    oppose = [t for t, v in results.items() if "SELECT" in v["per_split"]
              and "C3" in v["per_split"]["SELECT"]["cond"]
              and v["per_split"]["SELECT"]["cond"]["C3"]["damage"] < 0]
    common = sorted(set(sel_s) & set(sel_t))
    gen = [abs(sel_s[t] - sel_t[t]) for t in common]
    rho = float(spearmanr([sel_s[t] for t in common], [sel_t[t] for t in common]).statistic) if len(common) >= 3 else float("nan")
    preds = {
        'pred_a_the_two_controls_measure_different_things': bool(differ and float(np.median(differ)) >= BARS["differ"]),
        'pred_b_the_preserving_control_behaves_as_a_positive_control': bool(
            pos_s and float(np.median(list(pos_s.values()))) <= BARS["positive_ctrl"]),
        'pred_c_the_writer_opposes_the_copy_answer': bool(
            results and len(oppose) / len(results) >= BARS["oppose_frac"]),
        'pred_d_selectivity_generalises_to_held_out_splits': bool(gen and float(np.median(gen)) <= BARS["generalise"]),
        'pred_e_the_behaviour_ordering_is_stable': bool(not np.isnan(rho) and rho >= BARS["stable"]),
    }
    nulls = {
        "a_null_metrics_agree": bool(differ and float(np.median(differ)) <= NULLS["differ_le"]),
        "b_null_preserving_control_is_not_positive": bool(
            pos_s and float(np.median(list(pos_s.values()))) >= NULLS["positive_ctrl_ge"]),
        "c_null_writer_does_not_oppose_copy": bool(
            results and len(oppose) / len(results) <= NULLS["oppose_frac_le"]),
        "d_null_does_not_generalise": bool(gen and float(np.median(gen)) >= NULLS["generalise_ge"]),
        "e_null_ordering_is_noise": bool(not np.isnan(rho) and rho <= NULLS["stable_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "tasks": tasks, "splits": list(SPLITS), "smoke": SMOKE,
              "summary": {"median_metric_difference": float(np.median(differ)) if differ else None,
                          "median_positive_control": float(np.median(list(pos_s.values()))) if pos_s else None,
                          "n_opposing_copy": len(oppose), "n_behaviours": len(results), "opposing": oppose,
                          "median_select_test_gap": float(np.median(gen)) if gen else None,
                          "spearman_select_test": rho, "n_common": len(common),
                          "selectivity_SELECT": sel_s, "selectivity_TEST": sel_t,
                          "selectivity_OOD": col("OOD", "selectivity"),
                          "positive_control_SELECT": pos_s, "ratio_old_style_SELECT": old_s},
              "tasks_detail": results,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1700])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: P re-specified as a POSITIVE control (it preserves the causal variable); "
              f"selectivity scored on the copy control, on held-out TEST/OOD splits")
        sys.exit(0)
    main()
