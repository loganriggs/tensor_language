"""circuit_battery_selectivity_null_calibration -- what does the re-specified selectivity metric read for a component that is NOT the writer?

SS2861 (protocol v6) re-specified selectivity as |d_C| / max(d_A1, .5) over the copy control, after SS2860 showed the previous metric
was pinned near 1 by a control that preserves the causal variable. All five of its predictions held: the two metrics differ by a median
.574, P behaves as a positive control at .056, the writer opposes the copy answer on 7 of 7, the score generalises to held-out TEST
(gap .091) and the behaviour ordering is stable (Spearman .714).

**But .27-.63 has no meaning yet.** The campaign's .25 "selective" bar was calibrated for the OLD metric and carrying it over to the new
one is a category error, which SS2861 states and does not commit. This rung supplies the missing null the honest way: it computes the
SAME metric with every one of the 36 components standing in as the writer, so attn8's value is read against the distribution over
components rather than against a bar inherited from a different measurement.

# BQGATE: EXPERIMENT  pred_a_the_identified_writer_is_an_outlier_on_the_new_metric
#                     pred_b_the_null_distribution_is_not_degenerate
#                     pred_c_the_inherited_bar_is_not_the_right_bar
#                     pred_d_the_writers_rank_is_stable_across_held_out_splits
#                     pred_e_inert_components_are_gated_out

SIGN CONVENTION: d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that condition's own answer, NEGATIVE = it HELPS. selectivity =
|d_C| / max(d_A1, .5), LOWER = MORE SELECTIVE. No CE and no SS312 L2; nothing installs. (Frontier: L2 is CE ADDED ABOVE THE REAL MODEL,
LOWER IS BETTER, SS2135; norm-2304 at 2.6735, SS2125 stands.)
Preregistration: polynomial_causal/CIRCUIT_BATTERY_SELECTIVITY_NULL_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/"
            "circuit_battery_selectivity_null_calibration.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_SELECTIVITY_NULL_PREREGISTRATION.md"
V6RES = ROOT / "circuit_battery_variable_dependence_selectivity_results.json"
CALIB = ROOT / "circuit_battery_calibrated_selectivity_results.json"
RUNG = "circuit_battery_selectivity_null_calibration"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "ca5919eb1db6b6c91fbb5a27f9910442324cfaf51d36a67be358b8e0d2b10c58", V6RES: "940c1dc148d89d6458c6eecf1052f8f49701d1ffa149a8b3a0bffc4bd28d7e74",
          CALIB: "bb493ffa74ac41381155a8c72a915aa92cc6714200fee0b9e19e590be892ee4f",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
ENC = BANK.ENC
PER_CELL = 4 if SMOKE else 16
SPLITS = ("SELECT", "TEST")
WRITER = ("attn", 8)
LIVE_MIN = 0.10          # admissibility: a stand-in must DO something before its ratio is eligible (SS2820's lesson)
BARS = {"percentile": 0.20, "spread": 0.30, "bar_wrong": 4, "rank_rho": 0.50, "floor": 0.5}
NULLS = {"percentile_ge": 0.50, "spread_le": 0.10, "bar_wrong_le": 1, "rank_rho_le": 0.20}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    v6 = json.load(open(V6RES))
    tasks = sorted(v6["tasks"])
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    comps = [(kd, l) for l in range(NLAY) for kd in ("attn", "mlp")]
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)

        def pairs(fam, sp):
            if fam == "C3":
                return [v for r in rows if r["family"] == "C" and r["split"] == sp
                        for v in [CCR.variants(r).get("v2_triple")] if v]
            return [(r["base_text"], r["base_answer"]) for r in rows
                    if r["family"] == fam and r["split"] == sp]

        def measure(ps, stand_in):
            keep = [(x, y) for x, y in ps
                    if len(ENC.encode(y)) == 1 and ENC.encode(x + y) == ENC.encode(x) + ENC.encode(y)]
            if not keep:
                return float("nan")
            by_len = {}
            for x, y in keep:
                by_len.setdefault(len(ENC.encode(x)), []).append((x, y))
            dm = []
            for grp in by_len.values():
                for i in range(0, len(grp), 32):
                    ch = grp[i:i + 32]
                    ids = torch.tensor([ENC.encode(x) for x, _ in ch], device=DEV)
                    fin = torch.full((len(ch),), ids.size(1) - 1, device=DEV, dtype=torch.long)
                    ans = torch.tensor([ENC.encode(y)[0] for _, y in ch], device=DEV)
                    lg = CB.run(m, ids, fin); fwd[0] += 1
                    readers = [(kd, l) for l in range(stand_in[1], NLAY) for kd in ("attn", "mlp")
                               if not (l == stand_in[1] and kd == stand_in[0])]
                    lg2 = CB.run(m, ids, fin, writer=stand_in, removed=tuple(readers) + ("direct",)); fwd[0] += 1
                    dm.append((CB.margins(lg, ans, cand) - CB.margins(lg2, ans, cand)).cpu().numpy())
            return float(np.concatenate(dm).mean())

        per_split = {}
        for sp in SPLITS:
            a1p, c3p = pairs("A1", sp), pairs("C3", sp)
            per_comp = {}
            for c in comps:
                da1 = measure(a1p, c)
                dc = measure(c3p, c)
                live = abs(da1) >= LIVE_MIN
                per_comp[f"{c[0]}{c[1]}"] = {"d_A1": da1, "d_C": dc, "live": bool(live),
                                             "selectivity": (abs(dc) / max(da1, BARS["floor"])) if live else None}
            per_split[sp] = per_comp
        results[tid] = per_split
        w = f"{WRITER[0]}{WRITER[1]}"
        vals = [v["selectivity"] for v in per_split["SELECT"].values() if v["selectivity"] is not None]
        me = per_split["SELECT"][w]["selectivity"]
        pct = (float(np.mean([v < me for v in vals])) if (vals and me is not None) else float("nan"))
        print(f"[null] {tid:30s} attn8 sel={me if me is None else round(me,3)} "
              f"percentile={pct:.3f} live_components={len(vals)}/{len(comps)}", flush=True)

    w = f"{WRITER[0]}{WRITER[1]}"
    pcts, spreads, below_bar, ranks = {}, {}, {}, {}
    for tid, ps in results.items():
        vals = {k: v["selectivity"] for k, v in ps["SELECT"].items() if v["selectivity"] is not None}
        me = vals.get(w)
        if me is None:
            continue
        pcts[tid] = float(np.mean([v < me for v in vals.values()]))
        spreads[tid] = float(np.percentile(list(vals.values()), 90) - np.percentile(list(vals.values()), 10))
        below_bar[tid] = int(sum(1 for v in vals.values() if v <= 0.25))
        t_vals = {k: v["selectivity"] for k, v in ps["TEST"].items() if v["selectivity"] is not None}
        common = sorted(set(vals) & set(t_vals))
        ranks[tid] = (float(spearmanr([vals[k] for k in common], [t_vals[k] for k in common]).statistic)
                      if len(common) >= 5 else float("nan"))
    med_pct = float(np.median(list(pcts.values()))) if pcts else float("nan")
    med_spread = float(np.median(list(spreads.values()))) if spreads else float("nan")
    med_below = float(np.median(list(below_bar.values()))) if below_bar else float("nan")
    rho = [v for v in ranks.values() if not np.isnan(v)]
    med_rho = float(np.median(rho)) if rho else float("nan")
    preds = {
        'pred_a_the_identified_writer_is_an_outlier_on_the_new_metric': bool(med_pct <= BARS["percentile"]),
        'pred_b_the_null_distribution_is_not_degenerate': bool(med_spread >= BARS["spread"]),
        'pred_c_the_inherited_bar_is_not_the_right_bar': bool(med_below >= BARS["bar_wrong"]),
        'pred_d_the_writers_rank_is_stable_across_held_out_splits': bool(med_rho >= BARS["rank_rho"]),
        'pred_e_inert_components_are_gated_out': bool(
            any(v["selectivity"] is None for ps in results.values() for v in ps["SELECT"].values())),
    }
    nulls = {
        "a_null_writer_is_typical": bool(med_pct >= NULLS["percentile_ge"]),
        "b_null_every_component_scores_alike": bool(med_spread <= NULLS["spread_le"]),
        "c_null_inherited_bar_is_fine": bool(med_below <= NULLS["bar_wrong_le"]),
        "d_null_component_ranking_is_noise": bool(med_rho <= NULLS["rank_rho_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "live_min": LIVE_MIN, "tasks": tasks, "splits": list(SPLITS), "smoke": SMOKE,
              "summary": {"median_writer_percentile": med_pct, "writer_percentile": pcts,
                          "median_null_spread_p90_p10": med_spread, "null_spread": spreads,
                          "median_components_under_inherited_bar": med_below,
                          "components_under_inherited_bar": below_bar,
                          "median_component_rank_rho_select_test": med_rho, "component_rank_rho": ranks},
              "tasks_detail": results,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1500])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: the v6 selectivity metric computed with all 36 components standing in as the "
              f"writer, so attn8 is read against a distribution rather than an inherited bar")
        sys.exit(0)
    main()
