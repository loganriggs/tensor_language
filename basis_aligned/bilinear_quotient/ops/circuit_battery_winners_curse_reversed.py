"""circuit_battery_winners_curse_reversed -- SS2869 left an asymmetry unexplained; this reverses the direction to see if it is real.

SS2869 selected on FIT and evaluated on OOD and found selection beat a random live component by **-.240**, against only **-.088** when
evaluated on TEST at the same sample size (SS2867). A held-out-VOCABULARY population separated the selected component from a random one
MORE cleanly than the nearer TEST population did. SS2869 recorded that as unexpected and explicitly did not explain it.

There are two accounts. Either the effect is a property of the OOD population as an EVALUATION set -- it spreads components out more, so
any real difference shows up larger -- or it is directional, an artefact of selecting on FIT specifically. Reversing the roles
distinguishes them: **select on OOD, evaluate on FIT.** If the advantage stays large, the OOD population is doing the work as a
population and the direction does not matter; if it collapses to SS2867's -.088, the asymmetry was directional and SS2869's -.240
should not be read as a property of the metric.

Bars carried over verbatim from the winner's-curse preregistration; PER_CELL=24 as in SS2867 and SS2869, so all three are comparable.

# BQGATE: EXPERIMENT  pred_a_selection_is_inflated
#                     pred_b_the_selected_component_does_not_beat_the_named_writer
#                     pred_c_selection_still_carries_signal_against_a_random_component
#                     pred_d_the_argmin_does_not_reproduce
#                     pred_e_inert_components_are_gated_out

SIGN CONVENTION: d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that condition's own answer, NEGATIVE = it HELPS. selectivity =
|d_C| / max(d_A1, .5), LOWER = MORE SELECTIVE. No CE and no SS312 L2; nothing installs. (Frontier: L2 is CE ADDED ABOVE THE REAL MODEL,
LOWER IS BETTER, SS2135; norm-2304 at 2.6735, SS2125 stands.)
Preregistration: polynomial_causal/CIRCUIT_BATTERY_WINNERS_CURSE_REVERSED_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/"
            "circuit_battery_winners_curse_reversed.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_WINNERS_CURSE_REVERSED_PREREGISTRATION.md"
NULLRES = ROOT / "circuit_battery_selectivity_null_calibration_results.json"
CALIB = ROOT / "circuit_battery_calibrated_selectivity_results.json"
RUNG = "circuit_battery_winners_curse_reversed"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "2689bc9d6991d94a7e4eed2b453ff13a24ab759c8abfb091bc7f450288a6b6c9", NULLRES: "b4bf278a10fea17e656db30e2d138fb219dc3c8cf5319282b307df5b08aa6295",
          CALIB: "bb493ffa74ac41381155a8c72a915aa92cc6714200fee0b9e19e590be892ee4f",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
ENC = BANK.ENC
PER_CELL = 4 if SMOKE else 24
SPLITS = ("OOD", "FIT")
WRITER = ("attn", 8)
LIVE_MIN = 0.10          # admissibility: a stand-in must DO something before its ratio is eligible (SS2820's lesson)
BARS = {"inflation": 0.15, "vs_writer": 0.0, "vs_random": -0.05, "n_agree": 2, "floor": 0.5}
NULLS = {"inflation_le": 0.05, "vs_writer_le": -0.05, "vs_random_ge": 0.0, "n_agree_ge": 5}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    nullres = json.load(open(NULLRES))
    tasks = sorted(nullres["tasks"])
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
        fv = {k: v["selectivity"] for k, v in per_split["OOD"].items() if v["selectivity"] is not None}
        tv = {k: v["selectivity"] for k, v in per_split["FIT"].items() if v["selectivity"] is not None}
        pick = min(fv, key=fv.get) if fv else None
        print(f"[wc] {tid:30s} OOD-pick={str(pick):8s} sel={fv.get(pick, float('nan')):.3f} "
              f"eval={tv.get(pick, float('nan')):.3f} live={len(fv)}/{len(comps)}", flush=True)

    w = f"{WRITER[0]}{WRITER[1]}"
    rng = np.random.default_rng(20260904)
    picks, infl, vs_writer, vs_random, argmin_agree = {}, {}, {}, {}, {}
    for tid, ps in results.items():
        fv = {k: v["selectivity"] for k, v in ps["OOD"].items() if v["selectivity"] is not None}
        tv = {k: v["selectivity"] for k, v in ps["FIT"].items() if v["selectivity"] is not None}
        if not fv or not tv:
            continue
        pick = min(fv, key=fv.get)
        picks[tid] = pick
        if pick in tv:
            # LOWER = MORE SELECTIVE, so held-out MINUS selection is POSITIVE when selection was inflated
            infl[tid] = tv[pick] - fv[pick]
            if w in tv:
                vs_writer[tid] = tv[pick] - tv[w]                     # POSITIVE = the pick is WORSE than the named writer
            pool = [k for k in tv if k != pick]
            if pool:
                vs_random[tid] = tv[pick] - float(np.median([tv[k] for k in rng.choice(pool, size=min(len(pool), 12),
                                                                                       replace=False)]))
        argmin_agree[tid] = bool(pick == min(tv, key=tv.get))
    med_infl = float(np.median(list(infl.values()))) if infl else float("nan")
    med_vs_writer = float(np.median(list(vs_writer.values()))) if vs_writer else float("nan")
    med_vs_random = float(np.median(list(vs_random.values()))) if vs_random else float("nan")
    n_agree = sum(1 for v in argmin_agree.values() if v)
    preds = {
        'pred_a_selection_is_inflated': bool(med_infl >= BARS["inflation"]),
        'pred_b_the_selected_component_does_not_beat_the_named_writer': bool(med_vs_writer >= BARS["vs_writer"]),
        'pred_c_selection_still_carries_signal_against_a_random_component': bool(med_vs_random <= BARS["vs_random"]),
        'pred_d_the_argmin_does_not_reproduce': bool(n_agree <= BARS["n_agree"]),
        'pred_e_inert_components_are_gated_out': bool(
            any(v["selectivity"] is None for ps in results.values() for v in ps["OOD"].values())),
    }
    nulls = {
        "a_null_selection_is_honest": bool(med_infl <= NULLS["inflation_le"]),
        "b_null_selection_beats_the_named_writer": bool(med_vs_writer <= NULLS["vs_writer_le"]),
        "c_null_selection_is_worthless": bool(med_vs_random >= NULLS["vs_random_ge"]),
        "d_null_argmin_reproduces": bool(n_agree >= NULLS["n_agree_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "live_min": LIVE_MIN, "tasks": tasks, "splits": list(SPLITS), "smoke": SMOKE,
              "summary": {"fit_selected_component": picks,
                          "median_inflation_test_minus_fit": med_infl, "inflation": infl,
                          "median_pick_minus_named_writer_on_test": med_vs_writer, "vs_named_writer": vs_writer,
                          "median_pick_minus_random_on_test": med_vs_random, "vs_random": vs_random,
                          "n_argmin_agreeing": n_agree, "argmin_agreement": argmin_agree,
                          "n_behaviours": len(picks)},
              "tasks_detail": results,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1500])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: select the most-selective component on FIT, evaluate it on TEST, and price the "
              f"inflation against the named writer and a random component")
        sys.exit(0)
    main()
