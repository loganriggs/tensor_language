#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the thirty-eighth six spec-authored behaviours, plus one instrument.
"""v287: the AMENDED tier-3 battery over the THIRTY-EIGHTH batch (SIX cells) -- plus one instrument.
Opens the ` onto` readout, which no cell in the corpus has used, and is the first batch whose no-cue-reuse property
was established by the programmatic checker rather than by hand.

v263 3/7, v265 2/7, v267 2/5, v269 5/5, v277 1/8, v279 4/7, v285 2/7 (+instrument each).
WHY THE PROPERTY CLAIM IS DIFFERENT THIS TIME. v277's docstring registered "no cue verb in this batch is reused from
any counted sibling or control" on the strength of a hand check, and v281 proved that false: verb_preposition_by_with
shares `abided` with verb_preposition_of_by and `tinkered` with verb_preposition_through_with, and it leaked into
exactly those two cells (0.087 and 0.058) and nothing else. A corpus-wide audit run after that receipt found 404
distinct (cue -> readout token) mappings with 73 already shared by two or more cells -- a space far too large to
hand-check. Every cue word here was verified against every kinds=() tuple in ops/ by script: 12 cue words, zero reused.
Six cells, all NEW pairs, word-boundary quote-agnostic grep 0 hits (in/onto, to/onto, of/onto, at/onto, from/onto, on/off):
verb in_onto (dabbled/clambered), to_onto (alluded/latched), of_onto (reminded/piled), at_onto (jeered/spilled),
from_onto (refrained/tumbled); adjective on_off (intent/aloof).
WHAT THE ` onto` DESIGN IS FOR. Five of the six attach ONE new readout token to five DIFFERENT first prepositions, so
the batch varies the first member while holding the second fixed. Two batches (v277 1/8, v285 2/7) spent most of their
misses on row 2, and the rare-token reading of that was refuted at v279 (the same token landed on both sides of the
bar). If row 2 is a property of the readout token after all, a batch that holds the second token fixed across five
cells should show it as a block; if the five split, row 2 is a property of the cell and the token reading stays dead.
That is a cheap test of an already-refuted idea and it is stated as such: I expect the five to split.
CPU capability margins BEFORE enqueue (mean base / mean donor on the donor axis; A1 | A2; dropped rows; floor 1.0 on A1 per the v245 lesson):
  verb_preposition_in_onto A1:-3.72/+4.14 drop0 A2:-3.60/+4.00 drop0 C:-1.43/+0.93 drop2
  verb_preposition_to_onto A1:-5.17/+5.38 drop0 A2:-4.52/+4.78 drop0 C:-1.43/+0.93 drop2
  verb_preposition_of_onto A1:-6.88/+7.38 drop0 A2:-6.68/+7.08 drop0 C:-1.43/+0.93 drop2
  verb_preposition_at_onto A1:-4.43/+4.48 drop0 A2:-4.22/+4.23 drop0 C:-1.43/+0.93 drop2
  verb_preposition_from_onto A1:-3.93/+4.15 drop0 A2:-3.48/+4.03 drop0 C:-1.43/+0.93 drop2
  adjective_preposition_on_off A1:-3.04/+3.46 drop0 A2:-2.71/+3.06 drop0 C:-1.43/+0.93 drop2
  All SIX cleared the floor with zero dropped rows (weakest A1 3.04) -- the strongest capability profile of any batch
  today. Ninth batch on the probe path, authored WHILE v281 was running.
  CUE FRESHNESS VERIFIED PROGRAMMATICALLY: 12 cue words, zero reused, checked against every kinds=() tuple in ops/.
  THIS CHECK IS WHY THE CLAIM IS MADE HERE AND WAS WITHDRAWN FOR v277. v281 showed that v277's by_with shared BOTH its
  cues (`abided` with of_by, `tinkered` with through_with) and leaked into exactly those two cells; v277's docstring
  had claimed no cue reuse on the strength of a HAND check that missed both. The corpus has 404 distinct
  (cue -> readout token) mappings and 73 of them are already shared by two or more cells, so the hand check was
  sampling a space far too large for it. Batch size SIX; bars are the six-cell bars (3/5/5/4/4).
Protocol identical to v197/v201/.../v285. finiteness_selection is the instrument and is EXCLUDED from four-row counts.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 3 of the 6 new behaviours pass all four rows (v285 2/7, v279 4/7, v277 1/8). prior 55%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 5 of 6 (v285 5/7).   prior 60%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 5 of 6 (v285 7/7).                  prior 80%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 4 of 6 (v285 4/7).               prior 60%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 4 of 6 (v285 7/7).    prior 75%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02.
Accounting: four-row passes count only after a separability rung, and the passes from v279, v285 and this batch go
into ONE combined rung rather than three sweeps of the same 21 counted members -- the saving board proposal (ix) asks
for, taken inside the existing rules. With cue freshness now verified by script, a fused pass here would NOT have the
cue explanation and would be the first such case.
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~4 min.
Smoke: V287_SMOKE=<out.json> (CPU, V287_SMOKE_ROWS=4 per split, V287_SMOKE_NAMES=verb_preposition_in_onto).
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9l_v287_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_preposition_in_onto": "verb_preposition_in_onto", "verb_preposition_to_onto": "verb_preposition_to_onto",
         "verb_preposition_of_onto": "verb_preposition_of_onto", "verb_preposition_at_onto": "verb_preposition_at_onto",
         "verb_preposition_from_onto": "verb_preposition_from_onto", "adjective_preposition_on_off": "adjective_preposition_on_off",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 3, "k_row2": 5, "k_row3": 5, "k_row4": 4, "k_row5": 4, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec9l_v287", "behaviours": 9, "constructions": 5,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 9 * STEPS, "model_updates": 0, "fit_parameters": 9 * MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "rows" in r and "error" not in r}
    ok = bool(good)  # an empty or all-error receipt fails everything
    new = {n: good[n] for n in NEW if n in good}
    cnt = lambda row: sum(1 for n in new if new[n]["rows"][row])
    four = sum(1 for n in new if all(new[n]["rows"].values()))
    a = ok and four >= B["k_four"]
    b = ok and cnt("row2") >= B["k_row2"]
    c = ok and cnt("row3") >= B["k_row3"]
    d = ok and cnt("row4") >= B["k_row4"]
    e = ok and cnt("row5") >= B["k_row5"]
    f = ok and all(n in good and good[n].get("instr_n_units_match") and good[n].get("instr_dirB_abs_diff") is not None
                   and good[n]["instr_dirB_abs_diff"] <= B["instr_tol"] for n in INSTR)
    return {"pred_a_four_rows": bool(a), "pred_b_row2": bool(b), "pred_c_row3": bool(c), "pred_d_row4": bool(d),
            "pred_e_row5_amended": bool(e), "pred_f_instrument": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V287_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V287_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V287_SMOKE_NAMES", "verb_preposition_in_onto").split(",")]
    parent = json.loads(V196.read_text())["behaviours"]
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)
    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
            rows = {fam: g.rows_of(m, fam) for fam in ("A1", "A2", "P", "C")}
            V = dict(valid_only=True)   # capability-failure rows (donor does not beat base) are dropped, counted in n_dropped
            P = {"fit": g.prepare(backend, cut(fit_half(rows["A1"])), **V), "held": g.prepare(backend, cut(held_half(rows["A1"])), **V),
                 "held_dirA": g.prepare(backend, cut(rows["A1"][2::4]), **V), "held_dirB": g.prepare(backend, cut(rows["A1"][3::4]), **V),
                 "A2_fit": g.prepare(backend, cut(fit_half(rows["A2"])), **V), "A2_held": g.prepare(backend, cut(held_half(rows["A2"])), **V),
                 "P_held": g.prepare(backend, cut(held_half(rows["P"]))), "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))), "C_held": g.prepare(backend, cut(held_half(rows["C"])))}
            singles, ranked, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
            units = list(greedy["chosen"])
            e_fit, e_held, e_a, e_b = ext(P["fit"], units), ext(P["held"], units), ext(P["held_dirA"], units), ext(P["held_dirB"], units)
            mu1 = mu_of(P["fit"], units)
            q1 = g.block_diff_in_means(backend, P["fit"], units)
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C_fit"],), control_weight=LAM, mu=mu1)
            arms = {}
            for arm, qa in (("dim", q1), ("cdas", qc)):
                arms[arm] = {fam: dmg(P[k], units, qa, mu1) for fam, k in (("A1", "held"), ("A2", "A2_held"), ("P", "P_held"), ("C", "C_held"))}
                arms[arm]["extraction_held"] = round(ext(P["held"], units, q=qa) / e_held, 3) if abs(e_held) > 1e-6 else None
            mu2 = mu_of(P["A2_fit"], units)
            q2 = g.block_diff_in_means(backend, P["A2_fit"], units)
            a2_own = dmg(P["A2_held"], units, q2, mu2)
            a2_full = dmg(P["A2_held"], units, None, mu2)   # q=None: full-rank mean-ablation of the set = the construction's own ceiling
            a1_full = dmg(P["held"], units, None, mu1)
            share = lambda x, y: round(x / y, 4) if y else None
            d = arms["dim"]
            row5_share = share(a2_own["ce_damage"], a2_full["ce_damage"])
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,
                       "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                       "row4": arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX,
                       "row5": a2_own["ce_lb975"] > 0 and row5_share is not None and BARS["row5_share_band"][0] <= row5_share <= BARS["row5_share_band"][1]}
            R[n] = {"units": units, "n_units": len(units), "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]},
                    "direction_A": str(P["held_dirA"].rows[0].get("direction_id")), "direction_B": str(P["held_dirB"].rows[0].get("direction_id")),
                    "extraction_fit": e_fit, "extraction_held": e_held, "extraction_held_dirA": e_a, "extraction_held_dirB": e_b,
                    "arms": arms, "a2_own": a2_own, "a2_full_ceiling": a2_full, "a1_full_ceiling": a1_full,
                    "row5_share_own": row5_share, "row5_share_a1_direction": share(d["A2"]["ce_damage"], a2_full["ce_damage"]),
                    "a1_share_dim_over_full": share(d["A1"]["ce_damage"], a1_full["ce_damage"]), "old_row5_ratio": share(d["A2"]["ce_damage"], d["A1"]["ce_damage"]),
                    "instr_n_units_match": (n in parent and parent[n]["units"] == units), "instr_dirB_abs_diff": (round(abs(parent[n]["extraction_held_dirB"] - e_b), 3) if n in parent else None),
                    "rows": rows_ok, "n_dropped": {k: P[k].dropped for k in P}, "cdas_final_loss": (hist[-1] if hist else None), "n_rows": {k: len(P[k].rows) for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one behaviour must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "rows": {k: False for k in ("row2", "row3", "row4", "row5")}, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "singles_top16", "arms", "n_rows")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec9l_v287", "candidate_id": "corpus.unit_tier3_batch_amended_spec9l_v287", "bars": BARS,
              "protocol": {"split": "within-direction: fit rows[0::4]+rows[1::4], held rows[2::4]+rows[3::4]", "pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                           "cdas": {"steps": steps, "lr": LR, "complement_weight": CW, "control_weight": LAM, "controls": "own C FIT rows"},
                           "row5": "A2-own diff-in-means CE / A2 full-rank mean-ablation ceiling, LB975 > 0"},
              "four_row_passes": sorted(n for n, r in R.items() if "error" not in r and all(r["rows"].values())), "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "rows": {n: r["rows"] for n, r in R.items()}}, indent=2))


if __name__ == "__main__":
    main()
