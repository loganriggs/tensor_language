#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the twenty-fifth eight spec-authored behaviours, plus one instrument.
"""v251: the AMENDED tier-3 battery (v196/v197 protocol, unchanged) over the TWENTY-FIFTH batch of spec-authored behaviours (EIGHT) -- plus one instrument.

v204 4/8, v205 6/8, v207 4/8, v209 4/8, v211 3/8, v213 3/8, v215 5/8, v217 3/8, v219 5/8, v221 5/8, v223 3/8, v225 0/8, v227 4/8, v229 0/7,
v231 7/8, v233 3/8, v235 4/8, v237 2/8, v239 3/8, v241 4/6, v243 5/7, v245 2/6, v247 4/6, v249 3/8 (+instrument each) at ~30-45 GPU-s per behaviour.
THIRD batch under the v245 lesson (pre-screen floor 1.0 on A1, mean base/donor on the donor axis; one repair or drop). Row 4 held on 14/15
cells across v247/v249 (only correlative_or_and leaked), so the v245 leak is booked as a one-off; the row-2 held-direction bar (>= 0.80)
is now the binding miss (4 of 5 v249 misses). This batch stays on the SELECTION classes that cleared both (preposition selection from a verb
or adjective, correlatives, predicate category, argument type) and adds one suppletive-morphology cell.
Recipe since v231: GRAMMATICAL cues with readout token pairs never screened before -- from v251 on the duplicate grep is ORDER-INSENSITIVE
AND QUOTE-AGNOSTIC (bare tokens; 26 double-quoted modules were invisible to the v247/v249 grep, see the 23:52 ledger row):
of/with, from/for, on/for, for/of, but/and, better/worse, safe/safely, anyone/anywhere are all new pairs under the bare grep.
Authored through `circuit_fast_screen_behaviour_spec`, single-cue swap, matched final token, cue never final, 32 A1/A2/P/C rows, ABAB directions:
verb_preposition_of_with (`Near the lantern the pilot accused/charged the miller, of course,` -> of/with; verb_preposition family);
verb_preposition_from_for (`Near the lantern the pilot prevented/blamed the miller, of course,` -> from/for; verb_preposition family);
verb_preposition_on_for (`Near the lantern the pilot relied/applied, of course,` -> on/for; verb_preposition family);
adjective_preposition_for_of (`Near the lantern the pilot was responsible/capable, of course,` -> for/of; adjective_preposition family);
correlative_but_and (`Near the lantern not/both the pilot, of course,` -> but/and; correlative coordination, both_either counted, or_and 1101 in v249);
suppletive_comparative_better_worse (`Near the lantern the crate was good/bad, but the box, of course, was even` -> better/worse; degree family);
predicate_category_safe_safely (`Near the lantern the pilot seemed/moved, of course, quite` -> safe/safely; predicate-category family, quiet/quietly, slow/slowly);
animacy_place_anyone_anywhere (`Near the lantern the pilot never met/went, of course,` -> anyone/anywhere; animacy family, person vs place).
CPU capability margins BEFORE enqueue (mean base / mean donor on the donor axis; A1 | A2; dropped rows; floor 1.0 on A1 per the v245 lesson):
  verb_preposition_of_with A1:-3.83/+3.84 drop0 A2:-3.67/+3.72 drop0 C:-1.43/+0.93 drop2
  verb_preposition_from_for A1:-6.56/+6.73 drop0 A2:-5.94/+6.18 drop0 C:-1.43/+0.93 drop2
  verb_preposition_on_for A1:-4.30/+4.08 drop0 A2:-4.24/+3.85 drop0 C:-1.43/+0.93 drop2
  adjective_preposition_for_of A1:-5.32/+5.20 drop0 A2:-4.31/+4.33 drop0 C:-1.43/+0.93 drop2
  correlative_but_and A1:-2.75/+2.95 drop0 A2:-2.68/+2.74 drop0 C:-1.43/+0.93 drop2
  suppletive_comparative_better_worse A1:-1.42/+1.49 drop0 A2:-0.66/+0.59 drop0 C:-1.43/+0.93 drop2
  predicate_category_safe_safely A1:-3.00/+2.89 drop0 A2:-3.18/+2.99 drop0 C:-1.43/+0.93 drop2
  animacy_place_anyone_anywhere A1:-1.86/+1.35 drop0 A2:-2.13/+1.78 drop0 C:-1.43/+0.93 drop2
  All eight A1 cells are >= 1.35 with zero dropped rows (floor met). better_worse's A2 arm is WEAK (0.66/0.59, the storm/road/bridge frame):
  the ONE licensed repair (crate/lid frame ending "was far") made it worse (A2 0.12/0.03, 31 dropped) and was reverted; the original A2 is
  kept and DISCLOSED -- row 5 (A2-own / A2 ceiling) is at risk on that cell from a small ceiling, not from the mechanism. No drops.
  Batch size EIGHT; bars are the eight-cell bars (4/6/7/5/5 of 8).
Protocol identical to v197/v201/v204/.../v249 (greedy pool 40, target 0.88, min_gain 0.005, max 30; row 2 = MIN over the two held directions
>= 0.80; row 3 dim A1 CE LB975 > 0 and >= 0.10; row 4 constrained rank-1 own-C UB975 <= 0.01 on 16 held C rows; row 5 = A2-own / A2 full-rank
ceiling in [0.5, 1.4], LB975 > 0). finiteness_selection is re-run as the instrument and is EXCLUDED from four-row counts (v249 disclosure).
The receipt carries cdas extraction_held per behaviour; the proposed row-4 extraction floor (>= 0.80, board proposal (vi)) is read off it and
a pass below 0.80 is booked as HOLLOW (counted by protocol, not as a circuit); it is NOT a bar here (registered semantics change only through the board).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 4 of the 8 new behaviours pass all four rows (pre-screened batches 14/27 = 0.52). prior 50%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 6 of 8 (v249 5/8, v247 4/6).  prior 55%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 7 of 8 (31/31 of the >= 1.0 A1 cells across v241-v249 passed). prior 75%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 5 of 8 (v247 6/6, v249 7/8).                prior 65%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 5 of 8 (v247 6/6, v249 7/8).     prior 70%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02 (same seed,
                           split and code path: a determinism check, NOT an independent run outcome, disclosed as such).
Accounting: four-row passes here join the one-protocol standing. EVERY cell here has a family with counted or fused members
(verb_preposition 2 counted + v240; adjective_preposition 3 counted + v240; correlative: both_either; degree: 6 counted; predicate category:
quiet/quietly, slow/slowly; animacy: 4 counted) -- no cell counts before a separability rung under its full family control set (v244/v246 stems).
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~6 min.
A1/A2 preps use g.prepare(valid_only=True); `n_dropped` is in every receipt.
Smoke: V251_SMOKE=<out.json> (CPU, V251_SMOKE_ROWS=4 per split, V251_SMOKE_NAMES=correlative_but_and; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8y_v251_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_preposition_of_with": "verb_preposition_of_with", "verb_preposition_from_for": "verb_preposition_from_for",
         "verb_preposition_on_for": "verb_preposition_on_for", "adjective_preposition_for_of": "adjective_preposition_for_of",
         "correlative_but_and": "correlative_but_and", "suppletive_comparative_better_worse": "suppletive_comparative_better_worse",
         "predicate_category_safe_safely": "predicate_category_safe_safely", "animacy_place_anyone_anywhere": "animacy_place_anyone_anywhere",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 4, "k_row2": 6, "k_row3": 7, "k_row4": 5, "k_row5": 5, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec8y_v251", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V251_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V251_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V251_SMOKE_NAMES", "correlative_but_and").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec8y_v251", "candidate_id": "corpus.unit_tier3_batch_amended_spec8y_v251", "bars": BARS,
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
