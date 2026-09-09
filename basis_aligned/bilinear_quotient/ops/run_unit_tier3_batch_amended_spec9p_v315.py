#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the forty-third three spec-authored behaviours, plus one instrument.
"""v315: the AMENDED tier-3 battery over the FORTY-THIRD batch (THREE cells) -- plus one instrument.
Both are VERB_PARTICLE cells. v313 has just made that family the productive lane: it ran FOUR prepared members with
zero errors in 171 s and came out 4/4, so verb_particle_out_up is a counted circuit (128 -> 129) while the seven
verb_preposition passes still sit in the pending pool behind the memory wall.
v285 2/7, v287 5/6, v293 6/7, v301 4/6, v303 2/4, v311 1/3 (+instrument each).
WHAT v313 ADDED THAT CHANGES THE PRIOR HERE. In the OWN arm -- own C only, no sibling controls -- the particle members
leak into each other at 0.30 (out_up), 0.47 (out_down) and comparably for up_down: raw collateral between particle
behaviours is LARGE. Adding the family controls drove every one of those to <= 0.0096 while extraction moved from
1.03 to 1.006 and from 0.992 to 0.996, i.e. to zero cost. So this family is not separable because its members barely
interact; it is separable because a rank-1 direction that respects the siblings exists and is free. That is the
strongest form of the separability result in the corpus so far, and it is the reason to keep authoring here.
CELL SELECTION IS THE INTERESTING PART OF THIS BATCH. v311 recorded that the particle class is capable only where the
bare intransitive verb OBLIGES its particle. I authored FOUR cells on that theory, choosing verbs whose bare form is
close to impossible (keeled -> over, chimed -> in, petered -> out, dozed -> off), and the CPU screen REFUTED the theory
as I had stated it: those cells came in at A1 0.75/0.77 (over_in) and 0.57/0.65 (out_off), under the 1.0 floor, while
the two ordinary high-frequency pairs cleared it easily. Obligatoriness in my judgement is not what the model has; token
frequency of the collocation is the better predictor, and this batch is the two survivors.
CPU capability margins BEFORE enqueue (mean base / mean donor on the donor axis; A1 | A2; dropped rows; floor 1.0):
  verb_particle_away_up  (shied/sidled)    A1:-1.26/+1.28 drop0 A2:-0.90/+1.28 drop0 C:-1.43/+0.93 drop2
  verb_particle_down_in  (hunkered/caved)  A1:-2.16/+2.10 drop0 A2:-2.17/+1.96 drop0 C:-1.43/+0.93 drop2
  verb_particle_on_out   (pressed/blacked) A1:-2.23/+2.33 drop0 A2:-2.16/+2.28 drop0 C:-1.43/+0.93 drop2
  DROPPED, never entered ops/, FOUR of seven authored: verb_particle_over_in (keeled/chimed A1:-0.75/+0.77),
  verb_particle_out_off (petered/dozed A1:-0.57/+0.65), verb_particle_off_down (veered/broke A1:-0.70/+0.49) and
  verb_particle_in_up (barged/perked A1:-0.07/+0.15 with 28 of 32 rows dropped -- the least capable cell I have
  screened in any family). on_out at 2.23 and down_in at 2.16 are the two strongest A1 margins any particle cell has
  shown; the previous best was out_up at 2.64/2.41, which went on to pass all four rows and is now counted.
All three pairs and all six cue verbs were checked against every vocabulary=() and kinds=() tuple in ops/ with an
order-insensitive quote-agnostic grep on the GENERATED modules: away/up, down/in and on/out free, shied/sidled/hunkered/caved/pressed/blacked
free (one candidate was rewritten when `droned` came back already used by on_away).
Protocol identical to v197/v201/.../v313. finiteness_selection is the instrument and is EXCLUDED from four-row counts.
Batch size THREE; bars are the three-cell bars (2/2/3/2/2).
WHAT THIS BATCH ALSO INFORMS, WITHOUT A CODED PREDICATE. The seven screened cells span A1 from 0.15 to 2.33, so the
receipt will show whether floor MARGIN predicts four-row passing or only capability. I am not registering a bar on it
because v311's evidence already cuts against it: out_away cleared the floor at 1.07/1.28 -- the same band as away_up
here -- and then missed row 2 at held 0.79. The observation goes in the ledger; a bar would follow the readout.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 2 of the 3 new behaviours pass all four rows. This is the SAME bar v311 set and failed
                           at 1 of 3, restated deliberately: the class is 3 of 9 across v255/v257/v311, but two of the
                           three cells here sit above 2.1 on A1, a band whose only prior occupant (out_up at 2.64)
                           passed everything. If it fails again the honest reading is that margin does not carry.
                                                                                                          prior 50%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 2 of 3.        prior 60%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on all 3.                                prior 75%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 2 of 3.                    prior 65%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 2 of 3.         prior 70%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02.
Accounting: a pass here goes to a verb_particle separability rung whose family would be up to SIX members -- v313 measured
four prepared at 171 s with no error, so six is still under a ninth of the size where verb_preposition first lost members
to OOM. Unlike the preposition passes, these can be counted within the hour they land.
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~3 min.
Smoke: V315_SMOKE=<out.json> (CPU, V315_SMOKE_ROWS=4 per split, V315_SMOKE_NAMES=verb_particle_away_up).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v315_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_particle_away_up": "verb_particle_away_up", "verb_particle_down_in": "verb_particle_down_in",
         "verb_particle_on_out": "verb_particle_on_out",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 2, "k_row2": 2, "k_row3": 3, "k_row4": 2, "k_row5": 2, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec9p_v315", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V315_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V315_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V315_SMOKE_NAMES", "verb_particle_away_up").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec9p_v315", "candidate_id": "corpus.unit_tier3_batch_amended_spec9p_v315", "bars": BARS,
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
