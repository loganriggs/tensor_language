#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the tenth eight spec-authored behaviours, plus one instrument.
"""v221: the AMENDED tier-3 battery (v196/v197 protocol, unchanged) over the TENTH EIGHT spec-authored behaviours -- plus one instrument.

v204 4/8, v205 6/8, v207 4/8, v209 4/8, v211 3/8, v213 3/8, v215 5/8, v217 3/8, v219 5/8 (+instrument each) at ~30-40 GPU-s per
behaviour. Eight more, authored through `circuit_fast_screen_behaviour_spec`, single-cue swap, matched final token, cue never final,
32 A1/A2/P/C rows, ABAB directions, opening ONE NEW family and extending four: comparative_complement_superior (`The crate near the
lantern was superior/better, of course,` -> to/than: a Latinate comparative selects `to`, degree_complement family, joins the
superlative pair) and comparative_complement_inferior (`inferior/worse, of course,`, same family); noun_preposition_interest (`showed
interest/respect, of course,` -> in/for) and noun_preposition_belief (`expressed belief/need, of course,` -> in/for): a NOUN selects
its preposition (NEW family: the adjective_preposition pair selected by an adjective, this by a noun); relative_clause_agreement
(`Near the crate, the pilot/pilots who lifted the crate` -> has/have: agreement across a relative-clause intervener, agreement family)
and proximity_agreement_nor (`Neither the X nor the pilot/pilots near the crate` -> has/have: the nearer conjunct controls, agreement
family); purpose_vs_reason (`left early because/so, of course,` -> the/that: connective selects a finite clause vs a determiner,
NEW construction in the clause-type family); relative_case_who_whom (`thanked the X who/whom, of course,` -> lifted/the: the relative
pronoun's case selects a verb vs an object, case family, pronoun_case_coordination sibling).
Capability printed BEFORE the run on CPU (`g.prepare(valid_only=True)`, 32 rows per family): ZERO A1 and ZERO A2 rows dropped on
all eight; mean base/donor margins A1: relative_clause_agreement -4.94/+5.01, comparative_complement_inferior -3.42/+3.46,
comparative_complement_superior -3.22/+3.19, noun_preposition_belief -2.86/+3.06, noun_preposition_interest -2.27/+2.40,
relative_case_who_whom -1.82/+1.56, purpose_vs_reason -1.07/+1.09, proximity_agreement_nor -0.97/+0.95 (the WEAKEST cell, below
every previous batch's weakest; kept as a capability-limited probe -- an honest null on it is expected);
A2: relative_clause -4.17/+4.30, inferior -2.83/+2.86, belief -2.29/+2.37, interest -2.19/+2.26, superior -2.17/+2.07,
purpose -1.80/+1.70, who_whom -1.56/+1.09, nor -0.96/+0.88. No draft was replaced in this batch.
Protocol identical to v197/v201/v204/v205/v207/v209/v211/v213/v215/v217/v219 (greedy pool 40, target 0.88, min_gain 0.005, max 30;
row 2 = MIN over the two held directions >= 0.80; row 3 dim A1 CE LB975 > 0 and >= 0.10; row 4 constrained rank-1 own-C UB975 <= 0.01
on 16 held C rows; row 5 = A2-own / A2 full-rank ceiling in [0.5, 1.4], LB975 > 0). finiteness_selection is re-run as the instrument.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 4 of the 8 new behaviours pass all four rows (v204 4/8, v205 6/8, v207 4/8, v209 4/8, v211 3/8,
                           v213 3/8, v215 5/8, v217 3/8, v219 5/8 with the same recipe; proximity_agreement_nor and purpose_vs_reason
                           are the expected misses from their CPU margins).                                          prior 50%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 6 of 8.              prior 70%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 7 of 8 (v217 5/8, v219 8/8; bar kept).  prior 55%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 5 of 8.                          prior 50%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 5 of 8.               prior 65%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02 (same seed,
                           split and code path: a determinism check, NOT an independent run outcome, disclosed as such).
Accounting: four-row passes here join the one-protocol standing; every member has a sibling in this batch or the corpus
(comparative pair + the superlative pair, noun_preposition pair, the two agreement members + partitive/number_of/lexical_number_pp,
purpose_vs_reason + the complementizer/clause members, who_whom + pronoun_case_coordination), so all wait for the next separability
rung before counting.
An error receipt per behaviour counts as a miss on every row. ~35 s GPU each, ~5-6 min.
A1/A2 preps use g.prepare(valid_only=True); `n_dropped` is in every receipt (expected 0 everywhere).
Smoke: V221_SMOKE=<out.json> (CPU, V221_SMOKE_ROWS=4 per split, V221_SMOKE_NAMES=noun_preposition_interest; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8j_v221_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"comparative_complement_superior": "comparative_complement_superior", "comparative_complement_inferior": "comparative_complement_inferior",
         "noun_preposition_interest": "noun_preposition_interest", "noun_preposition_belief": "noun_preposition_belief",
         "relative_clause_agreement": "relative_clause_agreement", "proximity_agreement_nor": "proximity_agreement_nor",
         "purpose_vs_reason": "purpose_vs_reason", "relative_case_who_whom": "relative_case_who_whom",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 4, "k_row2": 6, "k_row3": 7, "k_row4": 5, "k_row5": 5, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec8j_v221", "behaviours": 9, "constructions": 2,
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
    smoke = os.environ.get("V221_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V221_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V221_SMOKE_NAMES", "noun_preposition_interest").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec8j_v221", "candidate_id": "corpus.unit_tier3_batch_amended_spec8j_v221", "bars": BARS,
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
