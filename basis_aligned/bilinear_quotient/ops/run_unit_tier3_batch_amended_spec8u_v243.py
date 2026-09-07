#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the twenty-first seven spec-authored behaviours, plus one instrument.
"""v243: the AMENDED tier-3 battery (v196/v197 protocol, unchanged) over the TWENTY-FIRST batch of spec-authored behaviours (SEVEN) -- plus one instrument.

v204 4/8, v205 6/8, v207 4/8, v209 4/8, v211 3/8, v213 3/8, v215 5/8, v217 3/8, v219 5/8, v221 5/8, v223 3/8, v225 0/8, v227 4/8, v229 0/7,
v231 7/8, v233 3/8, v235 4/8, v237 2/8, v239 3/8, v241 4/6 (+instrument each) at ~30-45 GPU-s per behaviour. SECOND batch under the v237
lesson (CPU capability read before enqueue; cells below 0.5/0.5 repaired once or dropped): v241 went 4/6 with preds 6/6 -- this batch is the
test of whether that was the pre-screen or one good draw. Recipe since v231: GRAMMATICAL cues with readout token pairs never screened before
(every pair here was grepped against every `vocabulary=` in the corpus). Authored through `circuit_fast_screen_behaviour_spec`, single-cue swap,
matched final token, cue never final, 32 A1/A2/P/C rows, ABAB directions:
agreement_do_does (`Near the lantern they/he, of course,` -> do/does; number family, auxiliary agreement -- `do` has never been a readout);
reflexive_number_itself_themselves (`Near the lantern it/they stood, of course, by` -> itself/themselves; number family, inanimate reflexive);
possessive_person_our_your (`Near the lantern we/you kept the crate, of course, in` -> our/your; person family, first-vs-second plural);
number_anaphor_other_others (`Near the lantern the pilot kept this/these, of course, and sold the` -> other/others; number family; the
  arithmetic-cue version was fully incapable at v241, this one cues on a bare demonstrative);
possessive_pronoun_number_hers_theirs (`Near the lantern she/they kept the crate, of course, since it was` -> hers/theirs; number family);
possessive_pronoun_person_ours_theirs (`Near the lantern we/they kept the crate, of course, since it was` -> ours/theirs; person family);
number_anaphor_it_them (`Near the lantern the pilot kept the crate/crates, of course, and later sold` -> it/them; number family, NOUN cue).
Five of seven are number-family members and two person-family members: the batch deliberately tests whether the number family keeps
separating as it densifies (v234 is re-checking 8 counted number members under 13 controls right now) -- a batch that passes four rows
and then fuses at separability is the informative outcome, not a loss.
CPU capability margins BEFORE enqueue (mean base / mean donor on the donor axis; A1 | A2; dropped rows):
  agreement_do_does A1:-4.29/+4.16 drop0 A2:-2.96/+2.78 drop0 C:-1.43/+0.93 drop2
  number_anaphor_other_others A1:-0.89/+0.95 drop0 A2:-0.57/+0.54 drop0 C:-1.43/+0.93 drop2
  reflexive_number_itself_themselves A1:-2.78/+2.74 drop0 A2:-3.18/+3.10 drop0 C:-1.43/+0.93 drop2
  possessive_person_our_your A1:-3.46/+3.48 drop0 A2:-3.72/+3.95 drop0 C:-1.43/+0.93 drop2
  possessive_pronoun_number_hers_theirs A1:-2.08/+2.04 drop0 A2:-2.12/+2.05 drop0 C:-1.43/+0.93 drop2
  possessive_pronoun_person_ours_theirs A1:-1.73/+1.72 drop0 A2:-1.92/+1.94 drop0 C:-1.43/+0.93 drop2
  number_anaphor_it_them A1:-4.06/+4.13 drop0 A2:-3.76/+3.62 drop0 C:-1.43/+0.93 drop2
  All seven A1 cells are >= 0.89/0.82 (five are >= 1.7/1.7) with zero dropped rows; every A2 cell >= 0.54/0.54. Repairs (one each, disclosed
  in the module docstrings): agreement_do_does A2 (`Whatever they near the pilot, of course,` had no verb, -0.15/+0.07 drop3 -> `The pilot asked
  what they, of course,` -2.96/+2.78); number_anaphor_other_others A2 (`and not the` -0.41/+0.40 -> `chose this, of course, rather than the`
  -0.57/+0.54). DROPPED after its one repair: wh_manner_how_why (A1 -0.67/+0.46 -> `and that is` -0.87/+0.82, but A2 fell to -0.49/+0.27
  drop2; the how/why readout stays open for a future spec with a stronger A2 frame). Batch size SEVEN; bars scaled from the eight-cell bars
  (4/6/7/5/5 of 8 -> 4/5/6/4/4 of 7).
Protocol identical to v197/v201/v204/.../v241 (greedy pool 40, target 0.88, min_gain 0.005, max 30; row 2 = MIN over the two held directions
>= 0.80; row 3 dim A1 CE LB975 > 0 and >= 0.10; row 4 constrained rank-1 own-C UB975 <= 0.01 on 16 held C rows; row 5 = A2-own / A2 full-rank
ceiling in [0.5, 1.4], LB975 > 0). finiteness_selection is re-run as the instrument. The receipt carries cdas extraction_held per behaviour;
the proposed row-4 extraction floor (>= 0.80, board proposal (vi)) is read off it and a pass below 0.80 is booked as HOLLOW (counted by
protocol, not as a circuit); it is NOT a bar here (registered semantics change only through the board).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 4 of the 7 new behaviours pass all four rows (pre-screened v241: 4/6; unscreened grammatical batches 2-7 of 8). prior 55%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 5 of 7.              prior 60%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 6 of 7 (v241 6/6 with all cells >= 1.6; two cells here are 0.9). prior 55%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 4 of 7 (v233 5/8, v235 5/8, v237 5/8, v239 4/8, v241 5/6). prior 60%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 4 of 7 (v237 7/8, v239 6/8, v241 6/6). prior 65%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02 (same seed,
                           split and code path: a determinism check, NOT an independent run outcome, disclosed as such).
Accounting: four-row passes here join the one-protocol standing. Every cell has counted family members (number x5, person x2) and counts only
after a separability rung under the full family control set (v236, after v234 lands).
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~5 min.
A1/A2 preps use g.prepare(valid_only=True); `n_dropped` is in every receipt.
Smoke: V243_SMOKE=<out.json> (CPU, V243_SMOKE_ROWS=4 per split, V243_SMOKE_NAMES=number_anaphor_it_them; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8u_v243_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"agreement_do_does": "agreement_do_does", "reflexive_number_itself_themselves": "reflexive_number_itself_themselves",
         "possessive_person_our_your": "possessive_person_our_your", "number_anaphor_other_others": "number_anaphor_other_others",
         "possessive_pronoun_number_hers_theirs": "possessive_pronoun_number_hers_theirs",
         "possessive_pronoun_person_ours_theirs": "possessive_pronoun_person_ours_theirs", "number_anaphor_it_them": "number_anaphor_it_them",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 4, "k_row2": 5, "k_row3": 6, "k_row4": 4, "k_row5": 4, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec8u_v243", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V243_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V243_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V243_SMOKE_NAMES", "number_anaphor_it_them").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec8u_v243", "candidate_id": "corpus.unit_tier3_batch_amended_spec8u_v243", "bars": BARS,
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
