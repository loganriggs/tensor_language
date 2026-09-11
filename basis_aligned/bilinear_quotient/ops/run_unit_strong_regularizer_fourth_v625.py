#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v625: swap which control enters the objective. Is row-4 selectivity the FIT, or the control?

WHAT v603 AND v611 ESTABLISHED. Row 4 is arms["cdas"]["C"]["ce_ub975"] <= 0.01, and the cdas direction is fitted
with controls=(C_fit,) at control_weight 30 -- optimised to spare the canonical control, then tested on it. Measured
on a second same-answer control absent from the objective, verdicts flipped five of ten (v603), and ALL FIVE of my
countable candidates failed (v611), at ratios of 2.8 to 9.4. Across the 15 cells now measured both ways the median
held-out-to-canonical ratio is 4.30.
WHAT IS NOT YET EXCLUDED, and why this rung exists. Two readings survive that evidence. (a) The direction spares the
canonical control BECAUSE it is in the objective, so row-4 selectivity is largely an artifact of the fit. (b) The
held-out control is structurally harder for these behaviours regardless of the objective. The 15-cell evidence
argues against (b) -- the held-out control PASSES row 4 on five of fifteen and returns a NEGATIVE upper bound on
four, meaning the direction sometimes improves it, which a uniformly harder control would not do -- but that is an
argument, not a test.
WHY THIS RUNG. v613 answered the swap question on ten cells and the answer was strong: with the held-out control in
the objective, verdicts agreed 10 of 10 and BOTH controls passed 8 of 10, against 11 of 25 agreement and 8 of 25
out-of-objective passes for the canonical arrangement. But the two sides of that comparison are unequal -- 25 cells
one way, 10 the other -- and the newer, more surprising half is the thin one. THE REASON THIS BATCH EXISTS. Both-pass in the strong arm has fallen monotonically across three batches: 8 of 10 in
v613, 6 of 10 in v619, 3 of 10 in v623. Pooled over 30 cells the arm is transfer 67 percent, both-pass 57 percent,
agreement 73 percent -- against 32 / 20 / 44 for the canonical arm, so the DIRECTION holds in every batch while the
magnitude keeps shrinking. Two readings: the decline is batch noise and regression from a lucky first batch, or it
is systematic and the later cells differ. A fourth batch discriminates: continued decline toward 3 or below argues
systematic, a bounce back toward 6 or above argues noise. Either way the pooled estimate gets an extra 9 cells.
TWO DEPARTURES FROM THE EARLIER BATCHES, BOTH REGISTERED. First, this batch is NINE cells and unbalanced -- four
canonical-row4 passers and five failers, where v613, v619 and v623 were five and five -- because only eleven
eligible cells remain and four of them are passers. An unbalanced batch cannot be compared cell-for-cell with the
others on the pass side, so I will report its passer and failer rates separately as well as pooled. Second, I
EXCLUDED person_agreement_be, whose readout is ' am' / ' are'. Neither token is literally in the avoided set, but
they are copula forms and sit next to the lane I am told to leave alone; after finding seven cells today whose
readouts were in that lane because I filtered on task names, the cautious call is to drop it and say so rather than
include it quietly.
This repeats the v613/v619/v623 arrangement on, five that passed row 4 under the canonical fit and five that failed, none appearing in v603, v611,
v613 or v615, all spec-authored with readouts outside the avoided lane. It is INDEPENDENT of the v617 refit running
alongside: v617 asks whether my five candidates recover, this asks whether the strong-regularizer effect holds on
cells I have not looked at.
THE TEST. Swap them. Fit the cdas direction against the HELD-OUT control (controls=(C3_fit,), same weight of 30) and
test on the CANONICAL one. Everything else is identical to v603, on the SAME TEN CELLS, so the numbers are directly
comparable with v603's. If reading (a) is right the pattern MIRRORS: the canonical control, now outside the
objective, starts failing cells it previously passed, and the held-out control, now inside, starts passing them.
If reading (b) is right the canonical control keeps passing whatever it passed before, because it was never the
objective that made it easy.
THIS CHANGES NO REGISTERED BAR AND NO CANONICAL SEMANTICS. It is one rung that fits against a different control to
find out what the fit is doing; the protocol's own control keeps its role everywhere else.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_v3_control_capable the control now IN the objective is capable: at most max_dropped = 4 of its 16 held rows
                        dropped. Same first check as before.                                          prior 90%
  pred_b_verdicts_agree at least k_agree = 8 of ten cells give the same canonical-control verdict as v603 recorded.
                        Under reading (a) this FAILS, because the canonical control has just lost the 30-weighted
                        protection that made it easy.                                                 prior 30%
  pred_c_passers_still_pass at least k_pass_agree = 4 of the five cells v603 recorded as canonical-control passers
                        still pass now that the canonical control is outside the objective.           prior 30%
  pred_d_failers_still_fail at least k_fail_agree = 4 of the five v603 failers still fail.            prior 55%
  pred_e_pattern_mirrors at least k_mirror = 5 cells now PASS on the control that is inside the objective while
                        FAILING on the one outside it -- the mirror image of what v603 found. This is the positive
                        form of reading (a) and the predicate I will report as the result.            prior 60%
HOW IT READS. e TRUE with b and c false: the pattern mirrors, row-4 selectivity tracks WHICH CONTROL IS IN THE
OBJECTIVE rather than the site, and the protocol's selectivity row measures the optimiser meeting its constraint.
That is a statement about the instrument, not about any receipt's arithmetic, and it would mean a four-row pass
licenses less than it appears to. e FALSE with b and c holding: the canonical control passes regardless of the
objective, reading (b) is right, the held-out control is simply harder for these behaviours, and v611's result says
my candidates are weak rather than that row 4 is. Both outcomes are worth the GPU; I expect the first and have said
so in the priors.
SCOPE. Ten cells, one swapped objective, rank 1 as registered. No bar change, no counting.
Smoke: V625_SMOKE=<out.json> (CPU, V625_SMOKE_ROWS=4, V625_SMOKE_NAMES=<cell>).
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
import circuit_fast_screen_control_v3_rows as r3
import circuit_fast_screen_canonical_control_v3 as control_v3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_strong_regularizer_fourth_v625_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"reflexive_object_control": "reflexive_object_control",
         "reflexive_object_control_plural": "reflexive_object_control_plural",
         "relative_animacy": "relative_animacy",
         "result_state_empty_full": "result_state_empty_full",
         "predicate_category_slow_slowly": "predicate_category_slow_slowly",
         "raising_extraposition_happened": "raising_extraposition_happened",
         "relative_locative_where_which": "relative_locative_where_which",
         "result_state_open_closed": "result_state_open_closed",
         "numeral_number": "numeral_number"}
# row 4 verdicts under the CANONICAL control when FITTED against it, from the v591-v607 lift receipts
V2_ROW4 = {
           "reflexive_object_control": True,
           "reflexive_object_control_plural": True,
           "relative_animacy": True,
           "result_state_empty_full": True,
           "predicate_category_slow_slowly": False,
           "raising_extraposition_happened": False,
           "relative_locative_where_which": False,
           "result_state_open_closed": False,
           "numeral_number": False,
           }
NEW = tuple(NAMES)
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_agree": 8, "k_pass_agree": 4, "k_fail_agree": 4, "max_dropped": 4, "k_mirror": 5}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_strong_regularizer_fourth_v625", "behaviours": 9, "constructions": 5,
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
    drop = max((new[n]["v3_dropped"] for n in new), default=99)
    a = ok and len(new) == len(NEW) and drop <= B["max_dropped"]
    agree = sum(1 for n in new if new[n]["row4_v2"] == new[n]["row4_v3"])
    b = ok and agree >= B["k_agree"]
    passers = [n for n in new if V2_ROW4[n]]
    failers = [n for n in new if not V2_ROW4[n]]
    c = ok and sum(1 for n in passers if new[n]["row4_v3"]) >= B["k_pass_agree"]
    d = ok and sum(1 for n in failers if not new[n]["row4_v3"]) >= B["k_fail_agree"]
    mirrored = sum(1 for n in new if new[n]["row4_v3"] and not new[n]["row4_v2"])
    e = ok and len(new) == len(NEW) and mirrored >= B["k_mirror"]
    return {"pred_a_v3_control_capable": bool(a), "pred_b_verdicts_agree": bool(b),
            "pred_c_passers_still_pass": bool(c), "pred_d_failers_still_fail": bool(d),
            "pred_e_pattern_mirrors": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V625_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V625_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V625_SMOKE_NAMES", "verb_preposition_against_for,finiteness_selection").split(",")]
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
                 "P_held": g.prepare(backend, cut(held_half(rows["P"]))), "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))), "C_held": g.prepare(backend, cut(held_half(rows["C"]))),
                 "C3_held": g.prepare(backend, cut(held_half(r3.rows_for(m, control_v3)))),
                 "C3_fit": g.prepare(backend, cut(fit_half(r3.rows_for(m, control_v3))))}
            singles, ranked, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
            units = list(greedy["chosen"])
            e_fit, e_held, e_a, e_b = ext(P["fit"], units), ext(P["held"], units), ext(P["held_dirA"], units), ext(P["held_dirB"], units)
            mu1 = mu_of(P["fit"], units)
            q1 = g.block_diff_in_means(backend, P["fit"], units)
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C3_fit"],), control_weight=LAM, mu=mu1)
            arms = {}
            for arm, qa in (("dim", q1), ("cdas", qc)):
                arms[arm] = {fam: dmg(P[k], units, qa, mu1) for fam, k in (("A1", "held"), ("A2", "A2_held"), ("P", "P_held"), ("C", "C_held"), ("C3", "C3_held"))}
                arms[arm]["extraction_held"] = round(ext(P["held"], units, q=qa) / e_held, 3) if abs(e_held) > 1e-6 else None
            mu2 = mu_of(P["A2_fit"], units)
            q2 = g.block_diff_in_means(backend, P["A2_fit"], units)
            a2_own = dmg(P["A2_held"], units, q2, mu2)
            a2_full = dmg(P["A2_held"], units, None, mu2)   # q=None: full-rank mean-ablation of the set = the construction's own ceiling
            a1_full = dmg(P["held"], units, None, mu1)
            share = lambda x, y: round(x / y, 4) if y else None
            d = arms["dim"]
            row5_share = share(a2_own["ce_damage"], a2_full["ce_damage"])
            row4_v2 = arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX      # canonical control: now the HELD-OUT one
            row4_v3 = arms["cdas"]["C3"]["ce_ub975"] <= C_UB_MAX   # v3: now the control IN the objective
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,
                       "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                       "row4": arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX,
                       "row5": a2_own["ce_lb975"] > 0 and row5_share is not None and BARS["row5_share_band"][0] <= row5_share <= BARS["row5_share_band"][1],
                       "row4_v3": row4_v3}
            R[n] = {"row4_v2": bool(row4_v2), "row4_v3": bool(row4_v3),
                    "c_ub_v2": arms["cdas"]["C"]["ce_ub975"], "c_ub_v3": arms["cdas"]["C3"]["ce_ub975"],
                    "v3_dropped": P["C3_held"].dropped, "v2_dropped": P["C_held"].dropped,
                    "units": units, "n_units": len(units), "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]},
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
    result = {"predictions": predictions, "schema": "unit_strong_regularizer_fourth_v625", "candidate_id": "corpus.unit_strong_regularizer_fourth_v625", "bars": BARS,
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
