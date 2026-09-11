#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v611: row 4 tests the control it was FITTED to spare. Measure the same quantity on a held-out control.

THE STRUCTURAL PROBLEM, read out of the code rather than assumed. Row 4 is
`arms["cdas"]["C"]["ce_ub975"] <= 0.01`. The cdas direction qc is produced by
`fit_block_subspace_constrained(..., controls=(P["C_fit"],), control_weight=LAM)` with LAM = 30 -- that is, the
direction is explicitly OPTIMISED to leave the canonical control undisturbed, and row 4 then checks whether it
succeeded on that same control. So row 4 currently measures how well the optimiser met the constraint it was given,
not whether the direction is specific in general. On top of that, all 88 spec-authored cells use ONE control
(`canonical_same_answer_nocturnal_completion`, answers " night"), so every selectivity verdict in the corpus rests
on that single behaviour.
WHY IT MATTERS NOW. Today's rungs made row 4 the general gate on countable behaviours -- 2/6 in noun_preposition,
3/8 across eight non-prepositional constructions, 1/5 under direction-matched fits -- while row 2 turned out to be
prepositional. If the thing gating the corpus is partly an artifact of testing the fit's own constraint, that has to
be known before more behaviours are counted on it.
WHAT THIS RUNG DOES. The SAME cdas direction, fitted exactly as now against the canonical control, is additionally
evaluated on a SECOND same-answer control that appears nowhere in the objective -- control v3, a temporal
subordinate frame ending "for the rest of the" -> " week", holding constant every property that makes v2 usable
(same-answer design, preferred completion, vocabulary disjoint from all targets, both sides ending on the same final
token) and varying only frame and completion. That is the first HELD-OUT selectivity measurement in this protocol.
The rows are built by `circuit_fast_screen_control_v3_rows`, which reproduces each cell's own C panel byte-identically
when run with the canonical control -- verified on four cells before this rung was written, per the standing lesson
that a module rebuilding an authority must reproduce the old digest through the new path.
FIVE CELLS, ALL ROW-4 PASSERS: the four amid behaviours (v575/v581, separable in v583) and
determiner_number_crates (v597, separable in v609). These are every countable candidate I hold. lexical_number was
WITHDRAWN before this rung -- it answers ' were'/' was', a readout in the lane I am told to avoid, which my
task-id-based batch filter missed; numeral_dual_both_all was dropped because v609 showed it fuses with
number_readout. So this rung tests exactly what I would otherwise put forward, and nothing else.
NOTHING IS COUNTED and no bar changes. The canonical control stays exactly as registered; v3 is measured alongside.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here). ALL FIVE CELLS HERE PASSED
ROW 4 under the canonical control, so unlike v603 this rung has no failers and the bars are one-sided ON PURPOSE:
  pred_a_v3_control_capable the held-out control clears its own capability -- at most max_dropped = 4 of its 16 held
                        rows dropped. Checked first; nothing else is readable without it.             prior 85%
  pred_b_verdicts_agree at least k_agree = 4 of the five keep the SAME row-4 verdict under the held-out control.
                        Since all five passed under the canonical one, this is the same statement as c and is
                        reported as one finding rather than two independent ones.                    prior 40%
  pred_c_passers_still_pass at least k_pass_agree = 4 of the five still pass row 4 on a control that was never in
                        the fit objective. THIS IS THE RUNG. v603 measured three of five passers flipping on
                        exactly this test, so I expect this to fail, and I am running it because the five are the
                        candidates I would otherwise put forward.                                    prior 40%
  pred_d_failers_still_fail vacuous here and registered as such: k_fail_agree = 0 because there are NO row-4 failers
                        in this set. It carries no information and I will not read it as evidence.   prior 100%
  pred_e_reproduces_v2_verdicts every cell reproduces the row-4 PASS its parent receipt recorded, so the comparison
                        is against booked numbers. In v603 this predicate caught a wrong entry in my own table.
                                                                                                      prior 90%
HOW IT READS. c TRUE: the five candidates' selectivity survives a control the fit never saw, their row-4 passes mean
what the protocol implies, and they are the strongest inventory this corpus has. c FALSE: their row 4 is
control-specific like the v603 sample, and the honest report is that I have five candidates whose LOCALISATION is
solid -- extraction, removal and the second construction all hold -- but whose SELECTIVITY rests on one control.
That would not retract a receipt; it would change what I put forward and how. Either way the result is reported as
one finding from b and c together, since with no failers in the set they are the same measurement.
a FALSE: the held-out control is unusable on these cells and the rung is void, not negative.
SCOPE. Ten cells, one additional control, rank 1 as registered; no bar change, no counting, and no claim that v3 is
a better control than v2 -- only that it is a different one the fit never saw.
Smoke: V611_SMOKE=<out.json> (CPU, V611_SMOKE_ROWS=4, V611_SMOKE_NAMES=<cell>).
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
OUT = ROOT / "circuits/followups/unit_candidate_heldout_control_v611_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_preposition_at_amid": "verb_preposition_at_amid",
         "verb_preposition_of_amid": "verb_preposition_of_amid",
         "verb_preposition_on_amid": "verb_preposition_on_amid",
         "verb_preposition_to_amid": "verb_preposition_to_amid",
         "determiner_number_crates": "determiner_number_crates"}
# every one of these cleared row 4 under the canonical control in its parent receipt
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(NAMES)
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_agree": 4, "k_pass_agree": 4, "k_fail_agree": 0, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_candidate_heldout_control_v611", "behaviours": 9, "constructions": 5,
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
    e = ok and len(new) == len(NEW) and all(new[n]["row4_v2"] == V2_ROW4[n] for n in new)
    return {"pred_a_v3_control_capable": bool(a), "pred_b_verdicts_agree": bool(b),
            "pred_c_passers_still_pass": bool(c), "pred_d_failers_still_fail": bool(d),
            "pred_e_reproduces_v2_verdicts": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V611_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V611_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V611_SMOKE_NAMES", "verb_preposition_against_for,finiteness_selection").split(",")]
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
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C_fit"],), control_weight=LAM, mu=mu1)
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
            row4_v2 = arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX
            row4_v3 = arms["cdas"]["C3"]["ce_ub975"] <= C_UB_MAX   # SAME bar, control NOT in the fit objective
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
    result = {"predictions": predictions, "schema": "unit_candidate_heldout_control_v611", "candidate_id": "corpus.unit_candidate_heldout_control_v611", "bars": BARS,
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
