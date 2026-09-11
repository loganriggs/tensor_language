#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v595: work the AUTHORED BACKLOG instead of authoring more cells.

WHY THIS RUNG EXISTS. The 03:32 tick reported the number I had been avoiding: circuit_latency shows ZERO terminals
in the last hour and it is right, because "terminal" moves only when a NEW COUNTED CIRCUIT lands, and every rung this
session has been a DAS receipt or a correction. The corpus count has sat at 139 all session while the
four-hypothesis tally went from 16 to 21 -- high rung throughput, zero circuit throughput.
WHERE THE INVENTORY ACTUALLY IS. Measured from ops/ this hour: 420 spec-authored cells carrying 286 distinct
cue -> token mappings, of which 245 appear in some separability FAMILIES list and 175 do not. Excluding my own probe
variants (coord_, cat_, sub_, rs_, the fX_ shape batches, lenmatched, disjoint, open_, attractor), sixteen
verb_preposition cells with plain distinct mappings have never been through a battery OR a separability pass. They
were authored, screened and then left. Eight of them are lifted here; the rest follow if this works.
WHY NOT MORE FRAMES. v573 settled that an hour ago: a restructured frame FUSES with its original (reach 0.95 to 1.02
in three of four directions), so authoring frame variants yields duplicates. The distinctness axis is the mapping,
and the mappings are already on disk.
WHAT THIS RUNG DOES AND DOES NOT DO. It produces units and the tier rows for eight behaviours. It does NOT count
them: the protocol requires a within-family separability pass against the counted verb_preposition set before the
corpus number moves, and that is the rung after this one. The count stays 139 whatever this returns.
The instrument member finiteness_selection rides along unchanged so a drift in the batteries is visible.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here). THESE BARS TEST THE INVERSION
v591 FOUND, IN THE DIRECTION IT WAS FOUND, WITH ROOM TO FAIL. v591 lifted eight cells from eight constructions and
row 2 passed SEVEN of eight -- against nine of 33 in the preposition and particle families -- while row 4 passed only
three of eight. That retired my claim that row 2 is the general gate. A single batch of eight is a thin basis for an
inversion, so this rung re-tests both halves, and four of its eight cells are SIBLINGS of v591's (benefactive, case,
comparative, countability), which gives the within-construction replication v591 could not do with one cell each:
  pred_a_four_rows      at least k_four = 1 of the eight clears ALL FOUR tier rows. v591 gave zero.  prior 40%
  pred_b_row2_mostly_free row 2 passes AT LEAST k_row2_min = 5 of the eight, replicating v591's seven of eight. It
                        FAILS if four or fewer pass, which would mean v591's row-2 result was a one-batch accident
                        and row 2 may be general after all.                                           prior 75%
  pred_c_row3           row 3 passes at least k_row3 = 4 of the eight, against five of eight in v591. prior 70%
  pred_d_row4_binds     row 4 passes AT MOST row4_max = 4 of the eight, replicating v591's three of eight and the
                        2 of 6 in noun_preposition. It FAILS if five or more pass, which would mean row 4 is not
                        the general constraint I have just called it.                                 prior 70%
  pred_e_row5_amended   at least k_row5 = 4 of the eight clear row 5.                                 prior 70%
  pred_f_instrument     finiteness_selection reproduces its parent receipt within instr_tol = 0.02.   prior 90%
HOW IT READS. b and d both holding: the inversion replicates on a second batch, row 2 is confirmed prepositional and
row 4 confirmed general, and the corpus statement that localisation is cheap while selectivity is scarce rests on 16
non-prepositional cells rather than eight. b FAILING: v591's row-2 result does not replicate and I should retract
the inversion rather than the original claim -- the outcome I most want this rung to be able to produce. d FAILING:
row 4 is not the general constraint, and the 2-of-6 in noun_preposition plus 3-of-8 in v591 were coincidence. The
four sibling pairs also give the first within-construction agreement check: if a construction's two cells disagree on
row 4, then a per-construction selectivity rate needs more than two cells and I will say so before quoting one.
SCOPE. Eight behaviours: four siblings of v591's cells and four new constructions. Units and tier rows only, no counting. Two cells in a construction is an AGREEMENT CHECK, not a rate -- a per-construction rate still needs five cells each, as the census rungs used.
Smoke: V595_SMOKE=<out.json> (CPU, V595_SMOKE_ROWS=4).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_crossconstruction2_v595_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"benefactive_preposition_send": "benefactive_preposition_send",
         "case_we_us": "case_we_us",
         "comparative_equative_as_than": "comparative_equative_as_than",
         "countability_much_many": "countability_much_many",
         "another_other_number": "another_other_number",
         "binding": "binding",
         "container_surface_into_onto": "container_surface_into_onto",
         "coordination_agreement": "coordination_agreement",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 1, "k_row2_min": 5, "row4_max": 4, "k_row3": 4, "k_row5": 4, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_crossconstruction2_v595", "behaviours": 9, "constructions": 5,
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
    b = ok and len(new) == len(NEW) and cnt("row2") >= B["k_row2_min"]
    c = ok and cnt("row3") >= B["k_row3"]
    d = ok and len(new) == len(NEW) and cnt("row4") <= B["row4_max"]
    e = ok and cnt("row5") >= B["k_row5"]
    f = ok and all(n in good and good[n].get("instr_n_units_match") and good[n].get("instr_dirB_abs_diff") is not None
                   and good[n]["instr_dirB_abs_diff"] <= B["instr_tol"] for n in INSTR)
    return {"pred_a_four_rows": bool(a), "pred_b_row2_mostly_free": bool(b), "pred_c_row3": bool(c),
            "pred_d_row4_binds": bool(d), "pred_e_row5_amended": bool(e), "pred_f_instrument": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V595_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V595_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V595_SMOKE_NAMES", "case_we_us,finiteness_selection").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_crossconstruction2_v595", "candidate_id": "corpus.unit_tier3_batch_crossconstruction2_v595", "bars": BARS,
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
