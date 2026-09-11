#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v593: a rescued direction is only a circuit if it carries rows 3, 4 and 5 too.

WHAT v587 SHOWED AND WHAT IT DID NOT. v587 found that the weak interchange direction IS carried: fitted on its own
direction it reached 0.80 or better in five of six cells, beat an equal-sized random half of the joint fit rows in
all six by 0.113 to 0.240, and used a DIFFERENT head set each way (Jaccard 0.056 to 0.571). So a row-2 failure is
the joint fit satisfying the strong direction, not the model refusing to carry the variable one way. But v587
measured EXTRACTION ONLY. Extraction says the direction-matched set reproduces the interchange; it says nothing
about whether removing that set damages the task, whether the set spares an unrelated behaviour, or whether it
carries the second construction. Those are rows 3, 4 and 5, and without them a rescued direction is a recovered
number rather than a circuit.
WHAT THIS RUNG DOES. The five cells v587 rescued are re-fitted on the weak direction and EVERY panel is restricted
to that same direction -- fit, held-out, A2, P and C all describe one direction, so the rows are not quietly mixing
a direction-matched fit with both-direction evidence. row2 is still computed and written to the receipt but is NOT
read by any predicate here, because both-direction symmetry is exactly what this rung is not claiming.
WHAT A PASS WOULD AND WOULD NOT MEAN. It would mean the 19 row-2 failures in the lifted backlog are a real source of
DIRECTION-SPECIFIC circuits, each a narrower object than a row-2 pass: a site carrying the variable one way, with its
own head set. It would NOT make them equivalent to symmetric circuits, and I will not book them as such. It also
halves the rows behind every row-3/4/5 number relative to a joint fit, so a failure could be sample size -- stated
here, and the reason pred_e checks the extraction reproduces v587 exactly before any row is believed.
NOTHING IS COUNTED. The count stays 139.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_rows345_hold at least k_rows345 = 3 of the five cells pass rows 3, 4 AND 5 together under the
                      direction-matched fit.                                                          prior 40%
  pred_b_row3         row 3 passes for at least k_row3 = 4 of the five -- removal on the direction-matched held set
                      damages the task with a lower bound above zero.                                 prior 70%
  pred_c_row4         row 4 passes for at least k_row4 = 3 of the five -- the C upper bound stays at or below 0.01
                      on the direction-matched control.                                               prior 55%
  pred_d_row5         row 5 passes for at least k_row5 = 3 of the five under the amendment.           prior 55%
  pred_e_reproduces_v587 every cell's held-out extraction reproduces its v587 weak-direction number within
                      tol = 0.05. Deterministic fits, so a drift means the panels are not the ones v587 measured and
                      no row here is interpretable.                                                   prior 85%
HOW IT READS. a true: direction-specific circuits are real, the largest untapped source of countable behaviours in
the backlog is the row-2 failures, and the protocol needs a direction-specific category rather than a pass/fail on
row 2. a FALSE with b true: the direction carries the variable and damages the task, but fails the control or the
second construction -- localisation without selectivity, which is a weaker and honest result. b FALSE: the rescued
extraction does not survive removal, and v587's numbers describe a subspace that reproduces the interchange without
being necessary for it -- the outcome that would most change how I read v587. e FALSE: panels drifted, nothing here
is comparable, and the rung is void rather than negative.
SCOPE. Five cells, one direction each, rank-free head sets; no counting, no symmetric claim.
Smoke: V593_SMOKE=<out.json> (CPU, V593_SMOKE_ROWS=4, V593_SMOKE_NAMES=<cell>).
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
OUT = ROOT / "circuits/followups/unit_direction_rows_v593_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"adjective_preposition_in_with": "adjective_preposition_in_with",
         "adjective_preposition_under_for": "adjective_preposition_under_for",
         "adjective_preposition_with_for": "adjective_preposition_with_for",
         "verb_preposition_from_with": "verb_preposition_from_with",
         "verb_preposition_in_amid": "verb_preposition_in_amid"}
# which held direction was WEAK under the joint fit (v587 parent numbers); the fit is matched to it
WEAK_IS_A = {"adjective_preposition_in_with": True, "adjective_preposition_under_for": False,
             "adjective_preposition_with_for": False, "verb_preposition_from_with": True,
             "verb_preposition_in_amid": False}
V587_WEAK = {"adjective_preposition_in_with": 0.850, "adjective_preposition_under_for": 0.908,
             "adjective_preposition_with_for": 0.875, "verb_preposition_from_with": 0.912,
             "verb_preposition_in_amid": 0.906}
NEW = tuple(NAMES)
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_rows345": 3, "k_row3": 4, "k_row4": 3, "k_row5": 3, "tol": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_direction_rows_v593", "behaviours": 9, "constructions": 5,
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
    r345 = sum(1 for n in new if all(new[n]["rows"][r] for r in ("row3", "row4", "row5")))
    a = ok and r345 >= B["k_rows345"]
    b = ok and cnt("row3") >= B["k_row3"]
    c = ok and cnt("row4") >= B["k_row4"]
    d = ok and cnt("row5") >= B["k_row5"]
    e = ok and len(new) == len(NEW) and all(
        abs(new[n]["extraction_held"] - V587_WEAK[n]) <= B["tol"] for n in new)
    return {"pred_a_rows345_hold": bool(a), "pred_b_row3": bool(b), "pred_c_row4": bool(c),
            "pred_d_row5": bool(d), "pred_e_reproduces_v587": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V593_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V593_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V593_SMOKE_NAMES", "verb_preposition_against_for,finiteness_selection").split(",")]
    parent = json.loads(V196.read_text())["behaviours"]
    # DIRECTION-MATCHED slices: v575 fits on rows[0::4]+rows[1::4] and holds rows[2::4]+rows[3::4].
    # Here every panel is restricted to the SINGLE direction whose extraction was weak under the joint fit.
    fit_half = lambda rows: rows[0::4] + rows[1::4]      # kept for the joint baseline only
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
            # DIRECTION-MATCHED panels: every family is restricted to the single direction that was WEAK
            # under the joint fit, so the fit, the held-out set, A2, P and C all describe the same direction.
            wa = WEAK_IS_A[n]
            fsl = (lambda r: r[0::4]) if wa else (lambda r: r[1::4])
            hsl = (lambda r: r[2::4]) if wa else (lambda r: r[3::4])
            P = {"fit": g.prepare(backend, cut(fsl(rows["A1"])), **V), "held": g.prepare(backend, cut(hsl(rows["A1"])), **V),
                 "held_dirA": g.prepare(backend, cut(rows["A1"][2::4]), **V), "held_dirB": g.prepare(backend, cut(rows["A1"][3::4]), **V),
                 "A2_fit": g.prepare(backend, cut(fsl(rows["A2"])), **V), "A2_held": g.prepare(backend, cut(hsl(rows["A2"])), **V),
                 "P_held": g.prepare(backend, cut(hsl(rows["P"]))), "C_fit": g.prepare(backend, cut(fsl(rows["C"]))), "C_held": g.prepare(backend, cut(hsl(rows["C"])))}
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
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,   # recorded, NOT read: both-direction symmetry is not what this rung claims
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
    result = {"predictions": predictions, "schema": "unit_direction_rows_v593", "candidate_id": "corpus.unit_direction_rows_v593", "bars": BARS,
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
