#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v627: the standing DAS targets, runnable for the first time, with selectivity measured on TWO controls.

WHY THIS HAS NOT RUN BEFORE. correlative_pair.both_vs_neither and possessive_number.adjacent_antecedent have been
named DAS targets in the standing direction for days. They predate BehaviourSpec and expose no SPEC attribute, so
the held-out-control row builder could not reach them, and after v611 a DAS rung measuring only the canonical
control would have produced exactly the kind of selectivity claim that four batches have now shown does not
transfer. The builder was extended to non-spec cells at 23:46 and its known-good check passes on both of these
cells -- each reproduces its own 32 C rows byte-identically under the canonical control -- so this is the first time
the direction can be run honestly rather than run and caveated.
WHAT IS FIXED IN ADVANCE. RANK 1, registered here, and a null is not permission to raise it. The cdas direction is
fitted against the WEEKLY control, because across 39 cells in four batches that arrangement gives selectivity which
transfers to an unseen control 64 percent of the time against 32 percent for the canonical arrangement -- an effect
present in every batch, with batch rates ranging 30 to 80 percent, so the direction is solid and the point estimate
is not tight. The CANONICAL control is then the held-out test. Both numbers are reported.
WHAT A PASS MEANS AND DOES NOT. Passing here means the rank-1 subspace carries the variable and spares TWO unrelated
behaviours, one of which the fit never saw. It does NOT make these counted circuits: the registered protocol fits
against the canonical control, so this is a DAS result under a fit the protocol does not currently use, and that
remains Codex's to decide. possessive_adjacent has five matched siblings and the standing protocol asks for the
across-sibling test; this rung does NOT do that, and the sibling pass is the next rung rather than something I am
quietly folding in.
TWO CELLS ONLY, so every bar below is small and I am not going to dress a two-cell result as a rate.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here; names and code rewritten for
this arrangement rather than inherited, after two rungs shipped a pred_c whose code and sentence disagreed):
  pred_a_v3_control_capable the weekly control is capable on these cells: at most max_dropped = 4 of its 16 held
                        rows dropped. Checked first, and on cells whose C panel had to be rebuilt it matters more
                        than usual.                                                                   prior 85%
  pred_b_verdicts_agree BOTH cells get the same row-4 verdict on both controls, k_agree = 2.          prior 60%
  pred_c_both_controls_pass at least k_both = 1 of the two passes row 4 on BOTH controls. This is the rung.
                                                                                                      prior 55%
  pred_d_canonical_passes at least k_canon = 1 passes on the CANONICAL control, which is outside the objective.
                                                                                                      prior 60%
  pred_e_weekly_passes  at least k_weekly = 1 passes on the WEEKLY control, which is inside it.        prior 75%
HOW IT READS. c TRUE: at least one standing target has a rank-1 subspace specific against two unrelated behaviours,
which is the strongest DAS evidence this corpus has produced, and the next step is the five matched siblings for
possessive_adjacent. c FALSE with e TRUE: the subspace spares the control it was fitted against and not the other,
which for these two cells would mean the same thing v611 found for the amid candidates. c and e both FALSE: the
sites do not support a specific rank-1 subspace at all, and a null here is NOT permission to raise the rank -- the
next move would be to report the null and leave the rank where it was registered.
SCOPE. Two cells, rank 1, one fit arrangement, two controls. No sibling test, no counting, no bar change.
Smoke: V627_SMOKE=<out.json> (CPU, V627_SMOKE_ROWS=4, V627_SMOKE_NAMES=<cell>).
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
OUT = ROOT / "circuits/followups/unit_das_standing_targets_v627_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"correlative_pair": "correlative_pair", "possessive_adjacent": "possessive_adjacent"}
# neither has a tier-3 row-4 verdict on record: both predate the lift batches, so there is no prior to reproduce
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(NAMES)
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_agree": 2, "k_both": 1, "k_canon": 1, "k_weekly": 1, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_das_standing_targets_v627", "behaviours": 9, "constructions": 5,
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
    c = ok and sum(1 for n in new if new[n]["row4_v2"] and new[n]["row4_v3"]) >= B["k_both"]
    d = ok and sum(1 for n in new if new[n]["row4_v2"]) >= B["k_canon"]
    e = ok and len(new) == len(NEW) and sum(1 for n in new if new[n]["row4_v3"]) >= B["k_weekly"]
    return {"pred_a_v3_control_capable": bool(a), "pred_b_verdicts_agree": bool(b),
            "pred_c_both_controls_pass": bool(c), "pred_d_canonical_passes": bool(d),
            "pred_e_weekly_passes": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V627_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V627_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V627_SMOKE_NAMES", "verb_preposition_against_for,finiteness_selection").split(",")]
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
                 "C3_held": g.prepare(backend, cut(held_half(r3.rows_any(m, control_v3)))),
                 "C3_fit": g.prepare(backend, cut(fit_half(r3.rows_any(m, control_v3))))}
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
    result = {"predictions": predictions, "schema": "unit_das_standing_targets_v627", "candidate_id": "corpus.unit_das_standing_targets_v627", "bars": BARS,
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
