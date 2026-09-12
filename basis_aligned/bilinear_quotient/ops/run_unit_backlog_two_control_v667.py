#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v667: the second control on eleven backlog cells -- the only thing between them and being proposable.

WHERE THE ELEVEN CAME FROM. v665 put thirteen never-counted four-row passers in one 16-member family; eleven
separated, with sibling leaks from 0.003 to 0.038 against a 0.05 bar. Two fell: gender_object_him_her fused with the
counted possessive_gender (mutual 0.095/0.083, the same gender axis under two different stem names), and
numeral_dual_both_all missed the family extraction floor by 0.0046 with a leak of only 0.008 -- a near-miss on a
different criterion, not a duplicate, and it is excluded here pending its own rerun rather than written off.
WHAT IS ACTUALLY MISSING, SWEPT NOT ASSUMED. I searched every receipt that names these eleven. All eleven carry only
cdas + dim -- single-control evidence. Four rows: on record. Separability: on record as of v665. The SECOND
same-answer control is the entire remaining gap, so this rung is that and nothing else. No new cells, no new axes.
WHY THE SECOND CONTROL IS NOT A FORMALITY HERE. These eleven were lifted in old tier3 batteries against the
canonical control alone, and the canonical control is a measured WEAK regularizer: fitting against it gives 32%
transfer against 64% for the weekly one. A cell that passed row 4 on the canonical control has cleared the easier
bar. The honest prior is that some of the eleven fail the weekly control, and pred_d is registered to allow that.
A DEAD ANCHOR I FOUND WHILE DERIVING THIS, AND FIXED. v659 defined a reproduction ANCHOR and a dict of its prior
values, then assigned `anc` and never used it: the anchor was never compared to anything. Every runner derived from
it inherited a reproduction guarantee that did not run. pred_f now compares the anchor's two control upper bounds
against the values v659 actually recorded (0.0075 canonical, 0.0096 weekly) within anchor_tol = 0.01, so the claim
that this code path reproduces the old one is tested rather than asserted.
HOW IT READS. d TRUE at six or more: the backlog yields at least six two-control-selective circuits and the query
route beats authoring by a wide margin. d FALSE: the backlog cells are systematically weaker than cells authored and
tested against both controls from the start, which would mean the old batteries' row 4 passes are not comparable to
recent ones -- a result about the CORPUS, and one I would need to carry back into every count that mixes them.
NOTHING IS COUNTED BY THIS FILE.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_controls_capable  every one of the eleven is measured and the worst v3 control drop is at most
                           max_dropped = 4 rows. Worked example: if ten cells drop 0 rows and aux_copy_so drops 5,
                           the max is 5, 5 <= 4 is false, pred_a is FALSE.                            prior 70%
  pred_b_all_reach         ALL eleven reach held-out extraction ext_min = 0.80. Worked example: ten at 0.88 or above
                           and equative_result at 0.79 makes pred_b FALSE -- a single miss breaks it.  prior 55%
  pred_c_siblings_reached  at least k_reach = 8 of the eleven reach 0.80. Worked example: nine at or above 0.80 and
                           two below, count 9, 9 >= 8 is true, pred_c is TRUE.                        prior 80%
  pred_d_reached_are_two_control_selective  at least k_both = 6 of those that reached 0.80 are selective on BOTH
                           controls, each C upper bound at or below c_ub_max = 0.01. Worked example: nine reach, and
                           of those seven have canonical AND weekly upper bounds at or under 0.01 while two pass only
                           the canonical, the count is 7, 7 >= 6 is true, pred_d is TRUE.             prior 55%
  pred_e_all_measured      all eleven produce rows rather than an error. Worked example: if both_either raises during
                           the v3 rebuild and lands as an error, len(new) is 10 not 11, pred_e is FALSE. prior 85%
  pred_f_anchor_reproduces the anchor possessive_person_our_your reproduces the control upper bounds v659 recorded --
                           canonical 0.0075 and weekly 0.0096 -- within anchor_tol = 0.01. Worked example: this run
                           returning 0.0081 and 0.0103 gives gaps of 0.0006 and 0.0007, both within tolerance, so
                           pred_f is TRUE; a canonical bound of 0.019 is a gap of 0.0115 and makes it FALSE. This is
                           the check v659 declared and never ran.                                     prior 90%
SCOPE. Eleven backlog cells plus one reproduction anchor, two same-answer controls. No counting, no new cells.
Smoke: V667_SMOKE=<out.json> (CPU) -- proves the code path runs; its numbers are not measurements.
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
import receipt_write
import circuit_fast_screen_control_v3_rows as r3
import circuit_fast_screen_canonical_control_v3 as control_v3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_backlog_two_control_v667_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {'animacy_place_anyone_anywhere': 'animacy_place_anyone_anywhere', 'countability_fewer_less': 'countability_fewer_less', 'equative_result': 'equative_result', 'case_he_him': 'case_he_him', 'durativity_until_by': 'durativity_until_by', 'rather_prefer': 'rather_prefer', 'let_want_complement': 'let_want_complement', 'both_either': 'both_either', 'category_selection_because_of': 'category_selection_because_of', 'raising_extraposition_happened': 'raising_extraposition_happened', 'aux_copy_so': 'aux_copy_so', 'possessive_person_our_your': 'possessive_person_our_your'}
ANCHOR = "possessive_person_our_your"
SIBLINGS = ['possessive_person_my_their', 'possessive_person_your_their']
# v637 measured the anchor under this exact arrangement: canonical +0.0075, weekly +0.0096, four rows clear,
# both controls passed. It is one of my three standing proposals and rides along to check reproduction.
V667_ANCHOR = {"canon_ub": 0.0075, "weekly_ub": 0.0096}
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(['animacy_place_anyone_anywhere', 'countability_fewer_less', 'equative_result', 'case_he_him', 'durativity_until_by', 'rather_prefer', 'let_want_complement', 'both_either', 'category_selection_because_of', 'raising_extraposition_happened', 'aux_copy_so'])
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"anchor_tol": 0.01, "ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_reach": 8, "k_both": 6, "tol": 0.05, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_backlog_two_control_v667", "behaviours": 9, "constructions": 5,
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
    anc = good.get(ANCHOR, {})
    f = ok and ANCHOR in good and all(
        anc.get(k) is not None and abs(anc[k] - V667_ANCHOR[j]) <= B["anchor_tol"]
        for k, j in (("c_ub_v2", "canon_ub"), ("c_ub_v3", "weekly_ub")))
    b = ok and all((new[n].get("extraction_held") or 0) >= EXT_MIN for n in new)
    reach = [n for n in new if (new[n].get("extraction_held") or 0) >= EXT_MIN]
    c = ok and len(reach) >= B["k_reach"]
    d = ok and sum(1 for n in reach if new[n]["row4_v2"] and new[n]["row4_v3"]) >= B["k_both"]
    e = ok and len(new) == len(NEW)
    return {"pred_a_controls_capable": bool(a), "pred_b_all_reach": bool(b),
            "pred_c_siblings_reached": bool(c), "pred_d_reached_are_two_control_selective": bool(d),
            "pred_e_all_measured": bool(e), "pred_f_anchor_reproduces": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V667_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V667_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V667_SMOKE_NAMES", "both_either,possessive_person_our_your").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_backlog_two_control_v667", "candidate_id": "corpus.unit_backlog_two_control_v667", "bars": BARS,
              "protocol": {"split": "within-direction: fit rows[0::4]+rows[1::4], held rows[2::4]+rows[3::4]", "pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                           "cdas": {"steps": steps, "lr": LR, "complement_weight": CW, "control_weight": LAM, "controls": "own C FIT rows"},
                           "row5": "A2-own diff-in-means CE / A2 full-rank mean-ablation ceiling, LB975 > 0"},
              "four_row_passes": sorted(n for n, r in R.items() if "error" not in r and all(r["rows"].values())), "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    written, used_fallback = receipt_write.write_receipt(out, result)
    if used_fallback:
        print('RECEIPT NOT AT ITS INTENDED PATH -- do not release until copied back', flush=True)
    print(json.dumps({"predictions": predictions, "rows": {n: r["rows"] for n, r in R.items()}}, indent=2))


if __name__ == "__main__":
    main()
