#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v675: one screen that fits BOTH objectives, and five fresh candidates judged by it.

WHY THIS MODULE EXISTS. v671 found that 12 of 26 counted cells FLIP their row-4 verdict depending on which control
sits in the fit objective, and v673 found the same of two of my own five proposals. So a single-objective screen
produces a verdict that is underdetermined for roughly 40-46% of cells. Until now getting both numbers meant two
separate runs of two different runners, comparable only because the known-good arm reproduced across them. This
runner fits BOTH objectives on the SAME greedy unit set in one run: controls=(C3_fit) gives the CANONICAL bound as
the held-out one, controls=(C_fit) gives the WEEKLY bound as the held-out one, and `both_objectives` is true only
when a cell clears the held-out control either way. That is the standard v671 implies, computed in one place.
HOW THE NEW CODE PATH IS CHECKED, AND IT IS THE STRICTEST CHECK IN THIS SEQUENCE. Lesson 7 says a module that
rebuilds an existing authority must reproduce the OLD digest through the NEW path. This module rebuilds TWO
authorities at once, so pred_a requires all four known-good cells to reproduce BOTH of their recorded numbers --
eight values in total, drawn from two different older single-arm runners (the C3-objective runs and the C-objective
runs), each within anchor_tol = 0.01. If the second arm were silently fitting the same objective as the first, the
weekly column would come back equal to the canonical one and pred_a would fail on every cell. That is the specific
failure this check is built to catch, and it cannot pass by accident.
THE FIVE CANDIDATES. Uncounted four-row passers on stems no counted cell occupies, none of them used in v665:
adj_adv_feel, aux_copy_ellipsis, countability_few_little, definiteness_anaphor_needed, determiner_number_crates.
Filters applied before selection: readouts in the avoided lane dropped (lexical_number_pp " were"/" was",
partitive_agreement and perfect_number " has"/" have", requested_subjunctive " be"/" was"); cells with no readable
vocabulary dropped, since a readout I cannot inspect is one I cannot certify; did_has_negation dropped for using
"has" as a CUE even though its readouts (" lift"/" lifted") are clean; and the whole gender stem excluded after v665
showed gender_object_him_her fusing with the counted possessive_gender across a stem-name boundary.
WHY THE BAR IS LOW AND HONEST. v667 put the single-objective yield on comparable backlog cells at 3 of 11. The
both-objectives standard is strictly harder, and v673 lost 2 of 5 cells that had already cleared one objective.
Two of five is the registered bar. I expect a thin yield; the value of the rung is the reusable screen plus honest
candidates, not a large number.
NOTHING IS COUNTED BY THIS FILE.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_known_good_reproduces_both  ALL FOUR known-good cells reproduce BOTH recorded bounds -- canonical from the
                       C3-objective runs and weekly from the C-objective runs -- each within anchor_tol = 0.01.
                       Worked example: correlative_or_and must return canonical near -0.0101 AND weekly near 0.0265;
                       if it returns -0.0101 and -0.0101 the two arms are fitting the same objective, the gap on the
                       weekly value is 0.0366, and pred_a is FALSE.                                   prior 75%
  pred_b_all_reach     ALL FIVE candidates reach held-out extraction ext_min = 0.80. Worked example: four at 0.88 or
                       above and adj_adv_feel at 0.76 makes pred_b FALSE.                             prior 70%
  pred_c_siblings_reached  at least k_reach = 4 of the five reach 0.80. Worked example: four reach and one does not,
                       count 4, 4 >= 4 is true, pred_c is TRUE.                                       prior 85%
  pred_d_pass_both_objectives  at least k_both = 2 of the five clear the HELD-OUT control under BOTH objectives.
                       Worked example: if countability_few_little and determiner_number_crates each have canonical
                       and weekly held-out bounds at or under 0.01 while the other three fail one side, the count is
                       2, 2 >= 2 is true, pred_d is TRUE.                                             prior 50%
  pred_e_all_measured  all five produce rows rather than an error. Worked example: a cell raising during the v3
                       rebuild leaves len(new) at 4 and makes pred_e FALSE.                           prior 85%
  pred_f_anchor_reproduces  the anchor possessive_person_our_your reproduces canonical -0.0489 and weekly 0.0096
                       within anchor_tol = 0.01. Worked example: -0.0421 and 0.0103 hold; a weekly value of 0.0004
                       is the IN-OBJECTIVE number from the C3 run, a gap of 0.0092 -- which only just holds, so the
                       four-cell pred_a above is the check that actually discriminates, not this one.  prior 85%
SCOPE. Five uncounted candidates, four known-good cells, both objectives fitted on one unit set per cell.
Smoke: V675_SMOKE=<out.json> (CPU) -- proves the code path runs; its numbers are not measurements.
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
OUT = ROOT / "circuits/followups/unit_two_objective_screen_v675_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {'adj_adv_feel': 'adj_adv_feel', 'aux_copy_ellipsis': 'aux_copy_ellipsis', 'countability_few_little': 'countability_few_little', 'definiteness_anaphor_needed': 'definiteness_anaphor_needed', 'determiner_number_crates': 'determiner_number_crates', 'correlative_or_and': 'correlative_or_and', 'possessive_gender': 'possessive_gender', 'possessive_number_his_their': 'possessive_number_his_their', 'possessive_person_our_your': 'possessive_person_our_your'}
ANCHOR = "possessive_person_our_your"
SIBLINGS = ['possessive_person_my_their', 'possessive_person_your_their']
# v637 measured the anchor under this exact arrangement: canonical +0.0075, weekly +0.0096, four rows clear,
# both controls passed. It is one of my three standing proposals and rides along to check reproduction.
V675_ANCHOR = {"canon_ub": -0.0489, "weekly_ub": 0.0096}
KG_RECORDED = {"correlative_or_and": (-0.0101, 0.0265), "possessive_gender": (0.0029, -0.0112),
               "possessive_number_his_their": (-0.0663, 0.0534), "possessive_person_our_your": (-0.0489, 0.0096)}
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(['adj_adv_feel', 'aux_copy_ellipsis', 'countability_few_little', 'definiteness_anaphor_needed', 'determiner_number_crates'])
KNOWN_GOOD = tuple(['correlative_or_and', 'possessive_gender', 'possessive_number_his_their', 'possessive_person_our_your'])
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"anchor_tol": 0.01, "ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_reach": 4, "k_both": 2, "k_kg": 6, "tol": 0.05, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_two_objective_screen_v675", "behaviours": 9, "constructions": 5,
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
    a = ok and all(n in good and good[n].get("canon_heldout") is not None and good[n].get("weekly_heldout") is not None
                   and abs(good[n]["canon_heldout"] - KG_RECORDED[n][0]) <= B["anchor_tol"]
                   and abs(good[n]["weekly_heldout"] - KG_RECORDED[n][1]) <= B["anchor_tol"] for n in KNOWN_GOOD) and len(good.get(KNOWN_GOOD[0], {})) > 0
    anc = good.get(ANCHOR, {})
    f = ok and ANCHOR in good and all(
        anc.get(k) is not None and abs(anc[k] - V675_ANCHOR[j]) <= B["anchor_tol"]
        for k, j in (("c_ub_v2", "canon_ub"), ("c_ub_v3", "weekly_ub")))
    b = ok and all((new[n].get("extraction_held") or 0) >= EXT_MIN for n in new)
    reach = [n for n in new if (new[n].get("extraction_held") or 0) >= EXT_MIN]
    c = ok and len(reach) >= B["k_reach"]
    d = ok and sum(1 for n in new if new[n]["both_objectives"]) >= B["k_both"]
    e = ok and len(new) == len(NEW)
    return {"pred_a_known_good_reproduces_both": bool(a), "pred_b_all_reach": bool(b),
            "pred_c_siblings_reached": bool(c), "pred_d_pass_both_objectives": bool(d),
            "pred_e_all_measured": bool(e), "pred_f_anchor_reproduces": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V675_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V675_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V675_SMOKE_NAMES", "adj_adv_feel,possessive_person_our_your").split(",")]
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
            qc2, hist2 = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C_fit"],), control_weight=LAM, mu=mu1)
            arms = {}
            for arm, qa in (("dim", q1), ("cdas", qc), ("cdas_canonobj", qc2)):
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
            canon_heldout = arms["cdas"]["C"]["ce_ub975"]           # weekly IN objective -> canonical is held out
            weekly_heldout = arms["cdas_canonobj"]["C3"]["ce_ub975"] # canonical IN objective -> weekly is held out
            row4_v2 = canon_heldout <= C_UB_MAX
            row4_v3 = weekly_heldout <= C_UB_MAX                     # BOTH numbers are now held-out
            both_objectives = bool(row4_v2 and row4_v3)
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,
                       "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                       "row4": arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX,
                       "row5": a2_own["ce_lb975"] > 0 and row5_share is not None and BARS["row5_share_band"][0] <= row5_share <= BARS["row5_share_band"][1],
                       "row4_v3": row4_v3}
            R[n] = {"row4_v2": bool(row4_v2), "row4_v3": bool(row4_v3), "both_objectives": both_objectives,
                    "canon_heldout": canon_heldout, "weekly_heldout": weekly_heldout,
                    "c_ub_v2": canon_heldout, "c_ub_v3": weekly_heldout,
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
    result = {"predictions": predictions, "schema": "unit_two_objective_screen_v675", "candidate_id": "corpus.unit_two_objective_screen_v675", "bars": BARS,
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
