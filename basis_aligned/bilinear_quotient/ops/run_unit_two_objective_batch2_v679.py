#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v679: the verified two-objective screen at scale -- twelve candidates across twelve thin axes.

WHY THIS IS A SCALING RUN AND NOT A NEW IDEA. v675 built and verified the two-objective screen: all four known-good
cells reproduced BOTH recorded bounds exactly, eight values from two different older single-arm runners, which rules
out the second arm silently fitting the same objective as the first. v677 then took six survivors through the
separability gate and all six passed. The pipeline is proven end to end and costs about 2.5 GPU-minutes per
surviving behaviour, with candidates sourced by QUERY from receipts already on disk rather than authored. This rung
simply runs more candidates through it.
HOW THE TWELVE WERE CHOSEN, AND WHAT WAS DELIBERATELY LEFT OUT. Sweeping every receipt for four-row passers that no
COUNTED tuple names, with avoided readouts and avoided CUES removed and cells with no readable vocabulary removed,
leaves 70. Half of those are verb_preposition and verb_particle cells. I am NOT screening those here: the verb stem
carries 38 counted cells, v661 measured a 50% duplication rate for candidates on an axis the family already covers,
and v665 showed fusion is within-axis. Putting thirty verb_preposition cells in one family would mostly measure
them fusing with each other. The twelve here sit on twelve distinct and comparatively thin axes -- adjective
finiteness, comparative frame, comparative complement, degree complement, modal complement, modal perfect form,
polarity quantifier, polarity person, reflexive number, wh adjunct, wh argument, noun preposition. The crowded verb
axes remain available later, but they need a design that expects fusion rather than one that would be surprised by it.
NO FUSION PREDICTION IS OFFERED, AND THAT IS DELIBERATE. I have twice registered a named pair as the likely fusion
on readout adjacency -- rather_prefer with let_want_complement in v665 (an EXACT shared readout pair) and
countability_few_little with definiteness_anaphor_needed in v677 -- and both times the named pair came back among
the cleanest in its family. The v321 result keeps holding where I bet against it: the cue-to-token MAPPING is a
behaviour's identity, the readout pair is not. Rather than make a third prediction from the same discredited signal,
I am making none. Separability is not tested by this rung in any case; it is the next gate if candidates survive.
WHAT THIS RUNG CAN AND CANNOT CONCLUDE. It measures ONLY whether each candidate clears the held-out control under
both objectives. A pass here makes a cell eligible for the separability gate, nothing more, and no cell becomes
proposable on this receipt alone. The current proposal to Codex stands at EIGHT and is not changed by this file.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_known_good_reproduces_both  ALL FOUR known-good cells reproduce BOTH recorded bounds -- canonical from the
                       C3-objective runs and weekly from the C-objective runs -- each within anchor_tol = 0.01.
                       Worked example: correlative_or_and must return canonical near -0.0101 AND weekly near 0.0265;
                       identical values in both columns would mean the two arms fitted the same objective, a gap of
                       0.0366 on the weekly value, and pred_a FALSE. This passed exactly in v675.     prior 90%
  pred_b_all_reach     ALL TWELVE candidates reach held-out extraction ext_min = 0.80. Worked example: eleven at
                       0.88 or above and wh_argument_selection at 0.74 makes pred_b FALSE. With twelve cells a
                       single miss breaks it, so I expect this to fail even if the batch is healthy. prior 35%
  pred_c_siblings_reached  at least k_reach = 10 of the twelve reach 0.80. Worked example: eleven reach and one does
                       not, count 11, 11 >= 10 is true, pred_c is TRUE.                               prior 80%
  pred_d_pass_both_objectives  at least k_both = 5 of the twelve clear the HELD-OUT control under BOTH objectives.
                       Worked example: if six clear both sides and six fail at least one, the count is 6, 6 >= 5 is
                       true, pred_d is TRUE. The rate to beat is v675's 3 of 5 and the counted corpus's 10 of 26;
                       five of twelve sits between them.                                              prior 55%
  pred_e_all_measured  all twelve produce rows rather than an error. Worked example: a cell raising during the v3
                       rebuild leaves len(new) at 11 and makes pred_e FALSE.                          prior 75%
  pred_f_anchor_reproduces  the anchor possessive_person_our_your reproduces canonical -0.0489 and weekly 0.0096
                       within anchor_tol = 0.01. Worked example: -0.0421 and 0.0103 hold; a weekly value of 0.0004
                       is the IN-OBJECTIVE number from the C3 run and makes pred_f FALSE.             prior 90%
SCOPE. Twelve uncounted candidates on twelve thin axes, four known-good cells, both objectives per cell. No counting,
no separability, no proposal changes.
Smoke: V679_SMOKE=<out.json> (CPU) -- proves the code path runs; its numbers are not measurements.
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
OUT = ROOT / "circuits/followups/unit_two_objective_batch2_v679_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {'adjective_finiteness': 'adjective_finiteness', 'comparative_frame': 'comparative_frame', 'comparative_complement_from_than': 'comparative_complement_from_than', 'degree_complement_than_of': 'degree_complement_than_of', 'modal_complement_to_lift': 'modal_complement_to_lift', 'modal_perfect_form': 'modal_perfect_form', 'polarity_any_some': 'polarity_any_some', 'polarity_anyone_everyone': 'polarity_anyone_everyone', 'reflexive_number_itself_themselves': 'reflexive_number_itself_themselves', 'wh_adjunct_when_where': 'wh_adjunct_when_where', 'wh_argument_selection': 'wh_argument_selection', 'noun_preposition_for_about': 'noun_preposition_for_about', 'correlative_or_and': 'correlative_or_and', 'possessive_gender': 'possessive_gender', 'possessive_number_his_their': 'possessive_number_his_their', 'possessive_person_our_your': 'possessive_person_our_your'}
ANCHOR = "possessive_person_our_your"
SIBLINGS = ['possessive_person_my_their', 'possessive_person_your_their']
# v637 measured the anchor under this exact arrangement: canonical +0.0075, weekly +0.0096, four rows clear,
# both controls passed. It is one of my three standing proposals and rides along to check reproduction.
V679_ANCHOR = {"canon_ub": -0.0489, "weekly_ub": 0.0096}
KG_RECORDED = {"correlative_or_and": (-0.0101, 0.0265), "possessive_gender": (0.0029, -0.0112),
               "possessive_number_his_their": (-0.0663, 0.0534), "possessive_person_our_your": (-0.0489, 0.0096)}
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(['adjective_finiteness', 'comparative_frame', 'comparative_complement_from_than', 'degree_complement_than_of', 'modal_complement_to_lift', 'modal_perfect_form', 'polarity_any_some', 'polarity_anyone_everyone', 'reflexive_number_itself_themselves', 'wh_adjunct_when_where', 'wh_argument_selection', 'noun_preposition_for_about'])
KNOWN_GOOD = tuple(['correlative_or_and', 'possessive_gender', 'possessive_number_his_their', 'possessive_person_our_your'])
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"anchor_tol": 0.01, "ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_reach": 10, "k_both": 5, "k_kg": 6, "tol": 0.05, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_two_objective_batch2_v679", "behaviours": 9, "constructions": 5,
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
        anc.get(k) is not None and abs(anc[k] - V679_ANCHOR[j]) <= B["anchor_tol"]
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
    smoke = os.environ.get("V679_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V679_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V679_SMOKE_NAMES", "polarity_any_some,possessive_person_our_your").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_two_objective_batch2_v679", "candidate_id": "corpus.unit_two_objective_batch2_v679", "bars": BARS,
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
