#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v262: is the "fusion" a property of the model, or of the control set I fit against? A CONTROL-SET ABLATION.

THE PHENOMENON, NOW REPLICATED IN BOTH FAMILIES. When a new cell that reuses a counted sibling's cue word joins that
sibling's control set, the INCUMBENT's fitted rank-1 direction stops being inert on the new cell -- by a lot -- while
the new cell's own direction barely notices the incumbent. Measured leaks (fam-arm |CE| into the partner):
    verb    with_about -> with_beyond  0.193   |  with_beyond -> with_about  0.056   (v258)
    verb    in_to      -> in_beyond    0.213   |  in_beyond   -> in_to       0.016   (v258)
    adj     for_of     -> for_beneath  0.203   |  for_beneath -> for_of      0.058   (v260)
    adj     in_about   -> in_beneath   0.139   |  in_beneath  -> in_about    0.009   (v260)
Four pairs, two families, ratios 3.5x to 15x, always in the same direction. Both of my accounts of it failed:
the CUE account was refuted as sufficient (four shared cue->token pairs do not fuse, incl. similar/to_of which shares
BOTH cue and readout; 11:52 row) and the FAMILY account was not confirmed either (the adjective arm came back 4 of 5
separable at v260, not 5 of 5 or 0 of 5, and its own reading rule called that unresolved).

WHAT IS ACTUALLY DIFFERENT ABOUT THE TWO DIRECTIONS. It is not the pair -- it is the OPTIMISATION. The incumbent is
fitted with the new cell IN its control set (the fam arm adds every sibling's A1 FIT prep as a control at weight
30 x n_controls), so its direction is being pushed to be inert on a cell that reuses its own cue; the new cell is
fitted with the incumbent in ITS control set, which is one control among many it never had to satisfy before. Those
are different problems, and a large leak in one direction is equally consistent with:
  (H1) REAL OVERLAP -- the incumbent's direction genuinely carries something the new cell responds to, and the control
       set is merely revealing it; or
  (H2) FIT ARTIFACT -- forcing inertness on a highly correlated control destabilises the incumbent's fit, and the leak
       is manufactured by the constraint rather than found by it.
No amount of further separability rungs distinguishes these, because every one of them adds controls. This rung
removes one.

THE ABLATION. For each of the four incumbents, three arms on the SAME units, seed and split:
    own                 own C_fit only (the existing baseline arm)
    fam                 own C_fit + EVERY sibling as control -- reproduces the parent rung
    fam_minus_partner   own C_fit + every sibling EXCEPT the cue-sharing partner
The partner is still EVALUATED in all three arms (the leak into it is measured); it is only removed from the CONTROLS.
That is the whole manipulation, and it is the one thing the separability recipe has never varied.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_instrument_reproduces -> the fam arm's leak into the partner is within 0.03 of the parent rung's value
      (0.193 / 0.213 / 0.139 / 0.203, hard-coded in PARENT_LEAK) for ALL FOUR. Same units, seed, split and code path,
      so this is a DETERMINISM CHECK on the new arm structure, not a result; if it fails the other three are void.
                                                                                                         prior 90%
  pred_b_partner_is_special_unconstrained -> in the OWN arm (own C_fit only, NO sibling controls at all) the leak into
      the partner exceeds the largest leak into any OTHER sibling by at least 0.05, on at least 3 of 4. This is H1: if
      the incumbent's direction really overlaps the partner, the partner is special even with nothing pushing on it.
      FIRST DRAFT OF THIS PREDICATE WAS VACUOUS AND THE CPU SMOKE CAUGHT IT: I had written "the leak into the partner
      is still > 0.05 with the partner removed from the controls", and the smoke returned 0.173 (constrained) -> 0.968
      (unconstrained). Of course it does -- dropping a constraint releases whatever it was suppressing, so that clause
      could not fail and would have been asserted rather than tested (standing lesson 4). Replaced before enqueue and
      disclosed here rather than quietly.                                                                prior 45%
  pred_c_extraction_recovers -> the fam_minus_partner arm's own extraction is at least 0.02 HIGHER than the fam arm's
      on at least 3 of 4. This is the H2 signature: if the partner-as-control was damaging the fit, removing it should
      buy back extraction. Note b and c are NOT exclusive -- both can hold, which would mean the overlap is real AND
      the constraint was costly.                                                                         prior 55%
  pred_d_other_siblings_stable -> removing the partner from the controls changes the leak into every OTHER sibling by
      at most 0.03, for all four. This is the specificity control: if pulling one control out of a 15-30 control set
      moves everything, the arm is measuring control-set size, not the partner.                          prior 75%
WHAT EACH OUTCOME MEANS, fixed now so it cannot be narrated later:
  b true, c false  -> H1: real overlap. The four fusions stand; the asymmetry is that the incumbent's direction always
                     carried the shared cue and nobody had built a cell that could show it.
  b false, c true  -> H2: fit artifact. The four "fusions" are manufactured by the control set, and EVERY fusion in the
                     corpus that was found by adding a correlated control needs re-reading -- including of_over /
                     over_with, which is where this whole line started. I would report that as a retraction of the
                     mechanism claim, not of the receipts.
  b true, c true   -> both: real overlap that the constraint also fights. The separability bar still means what it says
                     but the extraction bar is measuring the constraint's cost as well as the direction's quality.
  b false, c false -> neither; the leak is something else and I stop guessing and go back to interchange.
COST: 4 members x 3 arms, ~25-35 min. No new cells, no new capability checks; every unit is frozen from a parent receipt.
This rung CANNOT change the circuit count -- it re-fits directions for members that are already counted or already
fused, and adds no new cells. It is spent entirely on whether a number I have been reporting means what I said it means.
Smoke: V262_SMOKE=<out.json> (CPU, V262_SMOKE_ROWS=4; V262_SMOKE_NAMES=verb_preposition_in_to; steps 5).
BARS = {"cross_max": 0.05, "instr_tol": 0.03, "k_survives": 3, "recover_by": 0.02, "k_recovers": 3, "other_sib_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000
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
import run_unit_tier3_batch_amended_all_v197 as v197
import run_unit_tier3_batch_amended_spec8_v204 as v204
import run_unit_tier3_batch_amended_spec8b_v205 as v205
import run_unit_tier3_batch_amended_spec8c_v207 as v207
import run_unit_tier3_batch_amended_spec8d_v209 as v209
import run_unit_tier3_batch_amended_spec8e_v211 as v211
import run_unit_tier3_batch_amended_spec8f_v213 as v213
import run_unit_tier3_batch_amended_spec8g_v215 as v215
import run_unit_tier3_batch_amended_spec8h_v217 as v217
import run_unit_tier3_batch_amended_spec8i_v219 as v219
import run_unit_tier3_batch_amended_spec8j_v221 as v221
import run_unit_tier3_batch_amended_spec8k_v223 as v223
import run_unit_tier3_batch_amended_spec8l_v225 as v225
import run_unit_tier3_batch_amended_spec8m_v227 as v227
import run_unit_tier3_batch_amended_spec8n_v229 as v229
import run_unit_tier3_batch_amended_spec8o_v231 as v231
import run_unit_tier3_batch_amended_spec8p_v233 as v233
import run_unit_tier3_batch_amended_spec8q_v235 as v235
import run_unit_tier3_batch_amended_spec8r_v237 as v237
import run_unit_tier3_batch_amended_spec8s_v239 as v239
import run_unit_tier3_batch_amended_spec8t_v241 as v241
import run_unit_tier3_batch_amended_spec8u_v243 as v243
import run_unit_tier3_batch_amended_spec8v_v245 as v245
import run_unit_tier3_batch_amended_spec8w_v247 as v247
import run_unit_tier3_batch_amended_spec8x_v249 as v249
import run_unit_tier3_batch_amended_spec8y_v251 as v251
import run_unit_tier3_batch_amended_spec8z_v253 as v253
import run_unit_tier3_batch_amended_spec9a_v255 as v255
import run_unit_tier3_batch_amended_spec9b_v257 as v257
import run_unit_tier3_batch_amended_spec9d_v261 as v261
import run_unit_tier3_batch_amended_spec9e_v263 as v263
import run_unit_tier3_batch_amended_spec9g_v267 as v267
import run_unit_tier3_batch_amended_spec9f_v265 as v265
import run_unit_tier3_batch_amended_spec9c_v259 as v259
import run_unit_tier3_batch_amended_new9_v201 as v201
import run_unit_tier3_batch_amended_spec9h_v269 as v269

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v262_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
V201 = ROOT / "circuits/followups/unit_tier3_batch_amended_new9_v201_result.json"
V204 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8_v204_result.json"
V205 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8b_v205_result.json"
V207 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8c_v207_result.json"
V209 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8d_v209_result.json"
V211 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8e_v211_result.json"
V213 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8f_v213_result.json"
V215 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8g_v215_result.json"
V217 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8h_v217_result.json"
V219 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8i_v219_result.json"
V221 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8j_v221_result.json"
V223 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8k_v223_result.json"
V225 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8l_v225_result.json"
V227 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8m_v227_result.json"
V229 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8n_v229_result.json"
V231 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8o_v231_result.json"
V233 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8p_v233_result.json"
V235 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8q_v235_result.json"
V237 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8r_v237_result.json"
V239 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8s_v239_result.json"
V241 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8t_v241_result.json"
V243 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8u_v243_result.json"
V245 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8v_v245_result.json"
V247 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8w_v247_result.json"
V249 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8x_v249_result.json"
V251 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8y_v251_result.json"
V253 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8z_v253_result.json"
V255 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9a_v255_result.json"
V257 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9b_v257_result.json"
V261 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9d_v261_result.json"
V263 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9e_v263_result.json"
V267 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9g_v267_result.json"
V265 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9f_v265_result.json"
V259 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9c_v259_result.json"
V269 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9h_v269_result.json"
FAMILIES = {
    "verb_preposition": ["verb_preposition", "verb_preposition_insisted", "verb_preposition_to_at", "verb_preposition_with_about", "verb_preposition_of_with", "verb_preposition_from_for", "verb_preposition_on_about", "verb_preposition_to_about", "verb_preposition_at_from", "verb_preposition_on_with", "verb_preposition_for_at", "verb_preposition_to_with", "verb_preposition_at_to", "verb_preposition_from_about", "verb_preposition_by_for", "verb_preposition_of_by", "verb_preposition_in_to", "verb_preposition_over_with", "verb_preposition_on_under", "verb_preposition_of_over", "verb_preposition_toward_from", "verb_preposition_with_against", "verb_preposition_on_toward", "verb_preposition_about_into", "verb_preposition_into_with", "verb_preposition_of_into", "verb_preposition_about_against", "verb_preposition_through_with", "verb_preposition_of_beneath", "verb_preposition_at_under", "verb_preposition_in_beyond", "verb_preposition_with_beyond"],
    "adjective_preposition": ["adjective_preposition_fond", "adjective_preposition_similar", "identical_distinct_preposition", "adjective_preposition_of_at", "adjective_preposition_to_of", "adjective_preposition_for_of", "same_different_preposition", "adjective_preposition_in_about", "adjective_preposition_of_about", "adjective_preposition_at_about", "adjective_preposition_of_against", "adjective_preposition_at_within", "adjective_preposition_for_beneath", "adjective_preposition_in_beneath", "adjective_preposition_of_toward", "adjective_preposition_of_within"],
}
EVAL = ("verb_preposition_with_about", "verb_preposition_in_to", "adjective_preposition_in_about", "adjective_preposition_for_of", )
COUNTED = ()
PARTNER = {"verb_preposition_with_about": "verb_preposition_with_beyond", "verb_preposition_in_to": "verb_preposition_in_beyond", "adjective_preposition_in_about": "adjective_preposition_in_beneath", "adjective_preposition_for_of": "adjective_preposition_for_beneath"}
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES, **v243.NAMES, **v245.NAMES, **v247.NAMES, **v249.NAMES, **v251.NAMES, **v253.NAMES, **v255.NAMES, **v257.NAMES, **v259.NAMES, **v261.NAMES, **v263.NAMES, **v265.NAMES, **v267.NAMES, **v269.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V269, V267, V265, V263, V261, V259, V257, V255, V253, V251, V249, V247, V245, V243, V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
PRED_NAMES = ("pred_a_instrument_reproduces", "pred_b_partner_is_special_unconstrained",
              "pred_c_extraction_recovers", "pred_d_other_siblings_stable")
BARS = {"cross_max": 0.05, "instr_tol": 0.03, "k_survives": 3, "recover_by": 0.02, "k_recovers": 3, "other_sib_tol": 0.03, "partner_margin": 0.05}
PARENT_LEAK = {"verb_preposition_with_about": 0.1930, "verb_preposition_in_to": 0.2126,
               "adjective_preposition_in_about": 0.1392, "adjective_preposition_for_of": 0.2026}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v262", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "arms" in r and "fam_minus_partner" in r["arms"]}
    ok = len(good) == len(EVAL)
    lk = lambda n, arm: abs(good[n]["arms"][arm]["siblings"].get(PARTNER[n], 0.0))
    a = ok and all(good[n].get("instr_abs_diff") is not None
                   and good[n]["instr_abs_diff"] <= B["instr_tol"] for n in good)
    # b: in the UNCONSTRAINED own arm the partner must be SPECIAL -- its leak must exceed the largest leak into any
    # other sibling by margin. "Leak survives when the control is removed" would be vacuous: dropping a constraint
    # always releases what it was suppressing (the CPU smoke showed 0.173 -> 0.968), so that clause cannot fail.
    def special(n):
        sib = good[n]["arms"]["own"]["siblings"]
        others = [abs(v) for s, v in sib.items() if s != PARTNER[n]]
        return abs(sib.get(PARTNER[n], 0.0)) - (max(others) if others else 0.0)
    b = ok and sum(1 for n in good if special(n) >= B["partner_margin"]) >= B["k_survives"]
    c = ok and sum(1 for n in good
                   if good[n]["arms"]["fam_minus_partner"]["extraction"] is not None
                   and good[n]["arms"]["fam"]["extraction"] is not None
                   and good[n]["arms"]["fam_minus_partner"]["extraction"]
                       >= good[n]["arms"]["fam"]["extraction"] + B["recover_by"]) >= B["k_recovers"]
    d = ok and all(max([abs(good[n]["arms"]["fam_minus_partner"]["siblings"].get(s, 0.0)
                            - good[n]["arms"]["fam"]["siblings"].get(s, 0.0))
                        for s in good[n]["arms"]["fam"]["siblings"] if s != PARTNER[n]] or [0.0])
                   <= B["other_sib_tol"] for n in good)
    return dict(zip(PRED_NAMES, (bool(a), bool(b), bool(c), bool(d))))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V262_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V262_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V262_SMOKE_NAMES", "verb_preposition_in_to").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"], **json.loads(V243.read_text())["behaviours"], **json.loads(V245.read_text())["behaviours"], **json.loads(V247.read_text())["behaviours"], **json.loads(V249.read_text())["behaviours"], **json.loads(V251.read_text())["behaviours"], **json.loads(V253.read_text())["behaviours"], **json.loads(V255.read_text())["behaviours"], **json.loads(V257.read_text())["behaviours"], **json.loads(V259.read_text())["behaviours"], **json.loads(V261.read_text())["behaviours"], **json.loads(V263.read_text())["behaviours"], **json.loads(V265.read_text())["behaviours"], **json.loads(V267.read_text())["behaviours"], **json.loads(V269.read_text())["behaviours"]}
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]
    prep = {}
    for n in which:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
        a1, c = g.rows_of(m, "A1"), g.rows_of(m, "C")
        prep[n] = {"fit": g.prepare(backend, cut(fit_half(a1)), valid_only=True), "held": g.prepare(backend, cut(held_half(a1)), valid_only=True),
                   "C_fit": g.prepare(backend, cut(fit_half(c))), "C_held": g.prepare(backend, cut(held_half(c)))}

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def ce(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975")}

    R = {}
    for fam, mem in FAMILIES.items():
        for n in mem:
            if n not in evaluate:
                continue
            t1 = time.perf_counter()
            try:
                if "error" in prior.get(n, {"error": "missing"}):
                    raise RuntimeError(f"the parent receipts have no clean receipt for {n}")
                units = list(prior[n]["units"])
                P = prep[n]
                mu = mu_of(P["fit"], units)
                exact = g.recovery(P["held"], g.patched_axis(backend, P["held"], units))
                sibs = [s for s in mem if s != n and s in prep]
                arms = {}
                partner = PARTNER.get(n)
                fam_ctrl = (P["C_fit"],) + tuple(prep[s]["fit"] for s in sibs)
                minus_ctrl = (P["C_fit"],) + tuple(prep[s]["fit"] for s in sibs if s != partner)
                for arm, controls in (("own", (P["C_fit"],)), ("fam", fam_ctrl), ("fam_minus_partner", minus_ctrl)):
                    q, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
                                                               controls=controls, control_weight=LAM * len(controls), mu=mu)
                    a = {"extraction": round(g.recovery(P["held"], g.patched_axis(backend, P["held"], units, q=q)) / exact, 3) if abs(exact) > 1e-6 else None,
                         "A1": ce(P["held"], units, q, mu), "C": ce(P["C_held"], units, q, mu),
                         "siblings": {s: ce(prep[s]["held"], units, q, mu)["ce_damage"] for s in sibs}}
                    a["sib_abs_max"] = round(max([abs(v) for v in a["siblings"].values()] or [0.0]), 4)
                    arms[arm] = a
                sep = arms["fam"]["sib_abs_max"] <= BARS["cross_max"] and arms["fam"]["extraction"] is not None and arms["own"]["extraction"] \
                    and arms["fam"]["extraction"] >= BARS["keep"] * arms["own"]["extraction"] and arms["own"]["extraction"] >= BARS["ext_floor"]
                lk_fam = abs(arms["fam"]["siblings"].get(partner, 0.0))
                R[n] = {"family": fam, "units": units, "n_units": len(units), "exact_held": round(exact, 3), "arms": arms,
                        "partner": partner, "leak_fam": round(lk_fam, 4),
                        "leak_fam_minus": round(abs(arms["fam_minus_partner"]["siblings"].get(partner, 0.0)), 4),
                        "instr_abs_diff": round(abs(lk_fam - PARENT_LEAK[n]), 4),
                        "own_partner_margin": round(abs(arms["own"]["siblings"].get(partner, 0.0))
                                                    - max([abs(v) for s2, v in arms["own"]["siblings"].items()
                                                           if s2 != partner] or [0.0]), 4), "separable": bool(sep),
                        "parent_rows": prior[n]["rows"], "n_dropped": {k: P[k].dropped for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
            except Exception as exc:  # noqa: BLE001 - one member must not lose the batch
                R[n] = {"family": fam, "error": f"{type(exc).__name__}: {exc}", "separable": False, "seconds": round(time.perf_counter() - t1, 1)}
            print(fam, n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "arms")}),
                  "own", R[n].get("arms", {}).get("own", {}).get("extraction"), "fam", R[n].get("arms", {}).get("fam", {}).get("extraction"),
                  R[n].get("arms", {}).get("fam", {}).get("sib_abs_max"), round(time.perf_counter() - t0), "s", flush=True)

    predictions = PREDS(R)
    nsep = {fam: sum(1 for n in mem if n in R and R[n]["separable"]) for fam, mem in FAMILIES.items()}
    summary = {n: {"family": R[n]["family"], "separable": R[n]["separable"], "n_units": R[n].get("n_units"),
                   "own": (R[n]["arms"]["own"]["extraction"], R[n]["arms"]["own"]["sib_abs_max"], R[n]["arms"]["own"]["C"]["ce_ub975"]) if "arms" in R[n] else None,
                   "fam": (R[n]["arms"]["fam"]["extraction"], R[n]["arms"]["fam"]["sib_abs_max"], R[n]["arms"]["fam"]["C"]["ce_ub975"]) if "arms" in R[n] else None} for n in R}
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v262", "candidate_id": "corpus.unit_family_separability_spec_v262",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 ... v267 battery receipts (later receipts override earlier ones)", "evaluated": list(EVAL),
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
