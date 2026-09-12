#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v631: six structural variants that all pass -- one behaviour, or six counted six times?

WHAT v629 FOUND. All six variants of possessive number agreement -- adjacent, medial, verbfinal, long_simple,
argument, attractor -- reach held-out extraction 0.877 to 1.224 and pass BOTH controls, canonical bounds -0.0079 to
-0.0300 (every one negative) and weekly bounds -0.0006 to +0.0010. Two of them, argument and verbfinal, also clear
all four tier rows, so they pass EVERYTHING: four rows plus a held-out control. Nothing in this corpus had cleared
that combination before, and they are the strongest candidates today.
WHY I AM NOT PUTTING THEM FORWARD YET, AND EXPECT THIS RUNG TO SAY NO. The six share the readout pair ' their' /
' his' and the same cue, and differ only in what sits between cue and readout -- a PP modifier, an extra conjunct, a
longer modifier chain, an argument, an attractor noun. That is the configuration the corpus has twice shown fuses:
v573 found a restructured frame fusing with its original at reach 0.95 to 1.02, and v609 found a same-axis cell
whose family-constrained extraction collapsed to 0.359 because no direction both carried it and spared its family.
Counting structural variants of one behaviour as separate circuits is exactly the duplication the controlling goal
warns against, and a duplicate is invisible once it is in the count. So the honest prior is that these are ONE
behaviour measured six ways, and the value of this rung is that it can say so before anything is booked.
HOW IT CAN ANSWER EITHER WAY. All six sit in ONE family list with four counted number_readout members, so the
separability metric measures each variant's leak against the other five as well as against the counted four. If they
encode one direction their mutual leaks exceed cross_max = 0.05, or their family-constrained extraction collapses as
v609's did, and `separable` goes false. If they are genuinely distinct behaviours -- which would mean the model
carries a different direction for each cue-to-readout distance -- all six clear it while sitting beside each other.
The counted members are re-evaluated in the same run so a null cannot be blamed on the family.
NOTHING IS COUNTED BY THIS FILE whatever it returns, and the two all-rows cells pass under a fit against the weekly
control, which is not the registered objective. The count stays 139.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_new_separable ALL SIX variants are separable at cross_max = 0.05 while sitting in one family with each
                       other -- k_new = 6, so a single fusion breaks it. I expect this to FAIL and am registering
                       the low prior rather than a hedge.                                             prior 20%
  pred_b_counted_kept  at least k_kept = 3 of the four counted members stay separable in the same run, so a null on
                       the variants cannot be blamed on the family having stopped separating.         prior 85%
  pred_c_row4_kept     at least k_d = 2 members keep a C upper bound at or below c_ub_max = 0.01, so the control row
                       is still doing work in this batch.                                             prior 80%
  pred_d_instrument    every EVAL member reproduces its parent battery extraction within instr_tol = 0.03.
                                                                                                      prior 85%
HOW IT READS. a FALSE with b true -- the outcome I expect: the variants are one behaviour, possessive number
agreement is a SINGLE circuit with unusually strong evidence rather than six, and the right way to report v629 is
one behaviour robust across six structural variants. a TRUE: the model carries a distinct direction for each
cue-to-readout structure, which would be a genuine and surprising finding about how the behaviour is implemented and
would need its own write-up rather than a quiet increment of six. a PARTLY false: whichever variants fuse name the
structures that share a direction, and the split between them is the finer result -- for instance if argument and
verbfinal, the two that clear all four rows, separate from the other four while fusing with each other.
SCOPE. Six variants, four counted controls, one family. No counting.
Smoke: V631_SMOKE=<out.json> (CPU).
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
import run_unit_tier3_batch_amended_new9_v201 as v201
import run_unit_tier3_batch_amended_spec9p_v311 as v311
import run_unit_tier3_batch_amended_spec9p_v315 as v315
import run_unit_tier3_batch_amended_spec9p_v319 as v319
import run_unit_tier3_batch_amended_spec9p_v323 as v323
import run_unit_tier3_batch_amended_spec9p_v327 as v327
import run_unit_tier3_batch_amended_spec9p_v349 as v349
import run_unit_tier3_batch_amended_spec9m_v293 as v293
import run_unit_tier3_batch_amended_spec9p_v353 as v353
import run_unit_tier3_batch_amended_spec9e_v263 as v263
import run_unit_tier3_batch_amended_spec9f_v265 as v265
import run_unit_tier3_batch_amended_spec9p_v357 as v357
import run_unit_tier3_batch_amended_spec9p_v339 as v339
import run_unit_tier3_batch_amended_spec9p_v343 as v343
import run_unit_tier3_batch_amended_spec9p_v361 as v361
import run_unit_tier3_batch_amended_spec9p_v365 as v365
import run_unit_tier3_batch_backlog_v575 as v575
import run_unit_tier3_batch_backlog2_v581 as v581
import run_unit_tier3_batch_crossconstruction597_v597 as v597
import run_unit_tier3_batch_crossconstruction599_v599 as v599
import run_unit_tier3_batch_crossconstruction601_v601 as v601

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_possessive_variant_separability_v631_result.json"
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
V311 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v311_result.json"
V315 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v315_result.json"
V319 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v319_result.json"
V323 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v323_result.json"
V327 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v327_result.json"
V349 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v349_result.json"
V293 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9m_v293_result.json"
V353 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v353_result.json"
V263 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9e_v263_result.json"
V265 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9f_v265_result.json"
V357 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v357_result.json"
V339 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v339_result.json"
V343 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v343_result.json"
V361 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v361_result.json"
V575R = ROOT / "circuits/followups/unit_tier3_batch_backlog_v575_result.json"
V581R = ROOT / "circuits/followups/unit_tier3_batch_backlog2_v581_result.json"
V597R = ROOT / "circuits/followups/unit_tier3_batch_crossconstruction597_v597_result.json"
V599R = ROOT / "circuits/followups/unit_tier3_batch_crossconstruction599_v599_result.json"
V601R = ROOT / "circuits/followups/unit_tier3_batch_crossconstruction601_v601_result.json"
V365 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v365_result.json"
FAMILIES = {
    "number_readout": ['agreement_do_does', 'agreement_lifts_lift', 'demonstrative_number', 'demonstrative_number_that_those', 'possessive_adjacent', 'possessive_medial', 'possessive_verbfinal', 'possessive_long_simple', 'possessive_argument', 'possessive_attractor'],
}
COUNTED_LIST = ['agreement_do_does', 'agreement_lifts_lift', 'demonstrative_number', 'demonstrative_number_that_those']
COUNTED = tuple(COUNTED_LIST)
EVAL = tuple(COUNTED_LIST) + tuple(['possessive_adjacent', 'possessive_medial', 'possessive_verbfinal', 'possessive_long_simple', 'possessive_argument', 'possessive_attractor'])
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES, **v243.NAMES, **v245.NAMES, **v247.NAMES, **v249.NAMES, **v251.NAMES, **v253.NAMES, **v255.NAMES, **v257.NAMES, **v311.NAMES, **v315.NAMES, **v319.NAMES, **v323.NAMES, **v327.NAMES, **v349.NAMES, **v293.NAMES, **v353.NAMES, **v263.NAMES, **v265.NAMES, **v357.NAMES, **v339.NAMES, **v343.NAMES, **v361.NAMES, **v365.NAMES, **v575.NAMES, **v581.NAMES, **v597.NAMES, **v599.NAMES, **v601.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V597R, V599R, V601R, V581R, V575R, V365, V361, V343, V339, V357, V263, V265, V353, V293, V349, V327, V323, V319, V315, V311, V257, V255, V253, V251, V249, V247, V245, V243, V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
BARS = {"cross_max": 0.05, "keep": 0.8, "k_kept": 3, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 2, "k_new": 6, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_possessive_variant_separability_v631", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


G, H = "verb_particle_fg_up_down", "verb_particle_fh_up_down"
C, E = "verb_particle_fc_up_down", "verb_particle_fe_up_down"


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    a = ok and sum(1 for n in NEW_MEMBERS if sep(n)) >= B["k_new"]
    c = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    d = ok and all(n in good and good[n]["arms"]["own"]["extraction"] is not None
                   and abs(good[n]["arms"]["own"]["extraction"] - _parent_cdas(n)) <= B["instr_tol"] for n in EVAL)

    kept = sum(1 for n in COUNTED if sep(n))
    b = ok and kept >= B["k_kept"]
    return {"pred_a_new_separable": bool(a), "pred_b_counted_kept": bool(b),
            "pred_c_row4_kept": bool(c), "pred_d_instrument": bool(d)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V631_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V631_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V631_SMOKE_NAMES", "possessive_adjacent,possessive_medial").split(","))]
    prior = {**json.loads(V575R.read_text())["behaviours"], **json.loads(V581R.read_text())["behaviours"], **json.loads(V597R.read_text())["behaviours"], **json.loads(V599R.read_text())["behaviours"], **json.loads(V601R.read_text())["behaviours"], **json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"], **json.loads(V243.read_text())["behaviours"], **json.loads(V245.read_text())["behaviours"], **json.loads(V247.read_text())["behaviours"], **json.loads(V249.read_text())["behaviours"], **json.loads(V251.read_text())["behaviours"], **json.loads(V253.read_text())["behaviours"], **json.loads(V255.read_text())["behaviours"], **json.loads(V257.read_text())["behaviours"], **json.loads(V311.read_text())["behaviours"], **json.loads(V315.read_text())["behaviours"], **json.loads(V319.read_text())["behaviours"], **json.loads(V323.read_text())["behaviours"], **json.loads(V327.read_text())["behaviours"], **json.loads(V349.read_text())["behaviours"], **json.loads(V353.read_text())["behaviours"], **json.loads(V357.read_text())["behaviours"], **json.loads(V361.read_text())["behaviours"], **json.loads(V365.read_text())["behaviours"], **json.loads(V575R.read_text())["behaviours"], **json.loads(V581R.read_text())["behaviours"], **json.loads(V597R.read_text())["behaviours"], **json.loads(V599R.read_text())["behaviours"], **json.loads(V601R.read_text())["behaviours"]}
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
                for arm, controls in (("own", (P["C_fit"],)), ("fam", (P["C_fit"],) + tuple(prep[s]["fit"] for s in sibs))):
                    q, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
                                                               controls=controls, control_weight=LAM * len(controls), mu=mu)
                    a = {"extraction": round(g.recovery(P["held"], g.patched_axis(backend, P["held"], units, q=q)) / exact, 3) if abs(exact) > 1e-6 else None,
                         "A1": ce(P["held"], units, q, mu), "C": ce(P["C_held"], units, q, mu),
                         "siblings": {s: ce(prep[s]["held"], units, q, mu)["ce_damage"] for s in sibs}}
                    a["sib_abs_max"] = round(max([abs(v) for v in a["siblings"].values()] or [0.0]), 4)
                    arms[arm] = a
                sep = arms["fam"]["sib_abs_max"] <= BARS["cross_max"] and arms["fam"]["extraction"] is not None and arms["own"]["extraction"] \
                    and arms["fam"]["extraction"] >= BARS["keep"] * arms["own"]["extraction"] and arms["own"]["extraction"] >= BARS["ext_floor"]
                R[n] = {"family": fam, "units": units, "n_units": len(units), "exact_held": round(exact, 3), "arms": arms, "separable": bool(sep),
                        "parent_rows": prior[n]["rows"], "n_dropped": {k: P[k].dropped for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
            except Exception as exc:  # noqa: BLE001 - one member must not lose the batch
                R[n] = {"family": fam, "error": f"{type(exc).__name__}: {exc}", "separable": None, "seconds": round(time.perf_counter() - t1, 1)}
            print(fam, n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "arms")}),
                  "own", R[n].get("arms", {}).get("own", {}).get("extraction"), "fam", R[n].get("arms", {}).get("fam", {}).get("extraction"),
                  R[n].get("arms", {}).get("fam", {}).get("sib_abs_max"), round(time.perf_counter() - t0), "s", flush=True)

    predictions = PREDS(R)
    nsep = {fam: sum(1 for n in mem if n in R and R[n]["separable"]) for fam, mem in FAMILIES.items()}
    summary = {n: {"family": R[n]["family"], "separable": R[n]["separable"], "n_units": R[n].get("n_units"),
                   "own": (R[n]["arms"]["own"]["extraction"], R[n]["arms"]["own"]["sib_abs_max"], R[n]["arms"]["own"]["C"]["ce_ub975"]) if "arms" in R[n] else None,
                   "fam": (R[n]["arms"]["fam"]["extraction"], R[n]["arms"]["fam"]["sib_abs_max"], R[n]["arms"]["fam"]["C"]["ce_ub975"]) if "arms" in R[n] else None} for n in R}
    result = {"predictions": predictions, "schema": "unit_possessive_variant_separability_v631", "candidate_id": "corpus.unit_possessive_variant_separability_v631",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 ... v257 battery receipts (later receipts override earlier ones)", "evaluated": list(EVAL),
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
