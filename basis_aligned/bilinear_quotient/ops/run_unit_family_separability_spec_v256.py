#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v256: within-family separability for the five v263/v265 four-row passes under the verb_preposition family -- and the first rung authored
AFTER a fusion, with the fusion's cause registered as the question it tests.

WHAT v254 FOUND (09:12): 24 of 26 separable, but TWO members failed, the first non-separable verdicts in the preposition class after 60+
consecutive separable evals. verb_preposition_of_over (consisted/presided -> of/over) leaked 0.294 into verb_preposition_of_by and 0.134 into
verb_preposition_over_with; verb_preposition_over_with (presided/tampered -> over/with) leaked 0.129 back into of_over. The cause is not the
shared READOUT token: of_over and over_with share ` over`, but v248/v250/v252 established that a shared readout does not fuse (for_at vs to_at
0.005; at_to, which shares BOTH readouts with counted siblings, 0.006). It is the shared CUE WORD. of_over uses `presided` for its ` over` side,
which is exactly over_with's ` over` cue; and of_over uses `consisted` for its ` of` side, which is exactly of_by's ` of` cue. Both of of_over's
leaks are into the two cells that reuse one of its cue verbs, and into no others.
So the corpus-wide reading, stated as a hypothesis and not a conclusion: these directions are keyed to the CUE LEXEME, not to the answer token.
That is consistent with the standing "directions are cue-pair keyed" finding and it explains why 60+ shared-readout pairs stayed separable --
none of them reused a cue verb.
COUNT: of_over is a NEW cell that leaks into a COUNTED incumbent (over_with), so under the v242 rule it joins that incumbent, +0. over_with keeps
its count as the anchor of the now-fused pair -- it lost separability ONLY to the new cell that fuses with it, not to any pre-existing sibling,
so retracting it would leave a real circuit uncounted. DISTINCT CIRCUITS therefore stay at 99. The v222 rule (a counted member that loses
separability is RETRACTED) and the v242 rule (a new cell leaking into a counted incumbent joins it) give different answers in exactly this case
and BOTH are registered semantics, so the reading above is on the board as a question, not applied as a precedent.

THIS RUNG. Family: verb_preposition, 15 counted after v254 (verb_preposition on/to, insisted, to_at, with_about, of_with, from_for, on_about,
for_at, to_with, at_to, from_about, in_to, over_with, toward_from, with_against); control-only members at_from, on_with, to_about, by_for, of_by,
on_under, on_toward (v265's row-4 miss) and of_over (fused at v254, a control from here on).
  + verb_preposition_about_into (raved/delved -> about/into; v263)
  + verb_preposition_into_with (crashed/flirted -> into/with; v263)
  + verb_preposition_of_into (reeked/peered -> of/into; v263)
  + verb_preposition_about_against (quibbled/campaigned -> about/against; v265)
  + verb_preposition_through_with (rummaged/tinkered -> through/with; v265)
THE REGISTERED TEST OF THE CUE-LEXEME HYPOTHESIS: none of these five reuses a cue verb from any counted sibling or control -- raved, delved,
crashed, flirted, reeked, peered, quibbled, campaigned, rummaged, tinkered are all new to the family. THREE of them share a readout token with a
counted sibling (about_into and about_against share ` about` with with_about and on_about; into_with and through_with share ` with` with five
counted cells; of_into shares ` of` with of_with). If the cue-lexeme hypothesis is right, all five are separable despite those shared readouts,
and the v254 fusion stays explained by the cue reuse alone. If ANY of them fuses on a shared readout with no shared cue, the hypothesis is wrong
and the v254 fusion needs a different explanation -- that is the outcome that would refute it, and it is the reason this rung is worth running
rather than assuming the answer.
Recipe as v198/.../v254: per evaluated member, arm `own` (own C_fit control) and arm `fam` (own C_fit + EVERY sibling's A1 FIT prep,
control_weight 30 x len(controls)); evaluated on the member's held A1 (extraction), each sibling's held A1 (|CE| leak) and held C.
Units frozen from the parent receipts (v263/v265 for the new members; earlier batteries for the counted ones).
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (v206 floor).
EVAL = 5 new members + 15 counted re-checks (20 fits x 2 arms, 27 controls in the fam arm: ~2 h; the full re-check stands because board
proposal (ix), to sample it instead, is unanswered).
99 (after v254) -> up to 104 from this rung.

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here; bars in BARS):
    pred_a_new_separable   at least 3 of the 5 new members are separable (k_new). The cue-lexeme hypothesis predicts 5 of 5.        prior 65%
    pred_b_counted_kept    at least 12 of the 15 re-checked counted members remain separable (k_kept; v254 kept 14/15 with over_with
                           the only loss, v252 11/11).                                                                              prior 60%
    pred_c_row4_kept       fam-arm own-C UB975 <= 0.01 on at least 10 of the 20 evaluated members (k_d; v254 15/26, v252 7/13).     prior 55%
    pred_d_instrument      every evaluated member's own-arm extraction is within 0.03 of its parent battery's cdas extraction_held
                           (same units, seed, split and code path: a determinism check, NOT an independent run outcome).            prior 90%
Watch, not a bar: verb_preposition_with_about reached sib 0.047 at v254 against a 0.05 cross_max, and its leak partner is to_about, a control.
It is one control away from a retraction that would be caused entirely by control-set growth, which is the effect proposal (ix) is about.
Smoke: V256_SMOKE=<out.json> (CPU, V256_SMOKE_ROWS=4; V256_SMOKE_NAMES = a new member plus a re-check; steps 5).
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 10, "k_new": 3, "k_kept": 12, "instr_tol": 0.03}
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
import run_unit_tier3_batch_amended_spec9f_v265 as v265
import run_unit_tier3_batch_amended_spec9c_v259 as v259
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v256_result.json"
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
V265 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9f_v265_result.json"
V259 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9c_v259_result.json"
FAMILIES = {
    "verb_preposition": ["verb_preposition", "verb_preposition_insisted", "verb_preposition_to_at", "verb_preposition_with_about", "verb_preposition_of_with", "verb_preposition_from_for", "verb_preposition_on_about", "verb_preposition_to_about", "verb_preposition_at_from", "verb_preposition_on_with", "verb_preposition_for_at", "verb_preposition_to_with", "verb_preposition_at_to", "verb_preposition_from_about", "verb_preposition_by_for", "verb_preposition_of_by", "verb_preposition_in_to", "verb_preposition_over_with", "verb_preposition_on_under", "verb_preposition_of_over", "verb_preposition_toward_from", "verb_preposition_with_against", "verb_preposition_on_toward", "verb_preposition_about_into", "verb_preposition_into_with", "verb_preposition_of_into", "verb_preposition_about_against", "verb_preposition_through_with"],
}
EVAL = ("verb_preposition_about_into", "verb_preposition_into_with", "verb_preposition_of_into", "verb_preposition_about_against", "verb_preposition_through_with", "verb_preposition", "verb_preposition_insisted", "verb_preposition_to_at", "verb_preposition_with_about", "verb_preposition_of_with", "verb_preposition_from_for", "verb_preposition_on_about", "verb_preposition_for_at", "verb_preposition_to_with", "verb_preposition_at_to", "verb_preposition_from_about", "verb_preposition_in_to", "verb_preposition_over_with", "verb_preposition_toward_from", "verb_preposition_with_against", )
COUNTED = ("verb_preposition", "verb_preposition_insisted", "verb_preposition_to_at", "verb_preposition_with_about", "verb_preposition_of_with", "verb_preposition_from_for", "verb_preposition_on_about", "verb_preposition_for_at", "verb_preposition_to_with", "verb_preposition_at_to", "verb_preposition_from_about", "verb_preposition_in_to", "verb_preposition_over_with", "verb_preposition_toward_from", "verb_preposition_with_against", )
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES, **v243.NAMES, **v245.NAMES, **v247.NAMES, **v249.NAMES, **v251.NAMES, **v253.NAMES, **v255.NAMES, **v257.NAMES, **v259.NAMES, **v261.NAMES, **v263.NAMES, **v265.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V265, V263, V261, V259, V257, V255, V253, V251, V249, V247, V245, V243, V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 10, "k_new": 3, "k_kept": 12, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v256", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    new_sep = sum(1 for n in NEW_MEMBERS if sep(n)); kept = sum(1 for n in COUNTED if sep(n))
    a = ok and new_sep >= B["k_new"]
    b = ok and kept >= B["k_kept"]
    c = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    d = ok and all(n in good and good[n]["arms"]["own"]["extraction"] is not None
                   and abs(good[n]["arms"]["own"]["extraction"] - _parent_cdas(n)) <= B["instr_tol"] for n in EVAL)
    return {"pred_a_new_separable": bool(a), "pred_b_counted_kept": bool(b), "pred_c_row4_kept": bool(c), "pred_d_instrument": bool(d)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V256_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V256_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V256_SMOKE_NAMES", "verb_preposition_about_into,verb_preposition").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"], **json.loads(V243.read_text())["behaviours"], **json.loads(V245.read_text())["behaviours"], **json.loads(V247.read_text())["behaviours"], **json.loads(V249.read_text())["behaviours"], **json.loads(V251.read_text())["behaviours"], **json.loads(V253.read_text())["behaviours"], **json.loads(V255.read_text())["behaviours"], **json.loads(V257.read_text())["behaviours"], **json.loads(V259.read_text())["behaviours"], **json.loads(V261.read_text())["behaviours"], **json.loads(V263.read_text())["behaviours"], **json.loads(V265.read_text())["behaviours"]}
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
                R[n] = {"family": fam, "error": f"{type(exc).__name__}: {exc}", "separable": False, "seconds": round(time.perf_counter() - t1, 1)}
            print(fam, n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "arms")}),
                  "own", R[n].get("arms", {}).get("own", {}).get("extraction"), "fam", R[n].get("arms", {}).get("fam", {}).get("extraction"),
                  R[n].get("arms", {}).get("fam", {}).get("sib_abs_max"), round(time.perf_counter() - t0), "s", flush=True)

    predictions = PREDS(R)
    nsep = {fam: sum(1 for n in mem if n in R and R[n]["separable"]) for fam, mem in FAMILIES.items()}
    summary = {n: {"family": R[n]["family"], "separable": R[n]["separable"], "n_units": R[n].get("n_units"),
                   "own": (R[n]["arms"]["own"]["extraction"], R[n]["arms"]["own"]["sib_abs_max"], R[n]["arms"]["own"]["C"]["ce_ub975"]) if "arms" in R[n] else None,
                   "fam": (R[n]["arms"]["fam"]["extraction"], R[n]["arms"]["fam"]["sib_abs_max"], R[n]["arms"]["fam"]["C"]["ce_ub975"]) if "arms" in R[n] else None} for n in R}
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v256", "candidate_id": "corpus.unit_family_separability_spec_v256",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 ... v265 battery receipts (later receipts override earlier ones)", "evaluated": list(EVAL),
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
