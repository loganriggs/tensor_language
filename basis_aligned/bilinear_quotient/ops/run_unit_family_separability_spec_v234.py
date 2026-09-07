#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 + v217 + v227 receipts), recipe and bars fixed before the run.
"""v234: within-family separability for the v237 pass number_reflexive_myself_ourselves, the v239 passes agreement_lifts_lift,
possessive_pronoun_mine_yours and case_he_him, and the v241 passes possessive_number_his_their, reflexive_person_myself_himself,
object_pronoun_person_me_him and durativity_until_by, plus the re-check of every counted member of the four families they enter
(v212/v216/v220/v222 rule: a family that grows -- by a member OR by a control -- re-checks its counted members; standing until board proposal (v) is answered).

Families here (ALL members prepared as controls; only EVAL evaluated):
  number_readout (8 counted: reflexive_number, demonstrative_number, reciprocal, possessive_number, quantifier_number, object_pronoun_number,
    possessive_number_its_their, demonstrative_number_that_those (separable at v232); pronoun_number RETRACTED at v232 and kept as a control
    with possessive_verbfinal and pronoun_number_it_they) + number_reflexive_myself_ourselves (14 units 15:01/09:06/12:04/13:02/13:01; dim 0.823,
    cdas 1.017, C 0.0050, cext 0.995; v237) + agreement_lifts_lift (7 units 11:03/07:08/05:03/13:01/09:07; dim 0.535, cdas 0.675, C 0.0050,
    cext 0.996; v239). lifts_lift is the first LEXICAL-VERB agreement readout in the family (every other member reads a pronoun/determiner);
    myself_ourselves shares 09:06/12:04/13:02 with reflexive_number and the reflexive readout slot -- the v224 shared-readout rule: fusion is tested, not predicted.
    + possessive_number_his_their (9 units 09:06/12:04/15:01/10:01/10:05; dim 0.644, cdas 0.917, C 0.0044, cext 0.993; v241) -- a third
    possessive-number readout next to possessive_number and possessive_number_its_their (both counted): the family's densest sub-cluster.
  person_readout (5 counted: person_possessive, person_possessive_plural, object_pronoun_person_plural, reflexive_person_plural,
    person_reflexive_ourselves_themselves; controls reflexive_person, object_pronoun_person) + possessive_pronoun_mine_yours (11 units
    09:06/15:01/13:02/09:07/12:04; dim 0.339, cdas 0.578, C 0.0006, cext 0.990; v239) -- a first-vs-second person readout in a family whose
    counted members read first/third; cdas 0.578 is the weakest own extraction entering a separability rung since v206 (floor 0.50)
    + reflexive_person_myself_himself (15 units 12:04/09:06/15:01/05:03/10:01; dim 0.705, cdas 0.874, C 0.0072, cext 0.997; v241) and
    object_pronoun_person_me_him (18 units 09:06/13:02/12:04/13:01/06:01; dim 0.484, cdas 0.663, C −0.0041, cext 0.995; v241) -- the
    singular first-vs-third counterparts of reflexive_person_plural / object_pronoun_person_plural (both counted); the strongest fusion
    candidates of the rung. Control person_agreement_am_is (v241, rows 0111) is ADDED to the person control set.
  case (1 counted: relative_case_who_whose, counted at v224 as a genitive-case readout; controls pronoun_case_coordination (me/I, rows 4/5
    failed at v217) and relative_case_who_whom (row 4 only, v221)) + case_he_him (8 units 06:03/07:08/04:01/03:06/05:08; dim 0.863, cdas
    1.468, C 0.0091, cext 0.992; v239). The v239 docstring registered he_him as family-less; that was wrong on the record (who_whose was
    counted at v224) and is corrected here: he_him counts only if separable under the case family's control set.
  duration (2 counted: duration_point_since_for, duration_point_within_by) + durativity_until_by (12 units 06:03/09:01/07:08/14:08/08:01;
    dim 1.202, cdas 1.407, C 0.0088, cext 0.999; v241). until_by reads the VERB's durativity (waited/finished), its siblings read a
    duration-vs-point NOUN; the readout prepositions overlap (by) with within_by.
Recipe as v198/.../v232: per evaluated member, arm `own` (own C_fit control) and arm `fam` (own C_fit + EVERY sibling's A1 FIT prep,
control_weight 30 x len(controls)); evaluated on the member's held A1 (extraction), each sibling's held A1 (|CE| leak) and held C.
Units frozen from the parent receipts (v237/v239/v241 for the new members; earlier batteries for the counted ones).
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (v206 floor).
EVAL = 8 new members + 16 counted re-checks (24 fits x 2 arms; number_readout carries 13 controls, person 10, case 3, duration 2: ~45 min).
Count rule: each separable new EVAL member is a distinct circuit; a pair that leaks one way counts once; a re-checked counted member that
fails is RETRACTED (-1). 67 (after v237) -> up to 75.

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here; bars in BARS):
    pred_a_new_separable   at least 4 of the 8 new members are separable (k_new). he_him, lifts_lift, until_by and his_their are the
                           expected passes (distinct cue types from their siblings); myself_ourselves, myself_himself and me_him are the
                           fusion candidates (shared reflexive/object slots with counted plural members); mine_yours is the weak-extraction
                           case (0.578 own -- the 0.80 x own keep bar is easier at low extraction, so a pass there is read against the
                           0.50 floor, not celebrated).                                                                          prior 60%
    pred_b_counted_kept    at least 13 of the 16 re-checked counted members remain separable (k_kept; v232 kept 15/16).      prior 70%
    pred_c_row4_kept       fam-arm own-C UB975 <= 0.01 on at least 12 of the 24 evaluated members (k_d; v232 17/20).        prior 60%
    pred_d_instrument      every evaluated member's own-arm extraction is within 0.03 of its parent battery's cdas extraction_held
                           (same units, seed, split and code path: a determinism check, NOT an independent run outcome).  prior 90%
Priors are not moved by unit overlap (v214/v216/v220) nor by a shared readout token (withdrawn at v224).
Smoke: V234_SMOKE=<out.json> (CPU, V234_SMOKE_ROWS=4; all members are prepared as controls, V234_SMOKE_NAMES=case_he_him,relative_case_who_whose
are evaluated so a new member, a re-check and the instrument read are exercised; steps 5).
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
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v234_result.json"
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
FAMILIES = {
    "number_readout": ["reflexive_number", "pronoun_number", "demonstrative_number", "reciprocal", "possessive_number", "quantifier_number", "possessive_verbfinal", "object_pronoun_number", "possessive_number_its_their", "demonstrative_number_that_those", "pronoun_number_it_they", "number_reflexive_myself_ourselves", "agreement_lifts_lift", "possessive_number_his_their"],
    "person_readout": ["person_possessive", "person_possessive_plural", "reflexive_person", "object_pronoun_person", "reflexive_person_plural", "object_pronoun_person_plural", "person_reflexive_ourselves_themselves", "possessive_pronoun_mine_yours", "reflexive_person_myself_himself", "object_pronoun_person_me_him", "person_agreement_am_is"],
    "duration": ["duration_point_since_for", "duration_point_within_by", "durativity_until_by"],
    "case": ["relative_case_who_whose", "pronoun_case_coordination", "relative_case_who_whom", "case_he_him"],
}
EVAL = ("number_reflexive_myself_ourselves", "agreement_lifts_lift", "possessive_number_his_their", "possessive_pronoun_mine_yours", "reflexive_person_myself_himself", "object_pronoun_person_me_him", "durativity_until_by", "case_he_him", "reflexive_number", "demonstrative_number", "reciprocal", "possessive_number", "quantifier_number", "object_pronoun_number", "possessive_number_its_their", "demonstrative_number_that_those", "person_possessive", "person_possessive_plural", "object_pronoun_person_plural", "reflexive_person_plural", "person_reflexive_ourselves_themselves", "duration_point_since_for", "duration_point_within_by", "relative_case_who_whose", )
COUNTED = ("reflexive_number", "demonstrative_number", "reciprocal", "possessive_number", "quantifier_number", "object_pronoun_number", "possessive_number_its_their", "demonstrative_number_that_those", "person_possessive", "person_possessive_plural", "object_pronoun_person_plural", "reflexive_person_plural", "person_reflexive_ourselves_themselves", "duration_point_since_for", "duration_point_within_by", "relative_case_who_whose", )
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 12, "k_new": 4, "k_kept": 13, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v234", "members": n, "families": list(FAMILIES),
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
    smoke = os.environ.get("V234_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V234_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V234_SMOKE_NAMES", "case_he_him,relative_case_who_whose").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"]}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v234", "candidate_id": "corpus.unit_family_separability_spec_v234",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 ... v241 battery receipts (later receipts override earlier ones)", "evaluated": list(EVAL),
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
