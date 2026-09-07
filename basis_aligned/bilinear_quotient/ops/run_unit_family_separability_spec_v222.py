#!/usr/bin/env python3
# BQGATE: eight frozen predictions; families, EVAL set, units (v197 + v201 + v209 + v213 + v215 + v217 + v219 + v221 receipts), recipe and bars fixed before the run.
"""v222: within-family separability for v221's passes (comparative_complement_superior, comparative_complement_inferior, noun_preposition_interest, noun_preposition_belief, relative_clause_agreement) plus the re-evaluation of SIX counted members whose families grew.

v221 put the tenth eight spec-authored behaviours through the four-row battery. Families here: degree_complement = superlative_complement_best
(COUNTED at v220) + superlative_complement_largest (control) + comparative_complement_superior + comparative_complement_inferior;
noun_preposition = interest + belief (NEW family, no counted member); number_agreement = partitive_agreement, perfect_number,
lexical_number_pp (the v210-fused trio), number_of_agreement (COUNTED at v220), relative_clause_agreement, proximity_agreement_nor;
complementizer = verb_complementizer (control), noun_complementizer, doubt_belief_complementizer, for_complementizer_waited,
for_complementizer_arranged (all four COUNTED) + purpose_vs_reason; case = pronoun_case_coordination (v217 miss, control) +
relative_case_who_whom. v221 misses stay in their families as controls (a family grows when it gains a control too), so every counted
member of a grown family is re-checked (v212/v216/v220 precedent). Recipe as v198/.../v220: per evaluated member, arm `own` (own C_fit
control) and arm `fam` (own C_fit + EVERY sibling's A1 FIT prep, control_weight 30 x len(controls)); evaluated on the member's held A1
(extraction), each sibling's held A1 (|CE| leak) and held C. Units frozen from the parent receipts.
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (v206 floor).
All members are prepared; EVAL = the 5 v221 pass(es) + the six counted members. ~60-80 s per evaluated member.
Count rule: each separable new EVAL member is a distinct circuit (44 -> up to 49); a pair that leaks one way counts once (v216
aux-copy precedent); a re-checked counted member that fails is RETRACTED (-1 each).

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here):
    pred_a_comp_superior            comparative_complement_superior is separable from every other member of its family.            prior 50%
    pred_b_comp_inferior            comparative_complement_inferior is separable from every other member of its family.            prior 50%
    pred_c_np_interest              noun_preposition_interest is separable from every other member of its family.            prior 50%
    pred_d_np_belief                noun_preposition_belief is separable from every other member of its family.            prior 50%
    pred_e_relclause_agr            relative_clause_agreement is separable from every other member of its family.            prior 50%
    pred_f_counted_kept          all six counted members remain separable under their enlarged families.                prior 55%
    pred_g_row4_kept             fam-arm own-C UB975 <= 0.01 on >= 6 of the 11 evaluated members.                        prior 60%
    pred_h_instrument            every evaluated member's own-arm extraction is within +-0.03 of its parent-receipt row-4 (cdas)
                                 extraction_held (0.998/0.993/0.997/1.003/0.998/1.010/1.012/1.003/0.981/1.000/1.019; same units, seed, split
                                 and code path -- a determinism check, NOT an independent run outcome).                prior 90%
Priors are not moved by unit overlap in either direction (v214 separated shared-unit members; v216 fused them; v220 separated number_of
from a fused trio).
Smoke: V222_SMOKE=<out.json> (CPU, V222_SMOKE_ROWS=4; all members are prepared as controls, V222_SMOKE_NAMES=comparative_complement_superior,noun_complementizer
are evaluated so a new family, a re-check and the instrument read are exercised; steps 5).
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
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v222_result.json"
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
FAMILIES = {
    "degree_complement": ["superlative_complement_best", "superlative_complement_largest", "comparative_complement_superior", "comparative_complement_inferior"],
    "noun_preposition": ["noun_preposition_interest", "noun_preposition_belief"],
    "number_agreement": ["partitive_agreement", "perfect_number", "lexical_number_pp", "number_of_agreement", "relative_clause_agreement", "proximity_agreement_nor"],
    "complementizer": ["verb_complementizer", "noun_complementizer", "doubt_belief_complementizer", "for_complementizer_waited", "for_complementizer_arranged", "purpose_vs_reason"],
}
EVAL = ("comparative_complement_superior", "comparative_complement_inferior", "noun_preposition_interest", "noun_preposition_belief", "relative_clause_agreement", "superlative_complement_best", "number_of_agreement", "noun_complementizer", "doubt_belief_complementizer", "for_complementizer_waited", "for_complementizer_arranged",)
COUNTED = ("superlative_complement_best", "number_of_agreement", "noun_complementizer", "doubt_belief_complementizer", "for_complementizer_waited", "for_complementizer_arranged",)
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V221, V219, V215, V213):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 6, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v222", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    a = ok and sep("comparative_complement_superior")
    b = ok and sep("comparative_complement_inferior")
    c = ok and sep("noun_preposition_interest")
    d = ok and sep("noun_preposition_belief")
    e = ok and sep("relative_clause_agreement")
    f = ok and all(sep(n) for n in COUNTED)
    gg = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    h = ok and all(n in good and good[n]["arms"]["own"]["extraction"] is not None
                   and abs(good[n]["arms"]["own"]["extraction"] - _parent_cdas(n)) <= B["instr_tol"] for n in EVAL)
    return {"pred_a_comp_superior": bool(a), "pred_b_comp_inferior": bool(b), "pred_c_np_interest": bool(c), "pred_d_np_belief": bool(d), "pred_e_relclause_agr": bool(e), "pred_f_counted_kept": bool(f),
            "pred_g_row4_kept": bool(gg), "pred_h_instrument": bool(h)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V222_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V222_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V222_SMOKE_NAMES", "comparative_complement_superior,noun_complementizer").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"]}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v222", "candidate_id": "corpus.unit_family_separability_spec_v222",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 + v201 + v204 + v205 + v207 + v209 + v211 + v213 + v215 + v217 + v219 + v221 receipts (later receipts override earlier ones)", "evaluated": list(EVAL),
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
