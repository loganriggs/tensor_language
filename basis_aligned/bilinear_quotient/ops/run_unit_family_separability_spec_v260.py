#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v260: the ADJECTIVE arm of the family-vs-cue 2x2. Five new adjective_preposition cells, each of which REUSES ONE CUE
ADJECTIVE from a counted sibling at the same preposition, evaluated against the full adjective family (10 counted, 1
control). The verb arm (v258) is already in, and it came back MIXED -- so this rung is what decides whether the split
is about the family or about something else.

WHERE THIS SITS. v254/v256 found four fusing pairs, all sharing a (cue word -> readout token) mapping and all in
verb_preposition; enumerating every such shared mapping in the corpus turned up four MORE that do not fuse, three of
them adjective or cross-family, including adjective_preposition_similar / adjective_preposition_to_of which share BOTH
the cue `similar` and the readout ` to` and have been separable on every rung since v222. So the cue account is not
sufficient (11:52 ledger row, board note) and the surviving candidate was a FAMILY split.
v258 then ran the verb arm as a pre-registered test and split: with_beyond (reusing `dealt`) fused with with_about at
0.056, in_beyond (reusing `resulted`) stayed separable from in_to at 0.016. Registered as unresolved.
v258 ALSO turned up an asymmetry I did not predict and am not explaining here: the reverse leaks are 3-13x larger
(in_to -> in_beyond 0.213 vs in_beyond -> in_to 0.016; with_about -> with_beyond 0.193 vs 0.056; to_at -> at_under
0.151). The incumbent's fitted direction moves a lot when the new cell joins its control set, while the new cell's
direction barely notices the incumbent. That is a property of the fit, not necessarily of the model, and the rung that
separates those two is an ablation of the CONTROL SET (re-fit the incumbent with the new cell removed), not this one.
It is queued behind this rung, not folded into it.

THIS RUNG. Family: adjective_preposition, 10 counted after v254 (fond, similar, identical_distinct, of_at, to_of,
for_of, in_about, of_about, at_about, of_against; control same_different_preposition).
  + adjective_preposition_of_within (FOND/enclosed -> of/within);      `fond` is adjective_preposition_fond's ` of` cue
  + adjective_preposition_in_beneath (INTERESTED/buried -> in/beneath); `interested` is in_about's ` in` cue
  + adjective_preposition_at_within (SKILLED/contained -> at/within);   `skilled` is at_about's ` at` cue
  + adjective_preposition_of_toward (ASHAMED/hostile -> of/toward);     `ashamed` is of_about's ` of` cue
  + adjective_preposition_for_beneath (RESPONSIBLE/hidden -> for/beneath); `responsible` is for_of's ` for` cue
All five passed all four rows at v269 (6/6 preds, 5/5 four-row, held 0.83-0.86, cext 0.981-1.014, no hollow passes),
so the battery cannot be blamed for whatever this rung finds.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
    pred_a_new_separable   at least 3 of the 5 new members are separable (k_new). THE FAMILY ACCOUNT PREDICTS 5 of 5:
                           cue reuse fuses in the verb family and not in the adjective family. The CUE account predicts
                           0 of 5. This bar sits between them so that it discriminates rather than confirms; 3 or 4
                           separable is the messy middle and gets reported as such.                     prior 60%
    pred_b_counted_kept    at least 8 of the 10 re-checked counted members remain separable (k_kept). NOTE the verb arm
                           lost FOUR incumbents this way at v258, so this is not a formality: if the adjective
                           incumbents also destabilise when a cue-sharing cell joins their control set, the asymmetry
                           is general and the family split is about the NEW cells only.                 prior 55%
    pred_c_row4_kept       fam-arm own-C UB975 <= 0.01 on at least 7 of the 15 evaluated members (k_d; v258 11/22). prior 55%
    pred_d_instrument      every evaluated member's own-arm extraction is within 0.03 of its parent battery's cdas
                           extraction_held (same units, seed, split and code path: a determinism check, NOT an
                           independent run outcome).                                                    prior 90%
Reading rule, fixed now: pred_a at 5/5 with pred_b at 10/10 is the family account; pred_a at 0-1/5 is the cue account;
anything else is unresolved and will be reported as unresolved rather than narrated toward either.
COUNTING. Under the v242 rule a new cell separable from every sibling counts +1, and one that leaks into a counted
incumbent joins it, +0. So the family account here is worth up to +5 (104 -> 109) and the cue account is worth 0. That
the two accounts differ in the COUNT as well as the mechanism is stated so it is visible that I have a stake in the
outcome; the bars were set before the run and are not moving.
Recipe as v198/.../v258. Units frozen from the parent receipts (v269 for the new members; earlier batteries for the rest).
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (v206 floor).
EVAL = 5 new + 10 counted re-checks (15 fits x 2 arms, 15 controls in the fam arm: ~60-75 min; the full re-check stands
because board proposal (ix) is unanswered).
Smoke: V260_SMOKE=<out.json> (CPU, V260_SMOKE_ROWS=4; V260_SMOKE_NAMES = a new member plus a re-check; steps 5).
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 7, "k_new": 3, "k_kept": 8, "instr_tol": 0.03}
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
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v260_result.json"
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
    "adjective_preposition": ["adjective_preposition_fond", "adjective_preposition_similar", "identical_distinct_preposition", "adjective_preposition_of_at", "adjective_preposition_to_of", "adjective_preposition_for_of", "same_different_preposition", "adjective_preposition_in_about", "adjective_preposition_of_about", "adjective_preposition_at_about", "adjective_preposition_of_against", "adjective_preposition_at_within", "adjective_preposition_for_beneath", "adjective_preposition_in_beneath", "adjective_preposition_of_toward", "adjective_preposition_of_within"],
}
EVAL = ("adjective_preposition_at_within", "adjective_preposition_for_beneath", "adjective_preposition_in_beneath", "adjective_preposition_of_toward", "adjective_preposition_of_within", "adjective_preposition_fond", "adjective_preposition_similar", "identical_distinct_preposition", "adjective_preposition_of_at", "adjective_preposition_to_of", "adjective_preposition_for_of", "adjective_preposition_in_about", "adjective_preposition_of_about", "adjective_preposition_at_about", "adjective_preposition_of_against", )
COUNTED = ("adjective_preposition_fond", "adjective_preposition_similar", "identical_distinct_preposition", "adjective_preposition_of_at", "adjective_preposition_to_of", "adjective_preposition_for_of", "adjective_preposition_in_about", "adjective_preposition_of_about", "adjective_preposition_at_about", "adjective_preposition_of_against", )
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
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 7, "k_new": 3, "k_kept": 8, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v260", "members": n, "families": list(FAMILIES),
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
    smoke = os.environ.get("V260_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V260_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V260_SMOKE_NAMES", "adjective_preposition_at_within,adjective_preposition_fond").split(","))]
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v260", "candidate_id": "corpus.unit_family_separability_spec_v260",
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
