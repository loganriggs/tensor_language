#!/usr/bin/env python3
# BQGATE: eight frozen predictions; families, EVAL set, units (v201 + v207 + v209 + v213 + v215 receipts), recipe and bars fixed before the run.
"""v216: within-family separability for v215's passes -- the count test for aux_copy_so, aux_copy_ellipsis, doubt_belief_complementizer,
rarely_inversion, requested_subjunctive -- plus the re-evaluation of three COUNTED members under their now-enlarged families.

v215 put the seventh eight spec-authored behaviours through the four-row battery; its five four-row passes each have corpus siblings:
aux_copy_so / aux_copy_ellipsis (`... and so does/did`, `... and the X does/did` -- 11 of 13 units shared, NEW family, both members
new); doubt_belief_complementizer joins verb_complementizer (v201, not four-row: control) and noun_complementizer (v213, COUNTED at v214);
rarely_inversion joins negative_inversion (v207, COUNTED at v210), so_inversion (v209, control), neither_inversion (v209, COUNTED at
v210); requested_subjunctive joins mandative_subjunctive (v213, row 5 miss: control) and adjective_subjunctive (v215, row 4 miss:
control). Recipe as v198/v202/v206/v208/v210/v212/v214: per evaluated member, arm `own` (own C_fit control) and arm `fam` (own C_fit +
EVERY sibling's A1 FIT prep, control_weight 30 x len(controls)); evaluated on the member's held A1 (extraction), each sibling's held A1
(|CE| leak) and held C. Units frozen from the parent receipts.
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (v206 floor).
All twelve members are prepared; EVAL = the five v215 passes + negative_inversion, neither_inversion, noun_complementizer (counted
members re-checked under the enlarged control set, v212 precedent: a counted member that loses separability is RETRACTED).
~60-80 s per evaluated member (8 fits, ~10-12 min with the twelve prepares).
Count rule: each separable new EVAL member is a distinct circuit (33 -> up to 38); if aux_copy_so and aux_copy_ellipsis leak into each
other they are ONE aux-copy circuit with two surface constructions (counts +1). A re-checked counted member that fails is retracted (-1).
Prior lesson (v214): a shared unit list is NOT evidence against separability -- did_has_negation / modal_perfect_form shared four lead
units and separated at leak <= 0.022; priors here are not lowered for unit overlap.

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here):
    pred_a_aux_copy_so           aux_copy_so is separable from aux_copy_ellipsis.                                      prior 60%
    pred_b_aux_copy_ellipsis     aux_copy_ellipsis is separable from aux_copy_so.                                      prior 60%
    pred_c_doubt_belief          doubt_belief_complementizer is separable from verb_ and noun_complementizer.          prior 70%
    pred_d_rarely_inversion      rarely_inversion is separable from negative_, so_ and neither_inversion.              prior 65%
    pred_e_requested_subj        requested_subjunctive is separable from mandative_ and adjective_subjunctive.         prior 65%
    pred_f_counted_kept          negative_inversion, neither_inversion and noun_complementizer ALL remain separable under
                                 their enlarged families (each was separable at v210 / v214 against the smaller set).  prior 75%
    pred_g_row4_kept             fam-arm own-C UB975 <= 0.01 on >= 5 of the 8 evaluated members.                      prior 60%
    pred_h_instrument            every evaluated member's own-arm extraction is within +-0.03 of its parent-receipt row-4 (cdas)
                                 extraction_held (1.013 / 0.993 / 0.981 / 0.998 / 0.998 / 0.997 / 0.999 / 1.003; same units, seed,
                                 split and code path -- a determinism check, NOT an independent run outcome).          prior 90%
Smoke: V216_SMOKE=<out.json> (CPU, V216_SMOKE_ROWS=4; all twelve members are prepared as controls, V216_SMOKE_NAMES=aux_copy_so,noun_complementizer
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
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v216_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
V201 = ROOT / "circuits/followups/unit_tier3_batch_amended_new9_v201_result.json"
V204 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8_v204_result.json"
V205 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8b_v205_result.json"
V207 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8c_v207_result.json"
V209 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8d_v209_result.json"
V211 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8e_v211_result.json"
V213 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8f_v213_result.json"
V215 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8g_v215_result.json"
FAMILIES = {
    "aux_copy": ["aux_copy_so", "aux_copy_ellipsis"],
    "complementizer": ["verb_complementizer", "noun_complementizer", "doubt_belief_complementizer"],
    "inversion": ["negative_inversion", "so_inversion", "neither_inversion", "rarely_inversion"],
    "mood": ["mandative_subjunctive", "adjective_subjunctive", "requested_subjunctive"],
}
EVAL = ("aux_copy_so", "aux_copy_ellipsis", "doubt_belief_complementizer", "rarely_inversion", "requested_subjunctive",
        "negative_inversion", "neither_inversion", "noun_complementizer")
COUNTED = ("negative_inversion", "neither_inversion", "noun_complementizer")
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V215, V213, V209, V207):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 5, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v216", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    a = ok and sep("aux_copy_so")
    b = ok and sep("aux_copy_ellipsis")
    c = ok and sep("doubt_belief_complementizer")
    d = ok and sep("rarely_inversion")
    e = ok and sep("requested_subjunctive")
    f = ok and all(sep(n) for n in COUNTED)
    gg = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    h = ok and all(n in good and good[n]["arms"]["own"]["extraction"] is not None
                   and abs(good[n]["arms"]["own"]["extraction"] - _parent_cdas(n)) <= B["instr_tol"] for n in EVAL)
    return {"pred_a_aux_copy_so": bool(a), "pred_b_aux_copy_ellipsis": bool(b), "pred_c_doubt_belief": bool(c),
            "pred_d_rarely_inversion": bool(d), "pred_e_requested_subj": bool(e), "pred_f_counted_kept": bool(f),
            "pred_g_row4_kept": bool(gg), "pred_h_instrument": bool(h)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V216_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V216_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V216_SMOKE_NAMES", "aux_copy_so,noun_complementizer").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"]}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v216", "candidate_id": "corpus.unit_family_separability_spec_v216",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 + v201 + v204 + v205 + v207 + v209 + v211 + v213 + v215 receipts (later receipts override earlier ones)", "evaluated": list(EVAL),
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
