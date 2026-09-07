#!/usr/bin/env python3
# BQGATE: six frozen predictions; families, units (v197 + v201 + v204 + v205 receipts), recipe and bars fixed before the run.
"""v206: within-family separability for the v204/v205 spec-authored behaviours that have siblings in the standing.

v204 put reflexive_number, pronoun_number and demonstrative_number through all four rows; each reads the subject's NUMBER into a
different form (reflexive / subject pronoun / pro-form) and shares hub heads (attn 10:01, 15:01, 12:04, 09:06) with possessive_number
and quantifier_number, which are already in the standing. The gender pair (reflexive_gender four-row-minus-row-4, pronoun_gender
minus row 5) shares the same heads, as do v205's four-row object_pronoun_gender and possessive_gender. v205 also added
reciprocal (each/himself: a number readout in yet another form) and equative_result (as/less: comparative_frame's sibling). polarity_addition (too/either) is a sibling of polarity_licensing (anything/something). None can
be counted as a circuit before a separability pass (standing rule since v114/v198). Recipe as v198/v202: per member, arm `own`
(own C_fit control) and arm `fam` (own C_fit + every sibling's A1 FIT prep, control_weight 30 x len(controls)); evaluated on the
member's held A1 (extraction), each sibling's held A1 (|CE| leak) and held C. Units frozen from the v197/v201/v204/v205 receipts.
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (new floor: v202 called existential 'separable' at extraction 0.10 --
                         a direction that extracts nothing leaks nothing; the floor makes the verdict non-vacuous).
Families: number_readout = reflexive_number, pronoun_number, demonstrative_number, reciprocal, possessive_number,
quantifier_number, possessive_verbfinal (the last for the instrument); gender = reflexive_gender, pronoun_gender,
object_pronoun_gender, possessive_gender; polarity = polarity_addition, polarity_licensing; degree = comparative_frame,
equative_result. 15 members, ~55-90 s each (the fam arm carries one control prep per sibling).

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here):
    pred_a_gender_separable      separable on >= 3 of the 4 gender members (same cue, different readout; shared hub
                                 heads; v205's two are four-row, v204's two each miss one row).                    prior 45%
    pred_b_number_readouts       at least 3 of the 4 new number readouts (reflexive/pronoun/demonstrative/reciprocal)
                                 separable from the number_readout family.                                         prior 45%
    pred_c_addition_separable    polarity_addition separable from polarity_licensing.                              prior 55%
    pred_d_row4_kept             fam-arm own-C UB975 <= 0.01 on >= 8 of 15 members (v202: 4 of 8).                 prior 45%
    pred_f_degree_separable      equative_result separable from comparative_frame (same frame, as/less vs less/as). prior 50%
    pred_e_instrument            possessive_number and possessive_verbfinal are NOT separable from each other (v114/v202)
                                 and quantifier_number IS (v198/v202).                                              prior 70%
Circuit accounting: a separable four-row member is its own circuit; a non-separable member joins its sibling's circuit as a
construction; members that missed a row in v204/v205 are reported but not counted either way.
Smoke: V206_SMOKE=<out.json> (CPU, V206_SMOKE_ROWS=4, V206_SMOKE_NAMES=reflexive_gender,pronoun_gender; steps 5).
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
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v206_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
V201 = ROOT / "circuits/followups/unit_tier3_batch_amended_new9_v201_result.json"
V204 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8_v204_result.json"
V205 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8b_v205_result.json"
FAMILIES = {
    "number_readout": ["reflexive_number", "pronoun_number", "demonstrative_number", "reciprocal", "possessive_number", "quantifier_number", "possessive_verbfinal"],
    "gender": ["reflexive_gender", "pronoun_gender", "object_pronoun_gender", "possessive_gender"],
    "polarity": ["polarity_addition", "polarity_licensing"],
    "degree": ["comparative_frame", "equative_result"],
}
NEW_NUMBER = ("reflexive_number", "pronoun_number", "demonstrative_number", "reciprocal")
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES}
KNOWN_BAD, KNOWN_GOOD = ("possessive_number", "possessive_verbfinal"), ("quantifier_number",)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_d": 8, "k_b": 3, "k_a": 3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = sum(len(v) for v in FAMILIES.values())
    return {"candidate_id": "corpus.unit_family_separability_spec_v206", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    a = ok and sum(1 for n in FAMILIES["gender"] if sep(n)) >= B["k_a"]
    b = ok and sum(1 for n in NEW_NUMBER if sep(n)) >= B["k_b"]
    c = ok and sep("polarity_addition")
    d = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    e = ok and all(n in good and not good[n]["separable"] for n in KNOWN_BAD) and all(sep(n) for n in KNOWN_GOOD)
    return {"pred_a_gender_separable": bool(a), "pred_b_number_readouts": bool(b), "pred_c_addition_separable": bool(c),
            "pred_d_row4_kept": bool(d), "pred_e_instrument": bool(e), "pred_f_degree_separable": bool(ok and sep("equative_result"))}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V206_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V206_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = [n for n in members if not smoke or n in os.environ.get("V206_SMOKE_NAMES", "reflexive_gender,pronoun_gender").split(",")]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"]}
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
            if n not in which:
                continue
            t1 = time.perf_counter()
            try:
                if "error" in prior.get(n, {"error": "missing"}):
                    raise RuntimeError(f"v197/v201/v204/v205 have no clean receipt for {n}")
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v206", "candidate_id": "corpus.unit_family_separability_spec_v206",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 + v201 + v204 + v205 receipts (later receipts override earlier ones)",
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
