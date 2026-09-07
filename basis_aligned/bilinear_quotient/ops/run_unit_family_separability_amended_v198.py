#!/usr/bin/env python3
# BQGATE: five frozen predictions; families, units (v197 receipt), recipe and bars fixed before the run.
"""v198: within-family separability for the four-row passes v114 never tested, on the AMENDED (v197) sets.

v114 (05:19) decided the circuit count for 14 v112/v113 sets (possessive 5 -> 1, number 5 -> 4, correlative 3 -> 1). The amended
standing (v196/v197) adds four-row passes whose family structure is untested: the two LIST behaviours (numbered_list_choice,
numeric_sequence_choice: both greedy sets start attn:08:03 + attn:08:07 -- the 'attn8 writes the last salient item' mechanism),
the two DEGREE behaviours (degree_frame, degree_result: share attn:08:01, 07:08, 06:03), and animacy (who/which; no v112 cross
row exists, its nearest family by cue type is the number family on the hub heads 07:08 / 11:03).
Magnitudes printed from the v112 receipt BEFORE writing this (own-C-only fits, cross CE damage): degree_frame -> degree_result
0.017 / 0.012 back; numbered_list -> numeric_sequence 0.044 / -0.005 back; lexical_number_pp -> perfect_number 0.71 (the known
non-separable pair), quantifier -> narrative 0.001 (known separable). So the degree and list pairs are already near the 0.05
cross bar with NO sibling control; this rung asks whether they separate at ZERO extraction cost once the sibling is a control.
Recipe = v114's (rank 1, 120 steps, lr 0.05, seed 0, complement 1.0, own-C + siblings' A1 as controls at 30 each, mu = pooled
FIT mean), with two changes that follow v195-v197: units are FROZEN from the v197 receipt (amended sets), and the split is
within-direction (FIT rows[0::4]+rows[1::4] for every fit and control, HELD rows[2::4]+rows[3::4] for every evaluation).
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction.
Members: degree {degree_frame, degree_result}; list {numbered_list_choice, numeric_sequence_choice}; number+animacy
{lexical_number_pp, perfect_number, coordination_agreement, quantifier_number, narrative_tense, animacy} = 10 members, 2 fits
each, ~60 s/member (v114) -> ~10 min GPU. Requires the v197 receipt (enqueue after it lands; a missing member is an error receipt).

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here):
    pred_a_degree_separable      separable on BOTH degree members.                                     prior 75%
    pred_b_list_separable        separable on BOTH list members (same heads, different rank-1 axes).      prior 55%
    pred_c_animacy_separable     animacy separable from the number family AND at most 1 of the 5 number members has fam-arm
                                 |CE| on animacy's held A1 > 0.05.                                        prior 75%
    pred_d_row4_kept             fam-arm own-C UB975 <= 0.01 on >= 8 of 10 members.                      prior 60%
    pred_e_instrument            known-bad AND known-good reproduce with the amended units + one extra sibling: lexical_number_pp
                                 and perfect_number are NOT separable, and quantifier_number, narrative_tense,
                                 coordination_agreement ARE (v114's pattern).                             prior 70%
Circuit accounting: each separable member of a family is its own circuit; a non-separable pair is one circuit with two
constructions. Combined with v114 + v191 (possessive_number = sixth possessive construction) this fixes the amended count.
Smoke: V198_SMOKE=<out.json> (CPU, V198_SMOKE_ROWS=4, V198_SMOKE_NAMES=degree_frame,degree_result; steps 5).
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_amended_v198_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
FAMILIES = {
    "degree": ["degree_frame", "degree_result"],
    "list": ["numbered_list_choice", "numeric_sequence_choice"],
    "number_animacy": ["lexical_number_pp", "perfect_number", "coordination_agreement", "quantifier_number", "narrative_tense", "animacy"],
}
KNOWN_BAD, KNOWN_GOOD = ("lexical_number_pp", "perfect_number"), ("quantifier_number", "narrative_tense", "coordination_agreement")
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"cross_max": 0.05, "keep": 0.80, "c_ub_max": 0.01, "k_d": 8, "animacy_leak_max": 1}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = sum(len(v) for v in FAMILIES.values())
    return {"candidate_id": "corpus.unit_family_separability_amended_v198", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    leak = sum(1 for n in FAMILIES["number_animacy"] if n != "animacy" and n in good and abs(good[n]["arms"]["fam"]["siblings"].get("animacy", 0.0)) > B["cross_max"])
    a = ok and all(sep(n) for n in FAMILIES["degree"])
    b = ok and all(sep(n) for n in FAMILIES["list"])
    c = ok and sep("animacy") and leak <= B["animacy_leak_max"]
    d = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    e = ok and all(n in good and not good[n]["separable"] for n in KNOWN_BAD) and all(sep(n) for n in KNOWN_GOOD)
    return {"pred_a_degree_separable": bool(a), "pred_b_list_separable": bool(b), "pred_c_animacy_separable": bool(c),
            "pred_d_row4_kept": bool(d), "pred_e_instrument": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V198_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V198_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = [n for n in members if not smoke or n in os.environ.get("V198_SMOKE_NAMES", "degree_frame,degree_result").split(",")]
    prior = json.loads(V197.read_text())["behaviours"]
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]
    prep = {}
    for n in which:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{v197.NAMES[n]}")
        a1, c = g.rows_of(m, "A1"), g.rows_of(m, "C")
        prep[n] = {"fit": g.prepare(backend, cut(fit_half(a1))), "held": g.prepare(backend, cut(held_half(a1))),
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
                    raise RuntimeError(f"v197 has no clean receipt for {n}")
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
                    and arms["fam"]["extraction"] >= BARS["keep"] * arms["own"]["extraction"]
                R[n] = {"family": fam, "units": units, "n_units": len(units), "exact_held": round(exact, 3), "arms": arms, "separable": bool(sep),
                        "v197_rows": prior[n]["rows"], "seconds": round(time.perf_counter() - t1, 1)}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_amended_v198", "candidate_id": "corpus.unit_family_separability_amended_v198",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197",
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
