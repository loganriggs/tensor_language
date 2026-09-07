#!/usr/bin/env python3
# BQGATE: five frozen predictions; families, units (v197 + v201 receipts), recipe and bars fixed before the run.
"""v202: within-family separability for the v201 NEW behaviours that have siblings in the one-protocol standing.

v201 runs the amended battery over nine new behaviours; three of them have family siblings already in the standing and cannot be
counted as circuits until a separability pass says so (standing rule since v114/v198): polarity_licensing (never_vs_often) vs
polarity_state; possessive_attractor (animate attractor, possessive_number.animate_attractor) vs the possessive circuit
(possessive_number + possessive_verbfinal stand in for it: v114 fused the five possessive constructions and v191 added
possessive_number); existential (were_vs_was) vs the number family (quantifier_number, coordination_agreement -- both separable in
v198). The other six new behaviours (countability, dative, modal_remoteness, voice_frame, verb_complementizer, verb_preposition)
have no sibling in the standing and are counted from v201 alone (cross-behaviour collateral of the hub sets was separable at zero
cost in v80). Magnitudes: no cross-collateral receipt exists for any of the three new members (they were never in v112's table);
v198's number-family fam-arm sibling max|CE| was 0.023-0.035 on the separable members and 0.13-0.15 on the fused pair.
Recipe = v198's (rank 1, 120 steps, lr 0.05, seed 0, complement 1.0, own-C + siblings' A1 FIT rows as controls at 30 each, mu =
pooled FIT mean; within-direction split), units FROZEN from the v197 receipt (old members) and the v201 receipt (new members).
A1 preps use g.prepare(valid_only=True) as v201 does (attractor drops 1 A1 row, existential 0; the v197 members drop none, so
their numbers are on the same rows as v198).
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction.
Members: polarity {polarity_state, polarity_licensing}; possessive {possessive_number, possessive_verbfinal, possessive_attractor};
number {quantifier_number, coordination_agreement, existential} = 8 members, 2 fits each, ~60 s/member -> ~8 min GPU.
Requires the v201 receipt (enqueue after it lands; a missing or error member is an error receipt and fails its predicates).

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here):
    pred_a_polarity_separable    separable on BOTH polarity members (licensing 'never/often' vs state).     prior 50%
    pred_b_attractor_fused       possessive_attractor is NOT separable from the possessive circuit.          prior 60%
    pred_c_existential_separable existential separable from the number family AND neither number member has fam-arm |CE|
                                 on existential's held A1 > 0.05.                                            prior 55%
    pred_d_row4_kept             fam-arm own-C UB975 <= 0.01 on >= 5 of 8 members (v198: 5 of 10).          prior 50%
    pred_e_instrument            possessive_number and possessive_verbfinal are NOT separable from each other (v114's fused
                                 possessive circuit) and quantifier_number, coordination_agreement ARE (v198). prior 65%
Circuit accounting: a separable member is its own circuit; a non-separable member joins its sibling's circuit as a construction.
Smoke: V202_SMOKE=<out.json> (CPU, V202_SMOKE_ROWS=4, V202_SMOKE_NAMES=polarity_state,polarity_licensing; steps 5).
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
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_new_v202_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
V201 = ROOT / "circuits/followups/unit_tier3_batch_amended_new9_v201_result.json"
FAMILIES = {
    "polarity": ["polarity_state", "polarity_licensing"],
    "possessive": ["possessive_number", "possessive_verbfinal", "possessive_attractor"],
    "number": ["quantifier_number", "coordination_agreement", "existential"],
}
NAMES = {**v197.NAMES, **v201.NAMES}
KNOWN_BAD, KNOWN_GOOD = ("possessive_number", "possessive_verbfinal"), ("quantifier_number", "coordination_agreement")
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"cross_max": 0.05, "keep": 0.80, "c_ub_max": 0.01, "k_d": 5, "leak_max": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = sum(len(v) for v in FAMILIES.values())
    return {"candidate_id": "corpus.unit_family_separability_new_v202", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    leak = sum(1 for n in FAMILIES["number"] if n != "existential" and n in good and abs(good[n]["arms"]["fam"]["siblings"].get("existential", 0.0)) > B["cross_max"])
    a = ok and all(sep(n) for n in FAMILIES["polarity"])
    b = ok and "possessive_attractor" in good and not good["possessive_attractor"]["separable"]
    c = ok and sep("existential") and leak <= B["leak_max"]
    d = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    e = ok and all(n in good and not good[n]["separable"] for n in KNOWN_BAD) and all(sep(n) for n in KNOWN_GOOD)
    return {"pred_a_polarity_separable": bool(a), "pred_b_attractor_fused": bool(b), "pred_c_existential_separable": bool(c),
            "pred_d_row4_kept": bool(d), "pred_e_instrument": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V202_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V202_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = [n for n in members if not smoke or n in os.environ.get("V202_SMOKE_NAMES", "polarity_state,polarity_licensing").split(",")]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"]}
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
                    raise RuntimeError(f"v197/v201 have no clean receipt for {n}")
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
    result = {"predictions": predictions, "schema": "unit_family_separability_new_v202", "candidate_id": "corpus.unit_family_separability_new_v202",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 (old members) + v201 (new members)",
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
