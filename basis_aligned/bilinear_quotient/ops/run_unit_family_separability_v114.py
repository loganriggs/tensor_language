#!/usr/bin/env python3
# BQGATE: five frozen predictions; families, units (v112 receipt), recipe and bars fixed before the run.
"""v114: are the v112 cross-matrix families one circuit each, or separable behaviours at rank 1?

v112's 21x21 cross-collateral matrix (own-C-only fits) is family-structured: possessive x5 damage each other 0.35-0.64,
{lexical_number_pp, perfect_number, coordination_agreement, quantifier_number, narrative_tense} 0.42-0.71 (narrative ->
perfect with opposed sign), and correlative x3 + polarity_state 0.51-0.88; cross-family pairs are <= 0.05 with one
exception. v80 showed cross-behaviour collateral is separable at rank 1 when the other behaviours enter the objective
(cross-family). Here the same recipe is applied WITHIN each family: for each member, arm `own` = own-C EVEN control only
(v112's fit, the instrument) and arm `fam` = own-C EVEN + every sibling's A1 EVEN as controls (30 each), rank 1,
120 steps, lr 0.05, seed 0, complement 1.0, mu = pooled EVEN mean; head sets are frozen from the v112 receipt (no greedy).
Evaluated on ODD: own A1 extraction (fraction of exact), own-C CE-damage UB, max |CE damage| over siblings' ODD A1.
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction.
A family whose members are NOT separable shares its rank-1 axis: one circuit with several surface variants.

REGISTERED BEFORE THE RUN
    pred_a_possessive_one_circuit   separable on <= 1 of the 5 possessive members.    Worked: 0 True; 3 False.
    pred_b_number_separable         separable on >= 3 of the 5 number members.        Worked: 3 True; 1 False.
    pred_c_correlative_separable    separable on >= 2 of the 4 correlative members.   Worked: 2 True; 1 False.
    pred_d_row4_kept                fam-arm own-C UB <= 0.01 on >= 10 of 14.           Worked: 11 True; 7 False.
    pred_e_instrument               own arm reproduces v112's cdas arm on 14/14: own extraction within 0.05 and own-C UB
                                    within 0.01 (same units, recipe, seed). Worked: 0.83 vs 0.85 True; 0.70 vs 0.85 False.
    Prior: a 60%; b 35% (v54/hub-heads: the number axis is shared within the number family); c 40%; d 65%; e 85%.
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
import run_unit_tier3_batch_v112 as v112

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_v114_result.json"
V112 = ROOT / "circuits/followups/unit_tier3_batch_v112_result.json"
FAMILIES = {
    "possessive": ["possessive_adjacent", "possessive_argument", "possessive_long_simple", "possessive_medial", "possessive_verbfinal"],
    "number": ["lexical_number_pp", "perfect_number", "coordination_agreement", "quantifier_number", "narrative_tense"],
    "correlative": ["correlative_both_either", "correlative_both_neither", "correlative_either_neither", "polarity_state"],
}
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
CROSS_MAX, KEEP, C_UB_MAX, INSTR_EXT, INSTR_C, K_A, K_B, K_C, K_D = 0.05, 0.80, 0.01, 0.05, 0.01, 1, 3, 2, 10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 12000, 400000


def _plan():
    n = sum(len(v) for v in FAMILIES.values())
    return {"candidate_id": "corpus.unit_family_separability_v114", "members": n,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": n * 2 * STEPS, "model_updates": 0, "fit_parameters": n * 2 * 14 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    prior = json.loads(V112.read_text())["behaviours"]
    members = [m for f in FAMILIES.values() for m in f]
    modules = {n: importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}") for n in members}
    prep = {}
    for n, m in modules.items():
        a1 = g.rows_of(m, "A1")
        prep[n] = {"even": g.prepare(backend, a1[0::2]), "odd": g.prepare(backend, a1[1::2]),
                   "C": g.prepare(backend, g.rows_of(m, "C")[1::2]), "C_even": g.prepare(backend, g.rows_of(m, "C")[0::2])}

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def ce(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975")}

    R = {}
    for fam, mem in FAMILIES.items():
        for n in mem:
            t1 = time.perf_counter()
            units = list(prior[n]["units"])
            P = prep[n]
            mu = mu_of(P["even"], units)
            exact = g.recovery(P["odd"], g.patched_axis(backend, P["odd"], units))
            sibs = [s for s in mem if s != n]
            arms = {}
            for arm, controls in (("own", (P["C_even"],)), ("fam", (P["C_even"],) + tuple(prep[s]["even"] for s in sibs))):
                q, hist = g.fit_block_subspace_constrained(backend, P["even"], units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                           controls=controls, control_weight=LAM * len(controls), mu=mu)
                a = {"extraction": round(g.recovery(P["odd"], g.patched_axis(backend, P["odd"], units, q=q)) / exact, 3) if abs(exact) > 1e-6 else None,
                     "A1": ce(P["odd"], units, q, mu), "C": ce(P["C"], units, q, mu),
                     "siblings": {s: ce(prep[s]["odd"], units, q, mu)["ce_damage"] for s in sibs}}
                a["sib_abs_max"] = round(max(abs(v) for v in a["siblings"].values()), 4)
                arms[arm] = a
            sep = arms["fam"]["sib_abs_max"] <= CROSS_MAX and arms["fam"]["extraction"] is not None and arms["own"]["extraction"] \
                and arms["fam"]["extraction"] >= KEEP * arms["own"]["extraction"]
            v = prior[n]["arms"]["cdas"]
            R[n] = {"family": fam, "units": units, "exact_odd": round(exact, 3), "arms": arms, "separable": bool(sep),
                    "v112_cdas": {"extraction": v["extraction_odd"], "C_ub": v["C"]["ce_ub975"]}, "seconds": round(time.perf_counter() - t1, 1)}
            print(fam, n, "own", (arms["own"]["extraction"], arms["own"]["sib_abs_max"], arms["own"]["C"]["ce_ub975"]),
                  "fam", (arms["fam"]["extraction"], arms["fam"]["sib_abs_max"], arms["fam"]["C"]["ce_ub975"]), "sep", sep, round(time.perf_counter() - t0), "s", flush=True)

    nsep = {fam: sum(R[n]["separable"] for n in mem) for fam, mem in FAMILIES.items()}
    predictions = {
        'pred_a_possessive_one_circuit': nsep["possessive"] <= K_A,
        'pred_b_number_separable': nsep["number"] >= K_B,
        'pred_c_correlative_separable': nsep["correlative"] >= K_C,
        'pred_d_row4_kept': sum(R[n]["arms"]["fam"]["C"]["ce_ub975"] <= C_UB_MAX for n in R) >= K_D,
        'pred_e_instrument': all(R[n]["arms"]["own"]["extraction"] is not None and R[n]["v112_cdas"]["extraction"] is not None
                                 and abs(R[n]["arms"]["own"]["extraction"] - R[n]["v112_cdas"]["extraction"]) <= INSTR_EXT
                                 and abs(R[n]["arms"]["own"]["C"]["ce_ub975"] - R[n]["v112_cdas"]["C_ub"]) <= INSTR_C for n in R),
    }
    summary = {n: {"family": R[n]["family"], "separable": R[n]["separable"],
                   "own": (R[n]["arms"]["own"]["extraction"], R[n]["arms"]["own"]["sib_abs_max"], R[n]["arms"]["own"]["C"]["ce_ub975"]),
                   "fam": (R[n]["arms"]["fam"]["extraction"], R[n]["arms"]["fam"]["sib_abs_max"], R[n]["arms"]["fam"]["C"]["ce_ub975"])} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_family_separability_result_v1", "candidate_id": "corpus.unit_family_separability_v114",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R,
              "recipe": {"lambda": LAM, "steps": STEPS, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v112"},
              "bars": {"cross_max": CROSS_MAX, "keep": KEEP, "c_ub_max": C_UB_MAX, "instr_ext": INSTR_EXT, "instr_c": INSTR_C, "k": {"a": K_A, "b": K_B, "c": K_C, "d": K_D}},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
