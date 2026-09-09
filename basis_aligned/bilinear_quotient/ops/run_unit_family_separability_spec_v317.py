#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v317: verb_particle separability at SEVEN prepared -- two new passes, and the first rung where this family's own-arm
leak can be read across five evaluated members.

WHY. v313 counted verb_particle_out_up (128 -> 129) and, more usefully, showed WHY it counted: in the own arm -- own C
only, no sibling controls -- the particle members leak into each other at own sib-max 0.30 (out_up) and 0.47
(out_down), and the family controls drive those to <= 0.0096 while extraction moves 1.03 -> 1.006 and 0.992 -> 0.996.
This family is not separable because its members never touch; it is separable because a sibling-respecting rank-1
direction exists at no extraction cost. v315 then produced two more four-row passes (away_up n13 held 0.893, down_in
n9 held 0.922) and one three-row cell (on_out, row-4 miss), so the family reaches SEVEN prepared here: three counted
(up_down, out_down, out_up), two new, and two three-row controls (up_on from v311, on_out from v315).
COST. v313 ran four prepared in 171 s with zero errors. The measured verb_preposition curve (s/member = 19 + 9.3 x
family_size) puts seven at ~84 s/member, ~10 min -- an eighth of the 55-to-72-member sizes where verb_preposition
lost members to CUDA OOM. This is still the cheap lane, and it is the only lane that can count while proposal (ix) is
unanswered.
WHAT THE OWN ARM WILL SHOW. With five evaluated members the own-arm sib-max is no longer one number per pair: it is a
5 x 4 leak table for a family whose members share readout tokens in a known pattern (` up` in up_down/out_up/away_up,
` down` in up_down/out_down/down_in, ` out` in out_down/out_up/on_out, ` in` in down_in, ` away` in away_up, ` on` in
on_out). If the own-arm leak tracks SHARED READOUT TOKEN rather than family membership, that is the first quantitative
test of the cue-overlap account inside a single family, where every member shares the frame, the cue class and the P
control and only the token pair differs. I am NOT registering a bar on it: the account was built from six fusions and
the honest move is to read this table first and register the bar on the family that follows.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
    pred_a_new_separable   at least k_new = 1 of the 2 new members (away_up, down_in) is separable at the registered
                           cross bar. The family is 3 for 3 so far, but this is the first rung where a new member has
                           four siblings rather than two, and the control set grows with it.              prior 80%
    pred_b_counted_kept    at least k_kept = 2 of the 3 re-checked counted members remain separable. Their v313 fam
                           margins were 0.0096, 0.0049 and comparable, all far from the 0.05 bar; a loss here would be
                           the control-set-growth effect v309 established for adjective_preposition_for_toward, not a
                           fusion.                                                                        prior 85%
    pred_c_row4_kept       fam-arm own-C UB975 <= 0.01 on at least k_d = 2 of the 5 evaluated members.     prior 70%
    pred_d_instrument      every evaluated member's own-arm extraction is within 0.03 of its parent battery's cdas
                           extraction_held. Fails automatically if any member errors; at seven prepared an OOM would
                           put the wall four times lower than every measurement so far and the memory account would
                           need rewriting.                                                                prior 90%
COUNTING. Each separable new member is +1 on 129. A fused member joins its incumbent (+0) and, because these cells
were cue-verified fresh, would be the first fusion in the corpus with no cue-overlap explanation -- which would then
get the v309 control-set ablation before anything is claimed from it.
BARS = {"cross_max": 0.05, "keep": 0.8, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 2, "k_new": 1, "k_kept": 2, "instr_tol": 0.03}
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
import run_unit_tier3_batch_amended_new9_v201 as v201
import run_unit_tier3_batch_amended_spec9p_v311 as v311
import run_unit_tier3_batch_amended_spec9p_v315 as v315

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v317_result.json"
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
FAMILIES = {
    "verb_particle": ["verb_particle_up_down", "verb_particle_out_down", "verb_particle_out_up", "verb_particle_up_on", "verb_particle_away_up", "verb_particle_down_in", "verb_particle_on_out"],
}
EVAL = ("verb_particle_away_up", "verb_particle_down_in", "verb_particle_up_down", "verb_particle_out_down", "verb_particle_out_up", )
COUNTED = ("verb_particle_up_down", "verb_particle_out_down", "verb_particle_out_up", )
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES, **v243.NAMES, **v245.NAMES, **v247.NAMES, **v249.NAMES, **v251.NAMES, **v253.NAMES, **v255.NAMES, **v257.NAMES, **v311.NAMES, **v315.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V315, V311, V257, V255, V253, V251, V249, V247, V245, V243, V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
BARS = {"cross_max": 0.05, "keep": 0.8, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 2, "k_new": 1, "k_kept": 2, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v317", "members": n, "families": list(FAMILIES),
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
    smoke = os.environ.get("V317_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V317_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V317_SMOKE_NAMES", "verb_preposition_at_to,verb_preposition").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"], **json.loads(V243.read_text())["behaviours"], **json.loads(V245.read_text())["behaviours"], **json.loads(V247.read_text())["behaviours"], **json.loads(V249.read_text())["behaviours"], **json.loads(V251.read_text())["behaviours"], **json.loads(V253.read_text())["behaviours"], **json.loads(V255.read_text())["behaviours"], **json.loads(V257.read_text())["behaviours"], **json.loads(V311.read_text())["behaviours"], **json.loads(V315.read_text())["behaviours"]}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v317", "candidate_id": "corpus.unit_family_separability_spec_v317",
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
