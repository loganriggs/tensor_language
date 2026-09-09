#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v325: THE CUE HALF OF THE 2x2. verb_particle separability at THIRTEEN prepared, with three DESIGNED-TO-FUSE cells
whose cue -> token mappings are reused verbatim from counted circuits.

WHERE THIS SITS. v321 settled the other half and settled it against me: three cells sharing their ENTIRE readout pair
with a counted circuit -- differing only in the cue verbs -- were all separable from their originals (fam-arm leak into
the original -0.0033, -0.0076, -0.0093), with own-arm leak reaching 0.4067. A behaviour's identity is finer than its
token pair, and my authoring gate has been relaxed from "no repeated readout pair" to "no repeated (cue -> token)
mapping". This rung asks whether that relaxed gate is measuring anything either:
    verb_particle_away_down_r  shied/settled   shied -> ` away` from away_up, settled -> ` down` from out_down;
                                               its PAIR belongs to no counted cell.
    verb_particle_up_out_r     woke/found      woke -> ` up` from up_down, found -> ` out` from out_down; its PAIR
                                               belongs to out_up (blurted/brightened), which shares NO cue with it.
    verb_particle_down_in_r    calmed/caved    CONFOUNDED: its pair twin down_in is also a cue donor (caved -> ` in`),
                                               so both accounts predict the same partner. Reported, not counted as
                                               evidence either way.
All three passed all four rows at v323 (n 14, 4, 10; held 0.875, 1.023, 0.915).
THE PREDICTION IS FUSION, AND I AM REGISTERING IT AS SUCH. Every fusion in the corpus (six, zero surviving exceptions
after v309) has shared a cue -> token mapping, so if that account is right these cells are NOT new circuits. pred_a is
therefore registered at a LOW prior: a pass would be evidence against the only account that has ever explained a
fusion here, not a routine +3.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
    pred_a_new_separable   at least k_new = 2 of the 3 reuse cells is separable at the registered cross bar. I expect
                           this to FAIL. If it passes, the cue -> token mapping is not the identity either, no gate I
                           have is measuring anything, and the corpus's fusion account needs a different mechanism.
                                                                                                          prior 25%
    pred_b_counted_kept    at least k_kept = 7 of the 8 re-checked counted members remain separable. A donor losing
                           separability against the cell that borrowed its cue is the FUSION signature seen from the
                           other side, so a failure here is the same finding as pred_a failing, not a contradiction
                           of it. Both are reported.                                                      prior 60%
    pred_c_row4_kept       fam-arm own-C UB975 <= 0.01 on at least k_d = 5 of the 11 evaluated members.    prior 70%
    pred_d_instrument      every evaluated member's own-arm extraction is within 0.03 of its parent battery's cdas
                           extraction_held. Fails automatically if any member errors. Thirteen prepared is the largest
                           this family has run; v321 did ten in 1143 s with none.                          prior 85%
    pred_e_cue_beats_pair  in the OWN arm, verb_particle_up_out_r leaks into BOTH of its cue donors more than into its
                           pair twin, by at least cue_margin = 0.05:
                           min(|leak into up_down|, |leak into out_down|) - |leak into out_up| >= 0.05.
                           This is measured whether or not anything fuses, uses absolute values because a
                           per-behaviour direction sign is not fixed by the name order, and it separates the two
                           accounts by PARTNER rather than by outcome. Worked example: leaks 0.30 / 0.25 into the cue
                           donors and 0.10 into the pair twin give min(0.30, 0.25) - 0.10 = 0.15 >= 0.05, TRUE, the
                           cue account; 0.10 / 0.10 into the donors and 0.40 into the pair twin give -0.30, FALSE,
                           the pair account. v321 measured own-arm leaks of 0.25 to 0.47 between ordinary siblings,
                           so 0.05 is a tenth of the spread and not a bar the noise can clear on its own. prior 55%
COST. Thirteen prepared; the measured curve (19 + 9.3 x family_size) puts it near 30 min. That is the largest single
rung I have run in this family and it is worth it: it is the second half of a 2x2 whose first half already overturned
a gate I had applied to every cell for weeks.
COUNTING. A fused cell is +0 and CONFIRMS the account. A separable cell is +1 on 134 and REFUTES it, and would be
reported as a refutation in the same row as the count.
BARS = {"cross_max": 0.05, "keep": 0.8, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 5, "k_new": 2, "k_kept": 7, "instr_tol": 0.03, "cue_margin": 0.05}
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
import run_unit_tier3_batch_amended_spec9p_v319 as v319
import run_unit_tier3_batch_amended_spec9p_v323 as v323

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v325_result.json"
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
V319 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v319_result.json"
V323 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v323_result.json"
FAMILIES = {
    "verb_particle": ["verb_particle_up_down", "verb_particle_out_down", "verb_particle_out_up", "verb_particle_up_on", "verb_particle_away_up", "verb_particle_down_in", "verb_particle_on_out", "verb_particle_up_down_b", "verb_particle_away_up_b", "verb_particle_out_down_b", "verb_particle_away_down_r", "verb_particle_up_out_r", "verb_particle_down_in_r"],
}
EVAL = ("verb_particle_away_down_r", "verb_particle_up_out_r", "verb_particle_down_in_r", "verb_particle_up_down", "verb_particle_out_down", "verb_particle_out_up", "verb_particle_away_up", "verb_particle_down_in", "verb_particle_up_down_b", "verb_particle_away_up_b", "verb_particle_out_down_b", )
COUNTED = ("verb_particle_up_down", "verb_particle_out_down", "verb_particle_out_up", "verb_particle_away_up", "verb_particle_down_in", "verb_particle_up_down_b", "verb_particle_away_up_b", "verb_particle_out_down_b", )
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES, **v243.NAMES, **v245.NAMES, **v247.NAMES, **v249.NAMES, **v251.NAMES, **v253.NAMES, **v255.NAMES, **v257.NAMES, **v311.NAMES, **v315.NAMES, **v319.NAMES, **v323.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V323, V319, V315, V311, V257, V255, V253, V251, V249, V247, V245, V243, V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
BARS = {"cross_max": 0.05, "keep": 0.8, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 5, "k_new": 2, "k_kept": 7, "instr_tol": 0.03, "cue_margin": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v325", "members": n, "families": list(FAMILIES),
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
    tgt, donors, twin = "verb_particle_up_out_r", ("verb_particle_up_down", "verb_particle_out_down"), "verb_particle_out_up"
    e = False
    if tgt in good:
        sibs = good[tgt]["arms"]["own"]["siblings"]
        if all(s in sibs for s in donors + (twin,)):
            e = min(abs(sibs[d_]) for d_ in donors) - abs(sibs[twin]) >= B["cue_margin"]
    return {"pred_a_new_separable": bool(a), "pred_b_counted_kept": bool(b), "pred_c_row4_kept": bool(c),
            "pred_d_instrument": bool(d), "pred_e_cue_beats_pair": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V325_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V325_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V325_SMOKE_NAMES", "verb_preposition_at_to,verb_preposition").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"], **json.loads(V243.read_text())["behaviours"], **json.loads(V245.read_text())["behaviours"], **json.loads(V247.read_text())["behaviours"], **json.loads(V249.read_text())["behaviours"], **json.loads(V251.read_text())["behaviours"], **json.loads(V253.read_text())["behaviours"], **json.loads(V255.read_text())["behaviours"], **json.loads(V257.read_text())["behaviours"], **json.loads(V311.read_text())["behaviours"], **json.loads(V315.read_text())["behaviours"], **json.loads(V319.read_text())["behaviours"], **json.loads(V323.read_text())["behaviours"]}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v325", "candidate_id": "corpus.unit_family_separability_spec_v325",
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
