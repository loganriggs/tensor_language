#!/usr/bin/env python3
# BQGATE: four frozen predictions; families, EVAL set, units (v197 ... v251 receipts), recipe and bars fixed before the run.
"""v363: THE FRAME-CAPACITY RUNG. One mapping pair, five frames, every copy in the room.

WHAT IS BEING ASKED. Thirteen mapping pairs now support the frame-bound account, but every one of those tests compared
a copy against its ORIGINAL. This rung asks the different question: how many frames can ONE mapping pair carry before
the copies collide with EACH OTHER? woke -> ` up` and calmed -> ` down` now have four-row circuits in five sentence
frames, and all five are prepared here:
    verb_particle_fd_up_down  frame D  `Nobody doubted the {agent} {verb}, honestly,`     NEW, evaluated
    verb_particle_fe_up_down  frame E  `Word spread that the {agent} {verb}, evidently,`  NEW, evaluated
    verb_particle_up_down     frame A  `Near the {object} the {agent} {verb}, of course,` counted; control-only
    verb_particle_fb_up_down  frame B  `Everyone agreed the {agent} {verb}, frankly,`     control-only
    verb_particle_fc_up_down  frame C  `It turned out the {agent} {verb}, apparently,`    counted; control-only
Five prepared, about five minutes. Every copy of the pair is a control, so a capacity limit has every chance to show
as a fusion BETWEEN copies rather than against the original -- which is the whole point, since v341/v355/v359 already
showed copies clear their originals with own-arm leaks of 0.29 to 1.34 against fam-arm leaks under 0.024.
THE COMPANION CELLS FAILED THE BATTERY AND THAT IS ITSELF A RESULT. v361 also authored frames D and E for
cared/voted, which has counted circuits in frames A, B and C, and BOTH missed rows 2 and 4. So frame transfer is not a
property of a mapping pair -- cared/voted transfers to B and C and not to D and E, while woke/calmed transfers to all
four. Whatever the capacity is, it is a property of the (pair, frame) combination and has to be screened, not assumed.
That failure also means this rung tests one pair rather than two, and I would rather say so than quietly report a
narrower result as if it were the planned one.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
    pred_a_new_separable   at least k_new = 1 of the 2 new members is separable at the registered cross bar.
                                                                                                          prior 85%
    pred_b_own_leak_large  in the OWN arm -- own C only -- the SMALLEST leak from either new cell into ANY other copy
                           of its own mapping pair is still large: min over the two cells of
                           (max over the three older copies of |own-arm leak|) >= own_leak_min = 0.20.
                           Without this a separability pass is ambiguous between "the controls did real work" and
                           "these copies never touched". v359 measured own-arm donor leaks of 0.85 to 1.34 against
                           fam-arm leaks under 0.024, so 0.20 is far below every observation and can still fail.
                                                                                                          prior 80%
    pred_c_row4_kept       fam-arm own-C UB975 <= 0.01 on at least k_d = 2 of the 2 evaluated members.    prior 65%
    pred_d_instrument      each evaluated member's own-arm extraction is within 0.03 of its v361 cdas
                           extraction_held. Fails automatically on an error receipt.                      prior 90%
    pred_e_capacity        NEITHER new copy fuses with ANY older copy of the same pair: the largest fam-arm leak from
                           either of the two new cells into any of the frame-A, frame-B or frame-C copies stays under
                           cross_max = 0.05 in absolute value.
                           Worked example: leaks of 0.01, 0.02 and 0.03 from the D copy and 0.01, 0.01, 0.02 from the
                           E copy give a maximum of 0.03, TRUE -- one mapping pair supports at least five
                           distinguishable circuits and the frame multiplier has no ceiling at five. A leak of 0.09
                           from the D copy into the B copy, near v333's within-frame fusion of 0.0921, gives FALSE and
                           is the more interesting outcome: the model would have a BOUNDED number of distinguishable
                           ways to carry one mapping, the bound would be somewhere between three and five, and the
                           next rung would be to find which pairs of frames collide rather than to author more copies.
                                                                                                          prior 60%
COUNTING. Each separable new member is +1 on 152. A fusion between copies is +0 and bounds the frame multiplier, which
is worth more than the two circuits.
BARS = {"cross_max": 0.05, "keep": 0.8, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 2, "k_new": 1, "instr_tol": 0.03, "own_leak_min": 0.2}
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
import run_unit_tier3_batch_amended_spec9p_v327 as v327
import run_unit_tier3_batch_amended_spec9p_v349 as v349
import run_unit_tier3_batch_amended_spec9m_v293 as v293
import run_unit_tier3_batch_amended_spec9p_v353 as v353
import run_unit_tier3_batch_amended_spec9e_v263 as v263
import run_unit_tier3_batch_amended_spec9f_v265 as v265
import run_unit_tier3_batch_amended_spec9p_v357 as v357
import run_unit_tier3_batch_amended_spec9p_v339 as v339
import run_unit_tier3_batch_amended_spec9p_v343 as v343
import run_unit_tier3_batch_amended_spec9p_v361 as v361

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v363_result.json"
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
V327 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v327_result.json"
V349 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v349_result.json"
V293 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9m_v293_result.json"
V353 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v353_result.json"
V263 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9e_v263_result.json"
V265 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9f_v265_result.json"
V357 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v357_result.json"
V339 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v339_result.json"
V343 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v343_result.json"
V361 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v361_result.json"
FAMILIES = {
    "frame_capacity_up_down": ["verb_particle_fd_up_down", "verb_particle_fe_up_down", "verb_particle_up_down", "verb_particle_fb_up_down", "verb_particle_fc_up_down"],
}
EVAL = ("verb_particle_fd_up_down", "verb_particle_fe_up_down", )
COUNTED = ()
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES, **v211.NAMES, **v213.NAMES, **v215.NAMES, **v217.NAMES, **v219.NAMES, **v221.NAMES, **v223.NAMES, **v225.NAMES, **v227.NAMES, **v229.NAMES, **v231.NAMES, **v233.NAMES, **v235.NAMES, **v237.NAMES, **v239.NAMES, **v241.NAMES, **v243.NAMES, **v245.NAMES, **v247.NAMES, **v249.NAMES, **v251.NAMES, **v253.NAMES, **v255.NAMES, **v257.NAMES, **v311.NAMES, **v315.NAMES, **v319.NAMES, **v323.NAMES, **v327.NAMES, **v349.NAMES, **v293.NAMES, **v353.NAMES, **v263.NAMES, **v265.NAMES, **v357.NAMES, **v339.NAMES, **v343.NAMES, **v361.NAMES}


def _parent_cdas(n):
    # the parent battery receipt that produced this member's units (each behaviour name is unique across batteries)
    for path in (V361, V343, V339, V357, V263, V265, V353, V293, V349, V327, V323, V319, V315, V311, V257, V255, V253, V251, V249, V247, V245, V243, V241, V239, V237, V235, V233, V231, V229, V227, V225, V223, V221, V219, V217, V215, V213, V211, V209, V207, V205, V204, V201, V197):
        b = json.loads(path.read_text())["behaviours"].get(n)
        if b is not None and "arms" in b:
            return float(b["arms"]["cdas"]["extraction_held"])
    raise KeyError(n)
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
NEW_MEMBERS = tuple(n for n in EVAL if n not in COUNTED)
BARS = {"cross_max": 0.05, "keep": 0.8, "ext_floor": 0.5, "c_ub_max": 0.01, "k_d": 2, "k_new": 1, "instr_tol": 0.03, "own_leak_min": 0.2}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = len(EVAL)
    return {"candidate_id": "corpus.unit_family_separability_spec_v363", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


OLDER = ("verb_particle_up_down", "verb_particle_fb_up_down", "verb_particle_fc_up_down")


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    a = ok and sum(1 for n in NEW_MEMBERS if sep(n)) >= B["k_new"]
    c = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_d"]
    d = ok and all(n in good and good[n]["arms"]["own"]["extraction"] is not None
                   and abs(good[n]["arms"]["own"]["extraction"] - _parent_cdas(n)) <= B["instr_tol"] for n in EVAL)
    own_max, fam_max = [], []
    for n in NEW_MEMBERS:
        if n not in good:
            own_max, fam_max = [], []
            break
        osib = good[n]["arms"]["own"]["siblings"]; fsib = good[n]["arms"]["fam"]["siblings"]
        if not all(o in osib and o in fsib for o in OLDER):
            own_max, fam_max = [], []
            break
        own_max.append(max(abs(osib[o]) for o in OLDER))
        fam_max.append(max(abs(fsib[o]) for o in OLDER))
    b = bool(own_max) and min(own_max) >= B["own_leak_min"]
    e = bool(fam_max) and max(fam_max) < B["cross_max"]
    return {"pred_a_new_separable": bool(a), "pred_b_own_leak_large": bool(b), "pred_c_row4_kept": bool(c),
            "pred_d_instrument": bool(d), "pred_e_capacity": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V363_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V363_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = members  # every family member is prepared: the non-EVAL members serve as controls
    evaluate = [n for n in members if n in EVAL and (not smoke or n in os.environ.get("V363_SMOKE_NAMES", "verb_preposition_at_to,verb_preposition").split(","))]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"], **json.loads(V211.read_text())["behaviours"], **json.loads(V213.read_text())["behaviours"], **json.loads(V215.read_text())["behaviours"], **json.loads(V217.read_text())["behaviours"], **json.loads(V219.read_text())["behaviours"], **json.loads(V221.read_text())["behaviours"], **json.loads(V223.read_text())["behaviours"], **json.loads(V225.read_text())["behaviours"], **json.loads(V227.read_text())["behaviours"], **json.loads(V229.read_text())["behaviours"], **json.loads(V231.read_text())["behaviours"], **json.loads(V233.read_text())["behaviours"], **json.loads(V235.read_text())["behaviours"], **json.loads(V237.read_text())["behaviours"], **json.loads(V239.read_text())["behaviours"], **json.loads(V241.read_text())["behaviours"], **json.loads(V243.read_text())["behaviours"], **json.loads(V245.read_text())["behaviours"], **json.loads(V247.read_text())["behaviours"], **json.loads(V249.read_text())["behaviours"], **json.loads(V251.read_text())["behaviours"], **json.loads(V253.read_text())["behaviours"], **json.loads(V255.read_text())["behaviours"], **json.loads(V257.read_text())["behaviours"], **json.loads(V311.read_text())["behaviours"], **json.loads(V315.read_text())["behaviours"], **json.loads(V319.read_text())["behaviours"], **json.loads(V323.read_text())["behaviours"], **json.loads(V327.read_text())["behaviours"], **json.loads(V349.read_text())["behaviours"], **json.loads(V353.read_text())["behaviours"], **json.loads(V357.read_text())["behaviours"], **json.loads(V361.read_text())["behaviours"]}
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
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v363", "candidate_id": "corpus.unit_family_separability_spec_v363",
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
