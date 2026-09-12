#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v651: is the registered-objective result possessive-specific? Eight verb_preposition mappings, same fit.

THE ASYMMETRY THIS EXISTS TO TEST. Under the objective the protocol actually registers -- canonical control in the
fit at weight 30 -- exactly four cells in this corpus have been shown two-control selective while also clearing all
four tier rows: three possessive mappings (gender, number_its_their, person_our_your) in v637, and one correlative
(disjoint_either_not) in v641. Everything else measured under that objective has failed, including all five
candidates in v611 -- four amid verb_preposition cells and determiner_number_crates -- which failed the held-out
control at ratios of 2.8 to 9.4. So the only registered-objective survivors so far come from two families, and the
verb_preposition evidence against is drawn from FIVE cells that share one readout token.
WHAT THIS RUNG ADDS. Eight verb_preposition mappings that have never been two-control tested at all -- about_against,
about_beyond, about_for, about_into, against_for, at_across, at_from, at_into -- measured under the REGISTERED fit
with the weekly control held out. If some pass, the possessive result is not special and the registered protocol
certifies behaviours across families, which changes how I read the whole two-control sequence. If none pass, then on
current evidence the registered objective admits possessive and correlative mappings and essentially nothing from
the preposition families, and THAT is a fact about the protocol worth putting to Codex rather than a fact about
prepositions.
WHY I AM NOT PREDICTING EITHER WAY WITH CONFIDENCE. v611's five failures were all amid cells, a single readout
token, so they are weak evidence about verb_preposition as a family. Against that, verb_preposition cells carried
row 4 at 15 of 16 under the canonical control in the lift batches, which is the highest of any family -- but row 4
there was measured on the control the fit was optimised against, which is exactly the measurement v603 showed does
not transfer.
RANK 1, registered, and a null is not permission to raise it. Both control numbers reported per cell. No separability
test here: if any pass, that is the next rung, and I am writing it down now so a positive does not become a quiet
increment of eight.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_controls_capable the weekly control -- held out here -- is capable on all eight: at most max_dropped = 4 of
                        its 16 held rows dropped.                                                     prior 85%
  pred_b_all_reach      every one of the eight reaches held-out extraction ext_min = 0.80 under the registered fit.
                                                                                                      prior 80%
  pred_c_siblings_reached at least k_reach = 6 of the eight reach 0.80.                               prior 85%
  pred_d_reached_are_two_control_selective at least k_both = 3 of the eight are selective on BOTH controls. This is
                        the rung, and I put it near even: no verb_preposition cell has ever passed a held-out
                        control, but the five that failed shared one readout token.                   prior 40%
  pred_e_all_measured   zero cells error.                                                             prior 90%
HOW IT READS. d TRUE: the registered objective certifies behaviours outside the possessive family, the possessive
block is not special, and the separability rung follows for whichever pass. d FALSE with c true: the cells carry the
variable and none is two-control selective under the registered fit, which -- added to v611's five -- would make
thirteen verb_preposition cells tested and none passing, and would say the registered objective effectively excludes
this family. That is the more consequential outcome and it belongs on the board as a protocol observation, not as a
claim about prepositions being un-localisable.
SCOPE. Eight cells, registered objective, rank 1, two controls. No separability test, no counting.
Smoke: V651_SMOKE=<out.json> (CPU, V651_SMOKE_ROWS=4, V651_SMOKE_NAMES=<cell>).
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
import receipt_write
import circuit_fast_screen_control_v3_rows as r3
import circuit_fast_screen_canonical_control_v3 as control_v3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_uncounted_registered_objective_v651_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"additive_scope_lenmatched": "additive_scope_lenmatched",
         "adjective_preposition_from_about": "adjective_preposition_from_about",
         "both_either": "both_either",
         "coordination_or_nor": "coordination_or_nor",
         "definiteness_anaphor_needed": "definiteness_anaphor_needed",
         "deixis": "deixis",
         "existential_article_a_some": "existential_article_a_some",
         "numeral_dual_between_among": "numeral_dual_between_among"}
ANCHOR = "deixis"
SIBLINGS = ['additive_scope_lenmatched', 'adjective_preposition_from_about', 'both_either', 'coordination_or_nor', 'definiteness_anaphor_needed', 'existential_article_a_some', 'numeral_dual_between_among']
V651_ANCHOR = None
# selected by the CORRECTED filter: none of these appears in ANY separability rung, counted or otherwise
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(NAMES)
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_reach": 6, "k_both": 3, "tol": 0.05, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_uncounted_registered_objective_v651", "behaviours": 9, "constructions": 5,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 9 * STEPS, "model_updates": 0, "fit_parameters": 9 * MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "rows" in r and "error" not in r}
    ok = bool(good)  # an empty or all-error receipt fails everything
    new = {n: good[n] for n in NEW if n in good}
    cnt = lambda row: sum(1 for n in new if new[n]["rows"][row])
    four = sum(1 for n in new if all(new[n]["rows"].values()))
    drop = max((new[n]["v3_dropped"] for n in new), default=99)
    a = ok and len(new) == len(NEW) and drop <= B["max_dropped"]
    anc = new.get(ANCHOR, {})
    b = ok and all((new[n].get("extraction_held") or 0) >= EXT_MIN for n in new)
    reach = [n for n in new if (new[n].get("extraction_held") or 0) >= EXT_MIN]
    c = ok and len(reach) >= B["k_reach"]
    d = ok and sum(1 for n in reach if new[n]["row4_v2"] and new[n]["row4_v3"]) >= B["k_both"]
    e = ok and len(new) == len(NEW)
    return {"pred_a_controls_capable": bool(a), "pred_b_all_reach": bool(b),
            "pred_c_siblings_reached": bool(c), "pred_d_reached_are_two_control_selective": bool(d),
            "pred_e_all_measured": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V651_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V651_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V651_SMOKE_NAMES", "verb_preposition_against_for,finiteness_selection").split(",")]
    parent = json.loads(V196.read_text())["behaviours"]
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)
    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
            rows = {fam: g.rows_of(m, fam) for fam in ("A1", "A2", "P", "C")}
            V = dict(valid_only=True)   # capability-failure rows (donor does not beat base) are dropped, counted in n_dropped
            P = {"fit": g.prepare(backend, cut(fit_half(rows["A1"])), **V), "held": g.prepare(backend, cut(held_half(rows["A1"])), **V),
                 "held_dirA": g.prepare(backend, cut(rows["A1"][2::4]), **V), "held_dirB": g.prepare(backend, cut(rows["A1"][3::4]), **V),
                 "A2_fit": g.prepare(backend, cut(fit_half(rows["A2"])), **V), "A2_held": g.prepare(backend, cut(held_half(rows["A2"])), **V),
                 "P_held": g.prepare(backend, cut(held_half(rows["P"]))), "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))), "C_held": g.prepare(backend, cut(held_half(rows["C"]))),
                 "C3_held": g.prepare(backend, cut(held_half(r3.rows_any(m, control_v3)))),
                 "C3_fit": g.prepare(backend, cut(fit_half(r3.rows_any(m, control_v3))))}
            singles, ranked, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
            units = list(greedy["chosen"])
            e_fit, e_held, e_a, e_b = ext(P["fit"], units), ext(P["held"], units), ext(P["held_dirA"], units), ext(P["held_dirB"], units)
            mu1 = mu_of(P["fit"], units)
            q1 = g.block_diff_in_means(backend, P["fit"], units)
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C_fit"],), control_weight=LAM, mu=mu1)
            arms = {}
            for arm, qa in (("dim", q1), ("cdas", qc)):
                arms[arm] = {fam: dmg(P[k], units, qa, mu1) for fam, k in (("A1", "held"), ("A2", "A2_held"), ("P", "P_held"), ("C", "C_held"), ("C3", "C3_held"))}
                arms[arm]["extraction_held"] = round(ext(P["held"], units, q=qa) / e_held, 3) if abs(e_held) > 1e-6 else None
            mu2 = mu_of(P["A2_fit"], units)
            q2 = g.block_diff_in_means(backend, P["A2_fit"], units)
            a2_own = dmg(P["A2_held"], units, q2, mu2)
            a2_full = dmg(P["A2_held"], units, None, mu2)   # q=None: full-rank mean-ablation of the set = the construction's own ceiling
            a1_full = dmg(P["held"], units, None, mu1)
            share = lambda x, y: round(x / y, 4) if y else None
            d = arms["dim"]
            row5_share = share(a2_own["ce_damage"], a2_full["ce_damage"])
            row4_v2 = arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX      # canonical control: now the HELD-OUT one
            row4_v3 = arms["cdas"]["C3"]["ce_ub975"] <= C_UB_MAX   # v3: now the control IN the objective
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,
                       "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                       "row4": arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX,
                       "row5": a2_own["ce_lb975"] > 0 and row5_share is not None and BARS["row5_share_band"][0] <= row5_share <= BARS["row5_share_band"][1],
                       "row4_v3": row4_v3}
            R[n] = {"row4_v2": bool(row4_v2), "row4_v3": bool(row4_v3),
                    "c_ub_v2": arms["cdas"]["C"]["ce_ub975"], "c_ub_v3": arms["cdas"]["C3"]["ce_ub975"],
                    "v3_dropped": P["C3_held"].dropped, "v2_dropped": P["C_held"].dropped,
                    "units": units, "n_units": len(units), "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]},
                    "direction_A": str(P["held_dirA"].rows[0].get("direction_id")), "direction_B": str(P["held_dirB"].rows[0].get("direction_id")),
                    "extraction_fit": e_fit, "extraction_held": e_held, "extraction_held_dirA": e_a, "extraction_held_dirB": e_b,
                    "arms": arms, "a2_own": a2_own, "a2_full_ceiling": a2_full, "a1_full_ceiling": a1_full,
                    "row5_share_own": row5_share, "row5_share_a1_direction": share(d["A2"]["ce_damage"], a2_full["ce_damage"]),
                    "a1_share_dim_over_full": share(d["A1"]["ce_damage"], a1_full["ce_damage"]), "old_row5_ratio": share(d["A2"]["ce_damage"], d["A1"]["ce_damage"]),
                    "instr_n_units_match": (n in parent and parent[n]["units"] == units), "instr_dirB_abs_diff": (round(abs(parent[n]["extraction_held_dirB"] - e_b), 3) if n in parent else None),
                    "rows": rows_ok, "n_dropped": {k: P[k].dropped for k in P}, "cdas_final_loss": (hist[-1] if hist else None), "n_rows": {k: len(P[k].rows) for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one behaviour must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "rows": {k: False for k in ("row2", "row3", "row4", "row5")}, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "singles_top16", "arms", "n_rows")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_uncounted_registered_objective_v651", "candidate_id": "corpus.unit_uncounted_registered_objective_v651", "bars": BARS,
              "protocol": {"split": "within-direction: fit rows[0::4]+rows[1::4], held rows[2::4]+rows[3::4]", "pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                           "cdas": {"steps": steps, "lr": LR, "complement_weight": CW, "control_weight": LAM, "controls": "own C FIT rows"},
                           "row5": "A2-own diff-in-means CE / A2 full-rank mean-ablation ceiling, LB975 > 0"},
              "four_row_passes": sorted(n for n, r in R.items() if "error" not in r and all(r["rows"].values())), "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    written, used_fallback = receipt_write.write_receipt(out, result)
    if used_fallback:
        print('RECEIPT NOT AT ITS INTENDED PATH -- do not release the claim until it is copied back', flush=True)
    print(json.dumps({"predictions": predictions, "rows": {n: r["rows"] for n, r in R.items()}}, indent=2))


if __name__ == "__main__":
    main()
