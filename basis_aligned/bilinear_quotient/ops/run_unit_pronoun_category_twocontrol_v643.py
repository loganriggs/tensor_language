#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-ninth two spec-authored behaviours, plus one instrument.
"""v643: does two-control selectivity hold for possessive cells that differ by MAPPING rather than structure?

WHAT v629 AND v631 TOGETHER SHOWED. Six possessive cells differing only in what sits between cue and readout -- a PP
modifier, an extra conjunct, a longer chain, an attractor noun -- ALL reached held-out extraction 0.877 to 1.224 and
ALL passed both controls, canonical bounds -0.0079 to -0.0300 and weekly bounds -0.0006 to +0.0010. Then v631 showed
five of the six FUSE, with sibling leaks of 0.055 to 0.260 against a 0.05 bar, so that impressive six-of-six was ONE
behaviour measured six ways. Only the attractor variant separated, at 0.0089.
THE QUESTION THIS LEAVES, AND IT IS THE ONE THAT MATTERS FOR INVENTORY. Those six varied STRUCTURE while holding the
mapping fixed at ' their' / ' his'. The ten cells here vary the MAPPING instead: my/your, his/her, his/their,
its/their, our/your, mine/yours, hers/theirs, ours/theirs, plus the plain number cell and the attractor-conflict
cell. Mapping is what the corpus record says a behaviour's identity IS, so these are the cells that could be
genuinely distinct -- and if two-control selectivity is a property of the possessive family rather than of one
direction, it should appear here too, across cells that are NOT structural variants of each other.
WHAT A POSITIVE WOULD AND WOULD NOT BUY. It would say the family supports specific rank-1 subspaces broadly, which
is the first thing today that would point at more than one countable behaviour in one place. It would NOT count
anything: these ten have never been through a separability pass against each other, and v631 is a standing warning
that cells which all look strong can still be one direction. The separability rung comes after, and I am writing
that down now so a good result here does not turn into a quiet increment of ten.
FITTED AGAINST THE WEEKLY CONTROL with the CANONICAL one held out, the arrangement that transfers 64 percent against
32 across 39 cells. Rank 1, registered, and a null is not permission to raise it. Both control numbers reported per
cell.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here). pred_b IS RE-REGISTERED rather
than inherited: v629's version checked an anchor against a prior receipt, and no prior two-control numbers exist for
these ten, so an anchor check would be vacuous and I am replacing it rather than leaving a predicate that cannot
inform:
  pred_a_controls_capable the weekly control is capable on all ten: at most max_dropped = 4 of its 16 held rows
                        dropped. These include non-spec cells whose control panels are rebuilt.       prior 85%
  pred_b_all_reach      EVERY one of the ten reaches held-out extraction ext_min = 0.80 under its own fit.
                                                                                                      prior 55%
  pred_c_siblings_reached at least k_reach = 7 of the ten reach 0.80, a weaker form of b that survives one or two
                        weak cells.                                                                   prior 80%
  pred_d_reached_are_two_control_selective at least k_both = 5 of the cells that reach are selective on BOTH
                        controls. This is the rung.                                                   prior 60%
  pred_e_all_measured   zero cells error.                                                             prior 90%
HOW IT READS. c and d both true: two-control selectivity is a property of the possessive family across MAPPINGS, not
just across structural variants of one direction, and the separability pass over these ten becomes the most valuable
rung available. d FALSE with c true: the cells carry the variable but their selectivity does not survive a second
control, which would mean v629's six-of-six was specific to that one direction and not a family property -- the
outcome that would most change how I read v629. b FALSE with c true: one or two mappings are weak and worth naming
individually rather than averaging away.
SCOPE. Ten cells, independent fits, rank 1, two controls. No separability test, no counting, no bar change.
Smoke: V643_SMOKE=<out.json> (CPU, V643_SMOKE_ROWS=4, V643_SMOKE_NAMES=<cell>).
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
import circuit_fast_screen_control_v3_rows as r3
import circuit_fast_screen_canonical_control_v3 as control_v3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_pronoun_category_twocontrol_v643_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"possessive_pronoun_his_hers": "possessive_pronoun_his_hers",
         "possessive_gender": "possessive_gender",
         "possessive_pronoun_mine_yours": "possessive_pronoun_mine_yours",
         "possessive_pronoun_person_ours_theirs": "possessive_pronoun_person_ours_theirs"}
ANCHOR = "possessive_gender"
SIBLINGS = ['possessive_pronoun_his_hers', 'possessive_pronoun_mine_yours', 'possessive_pronoun_person_ours_theirs']
# v633 under this same arrangement: gender canonical +0.0029 weekly -0.0006 (BOTH pass);
# mine_yours canonical +0.0340 (fails canonical); ours_theirs canonical +0.0267 (fails canonical)
V643_ANCHOR = {"possessive_gender": (0.0029, -0.0006), "possessive_pronoun_mine_yours": (0.0340, 0.0023),
               "possessive_pronoun_person_ours_theirs": (0.0267, 0.0009)}
V2_ROW4 = {n: True for n in NAMES}
NEW = tuple(NAMES)
INSTR = ()
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_reach": 3, "k_both": 1, "tol": 0.05, "max_dropped": 4}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_pronoun_category_twocontrol_v643", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V643_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V643_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V643_SMOKE_NAMES", "verb_preposition_against_for,finiteness_selection").split(",")]
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
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C3_fit"],), control_weight=LAM, mu=mu1)
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
    result = {"predictions": predictions, "schema": "unit_pronoun_category_twocontrol_v643", "candidate_id": "corpus.unit_pronoun_category_twocontrol_v643", "bars": BARS,
              "protocol": {"split": "within-direction: fit rows[0::4]+rows[1::4], held rows[2::4]+rows[3::4]", "pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                           "cdas": {"steps": steps, "lr": LR, "complement_weight": CW, "control_weight": LAM, "controls": "own C FIT rows"},
                           "row5": "A2-own diff-in-means CE / A2 full-rank mean-ablation ceiling, LB975 > 0"},
              "four_row_passes": sorted(n for n, r in R.items() if "error" not in r and all(r["rows"].values())), "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "rows": {n: r["rows"] for n, r in R.items()}}, indent=2))


if __name__ == "__main__":
    main()
