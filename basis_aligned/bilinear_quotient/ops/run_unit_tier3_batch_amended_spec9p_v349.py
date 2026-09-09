#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-first four spec-authored behaviours, plus one instrument.
"""v349: the AMENDED tier-3 battery over the FIFTY-FIRST batch (FOUR cells) -- plus one instrument.
Frame-C copies of four MORE counted particle circuits, chosen so that none of them has a frame-B copy.
v285 2/7, ..., v335 2/2, v339 2/3, v343 3/3 (+instrument each).
WHY THESE FOUR AND NOT ANY FOUR. v341 measured that a cell reusing both cue -> token mappings of a counted circuit in
a different frame is separable from it even with that donor in the control set (fam-arm leak 0.0026 and 0.0243 against
own-arm leaks of 0.6577 and 1.0611 into the same donors). v345, running now, asks the next question -- whether the
frame-B and frame-C copies of ONE mapping pair are also distinct from each other -- and until that reports I am
deliberately authoring only cells whose mappings exist in frame A alone. Whatever v345 says about the B/C grain, these
four are copies of an A circuit and nothing else, so the batch does not depend on the answer:
    verb_particle_fc_out_up      blurted/brightened  [A: verb_particle_out_up, counted]
    verb_particle_fc_down_in     hunkered/caved      [A: verb_particle_down_in, counted]
    verb_particle_fc_up_down_b   cheered/knuckled    [A: verb_particle_up_down_b, counted]
    verb_particle_fc_away_up_b   faded/livened       [A: verb_particle_away_up_b, counted]
THE RUNG THAT FOLLOWS IS DESIGNED AROUND THE COST, not around convenience. These four will go into their OWN family
with their four frame-A donors as control-only members -- eight prepared, about twelve minutes -- rather than into the
fourteen-member replication family, where the same test would cost twenty-two prepared and most of an hour. The
donors have to be present or the test is meaningless; the rest of the replication family does not.
CPU capability margins BEFORE enqueue (mean base / mean donor; A1; dropped rows; floor 1.0):
  verb_particle_fc_away_up_b  (faded/livened)      A1:-3.29/+3.46 drop0
  verb_particle_fc_out_up     (blurted/brightened) A1:-3.29/+3.37 drop0
  verb_particle_fc_up_down_b  (cheered/knuckled)   A1:-2.72/+2.77 drop0
  verb_particle_fc_down_in    (hunkered/caved)     A1:-2.47/+2.38 drop0
  Four of four, and frame C is now seven for seven across both batches. Set against six of twenty for the novel-verb
  cells I authored this morning, the pattern is consistent: a verb already measured capable somewhere transfers, my
  judgement about unfamiliar verbs does not. Note that this is exactly the "reuse a known-capable verb" observation I
  refused to register as a rule at v323 -- it is now 14 of 14 across three frames, and it is still an observation,
  because the one time I registered a capability rule today it failed at 2 of 5. Worth acting on; not worth asserting.
  Frame C is also markedly MORE capable than frame A for the same mappings (away_up_b 3.29 here against 2.75 in A;
  shied/sidled fails frame B entirely at 0.61 having cleared A at 1.26), so capability is a property of the pair
  (frame, mapping) just as identity is.
Protocol identical to v197/v201/.../v343. finiteness_selection is the instrument and is EXCLUDED from four-row counts.
Batch size FOUR; bars are the four-cell bars (2/3/3/3/3).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 2 of the 4 new behaviours pass all four rows.                          prior 80%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 3 of 4.       prior 70%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 3 of 4.                      prior 80%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 3 of 4.                   prior 65%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 3 of 4.        prior 75%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02.
Accounting: passes are counted only after the eight-member rung finds them separable from their own frame-A donors.
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~3 min.
Smoke: V349_SMOKE=<out.json> (CPU, V349_SMOKE_ROWS=4 per split, V349_SMOKE_NAMES=verb_particle_fc_out_up).
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v349_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_particle_fc_out_up": "verb_particle_fc_out_up", "verb_particle_fc_down_in": "verb_particle_fc_down_in",
         "verb_particle_fc_up_down_b": "verb_particle_fc_up_down_b", "verb_particle_fc_away_up_b": "verb_particle_fc_away_up_b",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 2, "k_row2": 3, "k_row3": 3, "k_row4": 3, "k_row5": 3, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec9p_v349", "behaviours": 9, "constructions": 5,
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
    a = ok and four >= B["k_four"]
    b = ok and cnt("row2") >= B["k_row2"]
    c = ok and cnt("row3") >= B["k_row3"]
    d = ok and cnt("row4") >= B["k_row4"]
    e = ok and cnt("row5") >= B["k_row5"]
    f = ok and all(n in good and good[n].get("instr_n_units_match") and good[n].get("instr_dirB_abs_diff") is not None
                   and good[n]["instr_dirB_abs_diff"] <= B["instr_tol"] for n in INSTR)
    return {"pred_a_four_rows": bool(a), "pred_b_row2": bool(b), "pred_c_row3": bool(c), "pred_d_row4": bool(d),
            "pred_e_row5_amended": bool(e), "pred_f_instrument": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V349_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V349_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V349_SMOKE_NAMES", "verb_particle_fc_out_up").split(",")]
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
                 "P_held": g.prepare(backend, cut(held_half(rows["P"]))), "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))), "C_held": g.prepare(backend, cut(held_half(rows["C"])))}
            singles, ranked, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
            units = list(greedy["chosen"])
            e_fit, e_held, e_a, e_b = ext(P["fit"], units), ext(P["held"], units), ext(P["held_dirA"], units), ext(P["held_dirB"], units)
            mu1 = mu_of(P["fit"], units)
            q1 = g.block_diff_in_means(backend, P["fit"], units)
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C_fit"],), control_weight=LAM, mu=mu1)
            arms = {}
            for arm, qa in (("dim", q1), ("cdas", qc)):
                arms[arm] = {fam: dmg(P[k], units, qa, mu1) for fam, k in (("A1", "held"), ("A2", "A2_held"), ("P", "P_held"), ("C", "C_held"))}
                arms[arm]["extraction_held"] = round(ext(P["held"], units, q=qa) / e_held, 3) if abs(e_held) > 1e-6 else None
            mu2 = mu_of(P["A2_fit"], units)
            q2 = g.block_diff_in_means(backend, P["A2_fit"], units)
            a2_own = dmg(P["A2_held"], units, q2, mu2)
            a2_full = dmg(P["A2_held"], units, None, mu2)   # q=None: full-rank mean-ablation of the set = the construction's own ceiling
            a1_full = dmg(P["held"], units, None, mu1)
            share = lambda x, y: round(x / y, 4) if y else None
            d = arms["dim"]
            row5_share = share(a2_own["ce_damage"], a2_full["ce_damage"])
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,
                       "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                       "row4": arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX,
                       "row5": a2_own["ce_lb975"] > 0 and row5_share is not None and BARS["row5_share_band"][0] <= row5_share <= BARS["row5_share_band"][1]}
            R[n] = {"units": units, "n_units": len(units), "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]},
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec9p_v349", "candidate_id": "corpus.unit_tier3_batch_amended_spec9p_v349", "bars": BARS,
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
