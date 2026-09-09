#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the fifty-second six spec-authored behaviours, plus one instrument.
"""v353: the AMENDED tier-3 battery over the FIFTY-SECOND batch (SIX cells) -- plus one instrument.
Frame-B and frame-C copies of six counted PREPOSITION circuits. The preposition class is where the frame lane has
actually been yielding, and this batch tests that against the particle class's poor showing rather than assuming it.
v285 2/7, ..., v339 2/3, v343 3/3, v349 1/4 (+instrument each).
THE YIELD QUESTION THIS BATCH ANSWERS. Frame copies have gone 2 of 3 in frame B and 3 of 3 in the first frame-C batch,
then 1 of 4 in v349 -- and the difference between those batches is the READOUT CLASS, not the frame: v349 was four
particle copies and only blurted/brightened survived, with two of the failures missing ROW 2 at held 0.874 and 0.880
despite CPU A1 floors of 2.4 to 3.5. Every preposition copy authored so far (fb_about_for, fc_about_for) passed all
four rows. Six preposition copies at once is the test of whether that holds up, and it is the difference between the
frame lane being a general multiplier and being a preposition-only one.
CELLS (frame in brackets; each borrows both cue -> token mappings from a counted frame-A circuit):
  verb_preposition_fb_at_to       aimed/appealed        [B]   from verb_preposition_at_to
  verb_preposition_fb_from_for    prevented/blamed      [B]   from verb_preposition_from_for
  verb_preposition_fb_about_into  raved/delved          [B]   from verb_preposition_about_into
  verb_preposition_fc_at_over     scoffed/agonised      [C]   from verb_preposition_at_over
  verb_preposition_fc_by_from     swore/withdrew        [C]   from verb_preposition_by_from
  verb_preposition_fc_for_at      apologized/laughed    [C]   from verb_preposition_for_at
Three in each frame, and no mapping appears twice within a frame, so the batch respects the gate as it now stands:
a (cue -> token) mapping may be reused in a DIFFERENT frame, never within one.
CPU capability margins BEFORE enqueue (mean base / mean donor; A1; dropped rows; floor 1.0):
  verb_preposition_fb_about_into  (raved/delved)       A1:-4.94/+4.84 drop0
  verb_preposition_fb_from_for    (prevented/blamed)   A1:-4.00/+4.22 drop0
  verb_preposition_fc_at_over     (scoffed/agonised)   A1:-3.88/+3.51 drop0
  verb_preposition_fc_by_from     (swore/withdrew)     A1:-3.07/+3.03 drop0
  verb_preposition_fc_for_at      (apologized/laughed) A1:-2.75/+2.69 drop0
  verb_preposition_fb_at_to       (aimed/appealed)     A1:-2.49/+2.27 drop0
  Six of six, none below 2.27. I am NOT reading that as a forecast: v349's four cells were at 2.4 to 3.5 and produced
  one four-row pass. The floor says these can be studied; it has never said they will pass.
Protocol identical to v197/v201/.../v349. finiteness_selection is the instrument and is EXCLUDED from four-row counts.
Batch size SIX; bars are the six-cell bars used at v287 and v301 (3/5/5/4/4).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 3 of the 6 new behaviours pass all four rows. Frame copies are 6 of 10 so far and
                           every preposition one has passed; the particle ones are what dragged the rate.  prior 65%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 5 of 6. This is the row
                           v349's particle copies failed, so it is the one to watch.                       prior 55%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 5 of 6.                      prior 80%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 4 of 6.                    prior 65%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 4 of 6.         prior 75%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02.
Accounting: passes go to small purpose-built rungs, each containing the new cell and its own frame-A donor and nothing
else, per the v347 finding that control-set size is inert and the v345 finding that only mapping relatives matter.
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~4 min.
Smoke: V353_SMOKE=<out.json> (CPU, V353_SMOKE_ROWS=4 per split, V353_SMOKE_NAMES=verb_preposition_fb_at_to).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9p_v353_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_preposition_fb_at_to": "verb_preposition_fb_at_to", "verb_preposition_fb_from_for": "verb_preposition_fb_from_for",
         "verb_preposition_fb_about_into": "verb_preposition_fb_about_into", "verb_preposition_fc_at_over": "verb_preposition_fc_at_over",
         "verb_preposition_fc_by_from": "verb_preposition_fc_by_from", "verb_preposition_fc_for_at": "verb_preposition_fc_for_at",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 3, "k_row2": 5, "k_row3": 5, "k_row4": 4, "k_row5": 4, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec9p_v353", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V353_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V353_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V353_SMOKE_NAMES", "verb_preposition_fb_at_to").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec9p_v353", "candidate_id": "corpus.unit_tier3_batch_amended_spec9p_v353", "bars": BARS,
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
