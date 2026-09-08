#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the thirty-fourth five spec-authored behaviours, plus one instrument.
"""v269: the AMENDED tier-3 battery over the THIRTY-FOURTH batch (FIVE cells) -- plus one instrument. Like v267 this batch buys a MECHANISM,
not a count, and it is authored after the hypothesis v267 was built to support was REFUTED as a sufficient condition.

WHERE THE MECHANISM QUESTION STANDS (11:52 ledger row and board note). v254/v256 found four cell pairs whose fam-arm leak crosses or approaches
the 0.05 bar, and in every one the two cells share a (CUE WORD -> READOUT TOKEN) mapping: consisted->` of` (of_over/of_by, +0.294),
presided->` over` (of_over/over_with, +0.134/+0.149), worried->` about` (with_about/to_about, +0.053), insisted->` on` (insisted/on_toward, +0.035).
That looked like "directions are keyed to the cue lexeme". It is NOT sufficient: enumerating EVERY shared cue->token mapping in the corpus turns
up four more pairs that do NOT fuse -- similar->` to` (adjective_preposition_similar / adjective_preposition_to_of, +0.016 and +0.005, separable
on every rung since v222), suffered->` from` (from_about/at_from, -0.009), different->` from` (similar/same_different, -0.001). Four fuse, four do
not, and the non-fusers include a pair that shares BOTH the cue and the readout.
The margin explanation is already dead: the held-out recovery on the shared-cue SIDE is 0.910 for the biggest fuser, 0.888 for the clean
similar/to_of pair, 0.894 for the clean suffered pair and 0.838 for a fuser -- no ordering at all.
What survives is a FAMILY split: all four fusing pairs are in verb_preposition; all four clean ones are adjective_preposition or cross-family.

WHAT THIS BATCH DOES. Five cells that each REUSE ONE CUE ADJECTIVE from a counted adjective_preposition sibling, at the same preposition, with a
readout pair new to the corpus (word-boundary quote-agnostic grep: of/within, in/beneath, at/within, of/toward, for/beneath all 0 hits):
adjective_preposition_of_within (FOND/enclosed -> of/within); `fond` is adjective_preposition_fond's ` of` cue;
adjective_preposition_in_beneath (INTERESTED/buried -> in/beneath); `interested` is adjective_preposition_in_about's ` in` cue;
adjective_preposition_at_within (SKILLED/contained -> at/within); `skilled` is adjective_preposition_at_about's ` at` cue;
adjective_preposition_of_toward (ASHAMED/hostile -> of/toward); `ashamed` is adjective_preposition_of_about's ` of` cue;
adjective_preposition_for_beneath (RESPONSIBLE/hidden -> for/beneath); `responsible` is adjective_preposition_for_of's ` for` cue.
This is the exact manipulation v267 applied in the VERB family, applied in the ADJECTIVE family. v258 (running) has the verb arm's separability;
the rung after this one has the adjective arm's. Together they are a 2x2 on (family) x (cue reused), and they discriminate the two accounts:
  - FAMILY account: verb-family cue reuses fuse, adjective-family cue reuses stay separable -> v258 fuses, this batch's rung does not.
  - CUE account (already refuted as sufficient, kept as the alternative): both fuse, and the four existing clean pairs need their own explanation.
  - NEITHER: both stay separable, and the four observed fusions are about of_over/with_about specifically rather than about cue reuse at all --
    which is live, since three of the four fusing pairs involve of_over.
No outcome here is "confirmation": the batch was designed after the refutation, and its job is to split two accounts that both survive the data.
CPU capability margins BEFORE enqueue (mean base / mean donor on the donor axis; A1 | A2; dropped rows; floor 1.0 on A1 per the v245 lesson):
  adjective_preposition_of_within A1:-5.94/+6.28 drop0 A2:-4.43/+4.53 drop0 C:-1.43/+0.93 drop2
  adjective_preposition_in_beneath A1:-4.15/+4.66 drop0 A2:-3.14/+3.31 drop0 C:-1.43/+0.93 drop2
  adjective_preposition_at_within A1:-2.31/+2.53 drop0 A2:-1.13/+1.18 drop0 C:-1.43/+0.93 drop2
  adjective_preposition_of_toward A1:-4.12/+3.95 drop0 A2:-2.81/+2.64 drop0 C:-1.43/+0.93 drop2
  adjective_preposition_for_beneath A1:-4.96/+5.15 drop0 A2:-3.62/+3.64 drop0 C:-1.43/+0.93 drop2
  ALL FIVE cleared the floor with zero dropped rows (weakest A1 2.31); fifth batch on the probe path, and the first to pass the new NO-MODEL
  validation stage (added 10:58) before any capability check was spent. Nothing was dropped, so nothing was kept out of the repo this time.
  Batch size FIVE; bars are the five-cell bars (3/4/4/3/3).
Protocol identical to v197/v201/.../v267 (greedy pool 40, target 0.88, min_gain 0.005, max 30; row 2 = MIN over the two held directions
>= 0.80; row 3 dim A1 CE LB975 > 0 and >= 0.10; row 4 constrained rank-1 own-C UB975 <= 0.01 on 16 held C rows; row 5 = A2-own / A2 full-rank
ceiling in [0.5, 1.4], LB975 > 0). finiteness_selection is re-run as the instrument and is EXCLUDED from four-row counts (v249 disclosure).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 3 of the 5 new behaviours pass all four rows (pre-screened batches 41/76 = 0.54).       prior 60%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 4 of 5 (v267 4/5, v265 6/8).   prior 60%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 4 of 5 (v267 4/5, v265 8/8).                  prior 75%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 3 of 5 (v267 3/5, v265 5/8).               prior 65%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 3 of 5 (v267 4/5, v265 8/8).    prior 75%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02 (same seed,
                           split and code path: a determinism check, NOT an independent run outcome, disclosed as such).
The FUSION prediction is NOT registered as a bar here because this battery cannot measure it -- it is registered in the separability rung that
follows, and it is stated now only so it cannot be written later: under the family account these five stay separable from their cue partners.
Accounting: four-row passes count only after their separability rung. Under the family account these cells COUNT (+1 each if separable);
under the cue account they count +0. The batch is therefore not a pure mechanism spend like v267 was -- it can add circuits, and which outcome
adds them is exactly what is in question.
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~4 min.
A1/A2 preps use g.prepare(valid_only=True); `n_dropped` is in every receipt.
Smoke: V269_SMOKE=<out.json> (CPU, V269_SMOKE_ROWS=4 per split, V269_SMOKE_NAMES=adjective_preposition_at_within; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9h_v269_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"adjective_preposition_of_within": "adjective_preposition_of_within", "adjective_preposition_in_beneath": "adjective_preposition_in_beneath",
         "adjective_preposition_at_within": "adjective_preposition_at_within", "adjective_preposition_of_toward": "adjective_preposition_of_toward",
         "adjective_preposition_for_beneath": "adjective_preposition_for_beneath",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 3, "k_row2": 4, "k_row3": 4, "k_row4": 3, "k_row5": 3, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec9h_v269", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V269_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V269_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V269_SMOKE_NAMES", "adjective_preposition_at_within").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec9h_v269", "candidate_id": "corpus.unit_tier3_batch_amended_spec9h_v269", "bars": BARS,
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
