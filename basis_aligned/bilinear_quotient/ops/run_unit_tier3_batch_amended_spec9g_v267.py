#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v197 protocol, unchanged) over the thirty-third five spec-authored behaviours, plus one instrument.
"""v267: the AMENDED tier-3 battery over the THIRTY-THIRD batch (FIVE cells) -- plus one instrument. This batch exists to TEST A MECHANISM,
not to raise the circuit count, and every cell in it is expected to fuse.

WHY. v254 produced the first fusion in the preposition class: verb_preposition_of_over leaked 0.294 into verb_preposition_of_by and 0.134 into
verb_preposition_over_with, and over_with leaked 0.129 back. The leaks did not follow the shared READOUT token (v248/v250/v252: shared readouts
stay separable, including at_to sharing BOTH readouts with counted siblings at sib 0.006). They followed the shared CUE VERB: `presided` is the
` over` cue in both of_over and over_with; `consisted` is the ` of` cue in both of_over and of_by. HYPOTHESIS: these rank-1 directions are keyed
to the cue lexeme, not to the answer token.
v256 (running) tests one side of that: five cells that share a readout with counted siblings but reuse no cue verb; the hypothesis predicts all
five separable. That side can only fail to refute. THIS batch is the other side, and it is the one that can be wrong in the interesting direction.

THE MANIPULATION. Each cell REUSES ONE CUE VERB from a counted verb_preposition sibling, at the same preposition, with a readout pair that is new
to the corpus (word-boundary quote-agnostic grep: on/within, at/under, with/beyond, of/beneath, in/beyond all 0 hits):
verb_preposition_on_within (insisted/dwelled -> on/within); `insisted` is verb_preposition_insisted's ` on` cue;
verb_preposition_at_under (looked/buckled -> at/under); `looked` is verb_preposition_to_at's ` at` cue;
verb_preposition_with_beyond (dealt/stretched -> with/beyond); `dealt` is verb_preposition_with_about's ` with` cue;
verb_preposition_of_beneath (accused/cowered -> of/beneath); `accused` is verb_preposition_of_with's ` of` cue;
verb_preposition_in_beyond (resulted/persisted -> in/beyond); `resulted` is verb_preposition_in_to's ` in` cue.
A sixth cue reuse (prevented, from_for's ` from` cue) failed the CPU floor two-sidedly and was dropped; the experiment runs with five.
` within`, ` beneath` and ` beyond` are new or near-new readout tokens, so the OTHER cue in each pair is unattached to anything counted.

WHAT THIS BATTERY DECIDES AND WHAT IT DOES NOT. This rung only produces four-row passes; the fusion question is settled at the separability rung
that follows it. Registering the prediction here anyway, in the parent, so it cannot be written after the receipt: if directions are cue-keyed,
each passing cell fuses with EXACTLY the sibling whose cue it reuses (sib > 0.05 into that one and <= 0.05 into every other counted sibling), and
each therefore counts +0. If instead they come out separable from the cue-sharing sibling, the cue-lexeme reading of v254 is WRONG and the
of_over/over_with leak needs a different explanation -- which is the outcome I would rather find, because it would mean the corpus's 99 circuits
are not one cue-verb collision away from collapsing into each other.
CPU capability margins BEFORE enqueue (mean base / mean donor on the donor axis; A1 | A2; dropped rows; floor 1.0 on A1 per the v245 lesson):
  verb_preposition_on_within A1:-1.97/+2.21 drop0 A2:-1.94/+2.06 drop0 C:-1.43/+0.93 drop2
  verb_preposition_at_under A1:-1.95/+1.67 drop0 A2:-1.89/+1.87 drop0 C:-1.43/+0.93 drop2
  verb_preposition_with_beyond A1:-5.30/+5.26 drop0 A2:-4.89/+5.08 drop0 C:-1.43/+0.93 drop2
  verb_preposition_of_beneath A1:-6.18/+5.95 drop0 A2:-5.60/+5.65 drop0 C:-1.43/+0.93 drop2
  verb_preposition_in_beyond A1:-1.92/+2.08 drop0 A2:-1.85/+2.28 drop0 C:-1.43/+0.93 drop2
  Five cells at >= 1.67 with zero dropped rows; probed outside the repo (fourth batch on the probe path). ONE candidate never entered ops/:
  verb_preposition_from_within (prevented/stemmed: A1 0.15/0.46 drop12, A2 0.11/0.49 drop6 -- two-sided, dropped), so the `prevented` cue-reuse
  arm of this experiment is missing and the test runs with five cue reuses instead of six; disclosed rather than repaired, since a repair would
  change the cue verb and the cue verb IS the manipulation.
  A generator bug cost one probe cycle here: voc=(' on', ...) was written with a leading space already present, producing vocabulary=('  on',
  ' within') and a CandidateBankError ("punctuation is not its exact standalone continuation token") on every cell in the batch. Caught by the
  probe, outside the repo, before anything was committed. Batch size FIVE; bars are the five-cell bars (3/4/4/3/3).
Protocol identical to v197/v201/.../v265 (greedy pool 40, target 0.88, min_gain 0.005, max 30; row 2 = MIN over the two held directions
>= 0.80; row 3 dim A1 CE LB975 > 0 and >= 0.10; row 4 constrained rank-1 own-C UB975 <= 0.01 on 16 held C rows; row 5 = A2-own / A2 full-rank
ceiling in [0.5, 1.4], LB975 > 0). finiteness_selection is re-run as the instrument and is EXCLUDED from four-row counts (v249 disclosure).

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 3 of the 5 new behaviours pass all four rows (pre-screened batches 39/71 = 0.55).       prior 55%
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 4 of 5 (v265 6/8, v263 4/7).   prior 55%
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on at least 4 of 5 (v265 8/8, v263 7/7).                  prior 80%
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 3 of 5 (v265 5/8, v263 6/7).               prior 65%
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 3 of 5 (v265 8/8, v263 7/7).    prior 75%
  pred_f_instrument     -> finiteness_selection: n_units identical to v197 and extraction_held_dirB within +-0.02 (same seed,
                           split and code path: a determinism check, NOT an independent run outcome, disclosed as such).
No row is predicted from a capability margin here: that mapping was retired at 09:45 after v265 refuted its second form (the thinnest cell in the
batch passed row 2 with the batch's best held-out recovery while a thicker cell failed it).
Accounting: a four-row pass counts only after its separability rung, and every cell here is PREDICTED to count +0 there. That is the trade this
batch makes: five GPU-minutes and a rung of separability spent on a mechanism question rather than on the total.
An error receipt per behaviour counts as a miss on every row. ~40 s GPU each, ~4 min.
A1/A2 preps use g.prepare(valid_only=True); `n_dropped` is in every receipt.
Smoke: V267_SMOKE=<out.json> (CPU, V267_SMOKE_ROWS=4 per split, V267_SMOKE_NAMES=verb_preposition_on_within; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9g_v267_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"   # parent receipt for the instrument (name kept from v197)
NAMES = {"verb_preposition_on_within": "verb_preposition_on_within", "verb_preposition_at_under": "verb_preposition_at_under",
         "verb_preposition_with_beyond": "verb_preposition_with_beyond", "verb_preposition_of_beneath": "verb_preposition_of_beneath",
         "verb_preposition_in_beyond": "verb_preposition_in_beyond",
         "finiteness_selection": "finiteness"}
NEW = tuple(n for n in NAMES if n != "finiteness_selection")
INSTR = ("finiteness_selection",)
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 3, "k_row2": 4, "k_row3": 4, "k_row4": 3, "k_row5": 3, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_spec9g_v267", "behaviours": 9, "constructions": 5,
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
    smoke = os.environ.get("V267_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V267_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V267_SMOKE_NAMES", "verb_preposition_on_within").split(",")]
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
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_spec9g_v267", "candidate_id": "corpus.unit_tier3_batch_amended_spec9g_v267", "bars": BARS,
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
