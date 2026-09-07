#!/usr/bin/env python3
# BQGATE: five frozen predictions; behaviours, units (v197/v201 receipts), recipe and bars fixed before the run. Measurement only: the standing is NOT re-scored.
"""v203: the row-4 C-UB misses -- how many are instrument width, measured with the split-swap pooled to 32 held C rows?

Across v197 (24) and v201 (9), 10 of 42 rows were lost on row 4 (constrained rank-1 own-C CE UB975 <= 0.01 on 16 held C rows)
with a POINT estimate <= 0.011 (printed from the receipts before this file was written -- point / UB): additive_scope
0.0026/0.0124, animacy 0.0074/0.0139, coordination_agreement -0.0041/0.0123, possessive_long_simple -0.0025/0.0118,
possessive_number 0.0045/0.0122, countability -0.0019/0.0124, dative -0.0018/0.0114, modal_remoteness 0.0095/0.0179, existential
0.0108/0.0233, verb_complementizer 0.0059/0.0202. One miss is real: quantifier_number 0.0191/0.0351. Two passes are controls:
finiteness_selection -0.0004/0.0041, verb_preposition 0.0037/0.0075. The 12:44 board PROPOSAL (i) says: pool both halves' held C
rows (32) so the 0.01 bar sits above the instrument's +-0.01 resolution. This rung MEASURES what that proposal would read, without
re-scoring anything: for each behaviour, fit A = the battery's fit (A1 FIT half, own-C FIT half as control; evaluated on C HELD --
reproduces the receipt), fit B = the split-swap (A1 HELD half as fit rows, C HELD as control; evaluated on C FIT). Each C row is
evaluated by the fit that never saw it; the two 16-row evaluations pool to 32 held rows and the same bootstrap gives the point and
UB975. Units frozen from the parent receipts; recipe identical (rank 1, 120 steps, lr 0.05, seed 0, complement 1.0, lambda 30,
mu = pooled mean of the fit rows). 13 behaviours x 2 fits ~ 6 min GPU.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_instrument   -> fit A's C UB975 reproduces the parent receipt within 0.003 on 13/13 (same seed, rows, code path:
                         deterministic, disclosed as not an independent outcome).
  pred_b_width_halves -> pooled (UB975 - point) <= 0.75 x fit A's (UB975 - point) on >= 10 of 13 (sqrt 2 = 0.71).  prior 70%
  pred_c_width_misses -> of the 10 point <= 0.011 misses, pooled UB975 <= 0.01 on >= 5 (belief 6: the six with point <= 0.005;
                         animacy, modal, existential, complementizer stay out).                              prior 55%
  pred_d_real_stays   -> quantifier_number pooled UB975 > 0.01 AND both passing controls pooled UB975 <= 0.01.  prior 75%
  pred_e_swap_agrees  -> |fit B's C point on C FIT - fit A's C point on C HELD| <= 0.01 on >= 10 of 13 (the residual is noise,
                         not fit instability).                                                               prior 60%
Accounting: no row is re-scored here. If pred_c holds the proposal gains its evidence; the standing changes only when the
protocol owner accepts (i) on the board.
Smoke: V203_SMOKE=<out.json> (CPU, V203_SMOKE_ROWS=4, V203_SMOKE_NAMES=dative; steps 5).
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
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_row4_pooled_width_v203_result.json"
PARENTS = {"v197": ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json",
           "v201": ROOT / "circuits/followups/unit_tier3_batch_amended_new9_v201_result.json"}
NAMES = {**v197.NAMES, **v201.NAMES}
WIDTH = ("additive_scope", "animacy", "coordination_agreement", "possessive_long_simple", "possessive_number",
         "countability", "dative", "modal_remoteness", "existential", "verb_complementizer")
REAL, PASSING = ("quantifier_number",), ("finiteness_selection", "verb_preposition")
SETS = WIDTH + REAL + PASSING
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"instr_tol": 0.003, "width_ratio": 0.75, "k_b": 10, "c_ub_max": 0.01, "k_c": 5, "swap_tol": 0.01, "k_e": 10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60000, 2000000


def _plan():
    return {"candidate_id": "corpus.unit_row4_pooled_width_v203", "behaviours": len(SETS), "fits": 2 * len(SETS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * len(SETS) * STEPS, "model_updates": 0, "fit_parameters": 2 * len(SETS) * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "pooled" in r and "error" not in r}
    ok = bool(good)
    w = lambda s: s["ce_ub975"] - s["ce_damage"]
    a = ok and len(good) == len(SETS) and all(abs(good[n]["fitA"]["C"]["ce_ub975"] - good[n]["parent_C"]["ce_ub975"]) <= B["instr_tol"] for n in good)
    b = ok and sum(1 for n in good if w(good[n]["pooled"]) <= B["width_ratio"] * w(good[n]["fitA"]["C"])) >= B["k_b"]
    c = ok and sum(1 for n in WIDTH if n in good and good[n]["pooled"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_c"]
    d = ok and all(n in good and good[n]["pooled"]["ce_ub975"] > B["c_ub_max"] for n in REAL) and all(n in good and good[n]["pooled"]["ce_ub975"] <= B["c_ub_max"] for n in PASSING)
    e = ok and sum(1 for n in good if abs(good[n]["fitB"]["C"]["ce_damage"] - good[n]["fitA"]["C"]["ce_damage"]) <= B["swap_tol"]) >= B["k_e"]
    return {"pred_a_instrument": bool(a), "pred_b_width_halves": bool(b), "pred_c_width_misses": bool(c), "pred_d_real_stays": bool(d), "pred_e_swap_agrees": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V203_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V203_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    which = [n for n in SETS if not smoke or n in os.environ.get("V203_SMOKE_NAMES", "dative").split(",")]
    prior = {}
    for tag, path in PARENTS.items():
        for n, r in json.loads(path.read_text())["behaviours"].items():
            if n in SETS and n not in prior:
                prior[n] = dict(r, parent=tag)
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def summ(d):
        s = v51.summary(torch, d)
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "documents")}

    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            if "error" in prior.get(n, {"error": "missing"}):
                raise RuntimeError(f"no clean parent receipt for {n}")
            units = list(prior[n]["units"])
            V = dict(valid_only=True) if prior[n]["parent"] == "v201" else {}   # v201 preps dropped capability-failure rows; v197 did not
            m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
            a1, c = g.rows_of(m, "A1"), g.rows_of(m, "C")
            A = {"fit": g.prepare(backend, cut(fit_half(a1)), **V), "held": g.prepare(backend, cut(held_half(a1)), **V)}
            C = {"fit": g.prepare(backend, cut(fit_half(c))), "held": g.prepare(backend, cut(held_half(c)))}
            out, raw = {}, {}
            for tag, fit_key, eval_key in (("fitA", "fit", "held"), ("fitB", "held", "fit")):
                mu = mu_of(A[fit_key], units)
                q, hist = g.fit_block_subspace_constrained(backend, A[fit_key], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
                                                           controls=(C[fit_key],), control_weight=LAM, mu=mu)
                raw[tag] = {"C": v51.removal(backend, C[eval_key], units, q, mu), "A1": v51.removal(backend, A[eval_key], units, q, mu)}
                out[tag] = {"C": summ(raw[tag]["C"]), "A1": summ(raw[tag]["A1"]), "eval_rows": eval_key, "final_loss": (hist[-1] if hist else None)}
            pooled = {k: raw["fitA"]["C"][k] + raw["fitB"]["C"][k] for k in raw["fitA"]["C"]}
            pooled_a1 = {k: raw["fitA"]["A1"][k] + raw["fitB"]["A1"][k] for k in raw["fitA"]["A1"]}
            pc = prior[n]["arms"]["cdas"]["C"]
            R[n] = {"units": units, "n_units": len(units), "parent": prior[n]["parent"], "parent_C": {k: pc[k] for k in ("ce_damage", "ce_lb975", "ce_ub975")},
                    "parent_row4": prior[n]["rows"]["row4"], "group": ("width" if n in WIDTH else "real" if n in REAL else "passing"),
                    "fitA": out["fitA"], "fitB": out["fitB"], "pooled": summ(pooled), "pooled_A1": summ(pooled_a1),
                    "would_pass_row4_pooled": summ(pooled)["ce_ub975"] <= BARS["c_ub_max"],
                    "n_rows": {"A1_fit": len(A["fit"].rows), "A1_held": len(A["held"].rows), "C_fit": len(C["fit"].rows), "C_held": len(C["held"].rows)},
                    "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one behaviour must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k in ("parent_C", "pooled", "would_pass_row4_pooled", "error", "seconds")}),
              "fitA C", R[n].get("fitA", {}).get("C", {}).get("ce_ub975"), "fitB C", R[n].get("fitB", {}).get("C", {}).get("ce_damage"), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    table = {n: {"parent_point_ub": (R[n]["parent_C"]["ce_damage"], R[n]["parent_C"]["ce_ub975"]), "pooled_point_ub": (R[n]["pooled"]["ce_damage"], R[n]["pooled"]["ce_ub975"]),
                 "would_pass": R[n]["would_pass_row4_pooled"], "group": R[n]["group"]} for n in R if "error" not in R[n]}
    result = {"predictions": predictions, "schema": "unit_row4_pooled_width_v203", "candidate_id": "corpus.unit_row4_pooled_width_v203", "bars": BARS,
              "groups": {"width": list(WIDTH), "real": list(REAL), "passing": list(PASSING)}, "table": table, "behaviours": R,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 + v201 receipts",
                         "pooling": "fit A (A1 FIT, C FIT control) evaluated on C HELD + fit B (A1 HELD, C HELD control) evaluated on C FIT; per-document CE pooled, v51 bootstrap"},
              "note": "measurement of board proposal (i); no standing row is re-scored by this receipt",
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out_path = Path(smoke) if smoke else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "table": table, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
