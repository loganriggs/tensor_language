#!/usr/bin/env python3
# BQGATE: five frozen predictions; head sets copied from the v190/v113/v112 receipts (no re-selection); bars fixed before the run.
"""v194: re-normalise tier-3 row 5 by the construction's OWN full-rank ceiling (mean-ablation of the set), per construction.

v192/v193: of the battery's four row-5 misses, three (animacy, finiteness_selection, numbered_list_choice) are construction-keyed
rank-1 directions on shared heads (an A2-own diff-in-means direction repairs them), while numeric_sequence_choice's two directions
are SHARED (|cos| 0.60, A1 direction = 0.97x the A2-own) and its miss looked like a CE-scale artefact (A2 margins 2.3x smaller).
The row-5 bar 'A2 CE >= 0.5 x A1 CE' compares a construction against ANOTHER construction's ceiling. The right denominator is the
construction's own full-rank removal: the same set mean-ablated at full rank on the same rows (v51.removal with q=None).

Magnitudes printed BEFORE writing (CPU probe, 6 rows per split, ODD eval; disclosed -- preds a, b, d are partly pre-measured):
    numeric_sequence: A1 full 0.836 / rank-1 own 0.787 (share 0.94); A2 full 0.367 / own 0.328 (0.89); A2 by A1 0.321 -> the A2
                      CEILING is 0.44x the A1 ceiling; rank-1 captures 0.89 of it. The miss was the denominator.
    numbered_list:    A1 0.792 / 0.770 (0.97); A2 0.762 / 0.832 (1.09); A2 by A1 0.102 (0.13 of the A2 ceiling) -> keyed direction.
    animacy:          A1 0.620 / 0.758 (1.22); A2 0.482 / 0.250 (0.52); A2 by A1 0.199 (0.41 of ceiling).
    perfect_number (clean row-5 pass, control): A1 0.743 / 0.668 (0.90); A2 0.411 / 0.496 (1.21); A2 by A1 0.465 (1.13).
    finiteness_selection and quantifier_number: not probed (registered from the others).
Rank-1 removal can exceed the full-rank removal (shares 1.09-1.22): block-live rank-1 keeps the live complement (memory v153-v157),
so 'share' is a ratio of two different interventions, not a fraction of a total; bars are two-sided for that reason.

Behaviours: the four misses (animacy set from v190; finiteness_selection, numbered_list_choice from v113; numeric_sequence_choice from
v112) + two clean row-5 passes as controls (perfect_number 0.81x, quantifier_number 1.02x, v113 sets). Directions: g.block_diff_in_means
per (layer, 'heads') block fitted on the construction's EVEN rows; mu = that construction's pooled EVEN mean; eval on ODD rows.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_a1_share      -> A1 rank-1 own share (A1-own CE / A1 full-rank CE) within [0.6, 1.4] on all six behaviours.
  pred_b_a2_own_share  -> A2 rank-1 OWN share (A2-own CE / A2 full-rank CE) within [0.5, 1.4] on all six -- the amended row 5
                          holds everywhere once the direction is the construction's own and the denominator is its own ceiling.
  pred_c_ceiling_ratio -> A2 full-rank CE / A1 full-rank CE <= 0.5 on numeric_sequence_choice (the construction's ceiling, not
                          the direction, is what row 5 compared) and >= 0.6 on numbered_list_choice and finiteness_selection
                          (their ceilings match; the direction was the miss). Controls and animacy unbounded here.
  pred_d_a1_direction_share -> A2-by-A1-direction CE / A2 full-rank CE <= 0.5 on the three keyed behaviours (animacy,
                          finiteness_selection, numbered_list_choice) AND >= 0.5 on numeric_sequence_choice and both controls:
                          the A1-direction share separates 'keyed direction' from 'shared direction' cleanly at 0.5.
  pred_e_instrument    -> A1-own CE reproduces the parent receipts within +-0.06 on all six: animacy 0.670 (v190), finiteness 0.770,
                          numbered_list 0.816, perfect_number 0.669, quantifier_number 0.601 (v113), numeric_sequence 0.765 (v112).
Reading if a-d hold: row 5 of the battery gets two registered amendments -- the own-construction direction, and the construction's
own full-rank ceiling as denominator -- and the four 'misses' are one circuit each (rows 2-4 unchanged). This is a protocol
amendment for FUTURE lifts; v112/v113 partials are not re-scored retroactively.
Smoke: V194_SMOKE=<out.json> (CPU, V194_SMOKE_ROWS=4 per split, V194_SMOKE_NAMES=numbered_list_choice).
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
OUT = ROOT / "circuits/followups/unit_row5_ceiling_share_v194_result.json"
V113 = ROOT / "circuits/followups/unit_tier3_batch_row2_v113_result.json"
V112 = ROOT / "circuits/followups/unit_tier3_batch_v112_result.json"
V190 = ROOT / "circuits/followups/unit_tier3_batch_unlifted_v190_result.json"
NAMES = {"animacy": "animacy", "finiteness_selection": "finiteness", "numbered_list_choice": "control_choice", "numeric_sequence_choice": "sequence_control_choice",
         "perfect_number": "perfect_number", "quantifier_number": "quantifier_number"}
KEYED = ("animacy", "finiteness_selection", "numbered_list_choice")
SHARED = ("numeric_sequence_choice", "perfect_number", "quantifier_number")
PARENT = {"animacy": ("v190", 0.670), "finiteness_selection": ("v113", 0.770), "numbered_list_choice": ("v113", 0.816), "numeric_sequence_choice": ("v112", 0.765),
          "perfect_number": ("v113", 0.669), "quantifier_number": ("v113", 0.601)}
BARS = {"a1_share_band": [0.6, 1.4], "a2_own_share_band": [0.5, 1.4], "ceiling_ratio_low": {"numeric_sequence_choice": 0.5}, "ceiling_ratio_high": {"numbered_list_choice": 0.6, "finiteness_selection": 0.6},
        "a1_direction_share_cut": 0.5, "a1_ce_tol": 0.06}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3000, 96000


def _plan():
    return {"candidate_id": "corpus.unit_row5_ceiling_share_v194", "behaviours": 6, "constructions": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    names = [n for n in NAMES if n in R]
    ok = bool(names)  # an empty receipt fails every prediction (all() over nothing would pass)
    a = ok and all(inb(R[n]["a1_share"], B["a1_share_band"]) for n in names)
    b = ok and all(inb(R[n]["a2_own_share"], B["a2_own_share_band"]) for n in names)
    c = ok and all(R[n]["ceiling_ratio_a2_over_a1"] is not None and R[n]["ceiling_ratio_a2_over_a1"] <= v for n, v in B["ceiling_ratio_low"].items() if n in R) \
        and all(R[n]["ceiling_ratio_a2_over_a1"] is not None and R[n]["ceiling_ratio_a2_over_a1"] >= v for n, v in B["ceiling_ratio_high"].items() if n in R)
    d = ok and all(R[n]["a2_by_a1_share"] is not None and (R[n]["a2_by_a1_share"] <= B["a1_direction_share_cut"] if n in KEYED else R[n]["a2_by_a1_share"] >= B["a1_direction_share_cut"]) for n in names)
    e = ok and all(abs(R[n]["a1_own_ce"] - PARENT[n][1]) <= B["a1_ce_tol"] for n in names)
    return {"pred_a_a1_share": bool(a), "pred_b_a2_own_share": bool(b), "pred_c_ceiling_ratio": bool(c), "pred_d_a1_direction_share": bool(d), "pred_e_instrument": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V194_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V194_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    sets = {**{k: v["units"] for k, v in json.loads(V112.read_text())["behaviours"].items() if "units" in v},
            **{k: v["units"] for k, v in json.loads(V113.read_text())["behaviours"].items() if "units" in v},
            "animacy": json.loads(V190.read_text())["behaviours"]["animacy"]["units"]}
    which = [n for n in NAMES if not smoke or n in os.environ.get("V194_SMOKE_NAMES", "numbered_list_choice").split(",")]

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def rem(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    R = {}
    for name in which:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[name]}")
        units = list(sets[name])
        P = {fam: {"even": g.prepare(backend, cut(g.rows_of(m, fam)[0::2])), "odd": g.prepare(backend, cut(g.rows_of(m, fam)[1::2]))} for fam in ("A1", "A2")}
        q = {fam: g.block_diff_in_means(backend, P[fam]["even"], units) for fam in ("A1", "A2")}
        mu = {fam: mu_of(P[fam]["even"], units) for fam in ("A1", "A2")}
        arms = {f"{tgt}_by_{src}": rem(P[tgt]["odd"], units, q[src], mu[src]) for tgt in ("A1", "A2") for src in ("A1", "A2")}
        arms.update({f"{fam}_full": rem(P[fam]["odd"], units, None, mu[fam]) for fam in ("A1", "A2")})  # q=None: full-rank mean-ablation of the set
        ratio = lambda x, y: round(x / y, 4) if y else None
        R[name] = {"units": units, "n_units": len(units), "parent": PARENT[name][0], "class": "keyed" if name in KEYED else "shared",
                   "a1_own_ce": arms["A1_by_A1"]["ce_damage"], "a2_own_ce": arms["A2_by_A2"]["ce_damage"], "a2_by_a1_ce": arms["A2_by_A1"]["ce_damage"],
                   "a1_full_ce": arms["A1_full"]["ce_damage"], "a2_full_ce": arms["A2_full"]["ce_damage"],
                   "a1_share": ratio(arms["A1_by_A1"]["ce_damage"], arms["A1_full"]["ce_damage"]),
                   "a2_own_share": ratio(arms["A2_by_A2"]["ce_damage"], arms["A2_full"]["ce_damage"]),
                   "a2_by_a1_share": ratio(arms["A2_by_A1"]["ce_damage"], arms["A2_full"]["ce_damage"]),
                   "ceiling_ratio_a2_over_a1": ratio(arms["A2_full"]["ce_damage"], arms["A1_full"]["ce_damage"]),
                   "old_row5_ratio_a2_by_a1_over_a1_own": ratio(arms["A2_by_A1"]["ce_damage"], arms["A1_by_A1"]["ce_damage"]),
                   "arms": arms, "n_rows": {fam: {sp: len(P[fam][sp].rows) for sp in ("even", "odd")} for fam in ("A1", "A2")}}
        print(name, json.dumps({k: v for k, v in R[name].items() if k not in ("arms", "units")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_row5_ceiling_share_v194", "candidate_id": "corpus.unit_row5_ceiling_share_v194",
              "bars": BARS, "parent": PARENT, "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
