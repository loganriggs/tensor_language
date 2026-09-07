#!/usr/bin/env python3
# BQGATE: five frozen predictions; head sets copied from the v113/v112 receipts (no re-selection); bars fixed before the run.
"""v193: are the battery's row-5 misses (A2 CE damage < 0.5 x A1) construction-KEYED DIRECTIONS on SHARED heads?

v192 (animacy) showed the pattern once: the A1 head set carries the A2 construction by exact interchange (0.903), the A1-fitted
diff-in-means direction under-damages A2 (0.37x in CE) and an A2-own direction on the same heads reaches 0.367 (above the 0.5 x A1
bar). This rung asks whether the SAME signature explains the remaining row-5 misses of the tier-3 battery: finiteness_selection
(v113: A1 CE 0.770, A2 0.297, 8 heads, row 2 also fails at ext 0.641), numbered_list_choice (v113: A1 0.816, A2 0.100, 3 heads
attn:08 {03,07,04}), numeric_sequence_choice (v112: A1 0.765, A2 0.323, 3 heads). Sets are the receipts' sets -- no greedy here --
so the direction is the only thing that varies between the arms.

Magnitudes printed BEFORE writing the runner (CPU probe, 8 rows per split, ODD eval; disclosed because they are the same measurement
on a subset of the run's rows, so preds a-c are only partly independent run outcomes):
    A2 exact-set recovery on the A1 set: finiteness 0.564 (A1 0.640), numbered_list 0.869 (A1 0.906), numeric_sequence 0.850 (A1 0.834)
    numbered_list: A1-own CE 0.752, A2-by-A1 0.104, A2-OWN 0.806, A1-by-A2 0.434; per-block |cos(q_A1, q_A2)| = 0.067
    finiteness:    A1-own CE 0.772, A2-by-A1 0.337, A2-OWN 0.700, A1-by-A2 0.226; |cos| 0.35-0.54 (mean 0.44)
    numeric_sequence: not probed (registered from the other two).
Direction = g.block_diff_in_means per (layer, 'heads') block, fitted on EVEN rows of ONE construction; removal = v51.removal (both
interchange sides pooled) on the ODD rows of the construction named; mu = pooled EVEN mean of the fitting construction.

REGISTERED (all bars in BARS; a coded predicate is the sentence here):
  pred_a_shared_heads   -> A2 ODD exact-set recovery on the A1 set within [0.70, 1.20] for numbered_list_choice and
                           numeric_sequence_choice and within [0.40, 0.80] for finiteness_selection (the set is weak for BOTH
                           constructions there -- row 2 fails on A1 too).
  pred_b_own_direction_repairs_row5 -> on all three behaviours the A2-OWN direction's CE damage on A2 ODD lies within
                           [0.5, 1.3] x the A1-own CE damage on A1 ODD (two-sided: an own direction that OVER-damages 1.3x would
                           say the A2 removal is doing something else).
  pred_c_directions_are_construction_keyed -> on all three behaviours BOTH cross-transfers are weak: A2-by-A1 CE <= 0.65 x A2-own
                           and A1-by-A2 CE <= 0.65 x A1-own (probe: 0.13 / 0.58 numbered; 0.48 / 0.29 finiteness), AND the mean
                           per-block |cos(q_A1, q_A2)| <= 0.60 on all three.
  pred_d_cos_orders_transfer -> across the three behaviours the mean |cos| orders the A2-by-A1 / A2-own ratio monotonically
                           (Spearman = +1 over 3 points): lower cosine, weaker transfer. Three points; a coarse ordering test.
  pred_e_instrument     -> A1-own CE damage on A1 ODD reproduces the parent receipts within +-0.06: finiteness 0.770, numbered_list
                           0.816, numeric_sequence 0.765 (v112 protocol; same dim arm), and A1 ODD exact-set recovery within +-0.04 of
                           the receipts (0.641, 0.908, 0.832).
Reading if a+b+c hold: the tier-3 row 5 conflates 'same heads' with 'same rank-1 direction'; the misses are one circuit per
behaviour with a construction-keyed direction (memory: directions are cue-pair keyed) -- the battery's row 5 needs an
own-construction-direction arm registered as a protocol amendment (not a retroactive pass). If b fails, the A2 construction is
genuinely under-served by these heads' rank-1 write and the miss stands.
Smoke: V193_SMOKE=<out.json> (CPU, 4 rows per split, V193_SMOKE_ROWS).
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
OUT = ROOT / "circuits/followups/unit_row5_construction_keyed_direction_v193_result.json"
V113 = ROOT / "circuits/followups/unit_tier3_batch_row2_v113_result.json"
V112 = ROOT / "circuits/followups/unit_tier3_batch_v112_result.json"
NAMES = {"finiteness_selection": "finiteness", "numbered_list_choice": "control_choice", "numeric_sequence_choice": "sequence_control_choice"}
PARENT = {"finiteness_selection": ("v113", 0.770, 0.641), "numbered_list_choice": ("v113", 0.816, 0.908), "numeric_sequence_choice": ("v112", 0.765, 0.832)}
BARS = {"a2_ext_band": {"finiteness_selection": [0.40, 0.80], "numbered_list_choice": [0.70, 1.20], "numeric_sequence_choice": [0.70, 1.20]},
        "own_ratio_band": [0.5, 1.3], "cross_max_ratio": 0.65, "cos_max": 0.60, "a1_ce_tol": 0.06, "a1_ext_tol": 0.04}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3000, 96000


def _plan():
    return {"candidate_id": "corpus.unit_row5_construction_keyed_direction_v193", "behaviours": 3, "constructions": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    names = [n for n in NAMES if n in R]
    if not names: return {k: False for k in ("pred_a_shared_heads", "pred_b_own_direction_repairs_row5", "pred_c_directions_are_construction_keyed", "pred_d_cos_orders_transfer", "pred_e_instrument")}
    ratio = lambda x, y: (x / y) if (x is not None and y) else None
    a = all(inb(R[n]["a2_ext_odd_on_a1_set"], B["a2_ext_band"][n]) for n in names)
    b = all(inb(ratio(R[n]["a2_own_ce"], R[n]["a1_own_ce"]), B["own_ratio_band"]) for n in names)
    c = all((ratio(R[n]["a2_by_a1_ce"], R[n]["a2_own_ce"]) or 9) <= B["cross_max_ratio"] and (ratio(R[n]["a1_by_a2_ce"], R[n]["a1_own_ce"]) or 9) <= B["cross_max_ratio"]
            and R[n]["cos_mean"] <= B["cos_max"] for n in names)
    pts = [(R[n]["cos_mean"], ratio(R[n]["a2_by_a1_ce"], R[n]["a2_own_ce"])) for n in names]
    d = len(pts) == 3 and all(p[1] is not None for p in pts) and [p[1] for p in sorted(pts)] == sorted(p[1] for p in pts) and len({p[1] for p in pts}) == 3
    e = all(abs(R[n]["a1_own_ce"] - PARENT[n][1]) <= B["a1_ce_tol"] and abs(R[n]["a1_ext_odd"] - PARENT[n][2]) <= B["a1_ext_tol"] for n in names)
    return {"pred_a_shared_heads": bool(a), "pred_b_own_direction_repairs_row5": bool(b), "pred_c_directions_are_construction_keyed": bool(c),
            "pred_d_cos_orders_transfer": bool(d), "pred_e_instrument": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V193_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V193_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    sets = {**{k: v["units"] for k, v in json.loads(V112.read_text())["behaviours"].items() if "units" in v},
            **{k: v["units"] for k, v in json.loads(V113.read_text())["behaviours"].items() if "units" in v}}
    which = [n for n in NAMES if not smoke or n in os.environ.get("V193_SMOKE_NAMES", "numbered_list_choice,finiteness_selection").split(",")]

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
        cos = {f"{k[0]:02d}:{k[1]}": round(abs(float(torch.nn.functional.cosine_similarity(torch.as_tensor(q["A1"][k]).flatten().float(), torch.as_tensor(q["A2"][k]).flatten().float(), dim=0))), 4) for k in q["A1"]}
        arms = {f"{tgt}_by_{src}": rem(P[tgt]["odd"], units, q[src], mu[src]) for tgt in ("A1", "A2") for src in ("A1", "A2")}
        R[name] = {"units": units, "n_units": len(units), "parent": PARENT[name][0],
                   "a1_ext_odd": round(g.recovery(P["A1"]["odd"], g.patched_axis(backend, P["A1"]["odd"], units)), 3),
                   "a2_ext_odd_on_a1_set": round(g.recovery(P["A2"]["odd"], g.patched_axis(backend, P["A2"]["odd"], units)), 3),
                   "a1_own_ce": arms["A1_by_A1"]["ce_damage"], "a2_own_ce": arms["A2_by_A2"]["ce_damage"],
                   "a2_by_a1_ce": arms["A2_by_A1"]["ce_damage"], "a1_by_a2_ce": arms["A1_by_A2"]["ce_damage"],
                   "cos_mean": round(sum(cos.values()) / len(cos), 4), "cos_per_block": cos, "arms": arms,
                   "n_rows": {fam: {sp: len(P[fam][sp].rows) for sp in ("even", "odd")} for fam in ("A1", "A2")}}
        print(name, json.dumps({k: v for k, v in R[name].items() if k not in ("arms", "units")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_row5_construction_keyed_direction_v193", "candidate_id": "corpus.unit_row5_construction_keyed_direction_v193",
              "bars": BARS, "parent": PARENT, "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
