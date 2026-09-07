#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets read from the v190/v112 receipts, behaviours and bars fixed before the run; diff-in-means only (no fit).
"""v191: is possessive_number's v190 head set a NEW circuit, or the v112 possessive family's circuit again?

v190 lifted possessive_number to all four rows with the set {10:05, 12:04, 15:01, 09:06, 05:03, 09:07, 10:01, 11:01} (ODD ext 0.845,
dim A1 CE damage 0.592). The five v112 possessive behaviours (adjacent / argument / long_simple / medial / verbfinal) were localised
to sets built from the same heads -- every one of them contains {09:06, 05:03, 04:05, 15:01} and four of five contain 10:05, 10:01,
12:04 -- and their within-family cross damage in v112 was 0.27-0.58 against own 0.40-0.61. The standing rule (circuit battery
protocol): a set is counted as a circuit only after a within-family separability pass. Numbers printed before writing (from the
receipts): Jaccard(possessive_number, v112 set) = adjacent 5/11 = 0.45, argument 6/9 = 0.67, long_simple 5/10 = 0.50,
medial 6/10 = 0.60, verbfinal 6/9 = 0.67; v190 cross damage of possessive_number's direction on quantifier_number 0.317 (the number
axis is shared within the number family, hub-heads memory) and on animacy < 0.32 (cross_abs_max was 0.317).

Protocol (all diff-in-means, no fit; v51 mean-removal CE damage on ODD A1 rows; mu = pooled EVEN mean of the fitting behaviour):
  own      possessive_number set + its EVEN dim direction on possessive_number ODD A1 (must reproduce v190: 0.592).
  family   the same set + direction on each v112 possessive behaviour's ODD A1 (five numbers).
  number   the same on quantifier_number ODD A1;   non-number control: the same on animacy ODD A1.
  reverse  possessive_verbfinal's v112 set + its EVEN dim direction on possessive_number ODD A1 (own v112: 0.611) and on its own ODD A1.
  random   an equal-size random head set (seed 0) with its dim direction on possessive_number ODD A1.

REGISTERED BEFORE THE RUN
  pred_a_same_heads    Jaccard >= 0.5 with at least 3 of the 5 v112 sets. DETERMINISTIC from the receipts (4 of 5 as printed
                       above); recorded as a bookkeeping check, NOT an independent run outcome.
  pred_b_own_repro     own damage within 0.592 +- 0.05 (same seed, same rows).
  pred_c_family_shared family damage / own in 0.5-1.5 on >= 4 of 5 possessive behaviours (the SAME circuit: its direction removes
                       their behaviour as strongly as its own).       Worked: 0.35/0.59 = 0.59 True; 0.20/0.59 = 0.34 False.
  pred_d_controls      quantifier damage / own in 0.2-0.9 (number axis shared, but a weaker route) AND |animacy damage| / own <= 0.25
                       AND |random-set damage| / own <= 0.25.
  pred_e_reverse       verbfinal's set+direction damages possessive_number at 0.5-1.5 x its own-behaviour damage.
  Prior: a 100 % (deterministic); b 90 %; c 70 %; d 55 %; e 65 %.
  Smoke (CPU, 4 ODD/4 EVEN rows, possessive_number / verbfinal / animacy only): own 0.445, verbfinal 0.480 (1.08x), animacy 0.021 (0.05x),
  random -0.0003, reverse own 0.570 / on possessive_number 0.461 (0.81x). No bar was changed by the smoke (quantifier and the other four
  possessive behaviours were not in the smoke).
If pred_c AND pred_e hold, possessive_number is booked as the possessive circuit's SIXTH construction (tally +0 circuits, +1 construction);
if both fail it is a new circuit (+1).
"""
from __future__ import annotations

import importlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_possessive_number_separability_v191_result.json"
V190 = ROOT / "circuits/followups/unit_tier3_batch_unlifted_v190_result.json"
V112 = ROOT / "circuits/followups/unit_tier3_batch_v112_result.json"
FAMILY = {"possessive_adjacent": "possessive_adjacent", "possessive_argument": "possessive_argument",
          "possessive_long_simple": "possessive_long_simple", "possessive_medial": "possessive_medial",
          "possessive_verbfinal": "possessive_verbfinal"}
NAMES = {"possessive_number": "possessive_number", **FAMILY, "quantifier_number": "quantifier_number", "animacy": "animacy"}
BARS = {"jaccard_min": 0.5, "jaccard_min_sets": 3, "own_repro": [0.592, 0.05], "family_band": [0.5, 1.5], "family_min": 4,
        "quant_band": [0.2, 0.9], "animacy_abs_max": 0.25, "random_abs_max": 0.25, "reverse_band": [0.5, 1.5], "random_seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 12800


def _plan():
    return {"candidate_id": "corpus.unit_possessive_number_separability_v191", "behaviours": len(NAMES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    own = R["own"]
    ratio = lambda x: (x / own) if (own and x is not None) else None
    pred_a = sum(j >= B["jaccard_min"] for j in R["jaccard"].values()) >= B["jaccard_min_sets"]
    pred_b = inb(own, [B["own_repro"][0] - B["own_repro"][1], B["own_repro"][0] + B["own_repro"][1]])
    pred_c = sum(inb(ratio(R["family"][n]), B["family_band"]) for n in FAMILY) >= B["family_min"]
    pred_d = inb(ratio(R["quantifier"]), B["quant_band"]) and ratio(R["animacy"]) is not None and abs(ratio(R["animacy"])) <= B["animacy_abs_max"] and abs(ratio(R["random"])) <= B["random_abs_max"]
    pred_e = inb(R["reverse_on_possessive_number"] / R["reverse_own"], B["reverse_band"]) if R["reverse_own"] else False
    return {"pred_a_same_heads": bool(pred_a), "pred_b_own_repro": bool(pred_b), "pred_c_family_shared": bool(pred_c),
            "pred_d_controls": bool(pred_d), "pred_e_reverse": bool(pred_e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V191_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    r190 = json.loads(V190.read_text())["behaviours"]; r112 = json.loads(V112.read_text())["behaviours"]
    units_pn = list(r190["possessive_number"]["units"]); units_vf = list(r112["possessive_verbfinal"]["units"])
    v112_sets = {n: set(r112[n]["units"]) for n in FAMILY}
    jaccard = {n: round(len(set(units_pn) & s) / len(set(units_pn) | s), 3) for n, s in v112_sets.items()}
    all_heads = [f"attn:{l:02d}:head:{h:02d}" for l in range(18) for h in range(9)]
    rng = random.Random(BARS["random_seed"]); units_rand = rng.sample([u for u in all_heads if u not in units_pn], len(units_pn))
    cut = (lambda rows: rows[:int(os.environ.get("V191_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    names = {k: NAMES[k] for k in (("possessive_number", "possessive_verbfinal", "animacy") if smoke else NAMES)}
    mods = {n: importlib.import_module(f"circuit_fast_screen_candidate_{s}") for n, s in names.items()}
    prep = {n: {"even": g.prepare(backend, cut(g.rows_of(m, "A1")[0::2])), "odd": g.prepare(backend, cut(g.rows_of(m, "A1")[1::2]))} for n, m in mods.items()}

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    P = prep["possessive_number"]
    mu_pn = mu_of(P["even"], units_pn); q_pn = g.block_diff_in_means(backend, P["even"], units_pn)
    D = {"own": dmg(P["odd"], units_pn, q_pn, mu_pn)}
    D["family"] = {n: dmg(prep[n]["odd"], units_pn, q_pn, mu_pn) for n in FAMILY if n in prep}
    D["quantifier"] = dmg(prep["quantifier_number"]["odd"], units_pn, q_pn, mu_pn) if "quantifier_number" in prep else None
    D["animacy"] = dmg(prep["animacy"]["odd"], units_pn, q_pn, mu_pn)
    mu_r = mu_of(P["even"], units_rand); q_r = g.block_diff_in_means(backend, P["even"], units_rand)
    D["random"] = dmg(P["odd"], units_rand, q_r, mu_r)
    PV = prep["possessive_verbfinal"]
    mu_vf = mu_of(PV["even"], units_vf); q_vf = g.block_diff_in_means(backend, PV["even"], units_vf)
    D["reverse_own"] = dmg(PV["odd"], units_vf, q_vf, mu_vf)
    D["reverse_on_possessive_number"] = dmg(P["odd"], units_vf, q_vf, mu_vf)
    ce = lambda d: d["ce_damage"] if d else None
    R = {"jaccard": jaccard, "own": ce(D["own"]), "family": {n: ce(d) for n, d in D["family"].items()}, "quantifier": ce(D["quantifier"]),
         "animacy": ce(D["animacy"]), "random": ce(D["random"]), "reverse_own": ce(D["reverse_own"]),
         "reverse_on_possessive_number": ce(D["reverse_on_possessive_number"])}
    for n in FAMILY:
        R["family"].setdefault(n, None)
    print(json.dumps(R, indent=1), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_possessive_number_separability_v191", "candidate_id": "corpus.unit_possessive_number_separability_v191",
              "bars": BARS, "units": {"possessive_number": units_pn, "possessive_verbfinal_v112": units_vf, "random": units_rand},
              "measures": R, "detail": D, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
