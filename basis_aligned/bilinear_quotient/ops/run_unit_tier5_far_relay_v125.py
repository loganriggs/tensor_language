#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets fixed by the v124 receipt rule (far non-cue loss >= 0.30), positions by rule.
"""Tier-5: WHERE between the cue and t-3 is the cue relayed, on the four long-offset sets?

v124 clamped every layer's writes at the far NON-cue positions (p < t-3, same token in base and donor) and found it
inert on 6/10 sets but costly on the four whose cue sits 5-9 tokens back: narrative_tense 0.70, possessive_verbfinal
0.55, possessive_long_simple 0.53, possessive_argument 0.47 ('The director(s) packed the crate and trimmed ...' --
the possessor number is carried across the verb phrase). v120 searched for such a relay at t-1..t-3 with layers
0-8 and found none; v123 then found the near-position carrier (attn 05:03) for the possessives. This rung splits
the far non-cue positions into BEFORE the (last) cue and BETWEEN the cue and t-3, and bisects BETWEEN by band and
by position.

Design (donor side, ODD A1 rows, equal-length rows; the same clamp machinery as v124):
  cue      = last position p < t with base_ids[p] != donor_ids[p]
  BEFORE   = {p < cue : equal tokens}          BETWEEN = {cue < p < t-3 : equal tokens}
  arms     far_all (BEFORE + BETWEEN, all bands = v124's number)  before_all  between_all
           between_{early,mid,late,top}  between_first (cue+1 only)  between_rest (cue+2 .. t-4)
  loss     (donor - clamped) / (donor - base) on the donor answer axis; instrument = donor-valued clamp of far_all

Registered before the run (bars fixed; sets = v124 far_noncue_all >= 0.30):
  pred_a_instrument     |instrument| <= 0.02 on every set that runs
  pred_b_between_carries between_all >= 0.8 x far_all on >= 3/4                    Worked: 0.45 of 0.50 True; 0.35 False.
  pred_c_before_inert   |before_all| <= 0.15 on >= 3/4 (control capable of failing: 'The' precedes the cue)
  pred_d_early_band     between_early is the largest band and >= 0.5 x between_all on >= 3/4 (v124: early everywhere)
  pred_e_first_position between_first >= 0.5 x between_all on >= 2/4 (the token right after the cue does the relaying)
Prior: b, c likely; d is the registered guess (the alternative is the mid band, where 05:03 reads at t-1..t-3);
e is open -- a distributed relay along the verb phrase would fail it and I will report the per-position curve.
Note from the smoke (possessive_argument, 4 CPU rows): with cue offset 5 BETWEEN has ONE position (cue+1), so
between_first = between_all trivially there; e is informative only on the sets with between_len > 1
(possessive_long_simple, narrative_tense) and I report between_len_mean per set.

Smoke: V125_SMOKE=<out.json> runs on CPU with one set (V125_SMOKE_SET) and 4 rows.
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
import run_unit_tier3_batch_v112 as v112
import run_unit_tier4_expansion_batch_v115 as v115
import run_unit_tier5_cue_writer_band_v124 as v124

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_far_relay_v125_result.json"
V124 = ROOT / "circuits/followups/unit_tier5_cue_writer_band_v124_result.json"
BANDS = {"early": range(0, 5), "mid": range(5, 9), "late": range(9, 12), "top": range(12, 15)}
ALL = list(range(15))
FAR_GAP, SET_MIN = 3, 0.30
INSTR_TOL, BETWEEN_FRAC, BEFORE_MAX, BAND_FRAC, FIRST_FRAC = 0.02, 0.8, 0.15, 0.5, 0.5
K_B, K_C, K_D, K_E = 3, 3, 3, 2
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 10000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_far_relay_v125", "behaviours": 4,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def maps_of(row_ids, lists):
    n = max((len(x) for x in lists), default=0)
    return [{rid: x[j] for rid, x in zip(row_ids, lists) if len(x) > j} for j in range(n)]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V125_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    v124r = json.load(open(V124))
    sets = sorted(n for n, L in v124r["summary"].items() if L and (L.get("far_noncue_all") or 0) >= SET_MIN)
    if smoke:
        sets = [os.environ.get("V125_SMOKE_SET") or sets[0]]
    R = {}
    for n in sets:
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        rows = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["base_semantic_position"] == r["donor_semantic_position"]]
        info = []
        for r in rows:
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            if not diff:
                continue
            cue = diff[-1]
            eq = lambda p: r["base_ids"][p] == r["donor_ids"][p]
            info.append((r, cue, [p for p in range(cue) if eq(p)], [p for p in range(cue + 1, t - FAR_GAP) if eq(p)]))
        rows = [x[0] for x in info]
        if len(rows) < 4:
            R[n] = {"rows": len(rows), "skipped": "fewer than 4 usable rows"}; print(n, "skipped", flush=True); continue
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        before = maps_of(D.row_ids, [x[2] for x in info])
        between = maps_of(D.row_ids, [x[3] for x in info])
        first = maps_of(D.row_ids, [x[3][:1] for x in info])
        rest = maps_of(D.row_ids, [x[3][1:] for x in info])

        def loss(vals):
            per = [(d - p) / (d - b) for d, b, p in zip(prep.donor_axis, prep.base_axis, vals) if abs(d - b) > 1e-6]
            return round(sum(per) / len(per), 3) if per else None

        def arm(maps, layers, side):
            if not maps:
                return None
            h, mm = v124.slot_caches(backend, side, maps, ALL)
            return loss(v124.clamped_margins(backend, D, layers, h, mm))

        L = {"instrument": arm(before + between, ALL, D), "far_all": arm(before + between, ALL, B),
             "before_all": arm(before, ALL, B), "between_all": arm(between, ALL, B)}
        for band, ls in BANDS.items():
            L[f"between_{band}"] = arm(between, list(ls), B)
        L["between_first"] = arm(first, ALL, B)
        L["between_rest"] = arm(rest, ALL, B)
        R[n] = {"units": S[n], "rows": len(rows), "cue_offset_mean": round(sum(x[0]["donor_semantic_position"] - x[1] for x in info) / len(info), 2),
                "between_len_mean": round(sum(len(x[3]) for x in info) / len(info), 2), "loss": L, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, L, round(time.perf_counter() - t0), "s", flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    L_ = lambda n, k: R[n]["loss"].get(k)
    inst = [n for n in ran if L_(n, "instrument") is not None and abs(L_(n, "instrument")) <= INSTR_TOL]
    carries = [n for n in ran if L_(n, "far_all") and L_(n, "between_all") is not None and L_(n, "between_all") >= BETWEEN_FRAC * L_(n, "far_all")]
    before_ok = [n for n in ran if L_(n, "before_all") is not None and abs(L_(n, "before_all")) <= BEFORE_MAX]
    largest = {n: max(BANDS, key=lambda b: L_(n, f"between_{b}") if L_(n, f"between_{b}") is not None else -9) for n in ran}
    early = [n for n in ran if largest[n] == "early" and L_(n, "between_all") and L_(n, "between_early") >= BAND_FRAC * L_(n, "between_all")]
    firsts = [n for n in ran if L_(n, "between_all") and L_(n, "between_first") is not None and L_(n, "between_first") >= FIRST_FRAC * L_(n, "between_all")]
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_between_carries": len(carries) >= K_B,
        "pred_c_before_inert": len(before_ok) >= K_C,
        "pred_d_early_band": len(early) >= K_D,
        "pred_e_first_position": len(firsts) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_far_relay_v125", "candidate_id": "corpus.unit_tier5_far_relay_v125",
              "bars": {"set_min": SET_MIN, "far_gap": FAR_GAP, "instr_tol": INSTR_TOL, "between_frac": BETWEEN_FRAC, "before_max": BEFORE_MAX,
                       "band_frac": BAND_FRAC, "first_frac": FIRST_FRAC, "K": [K_B, K_C, K_D, K_E]},
              "sets": sets, "largest_band": largest,
              "counts": {"ran": len(ran), "instrument": len(inst), "between_carries": len(carries), "before_inert": len(before_ok),
                         "early": len(early), "first": len(firsts)},
              "summary": {n: R[n].get("loss") for n in R}, "behaviours": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
