#!/usr/bin/env python3
# BQGATE: frozen predictions; head set (from the v9/v12 receipts), verb pairs, seeds and bars fixed before the run.
"""v13: are dative's two directions an abstract to/for variable, or verb identity?

v12 (`unit_a2_direction_v12_result.json`) found that the dative_v2 head set carries A1 (sent/reserved)
and A2 (mailed/prepared) on DIFFERENT per-block axes (median |cos| below 0.50), and that their rank-2
union serves both held-out families. Two readings survive: (i) the set carries an abstract
recipient/benefactive variable that happens to be written on two axes for two frames; (ii) each
axis is the identity of the verb pair, and a third pair gets a third axis. The A1 and A2 frames
differ in BOTH verb and frame, so v12 cannot tell them apart. This run holds the A1 frame fixed and
swaps only the verb pair: v3 = handed/bought, v4 = gave/kept, substituted into the A1 rows and
re-tokenized (all single tokens, same length, same final token, so positions are unchanged).
Rows are the module's own A1 rows with the verb replaced; nothing else varies.

  head set (fixed): attn:14:08, 07:08, 06:03, 13:08, 11:03 (dative_v2, v9 receipt)
  d1 = block dim on A1 even (sent/reserved); d2 = block dim on A2 even (mailed/prepared)
  d3 = block dim on v3 even; d4 = block dim on v4 even; all batteries on ODD rows only
  Band [0.50, 1.20]; complement <= 0.30; random rank-matched seed 1; cos bar 0.50; effect floor 0.10.

REGISTERED BEFORE THE RUN
    pred_a_exact_set_transfers     the fixed head set's exact interchange recovers >= 0.50 on v3 odd
                                   AND v4 odd rows (prerequisite; fractions below are undefined
                                   otherwise). Worked example: 0.71 and 0.66 -> True; v4 0.31 -> False.
    pred_b_new_pairs_own_axis      d3 on v3 odd and d4 on v4 odd both in band with complement <= 0.30.
                                   Worked example: 0.98 / 0.02 and 1.01 / 0.00 -> True; d4 0.44 -> False.
    pred_c_abstract_axis           union(d1, d2) (rank 2 per block) serves v3 odd AND v4 odd in band
                                   with complement <= 0.30. Worked example: 0.88 / 0.10 and 0.93 / 0.05
                                   -> True; v3 0.35 / 0.55 -> False.
    pred_d_verb_identity           at least one new pair is NOT served: union(d1, d2) fraction < 0.35
                                   on it AND that pair's own direction has median per-block |cos| < 0.50
                                   to BOTH d1 and d2. Worked example: union 0.22, medians 0.31 / 0.28
                                   -> True; union 0.22 but median to d1 0.63 -> False.
    pred_e_same_frame_same_axis    d1 (sent/reserved, same bare frame) serves v3 odd AND v4 odd in band
                                   with complement <= 0.30, i.e. the frame, not the verb, picks the axis.
                                   Worked example: 0.90 / 0.06 and 0.85 / 0.08 -> True; v4 0.40 -> False.
    Priors. a expected (the set was chosen for to/for, and these verbs are ordinary datives). b expected.
    c and e are the abstract-variable reading, d the verb-identity reading; they are complementary
    tests and I am unsure which holds. If c fails but e holds, the answer is FRAME identity (A1's axis
    is the bare frame's axis regardless of verb) -- a third reading, reported as such.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_dative as m_dative
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_dative_verb_axis_v13_result.json"
UNITS = ["attn:14:head:08", "attn:07:head:08", "attn:06:head:03", "attn:13:head:08", "attn:11:head:03"]
PAIRS = {"v3": ("handed", "bought"), "v4": ("gave", "kept")}   # (recipient verb, benefactive verb)
A1_VERBS = ("sent", "reserved")
LO, HI, COMP_BAR, COS_BAR, EXACT_BAR, NOT_SERVED = 0.50, 1.20, 0.30, 0.50, 0.50, 0.35
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 12000


def _plan():
    return {"candidate_id": "dative.unit_verb_axis_v13", "units": UNITS, "pairs": PAIRS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _swap_verbs(rows, pair):
    """A1 rows with the verb pair replaced and ids re-derived; asserts same length and final token."""
    out = []
    for r in rows:
        n = copy.deepcopy(r)
        for side in ("base", "donor"):
            text = n[f"{side}_text"]
            for old, new in zip(A1_VERBS, pair):
                text = text.replace(f" {old} ", f" {new} ")
            assert text != n[f"{side}_text"], text
            ids = builder.ENCODING.encode(text)
            assert len(ids) == len(n[f"{side}_ids"]) and ids[-1] == n[f"{side}_ids"][-1], text
            for tok, key in ((n[f"{side}_answer"], "answer"), (n[f"{side}_foil"], "foil")):
                assert builder.ENCODING.encode(text + tok) == ids + [n[f"{side}_{key}_id"]], text
            n[f"{side}_text"], n[f"{side}_ids"] = text, ids
        n["row_id"] = f"{n['row_id']}:{pair[0]}_{pair[1]}"
        out.append(n)
    return out


def _band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    a1, a2 = g.rows_of(m_dative, "A1"), g.rows_of(m_dative, "A2")
    fams = {"a1": a1, "a2": a2, **{k: _swap_verbs(a1, pair) for k, pair in PAIRS.items()}}
    p = {}
    for k, rows in fams.items():
        p[f"{k}_fit"] = g.prepare(backend, rows[0::2], valid_only=True)
        p[f"{k}_odd"] = g.prepare(backend, rows[1::2], valid_only=True)
    d = {k: g.block_diff_in_means(backend, p[f"{k}_fit"], UNITS) for k in fams}
    union12 = g.block_union(d["a1"], d["a2"])
    r1 = g.block_random_subspace(backend, UNITS, rank=1, seed=1)
    r2 = g.block_random_subspace(backend, UNITS, rank=2, seed=1)
    dirs = {"d1": (d["a1"], r1), "d2": (d["a2"], r1), "d3": (d["v3"], r1), "d4": (d["v4"], r1),
            "union12": (union12, r2)}
    B = {dn: {ev: g.block_direction_battery(backend, p[f"{ev}_odd"], UNITS, q, r) for ev in fams}
         for dn, (q, r) in dirs.items()}
    cos = {f"{a}_{b}": g.block_cosines(d[a], d[b]) for a, b in
           (("v3", "a1"), ("v3", "a2"), ("v4", "a1"), ("v4", "a2"), ("v3", "v4"), ("a1", "a2"))}
    med = {k: statistics.median(v.values()) for k, v in cos.items()}
    exact = {ev: B["d1"][ev]["exact_set"] for ev in fams}
    print("exact", {k: round(v, 2) for k, v in exact.items()}, "median_cos", {k: round(v, 2) for k, v in med.items()},
          {dn: {ev: (round(b["subspace_fraction"] or 0, 2), round(b["complement_fraction"] or 0, 2))
                for ev, b in bs.items()} for dn, bs in B.items()}, flush=True)

    def _verb_identity(k, dk):
        return (B["union12"][k]["subspace_fraction"] or 0) < NOT_SERVED \
            and med[f"{k}_a1"] < COS_BAR and med[f"{k}_a2"] < COS_BAR
    predictions = {
        'pred_a_exact_set_transfers': all(exact[k] >= EXACT_BAR for k in PAIRS),
        'pred_b_new_pairs_own_axis': _band(B["d3"]["v3"]) and _band(B["d4"]["v4"]),
        'pred_c_abstract_axis': all(_band(B["union12"][k]) for k in PAIRS),
        'pred_d_verb_identity': any(_verb_identity(k, dk) for k, dk in (("v3", "d3"), ("v4", "d4"))),
        'pred_e_same_frame_same_axis': all(_band(B["d1"][k]) for k in PAIRS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_dative_verb_axis_result_v1",
              "candidate_id": "dative.unit_verb_axis_v13", "semantics": "block_live", "units": UNITS,
              "pairs": PAIRS, "bars": {"band": [LO, HI], "complement": COMP_BAR, "cos": COS_BAR,
                                       "exact": EXACT_BAR, "not_served": NOT_SERVED},
              "dropped": {k: v.dropped for k, v in p.items()}, "exact_set_odd": exact,
              "median_cos": med, "cos": cos, "batteries": B,
              "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
