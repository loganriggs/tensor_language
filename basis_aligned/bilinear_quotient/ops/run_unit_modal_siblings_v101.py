#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v97 final), recipe (v99), sibling templates and bars fixed before the run.
"""v101: matched siblings for modal_remoteness -- row 4 on the default side, and is the direction a remoteness axis?

modal_remoteness ships only the generic C (in-the-middle-of-the -> night), so row 4 (own matched-sibling C, CE UB <= 0.01)
is untested. Two siblings are built from the ODD A1 rows by string transform (answers, foils and ids unchanged, tokenization
re-checked):
    default sibling   "When the X closes early, the A route" -> "The X closes early, so the A route"      answer will  (no conditional cue)
    habitual sibling  "If the X closed early, the A route"   -> "Whenever the X closed early, the A route" answer would (past habitual, no if)
Direction: v99 recipe exactly (v97 10-head set; rank 1 per block; fit pooled EVEN A1; controls own C EVEN + six v80 A1 EVEN,
30 each; complement 1.0; 120 steps lr 0.05 seed 0; mu = pooled EVEN mean). Siblings are never fitted. Removal = mean-ablation
CE damage (nat), split by side. Hypotheses on the habitual sibling: H1 the direction is a remoteness (past-marked modal)
axis -> habitual 'would' is carried like the conditional 'would'; H2 it is if-conditional specific -> habitual 'would' spared.

REGISTERED BEFORE THE RUN (ODD rows; 16 sibling rows per side)
    pred_a_capability      both sibling sides natively prefer their answer on >= 0.80 of rows. Worked: 0.94/0.88 True; 0.94/0.60 False.
                           A side that fails is reported UNTESTED, not passed.
    pred_b_row4_default    default-sibling (will) CE damage UB <= 0.01. Worked: 0.003 (UB 0.009) True; 0.020 (UB 0.040) False.
    pred_c_habitual_carried (H1) habitual-sibling (would) damage >= 0.50 x the A1 would side. Worked: 0.60 vs 1.00 True; 0.10 vs 1.00 False.
    pred_d_habitual_spared  (H2) habitual-sibling damage <= 0.20 x the A1 would side. Worked: 0.10 vs 1.00 True; 0.60 vs 1.00 False.
                           c and d are exclusive; both False = partial transfer (reported as such).
    pred_e_reproduces      A1 ODD removal >= 0.40 and would side >= 3 x will side (v99 reproduction on this code path).
    Prior: a ~75%; b ~70%; c ~55%; d ~25%; e ~90%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import run_unit_polarity_selective_removal_v50 as v50
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_modal_siblings_v101_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
CAP_MIN, ROW4_UB, CARRY_FRAC, SPARE_FRAC, REM_MIN, SIDE_RATIO = 0.80, 0.01, 0.50, 0.20, 0.40, 3.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 30000


def _plan():
    return {"candidate_id": "corpus.unit_modal_siblings_v101", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * STEPS, "model_updates": 0, "fit_parameters": 10 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def sibling_rows(rows):
    """ODD A1 rows with both sides rewritten into the registered siblings; ids/answers/foils kept, tokenization re-checked."""
    import copy
    import re
    import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
    enc = builder.ENCODING
    out = []
    for r in rows:
        n = copy.deepcopy(r)
        for side in ("base", "donor"):
            t = n[f"{side}_text"]
            if t.startswith("If the "):
                text = "Whenever the " + t[len("If the "):]
            else:
                m = re.fullmatch(r"When the (.+) closes early, the (.+ route)", t)
                assert m, t
                text = f"The {m.group(1)} closes early, so the {m.group(2)}"
            ids = enc.encode(text)
            assert ids[-1] == n[f"{side}_ids"][-1], (side, text)
            for tok, key in ((n[f"{side}_answer"], "answer"), (n[f"{side}_foil"], "foil")):
                assert enc.encode(text + tok) == ids + [n[f"{side}_{key}_id"]], (side, text, tok)
            n[f"{side}_text"], n[f"{side}_ids"] = text, ids
            n[f"{side}_semantic_position"] = n[f"{side}_prediction_position"] = len(ids) - 1
        n["row_id"] = n["row_id"] + ":sib"
        out.append(n)
    return out


def sides(torch, prep, d):
    k = len(prep.base_batch.row_ids)
    return {"base_side": v51.summary(torch, {kk: v[:k] for kk, v in d.items()}), "donor_side": v51.summary(torch, {kk: v[k:] for kk, v in d.items()})}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    units = json.loads(V97.read_text())["final"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}

    a1 = g.rows_of(m_modal, "A1")
    pool = g.prepare(backend, a1[0::2])
    even_c = g.prepare(backend, g.rows_of(m_modal, "C")[0::2])
    odd = {f: g.prepare(backend, g.rows_of(m_modal, f)[1::2]) for f in ("A1", "C")}
    mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache) for rid in pool.base_batch.row_ids]).mean(0) for u in units}
    controls = (even_c,) + tuple(cross_even.values())
    q, hist = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                               controls=controls, control_weight=LAM * len(controls), mu=mu)
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)

    sib = g.prepare(backend, sibling_rows(odd["A1"].rows))
    ans = {"base": a1[1]["base_answer"].strip(), "donor": a1[1]["donor_answer"].strip()}
    assert ans == {"base": "will", "donor": "would"}, ans
    assert sib.rows[0]["base_text"].startswith("The ") and sib.rows[0]["donor_text"].startswith("Whenever the "), sib.rows[0]["base_text"]
    # capability: base_axis = -(answer - foil) so base prefers its answer when < 0; donor_axis = answer - foil
    cap = {"will_default": sum(1 for x in sib.base_axis if x < 0) / len(sib.base_axis),
           "would_habitual": sum(1 for x in sib.donor_axis if x > 0) / len(sib.donor_axis)}

    rem = {f: v51.removal(backend, p, units, q, mu) for f, p in (("A1", odd["A1"]), ("C", odd["C"]), ("siblings", sib))}
    R = {f: v51.summary(torch, d) for f, d in rem.items()}
    R["A1"].update(sides(torch, odd["A1"], rem["A1"]))
    R["siblings"].update(sides(torch, sib, rem["siblings"]))
    R["random_siblings"] = v51.summary(torch, v51.removal(backend, sib, units, q_rand, mu))
    will, would = R["A1"]["base_side"]["ce_damage"], R["A1"]["donor_side"]["ce_damage"]
    dflt, habit = R["siblings"]["base_side"], R["siblings"]["donor_side"]
    cap_ok = {k: v >= CAP_MIN for k, v in cap.items()}
    predictions = {
        'pred_a_capability': all(cap_ok.values()),
        'pred_b_row4_default': cap_ok["will_default"] and dflt["ce_ub975"] <= ROW4_UB,
        'pred_c_habitual_carried': cap_ok["would_habitual"] and habit["ce_damage"] >= CARRY_FRAC * would,
        'pred_d_habitual_spared': cap_ok["would_habitual"] and habit["ce_damage"] <= SPARE_FRAC * would,
        'pred_e_reproduces': R["A1"]["ce_damage"] >= REM_MIN and would >= SIDE_RATIO * will,
    }
    untested = [k for k, v in cap_ok.items() if not v]
    summary = {"capability": cap, "untested_sides": untested,
               "removal": {f: (round(R[f]["ce_damage"], 3), round(R[f]["ce_lb975"], 3), round(R[f]["ce_ub975"], 3)) for f in ("A1", "C", "siblings", "random_siblings")},
               "a1_sides": {"will": round(will, 3), "would": round(would, 3)},
               "sibling_sides": {"will_default": (round(dflt["ce_damage"], 3), round(dflt["ce_lb975"], 3), round(dflt["ce_ub975"], 3)),
                                 "would_habitual": (round(habit["ce_damage"], 3), round(habit["ce_lb975"], 3), round(habit["ce_ub975"], 3))},
               "habitual_over_would": round(habit["ce_damage"] / max(would, 1e-6), 3),
               "sibling_texts": [sib.rows[0]["base_text"], sib.rows[0]["donor_text"]]}
    result = {"predictions": predictions, "schema": "circuit_unit_modal_siblings_result_v1", "candidate_id": "corpus.unit_modal_siblings_v101",
              "units": units, "answers": ans, "summary": summary, "removal": R, "history": hist,
              "bars": {"cap_min": CAP_MIN, "row4_ub": ROW4_UB, "carry_frac": CARRY_FRAC, "spare_frac": SPARE_FRAC, "rem_min": REM_MIN, "side_ratio": SIDE_RATIO},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
