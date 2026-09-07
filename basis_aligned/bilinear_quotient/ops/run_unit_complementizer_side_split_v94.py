#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v90 hub16), direction recipe (v91), verb panel, siblings and bars fixed before the run.
"""v94: side-split removal -- is the complementizer direction a one-sided INTERROGATIVE marker whose absence reads as 'that'?

v93 (verb panel) found A1-frame removal 1.07-1.27 nat for every one of 13 declarative verbs (seen or unseen, including the
sibling's noted / replied), no dependence on the model's own that-margin (Spearman -0.10), the that-only sibling C at 0.021 and a
whether-only sibling C2 (asked / inquired) at 2.017. `v51.removal` pools BOTH sides of each row, and every A1-frame panel row has
donor side asked -> whether: the flat ~1.1 is therefore what (~0 on the that side + ~2.1 on the whether side) / 2 looks like.
That is the signature of a one-sided marker: the direction carries 'interrogative verb here'; declarative verbs have nothing on it
to remove, and 'that' is what the model says when the marker is absent. This run splits every removal by side (document order in
`v51.removal` is base side then donor side) with the same direction (v91 recipe, seed 0; pred_a is an exact reproduction).
Unseen interrogatives: C3 = C rows with noted -> debated, replied -> checked, answer / foil swapped (whether-only, verbs the
direction never saw); well-formed only if the model prefers 'whether' on >= 75% of base rows (else pred_e is False = untested).

REGISTERED BEFORE THE RUN (removal = CE damage in nat on the row's answer; ODD rows; sides split)
    pred_a_reproduce        pooled xdas A1 odd removal within 0.02 of v91's 1.171 (read from the receipt).
    pred_b_one_sided_a1     A1 odd: that side (remarked -> that) <= 0.10 AND whether side (wondered -> whether) >= 1.50. Worked: 0.05 / 2.2 True; 0.4 / 1.8 False.
    pred_c_that_side_flat   that-side damage <= 0.10 for ALL 13 panel verbs (seen or unseen declaratives). Worked: max 0.06 True; one at 0.3 False.
    pred_d_whether_side     whether-side damage (asked -> whether, the panel's shared donor; wondered for the remarked row) >= 1.50 on ALL 13 panel rows AND both C2 sides (asked, inquired) >= 1.50.
    pred_e_unseen_interrog  C3 well-formed AND both C3 sides (debated, checked -> whether) >= 0.30 x the A1 whether side with lb975 > 0.
    Prior: a True; b ~85%; c ~75%; d ~75%; e ~50% (the marker may be lexical to the pool's interrogatives; v15's fourth-map
    'questioned' transferred at 0.70 pooled, i.e. ~1.4 on its whether side, which argues for generalisation).
    b, c, d True: the complementizer circuit's rank-1 direction is the interrogative marker and the tier list's row 4 with a
    that-only sibling is a test of the DEFAULT side's inertness (real at 0.02 nat, 1% of the whether-side effect); a whether-only
    sibling is a second A1 by construction. b False: the direction is two-sided and v93's flat panel needs another explanation.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_complementizer_side_split_v94_result.json"
V90 = ROOT / "circuits/followups/unit_complementizer_greedy_continuation_v90_result.json"
V91 = ROOT / "circuits/followups/unit_complementizer_hub16_full_specificity_v91_result.json"
NAME = "verb_complementizer"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
PANEL = ("remarked", "said", "insisted", "declared", "noted", "replied", "stated", "claimed", "argued", "announced", "added", "explained", "admitted")
SEEN = {"remarked", "said", "insisted"}
UNSEEN = ("stated", "claimed", "argued", "announced", "added", "explained", "admitted")
C2 = {"noted": "asked", "replied": "inquired"}
C3 = {"noted": "debated", "replied": "checked"}
REPRO, THAT_MAX, WHETHER_MIN, WELLFORMED, C3_FRAC = 0.02, 0.10, 1.50, 0.75, 0.30
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 500, 30000


def _plan():
    return {"candidate_id": "corpus.unit_complementizer_side_split_v94", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * STEPS, "model_updates": 0, "fit_parameters": 19 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    units = json.loads(V90.read_text())["sets"][NAME]["hub16"]
    v91_a1 = json.loads(V91.read_text())["sets"][NAME]["removal"]["xdas"]["A1"]["ce_damage"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    m = modules[NAME]
    a1 = g.rows_of(m, "A1")
    maps = v15.SETS[NAME][2]
    pool = g.prepare(backend, a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]])
    even_c = g.prepare(backend, g.rows_of(m, "C")[0::2])
    xc = (even_c,) + tuple(g.prepare(backend, g.rows_of(mm, "A1")[0::2]) for n, mm in modules.items() if n != NAME)
    mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache) for rid in pool.base_batch.row_ids]).mean(0) for u in units}
    q, _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu)

    def rem(rows):
        prep = g.prepare(backend, rows)
        d = v51.removal(backend, prep, units, q, mu)
        n = len(prep.base_batch.row_ids)
        s = v51.summary(torch, d)
        s["base_side"] = v51.summary(torch, {k: v[:n] for k, v in d.items()})
        s["donor_side"] = v51.summary(torch, {k: v[n:] for k, v in d.items()})
        s["dropped_by_prepare"] = getattr(prep, "dropped", 0)
        return s

    def wf_rows(rows):
        out = backend.native(g.batch_of(rows, "base"), capture=False).answer_foil
        return [a - f for a, f in out]

    a1_odd = rem(a1[1::2])                      # base remarked -> that, donor wondered -> whether
    R = {"A1_odd": a1_odd, "C_that_only_odd": rem(g.rows_of(m, "C")[1::2]), "panel": {}}
    for v in PANEL:
        # base v -> that, donor asked -> whether (the remarked row is the A1 odd half itself: donor wondered -> whether)
        rows = a1[1::2] if v == "remarked" else g.lexical_variant(a1, {"wondered": "asked", "remarked": v})[1::2]
        R["panel"][v] = {"seen_in_pool": v in SEEN, "removal_odd": rem(rows)}
    c_rows = g.rows_of(m, "C")
    for label, mp in (("C2_seen_interrogative", C2), ("C3_unseen_interrogative", C3)):
        rows = g.swap_answer_foil(g.lexical_variant(c_rows, mp))
        wm = wf_rows(rows[1::2])
        R[label] = {"mapping": mp, "fraction_whether_preferred": sum(1 for x in wm if x > 0) / len(wm), "natural_margin_mean": sum(wm) / len(wm), "removal_odd": rem(rows[1::2])}
    print({v: (round(p["removal_odd"]["base_side"]["ce_damage"], 3), round(p["removal_odd"]["donor_side"]["ce_damage"], 3)) for v, p in R["panel"].items()}, round(time.perf_counter() - t0), "s", flush=True)

    P = R["panel"]
    that_side = {v: P[v]["removal_odd"]["base_side"]["ce_damage"] for v in PANEL}
    whether_side = {v: P[v]["removal_odd"]["donor_side"]["ce_damage"] for v in PANEL}
    a1_that, a1_whether = a1_odd["base_side"]["ce_damage"], a1_odd["donor_side"]["ce_damage"]
    c2, c3 = R["C2_seen_interrogative"], R["C3_unseen_interrogative"]
    c2_sides = (c2["removal_odd"]["base_side"], c2["removal_odd"]["donor_side"])
    c3_sides = (c3["removal_odd"]["base_side"], c3["removal_odd"]["donor_side"])
    predictions = {
        'pred_a_reproduce': abs(a1_odd["ce_damage"] - v91_a1) <= REPRO,
        'pred_b_one_sided_a1': a1_that <= THAT_MAX and a1_whether >= WHETHER_MIN,
        'pred_c_that_side_flat': all(x <= THAT_MAX for x in that_side.values()),
        'pred_d_whether_side': all(x >= WHETHER_MIN for x in whether_side.values()) and all(s["ce_damage"] >= WHETHER_MIN for s in c2_sides),
        'pred_e_unseen_interrog': c3["fraction_whether_preferred"] >= WELLFORMED and all(s["ce_damage"] >= C3_FRAC * a1_whether and s["ce_lb975"] > 0 for s in c3_sides),
    }
    summary = {"A1_odd_pooled": round(a1_odd["ce_damage"], 3), "v91_A1_odd": round(v91_a1, 3), "A1_that_side": round(a1_that, 3), "A1_whether_side": round(a1_whether, 3),
               "C_that_only": {k: round(R["C_that_only_odd"][k]["ce_damage"], 3) for k in ("base_side", "donor_side")},
               "panel_that_side": {v: round(x, 3) for v, x in that_side.items()}, "panel_whether_side_asked": {v: round(x, 3) for v, x in whether_side.items()},
               "C2": {"whether_pref": round(c2["fraction_whether_preferred"], 2), "sides": [round(s["ce_damage"], 3) for s in c2_sides]},
               "C3": {"whether_pref": round(c3["fraction_whether_preferred"], 2), "margin": round(c3["natural_margin_mean"], 2), "sides": [round(s["ce_damage"], 3) for s in c3_sides], "lbs": [round(s["ce_lb975"], 3) for s in c3_sides]}}
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_side_split_result_v1", "candidate_id": "corpus.unit_complementizer_side_split_v94",
              "units": units, "summary": summary, "detail": R,
              "bars": {"repro": REPRO, "that_max": THAT_MAX, "whether_min": WHETHER_MIN, "wellformed": WELLFORMED, "c3_frac": C3_FRAC},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
