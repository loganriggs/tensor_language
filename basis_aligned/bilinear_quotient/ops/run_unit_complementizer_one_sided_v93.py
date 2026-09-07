#!/usr/bin/env python3
# BQGATE: five frozen predictions; set (v90 hub16), direction recipe (v91), verb panel and bars fixed before the run.
"""v93: why is the that-only sibling inert? The complementizer direction's removal damage is keyed to the VERB, tested on a verb panel.

Puzzle from v86/v91: the full-specificity direction on the complementizer set removes 1.17 nat from A1 odd rows (all of which are
base-'that' rows -- rows alternate direction, so the ODD half is remarked -> that and the EVEN fit half is wondered -> whether),
transfers to the unseen fourth map (declared -> that) at 0.70, and yet the matched sibling C (noted / replied -> that) takes only
0.019 (real: LB 0.005 at 64 docs). remarked, declared and noted are all declarative verbs in the same frame, so the residual's
size is a property of the verb, not of the sibling's frame or reporter swap. Two candidate readings, tested on a panel of
A1-frame variants (`g.lexical_variant(A1, {wondered: asked, remarked: V})`, ODD rows = V -> that):
  H1 (verb-keyed): the direction carries the complementizer preference of specific verbs; noted / replied are simply verbs it does
     not carry, so the sibling is inert because of WHICH verbs the sibling uses, and the same verbs are inert in the A1 frame.
  H2 (margin-keyed): removal damage scales with how strongly the model itself prefers 'that' after the verb -- weak-preference
     verbs have little for the direction to remove -- so damage tracks the natural margin across verbs.
Panel: seen-in-pool declaratives remarked / said / insisted; the v15 fourth-map verb declared; the sibling verbs noted / replied;
unseen stated / claimed / argued / announced / added / explained / admitted. Also the whether side on pool verbs: sibling C2 =
C rows with noted -> asked, replied -> inquired and answer / foil swapped (`g.swap_answer_foil`), ODD rows.

REGISTERED BEFORE THE RUN (removal = CE damage in nat on the row's answer, ODD rows of each variant; direction = v91 recipe, seed 0)
    pred_a_reproduce        xdas A1 odd removal within 0.02 of v91's 1.171 (read from the receipt).
    pred_b_verb_keyed       A1-frame removal with remarked -> noted AND remarked -> replied both <= 0.10 (the sibling's inertness is
                            the verbs', not the frame's). Worked: 0.03 / 0.05 True; 0.25 False.
    pred_c_margin_tracks    Spearman rank correlation between natural base margin (that - whether) and removal damage across the
                            12-verb panel >= 0.60. Worked: rho 0.75 True; 0.3 False.
    pred_d_unseen_transfer  at least 3 of the 7 unseen verbs show removal >= 0.30 x remarked's with lb975 > 0. Worked: 4 of 7 True.
    pred_e_whether_side     C2 (asked / inquired -> whether, ODD rows) well-formed (model prefers 'whether' on >= 75% of base rows)
                            AND removal >= 0.50 x remarked's with lb975 > 0. Worked: 0.9 vs 1.17 True; 0.4 False.
    Prior: a True; b ~75%; c ~50%; d ~60%; e ~75%.
    b True, c False: verb-keyed lexical direction -- row 4 with THIS sibling tests two verbs the direction never carried, and the
    residual 0.019 is what generalisation to those verbs looks like at rank 1; a stronger row-4 sibling would use a verb the model
    treats as declarative but the direction is inert on. b True, c True: the damage is margin-limited and noted / replied are weak
    'that' verbs; the sibling is a low-power control. b False: the frame or the reporter swap, not the verbs, explains the sibling.
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
OUT = ROOT / "circuits/followups/unit_complementizer_one_sided_v93_result.json"
V90 = ROOT / "circuits/followups/unit_complementizer_greedy_continuation_v90_result.json"
V91 = ROOT / "circuits/followups/unit_complementizer_hub16_full_specificity_v91_result.json"
NAME = "verb_complementizer"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
PANEL = ("remarked", "said", "insisted", "declared", "noted", "replied", "stated", "claimed", "argued", "announced", "added", "explained", "admitted")
SEEN = {"remarked", "said", "insisted"}
UNSEEN = ("stated", "claimed", "argued", "announced", "added", "explained", "admitted")
C2 = {"noted": "asked", "replied": "inquired"}
REPRO, INERT, RHO, UNSEEN_FRAC, N_UNSEEN, WELLFORMED, C2_FRAC = 0.02, 0.10, 0.60, 0.30, 3, 0.75, 0.50
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 500, 30000


def _plan():
    return {"candidate_id": "corpus.unit_complementizer_one_sided_v93", "lambda": LAM,
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
        s = v51.summary(torch, v51.removal(backend, prep, units, q, mu))
        s["dropped_by_prepare"] = getattr(prep, "dropped", 0)
        return s

    def wf_rows(rows):
        out = backend.native(g.batch_of(rows, "base"), capture=False).answer_foil
        return [a - f for a, f in out]

    def spearman(x, y):
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
            for pos, i in enumerate(order):
                r[i] = pos
            return r
        rx, ry = ranks(x), ranks(y); n = len(x)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else 0.0

    odd_a1 = a1[1::2]
    R = {"A1_odd": rem(odd_a1), "C_that_only_odd": rem(g.rows_of(m, "C")[1::2]), "panel": {}}
    for v in PANEL:
        rows = a1[1::2] if v == "remarked" else g.lexical_variant(a1, {"wondered": "asked", "remarked": v})[1::2]
        margins = wf_rows(rows)
        R["panel"][v] = {"seen_in_pool": v in SEEN, "natural_margin_mean": sum(margins) / len(margins),
                         "fraction_that_preferred": sum(1 for x in margins if x > 0) / len(margins), "removal_odd": rem(rows)}
    c2_rows = g.swap_answer_foil(g.lexical_variant(g.rows_of(m, "C"), C2))
    c2_m = wf_rows(c2_rows[1::2])
    R["C2_whether_side"] = {"mapping": C2, "fraction_whether_preferred": sum(1 for x in c2_m if x > 0) / len(c2_m), "natural_margin_mean": sum(c2_m) / len(c2_m), "removal_odd": rem(c2_rows[1::2])}
    print({v: (round(p["natural_margin_mean"], 2), round(p["removal_odd"]["ce_damage"], 3)) for v, p in R["panel"].items()}, round(time.perf_counter() - t0), "s", flush=True)

    P = R["panel"]
    dmg = lambda v: P[v]["removal_odd"]["ce_damage"]
    ref = dmg("remarked")
    rho = spearman([P[v]["natural_margin_mean"] for v in PANEL], [dmg(v) for v in PANEL])
    unseen_hits = [v for v in UNSEEN if dmg(v) >= UNSEEN_FRAC * ref and P[v]["removal_odd"]["ce_lb975"] > 0]
    c2 = R["C2_whether_side"]
    predictions = {
        'pred_a_reproduce': abs(R["A1_odd"]["ce_damage"] - v91_a1) <= REPRO,
        'pred_b_verb_keyed': dmg("noted") <= INERT and dmg("replied") <= INERT,
        'pred_c_margin_tracks': rho >= RHO,
        'pred_d_unseen_transfer': len(unseen_hits) >= N_UNSEEN,
        'pred_e_whether_side': c2["fraction_whether_preferred"] >= WELLFORMED and c2["removal_odd"]["ce_damage"] >= C2_FRAC * ref and c2["removal_odd"]["ce_lb975"] > 0,
    }
    summary = {"A1_odd": round(R["A1_odd"]["ce_damage"], 3), "v91_A1_odd": round(v91_a1, 3), "C_that_only": round(R["C_that_only_odd"]["ce_damage"], 3),
               "panel": {v: {"margin": round(P[v]["natural_margin_mean"], 2), "that_pref": round(P[v]["fraction_that_preferred"], 2), "removal": round(dmg(v), 3), "lb": round(P[v]["removal_odd"]["ce_lb975"], 3)} for v in PANEL},
               "spearman_margin_vs_removal": round(rho, 3), "unseen_hits": unseen_hits,
               "C2": {"whether_pref": round(c2["fraction_whether_preferred"], 2), "removal": round(c2["removal_odd"]["ce_damage"], 3), "lb": round(c2["removal_odd"]["ce_lb975"], 3)}}
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_verb_panel_result_v1", "candidate_id": "corpus.unit_complementizer_one_sided_v93",
              "units": units, "summary": summary, "detail": R,
              "bars": {"repro": REPRO, "inert": INERT, "rho": RHO, "unseen_frac": UNSEEN_FRAC, "n_unseen": N_UNSEEN, "wellformed": WELLFORMED, "c2_frac": C2_FRAC},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
