#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets from the v190 receipt + one A2-EVEN greedy with the v113 parameters; bars fixed before the run.
"""v192: animacy's row-5 miss (v190: A2 CE damage 0.247 < 0.5 x A1 0.670) -- a second circuit, or a CE-metric asymmetry?

Magnitudes printed before writing (v190 receipt + CPU probe on the ODD rows): the A1 head set {08:08, 05:03, 05:06, 09:07, 11:06,
06:03, 07:00, 04:05, 11:02, 10:03} EXACT-patched recovers A2 ODD at 0.903 (A1 ODD 0.808) -- the heads carry the A2 construction
(record_frame: 'The list also names the maple, the one -> which / director -> who') at least as well as A1 (notes_frame). The dim
direction's MARGIN damage on A2 is 1.25 vs 1.81 on A1 (0.69x, above the 0.5 rule) while its CE damage is 0.247 vs 0.670 (0.37x).
A2 base margins are small (ODD base axis -0.57, -0.45, +1.15, -1.95, -1.36 ...; A1: -1.8, -2.5, -1.0, -4.0, -3.2 ...) and donor margins
large (5.2-6.9 vs 2.8-4.7): mean-removal moves each row toward the pooled mean (near the decision boundary), so a row that already sits
near the boundary loses little CE however far its margin moves -- that was the pre-smoke story; the smoke turned it round (see pred_d):
A2's documents have LARGER margins (donor side 5.2-6.9 vs A1's 2.8-4.7) and CE saturates there. Hypothesis: row 5 misses because CE
damage is a saturating function of the margin, not because A2 has other carriers.

Protocol: (i) greedy on A2 EVEN with the v113 parameters (pool 20, target 0.88, min_gain 0.02, <= 14 heads) -> A2 set; exact ODD A2
extraction; Jaccard with the A1 set. (ii) A1 set + A1-EVEN dim direction: v51 mean-removal on A1 ODD and A2 ODD (CE and margin damage,
per-row CE kept). (iii) A2 set + A2-EVEN dim direction on A2 ODD (own-construction direction). (iv) Spearman rank correlation between the
per-row CE damage and the row's own margin magnitude (|base axis| for base-side rows, |donor axis| for donor-side rows; v51
pools both interchange sides) over the 64 pooled ODD documents (A1 + A2, 16 rows x 2 sides each) under the A1 set/direction.

REGISTERED BEFORE THE RUN
  pred_a_a2_set        the A2-EVEN greedy set reaches ODD A2 extraction in 0.80-1.20 AND its Jaccard with the A1 set is in 0.3-1.0.
  pred_b_margin_rule   A1 set/direction: A2 margin damage / A1 margin damage in 0.5-1.0 (v190: 0.69; same rows, same seed -> reproduces).
  pred_c_own_direction A2 set + A2-own direction CE damage on A2 ODD stays in 0.10-0.34 (< 0.5 x 0.670: an own direction does NOT
                       repair row 5; capable of failing at >= 0.335).
  pred_d_ce_is_saturating Spearman(per-document CE damage, two-class softplus prediction softplus(-(m - dm)) - softplus(-m) from the
                       document's own-side margin m and its measured margin damage dm) over the 64 pooled ODD documents in 0.6-1.0.
                       PRE-SMOKE this pred was 'Spearman(CE damage, |own margin|) in 0.4-1.0' (CE damage grows with the base margin);
                       the 4-row smoke REFUTED it (-0.24): A1's CE damage sits on its DONOR-side documents (1.06-1.69 nats, margins
                       2.8-4.7) while A2's donor documents with the LARGEST margins (5.2-6.9) lose only 0.05-0.13 nats for a margin
                       damage of 0.9. CE is a saturating function of the margin, so the same margin damage costs least where the
                       margin is largest -- the opposite of what I registered. What I now believe is the softplus form; the
                       refuted bar is disclosed here and the raw Spearman is reported.
  pred_e_instrument    A1 CE damage within 0.670 +- 0.05 and A1 ODD exact extraction within 0.808 +- 0.03.
  Prior: a 60 %; b 90 %; c 55 %; d 60 %; e 90 %.
  Smoke (CPU, 4 EVEN/4 ODD rows per construction, greedy pool 3 / 3 heads -- protocol-sized only for the removal arms): a1_ce 0.714,
  a2_ce 0.109, margins 1.69 / 0.90 (0.54x), a2_own_ce 0.077 (3-head smoke set), Spearman(CE, softplus pred) 0.63 on 16 documents
  (the two-class prediction under-reads A1's donor-side CE 1.06-1.69 by 3-8x: rank order is what is registered, not the value).
Reading if a, b, d hold and c holds: animacy is ONE circuit with two constructions and row 5 needs the margin metric for constructions
whose base sits near the boundary (a protocol note for the battery, not a new circuit). If a fails (A2 set disjoint / no extraction),
A2 has other carriers and the miss was real.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_animacy_a2_metric_v192_result.json"
V190 = ROOT / "circuits/followups/unit_tier3_batch_unlifted_v190_result.json"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 20, 0.88, 0.02, 14
BARS = {"a2_ext_band": [0.80, 1.20], "jaccard_band": [0.3, 1.0], "margin_ratio_band": [0.5, 1.0], "own_ce_band": [0.10, 0.34],
        "spearman_band": [0.6, 1.0], "a1_ce": [0.670, 0.05], "a1_ext": [0.808, 0.03]}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000


def _plan():
    return {"candidate_id": "corpus.unit_animacy_a2_metric_v192", "behaviours": 1, "constructions": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def PREDS(R):
    B = BARS
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    return {"pred_a_a2_set": bool(inb(R["a2_set_ext_odd"], B["a2_ext_band"]) and inb(R["jaccard"], B["jaccard_band"])),
            "pred_b_margin_rule": bool(inb(R["margin_ratio_a2_over_a1"], B["margin_ratio_band"])),
            "pred_c_own_direction": bool(inb(R["a2_own_ce"], B["own_ce_band"])),
            "pred_d_ce_is_saturating": bool(inb(R["spearman_ce_vs_softplus_pred"], B["spearman_band"])),
            "pred_e_instrument": bool(inb(R["a1_ce"], [B["a1_ce"][0] - B["a1_ce"][1], B["a1_ce"][0] + B["a1_ce"][1]]) and
                                      inb(R["a1_set_ext_odd"], [B["a1_ext"][0] - B["a1_ext"][1], B["a1_ext"][0] + B["a1_ext"][1]]))}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V192_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    pool, max_units = (int(os.environ.get("V192_SMOKE_POOL", "3")), 3) if smoke else (POOL, MAX_UNITS)
    cut = (lambda rows: rows[:int(os.environ.get("V192_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    m = importlib.import_module("circuit_fast_screen_candidate_animacy")
    P = {fam: {"even": g.prepare(backend, cut(g.rows_of(m, fam)[0::2])), "odd": g.prepare(backend, cut(g.rows_of(m, fam)[1::2]))} for fam in ("A1", "A2")}
    units_a1 = list(json.loads(V190.read_text())["behaviours"]["animacy"]["units"])

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def rem(p, units, q, mu):
        d = v51.removal(backend, p, units, q, mu); s = v51.summary(torch, d)
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}, [float(x) for x in d["ce"]], [float(x) for x in d["margin"]]

    singles, ranked, greedy = g.greedy_heads(backend, P["A2"]["even"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
    units_a2 = list(greedy["chosen"])
    R = {"units_a1": units_a1, "units_a2": units_a2, "jaccard": round(len(set(units_a1) & set(units_a2)) / len(set(units_a1) | set(units_a2)), 3),
         "a2_set_ext_even": round(g.recovery(P["A2"]["even"], g.patched_axis(backend, P["A2"]["even"], units_a2)), 3),
         "a2_set_ext_odd": round(g.recovery(P["A2"]["odd"], g.patched_axis(backend, P["A2"]["odd"], units_a2)), 3),
         "a1_set_ext_odd": round(g.recovery(P["A1"]["odd"], g.patched_axis(backend, P["A1"]["odd"], units_a1)), 3),
         "a1_set_ext_odd_a2": round(g.recovery(P["A2"]["odd"], g.patched_axis(backend, P["A2"]["odd"], units_a1)), 3),
         "a2_set_ext_odd_a1": round(g.recovery(P["A1"]["odd"], g.patched_axis(backend, P["A1"]["odd"], units_a2)), 3)}
    mu1 = mu_of(P["A1"]["even"], units_a1); q1 = g.block_diff_in_means(backend, P["A1"]["even"], units_a1)
    s_a1, ce_a1, dm_a1 = rem(P["A1"]["odd"], units_a1, q1, mu1); s_a2, ce_a2, dm_a2 = rem(P["A2"]["odd"], units_a1, q1, mu1)
    mu2 = mu_of(P["A2"]["even"], units_a2); q2 = g.block_diff_in_means(backend, P["A2"]["even"], units_a2)
    s_own, _, _ = rem(P["A2"]["odd"], units_a2, q2, mu2)
    # v51.removal pools BOTH interchange sides (base rows, then donor rows): pair each side with its own margin magnitude
    mags = lambda p: [abs(float(x)) for x in p.base_axis] + [abs(float(x)) for x in p.donor_axis]
    base_abs = mags(P["A1"]["odd"]) + mags(P["A2"]["odd"])
    assert len(base_abs) == len(ce_a1) + len(ce_a2), (len(base_abs), len(ce_a1), len(ce_a2))
    # two-class softplus prediction of the CE change from the document's own margin m and its measured margin damage dm
    sp = lambda z: math.log1p(math.exp(-abs(z))) + max(z, 0.0)
    ce_pred = [sp(-(m - dm)) - sp(-m) for m, dm in zip(base_abs, dm_a1 + dm_a2)]
    R.update({"a1_ce": s_a1["ce_damage"], "a2_ce": s_a2["ce_damage"], "a1_margin": s_a1["margin_damage"], "a2_margin": s_a2["margin_damage"],
              "margin_ratio_a2_over_a1": round(s_a2["margin_damage"] / s_a1["margin_damage"], 3) if s_a1["margin_damage"] else None,
              "ce_ratio_a2_over_a1": round(s_a2["ce_damage"] / s_a1["ce_damage"], 3) if s_a1["ce_damage"] else None,
              "a2_own_ce": s_own["ce_damage"], "a2_own_margin": s_own["margin_damage"],
              "spearman_ce_vs_abs_base": round(spearman(ce_a1 + ce_a2, base_abs), 3), "spearman_ce_vs_softplus_pred": round(spearman(ce_a1 + ce_a2, ce_pred), 3),
              "n_rows_pooled": len(base_abs), "mag_mean_a1_donor_side": round(sum(base_abs[len(ce_a1)//2:len(ce_a1)]) / (len(ce_a1)//2), 3), "mag_mean_a2_donor_side": round(sum(base_abs[len(ce_a1)+len(ce_a2)//2:]) / (len(ce_a2)//2), 3),
              "mag_mean_a1": round(sum(base_abs[:len(ce_a1)]) / len(ce_a1), 3), "mag_mean_a2": round(sum(base_abs[len(ce_a1):]) / len(ce_a2), 3),
              "detail": {"a1": s_a1, "a2": s_a2, "a2_own": s_own, "per_row_ce_a1": [round(x, 3) for x in ce_a1], "per_row_ce_a2": [round(x, 3) for x in ce_a2], "per_row_margin_damage_a1": [round(x, 3) for x in dm_a1], "per_row_margin_damage_a2": [round(x, 3) for x in dm_a2], "per_row_ce_pred": [round(x, 3) for x in ce_pred], "per_row_margin_mag": [round(x, 3) for x in base_abs],
                         "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]}}})
    print(json.dumps({k: v for k, v in R.items() if k != "detail"}, indent=1), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_animacy_a2_metric_v192", "candidate_id": "corpus.unit_animacy_a2_metric_v192", "bars": BARS,
              "protocol": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
