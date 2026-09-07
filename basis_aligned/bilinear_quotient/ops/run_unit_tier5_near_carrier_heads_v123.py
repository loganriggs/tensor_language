#!/usr/bin/env python3
# BQGATE: six frozen predictions; band per set taken from the v122 receipt (argmax band); heads-only true greedy (pre-enqueue redesign disclosed in the docstring).
"""Tier-5 step 2: WHICH units inside the v122 writer band carry the cue to the near non-cue positions?

v122 (receipt) clamped every head + MLP of one layer band at the near non-cue positions (t-1..t-3, offsets that are
cue positions excluded row-wise) and found the writer band: EARLY (layers 0-4) for lexical_number_pp (0.246 of 0.295),
perfect_number (0.262 / 0.312), possessive_medial (0.457 / 0.517), quantifier_number (0.580 / 0.696); MID (5-8) for
possessive_argument (0.183 / 0.313), possessive_verbfinal (0.179 / 0.277), narrative_tense (0.123 / 0.183). The
controls were spared (<= 0.114). At a near NON-cue position base and donor share the token, so inside the model the
only way that position can differ is attention from the cue (directly, or through an earlier position that itself
read the cue). The hypothesis: a FEW heads of the band read the cue at the near position and the band's MLPs
amplify; a small unit set reproduces the band clamp.

Design (donor side, ODD A1 rows to select, EVEN A1 rows to evaluate; same row filter and cue exclusion as v122):
  band(set)   = argmax band of the v122 receipt (early / mid) for the 7 target sets (v122 loss[all] >= 0.15)
  band_loss   = all heads + MLPs of the band clamped to base at the near non-cue positions (v122's measurement)
  heads_band  = only the HEADS of the band clamped (MLPs live). For the early band this must equal band_loss up to
                the MLPs' own read of the clamped residual (a consistency check); for the mid band the residual
                entering layer 5 already differs, so heads_band < band_loss is possible.
  single(h)   = loss when only head h of the band is clamped
  greedy      = TRUE marginal greedy over the band's heads: at each step the head whose addition gives the largest
                cumulative loss is added, until GREEDY_FRAC x heads_band is reached (max K_MAX heads)
  transfer    = the ODD-selected greedy set clamped on the EVEN rows, over the EVEN heads_band
  controls    = the UNION of all selected heads clamped at the near non-cue positions of the 5 control sets

Pre-enqueue change, disclosed: the first draft ranked heads AND MLPs by single loss and added them in that order.
The CPU smoke on quantifier_number (4 odd rows) saturated at 0.355 of a 0.533 band clamp after mlp:01, mlp:00,
attn:00:head:03, attn:01:head:03 -- the MLPs at the near position amplify whatever the heads bring in, and a
singles ranking mixes the two roles. The redesign asks the head question only, with real marginal gains. The smoke
also showed cue offsets can sit at t-4 or earlier: a near position then differs through attention from any earlier
differing position, not only from t-1..t-3 cues.

Registered before the run (bars fixed):
  pred_a_instrument      exact donor-clamp of the band reproduces the donor (|loss| <= 0.02) on every set that ran
  pred_b_heads_carry     heads_band >= 0.7 x band_loss on >= 5/7 targets
  pred_c_sparse          greedy reaches >= 0.8 x heads_band with <= 4 heads on >= 5/7 targets
  pred_d_transfer        on EVEN rows the ODD-selected set reaches >= 0.6 x EVEN heads_band on >= 5/7 targets
  pred_e_controls_spared |loss| of the union clamp <= 0.15 on >= 4/5 controls that run
  pred_f_number_shared   the first greedy head is the same (layer, head) on >= 2 of the 3 number sets
                         (lexical_number_pp, perfect_number, quantifier_number)
Prior: b (early sets) and d likely; c is the sparsity question; f is a genuine question (v115 found the LATE number
heads shared; the early carrier need not be). If c fails the carrying is distributed over many early heads and I
will report the greedy curve, not lower the bar.

Smoke: V123_SMOKE=<out.json> runs on CPU with one set (V123_SMOKE_SET) and 4 rows per parity.
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
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_writer_band_v122 as v122

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_near_carrier_heads_v123_result.json"
V122 = ROOT / "circuits/followups/unit_tier5_near_writer_band_v122_result.json"
BANDS = {"early": range(0, 5), "mid": range(5, 9), "late": range(9, 12), "top": range(12, 15)}
OFFSETS = (1, 2, 3)
TARGET_MIN, INSTR_TOL, HEADS_FRAC, GREEDY_FRAC, K_MAX, TRANSFER_FRAC, CTRL_MAX = 0.15, 0.02, 0.7, 0.8, 4, 0.6, 0.15
K_B, K_C, K_D, K_E, K_F = 5, 5, 5, 4, 2
NUMBER = ("lexical_number_pp", "perfect_number", "quantifier_number")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3000, 60000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_near_carrier_heads_v123", "behaviours": 12,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def band_of(L):
    return max((k for k in BANDS if L.get(k) is not None), key=lambda k: L[k])


class Side:
    """Rows of one parity for one set: prepared batches, cue-excluded caches per offset, and a loss function."""

    def __init__(self, backend, rows, layers):
        self.rows = rows
        prep = g.prepare(backend, rows)
        self.D, self.B = prep.donor_batch, prep.base_batch
        self.donor_axis, self.base_axis = prep.donor_axis, prep.base_axis
        cue = {rid: {k for k in OFFSETS if r["base_ids"][r["base_semantic_position"] - k] != r["donor_ids"][r["donor_semantic_position"] - k]}
               for rid, r in zip(self.D.row_ids, rows)}
        self.cue_rows = sum(1 for c in cue.values() if c)
        self.hd, self.hb, self.md, self.mb = {}, {}, {}, {}
        for k in OFFSETS:
            keep = lambda cache, k=k: {key: v for key, v in cache.items() if k not in cue[key[0]]}
            self.hd[k] = keep(v120.head_cache(backend, v120.shifted(self.D, k), layers))
            self.hb[k] = keep(v120.head_cache(backend, v120.shifted(self.B, k), layers))
            self.md[k] = keep(v120.mlp_cache(backend, v120.shifted(self.D, k), layers))
            self.mb[k] = keep(v120.mlp_cache(backend, v120.shifted(self.B, k), layers))

    def loss(self, backend, units, donor=False):
        hs, ms = (self.hd, self.md) if donor else (self.hb, self.mb)
        hi = [(u, -k, hs[k]) for k in OFFSETS for u in units if u.startswith("attn")]
        mi = [(u, -k, ms[k]) for k in OFFSETS for u in units if u.startswith("mlp")]
        vals = v122.clamped_margins(backend, self.D, hi, mi)
        per = [(d - p) / (d - b) for d, b, p in zip(self.donor_axis, self.base_axis, vals) if abs(d - b) > 1e-6]
        return round(sum(per) / len(per), 3) if per else None


def rows_of(m, parity, smoke):
    a1 = g.rows_of(m, "A1")[parity::2]
    if smoke:
        a1 = a1[:4]
    return [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"]) and r["donor_semantic_position"] >= max(OFFSETS)
            and r["base_semantic_position"] == r["donor_semantic_position"]]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V123_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    v122r = json.load(open(V122))
    S = v115.sets()
    targets = {n: band_of(L) for n, L in v122r["summary"].items() if L and L.get("all") is not None and L["all"] >= TARGET_MIN}
    controls = [n for n, b in v122r["behaviours"].items() if b.get("kind") == "control"]
    if smoke:
        pick = os.environ.get("V123_SMOKE_SET") or sorted(targets)[0]
        targets = {pick: targets[pick]}; controls = controls[:1]
    R = {}
    for n, band in targets.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        layers = list(BANDS[band])
        units = [f"attn:{l:02d}:head:{h:02d}" for l in layers for h in range(g.N_HEADS)] + [f"mlp:{l:02d}" for l in layers]
        odd, even = rows_of(m, 1, smoke), rows_of(m, 0, smoke)
        if len(odd) < 4 or len(even) < 4:
            R[n] = {"band": band, "rows_odd": len(odd), "rows_even": len(even), "skipped": "fewer than 4 equal-length rows in a parity"}
            print(n, "skipped", flush=True); continue
        heads = [u for u in units if u.startswith("attn")]
        so = Side(backend, odd, layers)
        instrument = so.loss(backend, units, donor=True)
        band_loss = so.loss(backend, units)
        heads_band = so.loss(backend, heads)
        single = {u: so.loss(backend, [u]) for u in heads}
        chosen, cum = [], []
        remaining = list(heads)
        for _ in range(K_MAX):
            gains = {u: so.loss(backend, chosen + [u]) for u in remaining}
            best = max(remaining, key=lambda u: gains[u] if gains[u] is not None else -9)
            chosen.append(best); cum.append(gains[best]); remaining.remove(best)
            if heads_band and cum[-1] >= GREEDY_FRAC * heads_band:
                break
        se = Side(backend, even, layers)
        even_heads = se.loss(backend, heads)
        even_set = se.loss(backend, chosen)
        top = sorted(heads, key=lambda u: -(single[u] or -9))[:8]
        R[n] = {"band": band, "units": S[n], "rows_odd": len(odd), "rows_even": len(even), "cue_rows_odd": so.cue_rows,
                "instrument": instrument, "band_loss": band_loss, "heads_band": heads_band, "single_top8": {u: single[u] for u in top},
                "greedy": chosen, "greedy_cumulative": cum, "even_heads_band": even_heads, "even_greedy_loss": even_set,
                "seconds": round(time.perf_counter() - t1, 1)}
        print(n, band, "band", band_loss, "heads", heads_band, "greedy", chosen, cum, "even", even_heads, even_set, round(time.perf_counter() - t0), "s", flush=True)
    union = sorted({u for n in R if "skipped" not in R[n] for u in R[n]["greedy"]})
    union_layers = sorted({g.unit_layer(u) for u in union})
    C = {}
    for n in controls:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        odd = rows_of(m, 1, smoke)
        if len(odd) < 4 or not union:
            C[n] = {"rows": len(odd), "skipped": True}; continue
        sc = Side(backend, odd, union_layers)
        C[n] = {"rows": len(odd), "union_loss": sc.loss(backend, union)}
        print("control", n, C[n], flush=True)

    ran = [n for n in R if "skipped" not in R[n]]
    inst = [n for n in ran if R[n]["instrument"] is not None and abs(R[n]["instrument"]) <= INSTR_TOL]
    carry = [n for n in ran if R[n]["band_loss"] and R[n]["heads_band"] is not None and R[n]["heads_band"] >= HEADS_FRAC * R[n]["band_loss"]]
    sparse = [n for n in ran if R[n]["heads_band"] and R[n]["greedy_cumulative"][-1] >= GREEDY_FRAC * R[n]["heads_band"] and len(R[n]["greedy"]) <= K_MAX]
    transfer = [n for n in ran if R[n]["even_heads_band"] and R[n]["even_greedy_loss"] is not None
                and R[n]["even_greedy_loss"] >= TRANSFER_FRAC * R[n]["even_heads_band"]]
    ctrl_ran = [n for n in C if not C[n].get("skipped")]
    ctrl = [n for n in ctrl_ran if abs(C[n]["union_loss"]) <= CTRL_MAX]
    firsts = [R[n]["greedy"][0] for n in NUMBER if n in ran]
    shared = max((firsts.count(u) for u in set(firsts)), default=0)
    predictions = {
        "pred_a_instrument": len(inst) == len(ran) and bool(ran),
        "pred_b_heads_carry": len(carry) >= K_B,
        "pred_c_sparse": len(sparse) >= K_C,
        "pred_d_transfer": len(transfer) >= K_D,
        "pred_e_controls_spared": len(ctrl) >= K_E,
        "pred_f_number_shared": shared >= K_F,
    }
    result = {"predictions": predictions, "schema": "unit_tier5_near_carrier_heads_v123",
              "candidate_id": "corpus.unit_tier5_near_carrier_heads_v123",
              "bars": {"target_min": TARGET_MIN, "instr_tol": INSTR_TOL, "heads_frac": HEADS_FRAC, "greedy_frac": GREEDY_FRAC, "k_max": K_MAX,
                       "transfer_frac": TRANSFER_FRAC, "ctrl_max": CTRL_MAX, "K": [K_B, K_C, K_D, K_E, K_F], "offsets": list(OFFSETS)},
              "targets": targets, "controls": C, "union": union,
              "counts": {"ran": len(ran), "instrument": len(inst), "heads_carry": len(carry), "sparse": len(sparse),
                         "transfer": len(transfer), "controls_spared": len(ctrl), "controls_ran": len(ctrl_ran), "number_shared": shared},
              "summary": {n: [R[n].get("band_loss"), R[n].get("heads_band"), R[n].get("greedy"), R[n].get("greedy_cumulative"), R[n].get("even_heads_band"), R[n].get("even_greedy_loss")] for n in R},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
