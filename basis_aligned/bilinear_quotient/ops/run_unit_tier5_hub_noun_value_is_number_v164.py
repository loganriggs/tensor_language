#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: 11:03's head-noun read is a VALUE read of the noun's NUMBER, invariant to the noun's identity.

v163: 11:03 at t reads the head noun's key with a negative weight; that term carries ~half the head's effect. Which side of
the term carries the interchange -- the weight P[t,noun] or the value V_h[noun]? Printed before registering (5-6 rows per
parity, CPU): resid_add of image(P_base[t,noun] (V_donor - V_base)[noun]) alone gives 0.247 / 0.444 (0.264 / 0.400 on 6
rows) of the exact 11:03 interchange 0.488 / 0.644, the weight swap ((P_donor - P_base)[t,noun] V_base[noun]) 0.003 / -0.031,
the cross term -0.002 / 0.119. But |V_donor - V_base| at the noun is 330-387 against |V_base| 377-431 -- the noun token
differs ('director' vs 'directors'), so the swap carries identity as well as number. Control: swapping the noun's value
with the SAME-NUMBER value of a DIFFERENT noun (the next row in the same parity: 'leaders' <- 'rangers', |dv| 408-431,
a larger perturbation than the number swap) gives -0.001 / 0.002. Instrument: image(out_donor - out_base) at t equals the
exact 11:03 interchange (0.488 = 0.488). Quantifier (noun = t): value swap at t 0.093 / 0.087 of 0.305 / 0.292, weight ~0.
Sets: lexical seven p0+p1, quantifier seven p0+p1 (16 rows each).
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'; quantifier A1 p0/p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_instrument   all four runs: resid_add of image(out_donor - out_base) at t equals the exact 11:03 interchange within 0.005
  pred_b_value_side   lexical, both parities: noun value-swap / exact within 0.35-0.85 and |noun weight-swap| <= 0.05
  pred_c_identity_inv lexical, both parities: |same-number other-noun value-swap| <= 0.05 while its |dv| is within 0.8-1.5 x the number swap's |dv|
  pred_d_quantifier   quantifier, both parities: value-swap at t / exact within 0.15-0.5 and |weight-swap| <= 0.05
  pred_e_noun_dominant lexical, both parities: the noun's value-swap is within 2-50 x the largest other single key's value-swap (keys t-4..t)
Reported, unregistered: cross terms, per-key value swaps, |dv| per key, the control's per-row values.
Prior: a/b/c/d printed; e unprinted (other keys' values change only through context) -- the informative bar.
Smoke: V164_SMOKE=<out.json> -> CPU, 4 rows per parity.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_hub_noun_value_is_number_v164_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
INSTR_TOL, VAL_LO, VAL_HI, WT_MAX, CTRL_MAX, DV_LO, DV_HI, Q_LO, Q_HI, DOM_LO, DOM_HI, KEYS = 0.005, 0.35, 0.85, 0.05, 0.05, 0.8, 1.5, 0.15, 0.5, 2.0, 50.0, 5
LEX_NOUN_OFFSET = 3  # lexical noun's last token is t - 3 on each side (' near the <object>' follows)
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_noun_value_is_number_v164", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V164_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    P = {}
    DIRS = {}
    for n in ("lexical_number_pp", "quantifier_number"):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        P[n] = {"p0": g.prepare(backend, cut(a1[0::2])), "p1": g.prepare(backend, cut(a1[1::2]))}
        DIRS[n] = {r["row_id"]: r["direction_id"] for r in a1}
    attn = backend.model.transformer.h[LAYER].attn
    W = attn.c_proj.weight
    Wh = W[:, HUB_H * g.HEAD_DIM:(HUB_H + 1) * g.HEAD_DIM]
    R = {}
    img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
    runs = [("lex_seven", "p0"), ("lex_seven", "p1"), ("quant_seven", "p0"), ("quant_seven", "p1")]
    for sname, par in runs:
        n = SETS[sname][0]
        prep = P[n][par]
        batch, db = prep.base_batch, prep.donor_batch
        sem, semd = list(batch.semantic_positions), list(db.semantic_positions)
        rows = len(batch.row_ids)
        off = LEX_NOUN_OFFSET if sname == "lex_seven" else 0
        nb, nd = [p - off for p in sem], [p - off for p in semd]
        def rec(add=None, units=()):
            out = g.forward_units(backend, batch, units=units, donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        with torch.no_grad():
            PB, VB = v131.capture_with_clamp(backend, batch, [], [], LAYER)
            PD, VD = v131.capture_with_clamp(backend, db, [], [], LAYER)
            outb = torch.stack([PB[i, HUB_H, sem[i], :] @ VB[i, :, HUB_H, :] for i in range(rows)]).float().to(backend.device)
            outd = torch.stack([PD[i, HUB_H, semd[i], :] @ VD[i, :, HUB_H, :] for i in range(rows)]).float().to(backend.device)
            def key_terms(kk):
                kb, kd = [p - kk for p in sem], [p - kk for p in semd]
                if min(kb) < 0 or min(kd) < 0:
                    return None
                pb = torch.stack([PB[i, HUB_H, sem[i], kb[i]] for i in range(rows)]).to(backend.device)
                pd = torch.stack([PD[i, HUB_H, semd[i], kd[i]] for i in range(rows)]).to(backend.device)
                vb = torch.stack([VB[i, kb[i], HUB_H, :] for i in range(rows)]).to(backend.device)
                vd = torch.stack([VD[i, kd[i], HUB_H, :] for i in range(rows)]).to(backend.device)
                return pb, pd, vb, vd
            per_key = {}
            for kk in range(KEYS):
                kt = key_terms(kk)
                if kt is None:
                    break
                pb, pd, vb, vd = kt
                per_key[f"t-{kk}"] = {"value_swap": (pb[:, None] * (vd - vb)).float(), "weight_swap": ((pd - pb)[:, None] * vb).float(),
                                      "cross": ((pd - pb)[:, None] * (vd - vb)).float(), "dv": round(float((vd - vb).norm(dim=1).mean()), 1)}
            pb, pd, vb, vd = key_terms(off)
            perm = [(i + 1) % rows for i in range(rows)]
            vs = vb[perm]
            ctrl = (pb[:, None] * (vs - vb)).float()
            dv_ctrl = round(float((vs - vb).norm(dim=1).mean()), 1)
            dv_num = round(float((vd - vb).norm(dim=1).mean()), 1)
        exact = rec(units=[HUB])
        instr = rec(img(outd - outb))
        eff = {k: {"value_swap": rec(img(v["value_swap"])), "weight_swap": rec(img(v["weight_swap"])), "cross": rec(img(v["cross"])), "dv": v["dv"]} for k, v in per_key.items()}
        noun_key = f"t-{off}"
        S = {"rows": rows, "exact_hub": exact, "add_out_delta": instr, "by_key": eff, "noun_key": noun_key,
             "noun_value_swap": eff[noun_key]["value_swap"], "noun_weight_swap": eff[noun_key]["weight_swap"], "noun_cross": eff[noun_key]["cross"],
             "control_same_number_other_noun": rec(img(ctrl)), "dv_control": dv_ctrl, "dv_number": dv_num,
             "control_rotation": "row i takes row (i+1) mod n of the same parity"}
        S["value_share"] = round(S["noun_value_swap"] / exact, 3) if abs(exact) > 1e-6 else None
        others = [abs(v["value_swap"]) for k, v in eff.items() if k != noun_key]
        S["largest_other_key_value_swap"] = max(others) if others else None
        S["dv_ratio_control_over_number"] = round(dv_ctrl / dv_num, 3) if dv_num else None
        R[f"{sname}:{par}"] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    L = [R["lex_seven:p0"], R["lex_seven:p1"]]; Q = [R["quant_seven:p0"], R["quant_seven:p1"]]
    pred_a = all(abs(S["add_out_delta"] - S["exact_hub"]) <= INSTR_TOL for S in R.values())
    pred_b = all(S["value_share"] is not None and VAL_LO <= S["value_share"] <= VAL_HI and abs(S["noun_weight_swap"]) <= WT_MAX for S in L)
    pred_c = all(abs(S["control_same_number_other_noun"]) <= CTRL_MAX and S["dv_ratio_control_over_number"] is not None and DV_LO <= S["dv_ratio_control_over_number"] <= DV_HI for S in L)
    pred_d = all(S["value_share"] is not None and Q_LO <= S["value_share"] <= Q_HI and abs(S["noun_weight_swap"]) <= WT_MAX for S in Q)
    pred_e = all(S["largest_other_key_value_swap"] is not None and DOM_LO * S["largest_other_key_value_swap"] <= abs(S["noun_value_swap"]) <= DOM_HI * S["largest_other_key_value_swap"] for S in L)
    predictions = {"pred_a_instrument": pred_a, "pred_b_value_side": pred_b, "pred_c_identity_inv": pred_c, "pred_d_quantifier": pred_d, "pred_e_noun_dominant": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_noun_value_is_number_v164", "candidate_id": "corpus.unit_tier5_hub_noun_value_is_number_v164",
              "bars": {"instr_tol": INSTR_TOL, "value_band": [VAL_LO, VAL_HI], "weight_max": WT_MAX, "control_max": CTRL_MAX, "dv_band": [DV_LO, DV_HI], "quant_band": [Q_LO, Q_HI], "dominance_band": [DOM_LO, DOM_HI], "keys": KEYS},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
