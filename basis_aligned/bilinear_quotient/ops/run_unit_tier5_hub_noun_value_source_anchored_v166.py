#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: v165 re-run with the head noun located by the ' near' anchor (not t-3), plus v164's core bars at the corrected key.

v165 pred_a failed at 0.934 / 0.812 because 4 of 32 lexical rows are 6 tokens long by splitting the PP OBJECT (falcon,
orchard, willow, meadow), so t-3 is 'near' there and the embedding patch was a no-op (13/16 = 0.812). Here noun_last =
index(' near') - 1 on each side (the ' near' id is read from a 5-token row); the one truly misaligned row (p0 'rangers',
6 vs 5 tokens) is kept and disclosed (so the instrument ceiling is 15/16 = 0.94 on p0). Quantifier rows: noun = t.
Measures as v165 (per-unit interchange at the noun position: wte, attn:l, mlp:l for l = 0..10; 11:03's value-read effect
via image(P_base[t,noun] dV) injected at t, and the plain patched forward), plus v164's split of the noun term into weight /
value / cross and the same-number other-noun value-swap control at the corrected key.
Magnitudes from v165 (wrong key on 4/32 rows) and v164 (same): direct route 0.009 / -0.004, computed 0.96 / 1.01 of full,
mlp:01 0.50 / 0.35 of full, attn <= 0.021; value share 0.50 / 0.62, weight 0.006 / -0.011, control -0.001 / 0.003.
Sets: lexical seven p0+p1, quantifier seven p0+p1 (16 rows each).
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'; quantifier A1 p0/p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_instrument     lexical, both parities: the noun-embedding patch alone recovers the whole donor within 0.90-1.05 (p0 ceiling
                        15/16) and its value swap equals the full noun value swap within 0.03
  pred_b_not_direct     lexical, both parities: embed donor with all 22 noun writes clamped to base gives |value swap| <= 0.03 and
                        whole-model <= 0.20
  pred_c_computed       lexical, both parities: writes donor with embed base gives value swap within 0.8-1.2 x the full noun value swap
  pred_d_mlp01_source   lexical, both parities: mlp:01 at the noun is the largest single unit, its value swap is 0.3-0.75 of the full
                        noun value swap, and every attn:l at the noun has |value swap| <= 0.03
  pred_e_quant_attn07   quantifier, both parities: attn:07 at t is the largest single unit, its value swap is 0.3-0.7 of the full value
                        swap at t, and its whole-model recovery is within 0.20-0.60
  pred_f_value_side     lexical, both parities: noun value-swap / exact 11:03 interchange within 0.35-0.85 and |weight-swap| <= 0.05
  pred_g_identity_inv   lexical, both parities: |same-number other-noun value-swap| <= 0.05 while its |dv| is within 0.8-1.5 x the number swap's
Reported, unregistered: every unit's value swap / whole-model / |dV| fraction; sum of singles; cross term; the noun positions used.
Smoke: V166_SMOKE=<out.json> -> CPU, 4 rows per parity (~40 s).
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_noun_value_source_anchored_v166_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
SRC_LAYERS = 11
BARS = {"instr_whole": [0.90, 1.05], "instr_value_tol": 0.03, "direct_value_max": 0.03, "direct_whole_max": 0.20, "computed_band": [0.8, 1.2], "mlp01_band": [0.3, 0.75], "attn_value_max": 0.03, "attn07_band": [0.3, 0.7], "attn07_whole_band": [0.20, 0.60], "value_band": [0.35, 0.85], "weight_max": 0.05, "control_max": 0.05, "dv_band": [0.8, 1.5]}
LEX_NOUN_OFFSET = 3  # lexical noun's last token is t - 3 on each side (' near the <object>' follows)
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_noun_value_source_anchored_v166", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V166_SMOKE")
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
    model = backend.model
    UNITS = ["embed"] + [f"{k}:{l:02d}" for l in range(SRC_LAYERS) for k in ("attn", "mlp")]
    for sname, par in runs:
        n = SETS[sname][0]
        prep = P[n][par]
        batch, db = prep.base_batch, prep.donor_batch
        sem, semd = list(batch.semantic_positions), list(db.semantic_positions)
        rows = len(batch.row_ids)
        if sname == "lex_seven":
            near = next(r[2] for r in batch.token_rows if len(r) == 5)
            nb = [list(r).index(near) - 1 for r in batch.token_rows]
            nd = [list(r).index(near) - 1 for r in db.token_rows]
        else:
            nb, nd = list(sem), list(semd)
        ar = torch.arange(rows, device=backend.device)
        nb_t, nd_t = torch.tensor(nb, device=backend.device), torch.tensor(nd, device=backend.device)
        # --- donor captures at the donor's noun position: wte output, every layer's c_proj input, every layer's Down output
        cap = {}
        hs = [model.transformer.wte.register_forward_hook(lambda m, a, o: cap.__setitem__("embed", o[ar, nd_t].detach().clone()))]
        for l in range(SRC_LAYERS):
            hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(f"attn:{l:02d}", a[0][ar, nd_t].detach().clone())))
            hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(f"mlp:{l:02d}", o[ar, nd_t].detach().clone())))
        with torch.no_grad():
            PD, VD = v131.capture_with_clamp(backend, db, [], [], LAYER)
        for h_ in hs: h_.remove()
        # base captures (for the all-writes clamp)
        capb = {}
        hs = [model.transformer.wte.register_forward_hook(lambda m, a, o: capb.__setitem__("embed", o[ar, nb_t].detach().clone()))]
        for l in range(SRC_LAYERS):
            hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m, a, l=l: capb.__setitem__(f"attn:{l:02d}", a[0][ar, nb_t].detach().clone())))
            hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m, a, o, l=l: capb.__setitem__(f"mlp:{l:02d}", o[ar, nb_t].detach().clone())))
        with torch.no_grad():
            PB, VB = v131.capture_with_clamp(backend, batch, [], [], LAYER)
        for h_ in hs: h_.remove()

        def clamped_forward(spec):
            """spec: {unit: source} with source 'donor' or 'base'; clamps that unit's write at the base noun position. Returns (P, V, margins)."""
            hs = []
            def mk_set(src_unit):
                val = cap[src_unit] if spec[src_unit] == "donor" else capb[src_unit]
                return val
            if "embed" in spec:
                def eh(m, a, o):
                    o = o.clone(); o[ar, nb_t] = mk_set("embed").to(o.dtype); return o
                hs.append(model.transformer.wte.register_forward_hook(eh))
            for u in spec:
                if u == "embed": continue
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m, a, u=u):
                        v = a[0].clone(); v[ar, nb_t] = mk_set(u).to(v.dtype); return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m, a, o, u=u):
                        o = o.clone(); o[ar, nb_t] = mk_set(u).to(o.dtype); return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            try:
                with torch.no_grad():
                    Pp, Vp = v131.capture_with_clamp(backend, batch, [], [], LAYER)
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return Pp, Vp, round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        def rec(add=None, units=()):
            out = g.forward_units(backend, batch, units=units, donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        pb = PB[ar, HUB_H, torch.tensor(sem, device=backend.device), nb_t].to(backend.device)
        vb = VB[ar, nb_t, HUB_H, :].to(backend.device)
        vd = VD[ar, nd_t, HUB_H, :].to(backend.device)
        exact = rec(units=[HUB])
        full_value_swap = rec(img((pb[:, None] * (vd - vb)).float()))
        per_unit = {}
        for u in UNITS:
            Pp, Vp, whole = clamped_forward({u: "donor"})
            vp = Vp[ar, nb_t, HUB_H, :].to(backend.device)
            per_unit[u] = {"value_swap_at_t": rec(img((pb[:, None] * (vp - vb)).float())), "whole_model": whole,
                           "dv_frac": round(float(((vp - vb).norm(dim=1) / (vd - vb).norm(dim=1).clamp_min(1e-6)).mean()), 3)}
        all_writes = {u: "donor" for u in UNITS if u != "embed"}
        _, Vp, whole_w = clamped_forward({**all_writes})
        vp = Vp[ar, nb_t, HUB_H, :].to(backend.device)
        computed = {"value_swap_at_t": rec(img((pb[:, None] * (vp - vb)).float())), "whole_model": whole_w}
        _, Vp, whole_e = clamped_forward({"embed": "donor", **{u: "base" for u in UNITS if u != "embed"}})
        vp = Vp[ar, nb_t, HUB_H, :].to(backend.device)
        direct = {"value_swap_at_t": rec(img((pb[:, None] * (vp - vb)).float())), "whole_model": whole_e}
        _, Vp, whole_all = clamped_forward({u: "donor" for u in UNITS})
        vp = Vp[ar, nb_t, HUB_H, :].to(backend.device)
        everything = {"value_swap_at_t": rec(img((pb[:, None] * (vp - vb)).float())), "whole_model": whole_all,
                      "dv_frac": round(float(((vp - vb).norm(dim=1) / (vd - vb).norm(dim=1).clamp_min(1e-6)).mean()), 3)}
        wt = rec(img(((PD[ar, HUB_H, torch.tensor(semd, device=backend.device), nd_t].to(backend.device) - pb)[:, None] * vb).float()))
        cross = rec(img(((PD[ar, HUB_H, torch.tensor(semd, device=backend.device), nd_t].to(backend.device) - pb)[:, None] * (vd - vb)).float()))
        perm = [(i + 1) % rows for i in range(rows)]
        ctrl = rec(img((pb[:, None] * (vb[perm] - vb)).float()))
        dv_ratio = round(float((vb[perm] - vb).norm(dim=1).mean() / (vd - vb).norm(dim=1).mean().clamp_min(1e-6)), 3)
        S = {"rows": rows, "exact_hub": exact, "noun_positions_base": nb, "noun_positions_donor": nd,
             "value_share_of_exact": round(full_value_swap / exact, 3) if abs(exact) > 1e-6 else None, "noun_weight_swap": wt, "noun_cross": cross,
             "control_same_number_other_noun": ctrl, "dv_ratio_control_over_number": dv_ratio, "noun_value_swap_full": full_value_swap, "per_unit_at_noun": per_unit,
             "embed_only_writes_clamped": direct, "writes_only_embed_base": computed, "embed_and_all_writes": everything}
        R[f"{sname}:{par}"] = S
        print(sname, par, "exact", exact, "value_swap", full_value_swap, "embed", per_unit["embed"], "direct", direct, "computed", computed, "all", everything, flush=True)
        print({u: (v["value_swap_at_t"], v["whole_model"], v["dv_frac"]) for u, v in per_unit.items()}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_hub_noun_value_source_anchored_v166", "candidate_id": "corpus.unit_tier5_hub_noun_value_source_anchored_v166",
              "bars": BARS, "units_at_noun": UNITS,
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    L = [R["lex_seven:p0"], R["lex_seven:p1"]]; Q = [R["quant_seven:p0"], R["quant_seven:p1"]]
    B = BARS
    def share(S, u):
        f = S["noun_value_swap_full"]
        return S["per_unit_at_noun"][u]["value_swap_at_t"] / f if abs(f) > 1e-6 else float("nan")
    def largest(S):
        return max(S["per_unit_at_noun"], key=lambda u: abs(S["per_unit_at_noun"][u]["value_swap_at_t"]) if u != "embed" else -1)
    pred_a = all(B["instr_whole"][0] <= S["per_unit_at_noun"]["embed"]["whole_model"] <= B["instr_whole"][1]
                 and abs(S["per_unit_at_noun"]["embed"]["value_swap_at_t"] - S["noun_value_swap_full"]) <= B["instr_value_tol"] for S in L)
    pred_b = all(abs(S["embed_only_writes_clamped"]["value_swap_at_t"]) <= B["direct_value_max"] and S["embed_only_writes_clamped"]["whole_model"] <= B["direct_whole_max"] for S in L)
    pred_c = all(abs(S["noun_value_swap_full"]) > 1e-6 and B["computed_band"][0] <= S["writes_only_embed_base"]["value_swap_at_t"] / S["noun_value_swap_full"] <= B["computed_band"][1] for S in L)
    pred_d = all(largest(S) == "mlp:01" and B["mlp01_band"][0] <= share(S, "mlp:01") <= B["mlp01_band"][1]
                 and all(abs(v["value_swap_at_t"]) <= B["attn_value_max"] for u, v in S["per_unit_at_noun"].items() if u.startswith("attn")) for S in L)
    pred_e = all(largest(S) == "attn:07" and B["attn07_band"][0] <= share(S, "attn:07") <= B["attn07_band"][1]
                 and B["attn07_whole_band"][0] <= S["per_unit_at_noun"]["attn:07"]["whole_model"] <= B["attn07_whole_band"][1] for S in Q)
    pred_f = all(S["value_share_of_exact"] is not None and B["value_band"][0] <= S["value_share_of_exact"] <= B["value_band"][1] and abs(S["noun_weight_swap"]) <= B["weight_max"] for S in L)
    pred_g = all(abs(S["control_same_number_other_noun"]) <= B["control_max"] and B["dv_band"][0] <= S["dv_ratio_control_over_number"] <= B["dv_band"][1] for S in L)
    return {"pred_a_instrument": pred_a, "pred_f_value_side": pred_f, "pred_g_identity_inv": pred_g, "pred_b_not_direct": pred_b, "pred_c_computed": pred_c, "pred_d_mlp01_source": pred_d, "pred_e_quant_attn07": pred_e}


if __name__ == "__main__":
    main()
