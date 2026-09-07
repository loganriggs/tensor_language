#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: 11:03 at t reads the HEAD NOUN's key with a negative weight -- the lexical form of quantifier's self-read.

v160-v162: in quantifier rows the head noun is t and 11:03's negative self-weight there (P[t,t] -0.12) carries 0.61 of the
head's effect on the All side. In lexical rows the noun sits at t-3 ('The director near the harbor', t = 'harbor'). Printed
before registering (5 rows per parity, CPU): P[t, noun] = -0.18..-0.33 on every row (both parities), the noun-key term
P[t,noun] V_h[noun] has norm ~110 (the largest key term; 'near' at t-2: 58-73, self at t: 5-14), and resid_add of -image of
the noun-key term alone moves the margin 0.137 (p0, base plural) / 0.123 (p1, base singular) of the whole head's 0.224 /
0.226 -- two-sided, unlike quantifier's self-term. Per-key removals sum to 0.190 / 0.210 vs full 0.224 / 0.226.
Sets: lexical seven p0 and p1 (16 rows each; the 5 printed rows are 5 of each 16); quantifier p1 rerun for the contrast.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'; quantifier A1 p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_instrument   lexical, both parities: the per-key removal effects (keys t-4..t) sum to the full-output removal within 0.05
  pred_b_noun_weight  lexical, both parities: P[t, noun] < 0 on >= 75% of rows on the base AND the donor side, mean within -0.4..-0.1
  pred_c_noun_share   lexical, both parities: removal(noun-key term) / removal(full 11:03 output at t) within 0.4-0.8
  pred_d_noun_is_max  lexical, both parities: the noun key is the largest |P[t,k]| (k <= t) on >= 60% of base rows
  pred_e_two_sided    lexical: removal(noun-key term) within 0.08-0.2 on BOTH parities; quantifier p1 self-term removal within 0.03-0.15 (v160 reproduced)
Reported, unregistered: per-key weights and removals (t-4..t), 'near' (t-2) share, term norms, plural-vs-singular |P[t,noun]|
keyed by direction_id, donor-side weights.
Prior: b/c hold on the printed rows; d is the shakiest (one printed row had 'near' larger); e is the contrast with v162 pred_d.
Smoke: V163_SMOKE=<out.json> -> CPU, 4 rows per parity.
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_noun_key_read_v163_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
ADD_TOL, FRAC, MAX_FRAC, W_LO, W_HI, SHARE_LO, SHARE_HI, REM_LO, REM_HI, Q_LO, Q_HI, KEYS = 0.05, 0.75, 0.6, -0.4, -0.1, 0.4, 0.8, 0.08, 0.2, 0.03, 0.15, 5
LEX_NOUN_OFFSET = 3  # lexical noun's last token is t - 3 on each side (' near the <object>' follows)
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_noun_key_read_v163", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V163_SMOKE")
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
    runs = [("lex_seven", "p0"), ("lex_seven", "p1"), ("quant_seven", "p1")]
    for sname, par in runs:
        n = SETS[sname][0]
        prep = P[n][par]
        batch = prep.base_batch
        sem = list(batch.semantic_positions)
        rows = len(batch.row_ids)
        off = LEX_NOUN_OFFSET if sname == "lex_seven" else 0
        noun = [p - off for p in sem]
        noun_d = [p - off for p in prep.donor_batch.semantic_positions]
        plural_is_base = [DIRS[n][rid].startswith("plural") for rid in batch.row_ids]
        def rec(add=None):
            out = g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        with torch.no_grad():
            PB, VB = v131.capture_with_clamp(backend, batch, [], [], LAYER)
            PD, _ = v131.capture_with_clamp(backend, prep.donor_batch, [], [], LAYER)
            full = torch.stack([PB[i, HUB_H, sem[i], :] @ VB[i, :, HUB_H, :] for i in range(rows)]).float().to(backend.device)
            terms, weights, norms = {}, {}, {}
            for kk in range(KEYS):
                keys = [p - kk for p in sem]
                if min(keys) < 0:
                    break
                term = torch.stack([PB[i, HUB_H, sem[i], keys[i]] * VB[i, keys[i], HUB_H, :] for i in range(rows)]).float().to(backend.device)
                terms[f"t-{kk}"] = term
                weights[f"t-{kk}"] = [round(float(PB[i, HUB_H, sem[i], keys[i]]), 4) for i in range(rows)]
                norms[f"t-{kk}"] = round(float(term.norm(dim=1).mean()), 2)
            w_noun = torch.stack([PB[i, HUB_H, sem[i], noun[i]] for i in range(rows)]).float()
            w_noun_d = torch.stack([PD[i, HUB_H, prep.donor_batch.semantic_positions[i], noun_d[i]] for i in range(rows)]).float()
            noun_term = torch.stack([PB[i, HUB_H, sem[i], noun[i]] * VB[i, noun[i], HUB_H, :] for i in range(rows)]).float().to(backend.device)
            noun_is_max = torch.stack([(PB[i, HUB_H, sem[i], :sem[i] + 1].abs().argmax() == noun[i]) for i in range(rows)]).float()
        base = rec()
        eff = {k: round(rec(-img(v)) - base, 3) for k, v in terms.items()}
        eff_full = round(rec(-img(full)) - base, 3)
        eff_noun = round(rec(-img(noun_term)) - base, 3)
        pl = [float(w_noun[i]) if plural_is_base[i] else float(w_noun_d[i]) for i in range(rows)]
        sg = [float(w_noun_d[i]) if plural_is_base[i] else float(w_noun[i]) for i in range(rows)]
        S = {"rows": rows, "base": base, "remove_full": eff_full, "remove_noun_term": eff_noun, "remove_by_key": eff, "sum_by_key": round(sum(eff.values()), 3),
             "weights_by_key": weights, "norms_by_key": norms, "noun_offset": off,
             "p_t_noun_base": [round(float(x), 4) for x in w_noun], "p_t_noun_donor": [round(float(x), 4) for x in w_noun_d],
             "noun_neg_frac_base": round(float((w_noun < 0).float().mean()), 3), "noun_neg_frac_donor": round(float((w_noun_d < 0).float().mean()), 3),
             "p_t_noun_mean_base": round(float(w_noun.mean()), 4), "p_t_noun_mean_donor": round(float(w_noun_d.mean()), 4),
             "noun_is_max_frac": round(float(noun_is_max.mean()), 3), "noun_term_norm": round(float(noun_term.norm(dim=1).mean()), 2),
             "abs_p_t_noun_plural_mean": round(sum(abs(x) for x in pl) / rows, 4), "abs_p_t_noun_singular_mean": round(sum(abs(x) for x in sg) / rows, 4)}
        S["noun_share"] = round(eff_noun / eff_full, 3) if abs(eff_full) > 1e-6 else None
        S["near_share"] = round(eff.get("t-2", 0.0) / eff_full, 3) if (abs(eff_full) > 1e-6 and off) else None
        R[f"{sname}:{par}"] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    L0, L1, Q1 = R["lex_seven:p0"], R["lex_seven:p1"], R["quant_seven:p1"]
    pred_a = all(abs(L["sum_by_key"] - L["remove_full"]) <= ADD_TOL for L in (L0, L1))
    pred_b = all(L["noun_neg_frac_base"] >= FRAC and L["noun_neg_frac_donor"] >= FRAC and W_LO <= L["p_t_noun_mean_base"] <= W_HI for L in (L0, L1))
    pred_c = all(L["noun_share"] is not None and SHARE_LO <= L["noun_share"] <= SHARE_HI for L in (L0, L1))
    pred_d = all(L["noun_is_max_frac"] >= MAX_FRAC for L in (L0, L1))
    pred_e = all(REM_LO <= L["remove_noun_term"] <= REM_HI for L in (L0, L1)) and Q_LO <= Q1["remove_noun_term"] <= Q_HI
    predictions = {"pred_a_instrument": pred_a, "pred_b_noun_weight": pred_b, "pred_c_noun_share": pred_c, "pred_d_noun_is_max": pred_d, "pred_e_two_sided": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_noun_key_read_v163", "candidate_id": "corpus.unit_tier5_hub_noun_key_read_v163",
              "bars": {"add_tol": ADD_TOL, "frac": FRAC, "max_frac": MAX_FRAC, "weight_band": [W_LO, W_HI], "share_band": [SHARE_LO, SHARE_HI], "removal_band": [REM_LO, REM_HI], "quant_band": [Q_LO, Q_HI], "keys": KEYS},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
