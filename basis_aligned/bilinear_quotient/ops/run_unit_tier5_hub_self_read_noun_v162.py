#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: 11:03's negative self-weight is a NOUN property scaled by plurality; its self-term defends the plural side only.

Printed before registering (CPU): the diagonal of 11:03's pattern is most negative at the head noun in quantifier rows
('All of the dry directors': -0.113 at 'directors', -0.04..+0.14 elsewhere; noun is the row minimum on 16/16 rows, both
sides); in lexical rows the subject noun at position 1 ('The director near the harbor') is negative on 16/16 rows on both
sides, singular base -0.052 vs plural donor -0.085 (plural more negative 16/16), while the read position t ('harbor') is
+0.006. Tokenisation: some plurals split into stem + 's' (rows of 6 tokens), so the noun is read at its LAST token, t - 3 on
each side (quantifier: the noun is t), and plurality is keyed by direction_id (p0 rows are plural_to_singular, p1 singular_to_plural). At the lexical noun 11:03's output is 81% self-term (24.6 of 30.2) but resid_add of -image at the NOUN position is
margin-inert (0.000 on 6 rows), and quantifier's self-term removal at t is 0.073 on p1 (base 'All' = plural) vs 0.005 on p0
(base 'Each') -- the one-sided readout default of v95/v96, which v160 (p1 only) did not show. This rung registers the
noun claims on the UNSEEN p0 rows (the diagonal statistics above were read on p1) and both parities for the effects.
New code path: forward_units(resid_add_positions=...) (non-breaking default None) -- instrument-checked below.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'; quantifier A1 p0/p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_instrument   quantifier p1: removal of the self-term at t via resid_add_positions=[t] equals the default path within 0.001
  pred_b_noun_sign    quantifier p0: the head noun t is the diagonal minimum on >= 75% of rows (base and donor) with mean P[t,t] within -0.2..-0.07;
                      lexical p0: subject-noun P[n,n] < 0 on >= 75% of rows on both sides (n = the noun's last token)
  pred_c_plurality    lexical p0+p1 (32 rows): the PLURAL side's P[n,n] is more negative than the singular side's on >= 75% of rows, mean
                      difference within 0.015-0.08
  pred_f_gate_const   lexical p0+p1: the gate q2.k2/D at the noun is within 0.9-1.1 x between the plural and singular sides (q.k alone carries
                      plurality; smoke on 8 rows hinted 1.24 -- bar kept as registered)
  pred_d_one_sided    quantifier: removal of the self-term at t within 0.03-0.15 on p1 (base 'All') and within -0.02..+0.03 on p0 (base 'Each')
  pred_e_noun_inert   lexical p0 and p1: |self-term| / |11:03 output| at the noun within 0.6-0.95, yet |removal at the noun| <= 0.02 on both parities
Reported, unregistered: lexical removal of the self-term at t on both parities (p0 showed 0.035 on 6 rows, p1 -0.001 in v160),
removal of the whole 11:03 output at the noun, per-row diagonals, f1/f2 at the noun on both sides.
Prior: the 4-row smoke of a first draft (noun read at position 1 on both sides) showed no plurality difference on p0 -- the
tokenisation confound above; the pooled 32-row claim at the noun's last token is the registered form. b/c/e are the noun
hypothesis on unseen rows; d is the retrospective caveat to v160 made into a registered claim.
Smoke: V162_SMOKE=<out.json> -> CPU, 4 rows per parity.
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_self_read_noun_v162_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
INSTR_TOL, FRAC, NOUN_LO, NOUN_HI, DIFF_LO, DIFF_HI, GATE_LO, GATE_HI, P1_LO, P1_HI, P0_LO, P0_HI, SHARE_LO, SHARE_HI, INERT = 0.001, 0.75, -0.2, -0.07, 0.015, 0.08, 0.9, 1.1, 0.03, 0.15, -0.02, 0.03, 0.6, 0.95, 0.02
LEX_NOUN_OFFSET = 3  # lexical noun's last token is t - 3 on each side (' near the <object>' follows)
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_self_read_noun_v162", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V162_SMOKE")
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
    D = g.HEAD_DIM
    img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
    def capture(bt, pos_list):
        rows = len(bt.row_ids)
        got = {}
        def pre(_m, args):
            got["x"] = torch.stack([args[0][i, pos_list[i]] for i in range(rows)]).detach().clone()
        hnd = attn.c_q.register_forward_pre_hook(pre)
        try:
            Pm, Vm = v131.capture_with_clamp(backend, bt, [], [], LAYER)
        finally:
            hnd.remove()
        x = got["x"]
        hv = lambda lin: lin(x).view(rows, g.N_HEADS, D)[:, HUB_H, :].float()
        q, k, q2, k2 = [torch.nn.functional.rms_norm(hv(l), (D,)) for l in (attn.c_q, attn.c_k, attn.c_q2, attn.c_k2)]
        f1 = (q * k).sum(1) / D
        f2 = (q2 * k2).sum(1) / D
        pdiag = torch.stack([Pm[i, HUB_H, pos_list[i], pos_list[i]] for i in range(rows)]).float()
        selfterm = torch.stack([Pm[i, HUB_H, pos_list[i], pos_list[i]] * Vm[i, pos_list[i], HUB_H, :] for i in range(rows)]).float().to(backend.device)
        full = torch.stack([Pm[i, HUB_H, pos_list[i], :] @ Vm[i, :, HUB_H, :] for i in range(rows)]).float().to(backend.device)
        sem = list(bt.semantic_positions)
        diag_min_is_pos = torch.stack([(torch.diagonal(Pm[i, HUB_H])[:sem[i] + 1].argmin() == pos_list[i]) for i in range(rows)]).float()
        return {"f1": f1, "f2": f2, "p": pdiag, "self": selfterm, "full": full, "min_is_pos": diag_min_is_pos}
    for sname, (n, units, _) in SETS.items():
        if sname == "shared_four":
            continue
        for par in ("p0", "p1"):
            prep = P[n][par]
            batch = prep.base_batch
            sem = list(batch.semantic_positions)
            rows = len(batch.row_ids)
            noun = [p - LEX_NOUN_OFFSET for p in sem] if sname == "lex_seven" else sem
            noun_d = [p - LEX_NOUN_OFFSET for p in prep.donor_batch.semantic_positions] if sname == "lex_seven" else list(prep.donor_batch.semantic_positions)
            plural_is_base = torch.tensor([DIRS[n][rid].startswith("plural") for rid in batch.row_ids], device=backend.device)
            def rec(add=None, ps=None):
                out = g.forward_units(backend, batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                      resid_add=None if add is None else {LAYER: add}, resid_add_positions=ps)
                return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
            with torch.no_grad():
                B = capture(batch, noun)
                Dn = capture(prep.donor_batch, noun_d)
                T = capture(batch, sem)
            base = rec()
            S = {"rows": rows, "base": base,
                 "noun_p_base": [round(float(x), 4) for x in B["p"]], "noun_p_donor": [round(float(x), 4) for x in Dn["p"]],
                 "noun_p_mean_base": round(float(B["p"].mean()), 4), "noun_p_mean_donor": round(float(Dn["p"].mean()), 4),
                 "noun_neg_frac_base": round(float((B["p"] < 0).float().mean()), 3), "noun_neg_frac_donor": round(float((Dn["p"] < 0).float().mean()), 3),
                 "noun_is_diag_min_base": round(float(B["min_is_pos"].mean()), 3), "noun_is_diag_min_donor": round(float(Dn["min_is_pos"].mean()), 3),
                 "donor_more_negative_frac": round(float((Dn["p"] < B["p"]).float().mean()), 3), "mean_diff_base_minus_donor": round(float((B["p"] - Dn["p"]).mean()), 4),
                 "plural_p": [round(float(x), 4) for x in torch.where(plural_is_base, B["p"], Dn["p"])], "singular_p": [round(float(x), 4) for x in torch.where(plural_is_base, Dn["p"], B["p"])],
                 "f2_plural": [round(float(x), 4) for x in torch.where(plural_is_base, B["f2"], Dn["f2"])], "f2_singular": [round(float(x), 4) for x in torch.where(plural_is_base, Dn["f2"], B["f2"])],
                 "noun_positions_base": noun, "noun_positions_donor": noun_d, "plural_is_base_frac": round(float(plural_is_base.float().mean()), 3),
                 "f1_noun_base": round(float(B["f1"].mean()), 4), "f1_noun_donor": round(float(Dn["f1"].mean()), 4),
                 "f2_noun_base": round(float(B["f2"].mean()), 4), "f2_noun_donor": round(float(Dn["f2"].mean()), 4),
                 "f1_donor_more_negative_frac": round(float((Dn["f1"] < B["f1"]).float().mean()), 3),
                 "noun_self_norm": round(float(B["self"].norm(dim=1).mean()), 2), "noun_full_norm": round(float(B["full"].norm(dim=1).mean()), 2),
                 "remove_self_at_noun": round(rec(-img(B["self"]), noun) - base, 3), "remove_full_at_noun": round(rec(-img(B["full"]), noun) - base, 3),
                 "remove_self_at_t_positions_path": round(rec(-img(T["self"]), sem) - base, 3), "remove_self_at_t_default_path": round(rec(-img(T["self"])) - base, 3)}
            S["f2_side_ratio"] = round(S["f2_noun_donor"] / S["f2_noun_base"], 3) if S["f2_noun_base"] else None
            S["noun_self_share"] = round(S["noun_self_norm"] / S["noun_full_norm"], 3)
            R[f"{sname}:{par}"] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    Q0, Q1, L0, L1 = R["quant_seven:p0"], R["quant_seven:p1"], R["lex_seven:p0"], R["lex_seven:p1"]
    pred_a = abs(Q1["remove_self_at_t_positions_path"] - Q1["remove_self_at_t_default_path"]) <= INSTR_TOL
    pred_b = (Q0["noun_is_diag_min_base"] >= FRAC and Q0["noun_is_diag_min_donor"] >= FRAC and NOUN_LO <= Q0["noun_p_mean_base"] <= NOUN_HI
              and L0["noun_neg_frac_base"] >= FRAC and L0["noun_neg_frac_donor"] >= FRAC)
    pl = L0["plural_p"] + L1["plural_p"]; sg = L0["singular_p"] + L1["singular_p"]
    f2p = L0["f2_plural"] + L1["f2_plural"]; f2s = L0["f2_singular"] + L1["f2_singular"]
    plural_more_negative_frac = sum(a < b for a, b in zip(pl, sg)) / len(pl)
    plural_mean_diff = sum(b - a for a, b in zip(pl, sg)) / len(pl)
    gate_ratio = (sum(f2p) / len(f2p)) / (sum(f2s) / len(f2s)) if sum(f2s) else None
    pred_c = plural_more_negative_frac >= FRAC and DIFF_LO <= plural_mean_diff <= DIFF_HI
    pred_f = gate_ratio is not None and GATE_LO <= gate_ratio <= GATE_HI
    pred_d = P1_LO <= Q1["remove_self_at_t_default_path"] <= P1_HI and P0_LO <= Q0["remove_self_at_t_default_path"] <= P0_HI
    pred_e = all(SHARE_LO <= L["noun_self_share"] <= SHARE_HI and abs(L["remove_self_at_noun"]) <= INERT for L in (L0, L1))
    predictions = {"pred_a_instrument": pred_a, "pred_b_noun_sign": pred_b, "pred_c_plurality": pred_c, "pred_d_one_sided": pred_d, "pred_e_noun_inert": pred_e, "pred_f_gate_const": pred_f}
    result = {"predictions": predictions, "schema": "unit_tier5_hub_self_read_noun_v162", "candidate_id": "corpus.unit_tier5_hub_self_read_noun_v162",
              "bars": {"instr_tol": INSTR_TOL, "frac": FRAC, "noun_band": [NOUN_LO, NOUN_HI], "diff_band": [DIFF_LO, DIFF_HI], "gate_band": [GATE_LO, GATE_HI],
                       "p1_band": [P1_LO, P1_HI], "p0_band": [P0_LO, P0_HI], "share_band": [SHARE_LO, SHARE_HI], "inert": INERT, "lex_noun_offset": LEX_NOUN_OFFSET},
              "lexical_pooled": {"plural_more_negative_frac": round(plural_more_negative_frac, 3), "singular_minus_plural_mean": round(plural_mean_diff, 4),
                                 "gate_ratio_plural_over_singular": None if gate_ratio is None else round(gate_ratio, 3), "rows": len(pl)},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
