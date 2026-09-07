#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the number direction mlp:01 writes at the noun -- what does it do when dosed AT THE ANSWER POSITION t?

v167: mlp:01's noun write acts through one direction u (rank-1 0.93/0.96 of its effect). Dosed into mlp:01's write at t on
QUANTIFIER rows it was not verb-number-signed: the SINGULAR direction helped BOTH interchange directions (+0.24 each->all,
+0.25 all->each), the plural direction -0.05 / +0.13, and 11:03's own read of the push <= 0.05. bilin18's MLPs are bilinear
(Left(x) * Right(x)), so a fixed added direction contributes cross terms Left(u)*Right(x) + Left(x)*Right(u) whose sign
follows the current stream x -- a context-signed push. Test on LEXICAL rows (16 per parity): u = lexical p1 (singular ->
plural) diff-in-means of mlp:01's noun write, dose = mean |dm . u| (v167: 2408). Arms: +/- u at the NOUN (number-like
control: sign must follow the donor's number), +/- u at t, and a 0.5x / 2x dose of the singular direction at t; readouts
whole-model recovery and 11:03's value-read effect at t (image(P_base[t,noun] dV) added at t).
Smoke (4 rows per parity, disclosed): at the noun, toward-donor 1.07 / 0.91 of the full delta, wrong sign +0.10 / +0.18 (small
POSITIVE, not negative -- my prior of a mirrored push was wrong before registration); at t, singular 0.084 / 0.195 whole-model,
plural -0.022 / -0.015, half-dose -0.010 / -0.002, double dose 0.092 / 0.728 (superlinear on p1). 11:03's value read of the
noun cannot change under a push at t (0.000 by construction) -- the at-t effect is entirely downstream of the hub's noun read.
Sets: lexical seven p0+p1 (16 rows each).
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'.

Registered before the run:
  pred_a_noun_is_number   lexical, both parities: at the noun, u signed toward the donor's number recovers 0.5-1.2 of mlp:01's full
                          delta and the wrong sign |.| <= 0.3 of it
  pred_b_t_context_signed lexical, both parities: at t, the SINGULAR direction recovers >= 0.05 whole-model on BOTH parities (p1's donor
                          is plural, so a number-like push would give <= 0 there) and the plural direction <= 0.0 on both
  pred_c_t_superlinear    lexical, both parities: at t the singular response is superlinear in dose -- double / single within 2-6 and
                          half / single <= 0.5 (the bilinear cross-term hypothesis; the 4-row smoke gives 1.1 on p0, 3.7 on p1)
Reported, unregistered: every arm's whole-model and 11:03-read numbers.
Smoke: V168_SMOKE=<out.json> -> CPU, V168_SMOKE_ROWS rows per parity (default 4).
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_number_dir_at_t_v168_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
SRC_LAYERS = 11
SRC = "mlp:01"
SRC_L = 1
BARS = {"noun_band": [0.5, 1.2], "noun_wrong_abs_max": 0.3, "t_singular_min": 0.05, "t_plural_max": 0.0, "double_band": [2.0, 6.0], "half_max": 0.5, "_v166_bars": {"instr_whole": [0.90, 1.05], "instr_value_tol": 0.03, "direct_value_max": 0.03, "direct_whole_max": 0.20, "computed_band": [0.8, 1.2], "mlp01_band": [0.3, 0.75], "attn_value_max": 0.03, "attn07_band": [0.3, 0.7], "attn07_whole_band": [0.20, 0.60], "value_band": [0.35, 0.85], "weight_max": 0.05, "control_max": 0.05, "dv_band": [0.8, 1.5]}}
LEX_NOUN_OFFSET = 3  # lexical noun's last token is t - 3 on each side (' near the <object>' follows)
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_number_dir_at_t_v168", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V168_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V168_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    model = backend.model
    gen = torch.Generator(device="cpu").manual_seed(0)
    def setup(par):
        """per-parity setup in its own scope -- closures below bind THIS parity's batch/positions (a shared-scope loop bound them all to the last parity)."""
        n = SETS["lex_seven"][0]
        prep = P[n][par]
        batch, db = prep.base_batch, prep.donor_batch
        sem, semd = list(batch.semantic_positions), list(db.semantic_positions)
        rows = len(batch.row_ids)
        near = next(r[2] for r in batch.token_rows if len(r) == 5)
        nb = [list(r).index(near) - 1 for r in batch.token_rows]
        nd = [list(r).index(near) - 1 for r in db.token_rows]
        ar = torch.arange(rows, device=backend.device)
        nb_t, nd_t = torch.tensor(nb, device=backend.device), torch.tensor(nd, device=backend.device)
        cap = {}
        def capture_src(bt, pos_t):
            h_ = model.transformer.h[SRC_L].mlp.Down.register_forward_hook(lambda m, a, o: cap.__setitem__("m", o[ar, pos_t].detach().clone().float()))
            try:
                with torch.no_grad():
                    Pp, Vp = v131.capture_with_clamp(backend, bt, [], [], LAYER)
            finally:
                h_.remove()
            return Pp, Vp, cap["m"]
        PB, VB, mb = capture_src(batch, nb_t)
        PD, VD, md = capture_src(db, nd_t)
        dm = (md - mb).to(backend.device)                                   # (rows, 1152) mlp:01 write delta at the noun
        pb = PB[ar, HUB_H, torch.tensor(sem, device=backend.device), nb_t].to(backend.device)
        vb = VB[ar, nb_t, HUB_H, :].to(backend.device)
        vd = VD[ar, nd_t, HUB_H, :].to(backend.device)
        def rec(add=None, units=()):
            out = g.forward_units(backend, batch, units=units, donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        sem_t = torch.tensor(sem, device=backend.device)
        def add_at_noun(delta, pos_t=nb_t):
            """add delta (rows,1152) to mlp:01's write at pos_t (default the base noun); return 11:03 value-swap effect at t and whole-model recovery."""
            def hk(m, a, o):
                o = o.clone(); o[ar, pos_t] += delta.to(o.dtype); return o
            h_ = model.transformer.h[SRC_L].mlp.Down.register_forward_hook(hk)
            try:
                with torch.no_grad():
                    Pp, Vp = v131.capture_with_clamp(backend, batch, [], [], LAYER)
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                h_.remove()
            vp = Vp[ar, nb_t, HUB_H, :].to(backend.device)
            return {"value_swap_at_t": rec(img((pb[:, None] * (vp - vb)).float())),
                    "whole_model": round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)}
        U, Sv, _ = torch.linalg.svd(dm - dm.mean(0, keepdim=True) * 0, full_matrices=False)
        top_sv_share = round(float(Sv[0] ** 2 / (Sv ** 2).sum()), 3)
        u = dm.mean(0); u = u / u.norm()
        exact = rec(units=[HUB])
        full = rec(img((pb[:, None] * (vd - vb)).float()))
        return {"prep": prep, "u": u, "dm": dm, "add": add_at_noun, "rows": rows, "exact": exact, "full_value_swap": full,
                    "pb": pb, "vb": vb, "vd": vd, "sem_t": sem_t, "nb_t": nb_t}
    lex = {par: setup(par) for par in ("p0", "p1")}

    u_pl = lex["p1"]["u"]; dose = float((lex["p1"]["dm"] @ u_pl).abs().mean())
    for par in ("p0", "p1"):
        L = lex[par]; add = L["add"]; rows = L["rows"]
        donor_sign = -1.0 if par == "p0" else +1.0            # p0 = plural_to_singular (donor singular), p1 = singular_to_plural
        e = lambda sgn, k, pos: add((sgn * k * dose) * u_pl[None, :].expand(rows, -1), pos)
        S = {"rows": rows, "exact_hub": L["exact"], "full_value_swap": L["full_value_swap"], "dose": round(dose, 1), "donor_sign_of_u": donor_sign,
             "noun_toward_donor": e(donor_sign, 1.0, L["nb_t"]), "noun_wrong_sign": e(-donor_sign, 1.0, L["nb_t"]),
             "t_singular": e(-1.0, 1.0, L["sem_t"]), "t_plural": e(+1.0, 1.0, L["sem_t"]),
             "t_singular_half": e(-1.0, 0.5, L["sem_t"]), "t_singular_double": e(-1.0, 2.0, L["sem_t"]),
             "full_delta": add(L["dm"])}
        f = S["full_delta"]["value_swap_at_t"]
        S["noun_toward_donor_share"] = round(S["noun_toward_donor"]["value_swap_at_t"] / f, 3) if abs(f) > 1e-6 else None
        S["noun_wrong_sign_share"] = round(S["noun_wrong_sign"]["value_swap_at_t"] / f, 3) if abs(f) > 1e-6 else None
        R[f"lex_seven:{par}"] = S
        print(par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_hub_number_dir_at_t_v168", "candidate_id": "corpus.unit_tier5_hub_number_dir_at_t_v168",
              "bars": BARS, "source_unit": SRC, "direction": "lexical p1 diff-in-means of mlp:01 noun write (singular->plural)",
              "sets": {"lex_seven": {"family": SETS["lex_seven"][0], "units": list(SETS["lex_seven"][1])}}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    L = [R["lex_seven:p0"], R["lex_seven:p1"]]
    pred_a = all(S["noun_toward_donor_share"] is not None and B["noun_band"][0] <= S["noun_toward_donor_share"] <= B["noun_band"][1] and abs(S["noun_wrong_sign_share"]) <= B["noun_wrong_abs_max"] for S in L)
    pred_b = all(S["t_singular"]["whole_model"] >= B["t_singular_min"] and S["t_plural"]["whole_model"] <= B["t_plural_max"] for S in L)
    def ratio(S, k):
        d = S["t_singular"]["whole_model"]
        return S[k]["whole_model"] / d if abs(d) > 1e-6 else float("nan")
    pred_c = all(B["double_band"][0] <= ratio(S, "t_singular_double") <= B["double_band"][1] and ratio(S, "t_singular_half") <= B["half_max"] for S in L)
    return {"pred_a_noun_is_number": pred_a, "pred_b_t_context_signed": pred_b, "pred_c_t_superlinear": pred_c}


if __name__ == "__main__":
    main()
