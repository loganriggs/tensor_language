#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: mlp:01's write at the head noun -- is the number it computes a rank-1 direction, and does that direction transfer?

v166: the noun's number value read by 11:03 is computed at the noun position, first by mlp:01 (0.135 / 0.178 of the 0.256 /
0.421 noun value swap; whole-model 0.56 / 0.65). Here dm = mlp:01's (donor - base) write at the noun (|dm| ~ 3,600 per row),
u = normalised mean of dm within a parity (diff-in-means), and we ADD to mlp:01's write at the base noun position: dm (= the
v166 clamp), uu'dm (rank-1), dm - uu'dm (complement), the leave-one-out rank-1, the OTHER parity's u (sign-aligned), and a
random unit direction with the same projection. Read-outs: 11:03's value-read effect at t (image(P_base[t,noun] dV) added at
t) and the whole-model recovery. Transfer: the lexical p1 (singular->plural) direction added at mlp:01 at t on QUANTIFIER
rows (noun = t, same token both sides) with dose = lexical mean |dm . u|, signed toward the donor's verb number (+ for
each->all, - for all->each), against the wrong sign.
Disclosure: the lexical arms were measured on the full 16-row population on CPU before registration (the model is
deterministic, so those bars are confirmations, not predictions): rank-1 0.926 / 0.955 of the full delta, complement
0.022 / -0.006, LOO 0.79 / 0.84, other parity 0.79 / 0.90, |cos(u_p0, u_p1)| 0.912, random 0.000, top singular share of dm
only 0.43 / 0.47 (rowwise cos to u 0.67 / 0.69). The quantifier transfer was seen on 4 rows only and did NOT look naive-signed
(p0 signed -0.047 / wrong sign +0.197; p1 +0.167 / +0.106) -- pred_e is the naive-sign hypothesis and is expected to be
the informative bar.
Sets: lexical seven p0+p1 (16 rows each); quantifier seven p0+p1 for the transfer.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'; quantifier A1 p0/p1 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}'.

Registered before the run:
  pred_a_rank1        lexical, both parities: rank-1 diff-in-means carries 0.8-1.1 of mlp:01's full-delta value-swap effect and
                      the complement |.| <= 0.1 of it
  pred_b_loo          lexical, both parities: leave-one-out rank-1 carries 0.6-1.1 of the full delta
  pred_c_transfer     lexical, both parities: the other parity's direction carries 0.6-1.1 of the full delta and |cos| >= 0.8
  pred_d_random       lexical, both parities: a random direction with the same projection gives |value swap| <= 0.02
  pred_e_quant_sign   quantifier, both parities: the signed lexical direction at t recovers >= 0.10 whole-model and the wrong sign <= 0.0
Reported, unregistered: top singular share of dm, rowwise cos, 11:03-read vs whole-model on every arm, dose.
Smoke: V167_SMOKE=<out.json> -> CPU, V167_SMOKE_ROWS rows per parity (default 4).
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
OUT = ROOT / "circuits/followups/unit_tier5_hub_noun_mlp01_rank1_v167_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
SRC_LAYERS = 11
SRC = "mlp:01"
SRC_L = 1
BARS = {"rank1_band": [0.8, 1.1], "complement_max": 0.1, "loo_band": [0.6, 1.1], "transfer_band": [0.6, 1.1], "cos_min": 0.8, "random_max": 0.02, "quant_signed_min": 0.10, "quant_wrong_max": 0.0, "_v166_bars": {"instr_whole": [0.90, 1.05], "instr_value_tol": 0.03, "direct_value_max": 0.03, "direct_whole_max": 0.20, "computed_band": [0.8, 1.2], "mlp01_band": [0.3, 0.75], "attn_value_max": 0.03, "attn07_band": [0.3, 0.7], "attn07_whole_band": [0.20, 0.60], "value_band": [0.35, 0.85], "weight_max": 0.05, "control_max": 0.05, "dv_band": [0.8, 1.5]}}
LEX_NOUN_OFFSET = 3  # lexical noun's last token is t - 3 on each side (' near the <object>' follows)
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_noun_mlp01_rank1_v167", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V167_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V167_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
        def add_at_noun(delta):
            """add delta (rows,1152) to mlp:01's write at the base noun position; return 11:03 value-swap effect at t and whole-model recovery."""
            def hk(m, a, o):
                o = o.clone(); o[ar, nb_t] += delta.to(o.dtype); return o
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
                    "pb": pb, "vb": vb, "vd": vd}
    lex = {par: setup(par) for par in ("p0", "p1")}

    def setup_quant(par):
        """quantifier rows: noun = t on both sides (same token); add a lexical number direction to mlp:01's write at t."""
        n = SETS["quant_seven"][0]
        prep = P[n][par]
        batch = prep.base_batch
        rows = len(batch.row_ids)
        nb_t = torch.tensor(list(batch.semantic_positions), device=backend.device)
        ar = torch.arange(rows, device=backend.device)
        sem_t = nb_t
        with torch.no_grad():
            PB, VB = v131.capture_with_clamp(backend, batch, [], [], LAYER)
        pb = PB[ar, HUB_H, sem_t, nb_t].to(backend.device)
        vb = VB[ar, nb_t, HUB_H, :].to(backend.device)
        def rec(add=None, units=()):
            out = g.forward_units(backend, batch, units=units, donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        def add_at_noun(delta):
            def hk(m, a, o):
                o = o.clone(); o[ar, nb_t] += delta.to(o.dtype); return o
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
        return {"rows": rows, "add": add_at_noun, "exact": rec(units=[HUB]), "directions": [DIRS[n][rid] for rid in batch.row_ids]}
    # lexical p1 = singular_to_plural: +u_p1 pushes the noun toward PLURAL. Quantifier p0 (base 'Each' -> donor 'All') wants the
    # plural answer, so it gets +u_p1 * dose; quantifier p1 (base 'All' -> donor 'Each') gets -u_p1 * dose. Dose = lexical p1 mean |dm . u|.
    u_pl = lex["p1"]["u"]; dose = float((lex["p1"]["dm"] @ u_pl).abs().mean())
    for qpar, sign in (("p0", +1.0), ("p1", -1.0)):
        Qs = setup_quant(qpar)
        delta = (sign * dose) * u_pl[None, :].expand(Qs["rows"], -1)
        R[f"quant_seven:{qpar}"] = {"rows": Qs["rows"], "exact_hub": Qs["exact"], "direction_ids": sorted(set(Qs["directions"])), "dose": round(dose, 1),
                                    "lexical_plural_dir_signed": Qs["add"](delta), "wrong_sign": Qs["add"](-delta)}
        print(qpar, R[f"quant_seven:{qpar}"], flush=True)
    for par in ("p0", "p1"):
        L = lex[par]; other = lex["p1" if par == "p0" else "p0"]
        dm, u, add = L["dm"], L["u"], L["add"]
        proj = (dm @ u)[:, None] * u[None, :]
        rnd = torch.randn(dm.shape[1], generator=gen).to(backend.device); rnd = rnd / rnd.norm()
        rproj = (dm @ rnd)[:, None] * rnd[None, :]
        uo = other["u"]; uo = -uo if float(uo @ u) < 0 else uo             # geometric sign alignment across parities
        oproj = (dm @ uo)[:, None] * uo[None, :]
        # LOO diff-in-means: each row's direction from the other rows
        loo = torch.stack([(dm[[j for j in range(L["rows"]) if j != i]].mean(0)) for i in range(L["rows"])])
        loo = loo / loo.norm(dim=1, keepdim=True)
        lproj = (dm * loo).sum(1)[:, None] * loo
        S = {"rows": L["rows"], "exact_hub": L["exact"], "full_value_swap": L["full_value_swap"],
             "dm_norm_mean": round(float(dm.norm(dim=1).mean()), 1), "mb_norm_mean": None,
             "top_singular_share": round(float(torch.linalg.svdvals(dm)[0] ** 2 / (torch.linalg.svdvals(dm) ** 2).sum()), 3),
             "dim_cos_rowwise_mean": round(float(((dm @ u) / dm.norm(dim=1)).mean()), 3),
             "full_delta": add(dm), "rank1_dim": add(proj), "complement": add(dm - proj), "loo_rank1": add(lproj),
             "other_parity_dim": add(oproj), "random_dir": add(rproj),
             "cos_dim_p0_p1": round(abs(float(u @ other["u"])), 3)}
        S["rank1_share"] = round(S["rank1_dim"]["value_swap_at_t"] / S["full_delta"]["value_swap_at_t"], 3) if abs(S["full_delta"]["value_swap_at_t"]) > 1e-6 else None
        S["complement_share"] = round(S["complement"]["value_swap_at_t"] / S["full_delta"]["value_swap_at_t"], 3) if abs(S["full_delta"]["value_swap_at_t"]) > 1e-6 else None
        S["S_plus_C"] = round((S["rank1_share"] or 0) + (S["complement_share"] or 0), 3)
        R[f"lex_seven:{par}"] = S
        print(par, {k: v for k, v in S.items()}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_hub_noun_mlp01_rank1_v167", "candidate_id": "corpus.unit_tier5_hub_noun_mlp01_rank1_v167",
              "bars": BARS, "source_unit": SRC,
              "sets": {"lex_seven": {"family": SETS["lex_seven"][0], "units": list(SETS["lex_seven"][1])}}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    L = [R["lex_seven:p0"], R["lex_seven:p1"]]; Q = [R["quant_seven:p0"], R["quant_seven:p1"]]
    def fr(S, k):
        f = S["full_delta"]["value_swap_at_t"]
        return S[k]["value_swap_at_t"] / f if abs(f) > 1e-6 else float("nan")
    pred_a = all(B["rank1_band"][0] <= fr(S, "rank1_dim") <= B["rank1_band"][1] and abs(fr(S, "complement")) <= B["complement_max"] for S in L)
    pred_b = all(B["loo_band"][0] <= fr(S, "loo_rank1") <= B["loo_band"][1] for S in L)
    pred_c = all(B["transfer_band"][0] <= fr(S, "other_parity_dim") <= B["transfer_band"][1] and S["cos_dim_p0_p1"] >= B["cos_min"] for S in L)
    pred_d = all(abs(S["random_dir"]["value_swap_at_t"]) <= B["random_max"] for S in L)
    pred_e = all(S["lexical_plural_dir_signed"]["whole_model"] >= B["quant_signed_min"] and S["wrong_sign"]["whole_model"] <= B["quant_wrong_max"] for S in Q)
    return {"pred_a_rank1": pred_a, "pred_b_loo": pred_b, "pred_c_transfer": pred_c, "pred_d_random": pred_d, "pred_e_quant_sign": pred_e}


if __name__ == "__main__":
    main()
