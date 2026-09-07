#!/usr/bin/env python3
# BQGATE: six frozen predictions; doses and direction fixed from v167/v168; one-cue rows by rule.
"""Tier-5: the number direction u dosed AT t -- is the response a bilinear (even + odd in dose) curve, and where does it enter?

v168: at the noun, u (mlp:01's noun-write diff-in-means, singular -> plural) is a number direction (toward-donor 0.93/0.96 of
mlp:01's delta, wrong sign +0.16/+0.14); at t it is CONTEXT-SIGNED and superlinear: -u (singular) +0.045/+0.158 whole-model on
both parities, +u -0.024/-0.026, half -0.018/-0.007, double +0.104/+0.638. bilin18's MLPs are bilinear, Left(x)*Right(x): a
push d*u into the stream at t contributes d*(Left(u)*Right(x) + Left(x)*Right(u)) (odd in d, coefficient set by the current
stream x -> context-signed) plus d^2 * Left(u)*Right(u) (even in d, sign fixed by u alone). A linear number-axis mechanism
would give an odd response only. Test: dose d in {+-0.25, +-0.5, +-1, +-2, +-3} x 2408 of u into mlp:01's write at t on
both lexical parities (16 rows each); decompose the whole-model recovery f(d) into E(d) = (f(d)+f(-d))/2 and
O(d) = (f(d)-f(-d))/2 and fit f(d) = b d + a d^2 over the 10 doses (least squares, no intercept). Where it enters: capture
every downstream MLP's Down output at t (mlp:02..11) under +-1x and base, and report the even/odd norm split per layer
(unregistered; the FIRST bilinear layer after the push, mlp:02, must already carry an even part if the d^2 term is real).
Sets: lexical seven p0+p1 (16 rows each). Noun positions unused (push is at t = semantic position); no offset arithmetic.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 '{1: 16}'.
Smoke (CPU, 16 rows per parity = the full population, disclosed): the GPU run is a device reproduction of these numbers, and the bars
below encode what I believe AFTER seeing them (pred_c's original O(2x)/O(1x) band 0.5-3 failed on p1 at 3.6 and was dropped;
pred_a's small-dose clause and pred_e/pred_f were added). f(d) p0: -3x 0.121, -2x 0.104, -1x 0.045, -0.5x -0.018, -0.25x -0.008,
+0.25x -0.002, +0.5x -0.014, +1x -0.024, +2x 0.004, +3x 0.020; p1: -3x 0.633, -2x 0.639, -1x 0.158, -0.5x -0.007, -0.25x -0.004,
+0.25x -0.012, +0.5x -0.021, +1x -0.026, +2x -0.028, +3x 0.012. E(0.25/0.5/1/2/3x) p0 -0.005/-0.016/0.010/0.054/0.070,
p1 -0.008/-0.014/0.066/0.305/0.323; O(1/2/3x) p0 -0.035/-0.050/-0.051, p1 -0.092/-0.333/-0.310; quadratic fit R^2 0.883/0.882,
a 0.009/0.043, b -0.020/-0.118; second direction E(2x) 0.063/0.287, O(1x) -0.041/-0.089; random |f| <= 0.004 at 0.5x and 2x.
mlp:02's write at t at 1x splits even 6.3k / odd 8.9k (base write 15k) on both parities: the first bilinear layer already carries
the even part. The p0 arm at -1x (0.045) reproduces v168's t_singular (0.045) -- parent cross-check passed after one bug fix
(push() first used each parity's OWN u instead of the dosed p1 direction; caught by the cross-check).

Registered before the run:
  pred_a_even_part_changes_sign  lexical, both parities: E(0.25x) <= 0 and E(0.5x) <= 0 (small pushes of EITHER sign hurt), and
                                 E(2x) >= 0.05 and E(3x) >= 0.05 (a large even part; a linear number axis gives |E| <= 0.02)
  pred_b_even_grows              lexical, both parities: E(2x) / E(1x) within 2-12 (a d^2 term predicts 4; the margin readout is
                                 nonlinear, hence the band) -- evaluated only where E(1x) >= 0.01, else the parity counts as a fail
  pred_c_odd_context_signed      lexical, both parities: O(1x), O(2x) and O(3x) all <= -0.02 (the singular direction -u helps on BOTH
                                 parities at every dose >= 1x although p1's donor is plural)
  pred_d_quadratic_fit           lexical, both parities: the no-intercept quadratic fit over the 10 doses has R^2 >= 0.85 with a > 0
  pred_e_second_direction_same_curve  the p0-estimated direction (own diff-in-means, sign-aligned, |cos| ~0.91 to u) dosed the same way
                                 gives E(2x) >= 0.05 and O(1x) <= -0.02 on both parities (curve shape transfers across estimates)
  pred_f_random_direction_inert  a seeded random unit direction at 0.5x and 2x, both signs: |f| <= 0.03 on both parities (the
                                 small-dose hurt and the large-dose gain are direction-specific, not norm inflation at t)
Reported, unregistered: f(d) at every dose, a, b, O(2x)/O(1x) (the smoke gives 1.4 / 3.6 -- the odd part is superlinear on p1
too, so no ratio band is registered on it), per-layer even/odd norms of the downstream MLP writes at t at 1x, the push norm
relative to mlp:01's own write norm at t.
Smoke: V169_SMOKE=<out.json> -> CPU, V169_SMOKE_ROWS rows per parity (default 4).
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
import run_unit_tier5_near_value_source_v131 as v131

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_hub_number_dir_dose_curve_v169_result.json"
SRC, SRC_L, HUB, HUB_H, LAYER = "mlp:01", 1, "attn:11:head:03", 3, 11
DOSE = 2408.0                      # v167: mean |dm . u| of mlp:01's noun write (lexical p1)
MULTS = (0.25, 0.5, 1.0, 2.0, 3.0)
BARS = {"even_small_max": 0.0, "even_min": 0.05, "even_ratio_band": [2.0, 12.0], "even_ratio_floor": 0.01, "odd_max": -0.02, "r2_min": 0.85, "second_even_min": 0.05, "second_odd_max": -0.02, "random_abs_max": 0.03}
SETS = {"lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 4000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub_number_dir_dose_curve_v169", "behaviours": 1, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V169_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V169_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    m = importlib.import_module(f"circuit_fast_screen_candidate_{names['lexical_number_pp']}")
    a1 = g.rows_of(m, "A1")
    P = {"p0": g.prepare(backend, cut(a1[0::2])), "p1": g.prepare(backend, cut(a1[1::2]))}
    model = backend.model
    R = {}

    def setup(par):
        """per-parity setup in its own scope (v167 lesson: closures in a shared loop bind the last parity)."""
        prep = P[par]
        batch, db = prep.base_batch, prep.donor_batch
        rows = len(batch.row_ids)
        ar = torch.arange(rows, device=backend.device)
        sem_t = torch.tensor(list(batch.semantic_positions), device=backend.device)
        nb, nd = g.cue_positions(batch, db, which="last")
        nb_t, nd_t = torch.tensor(nb, device=backend.device), torch.tensor(nd, device=backend.device)
        cap = {}
        def capture(bt, pos_t, layers):
            hs = [model.transformer.h[l].mlp.Down.register_forward_hook((lambda l: lambda m_, a, o: cap.__setitem__(l, o[ar, pos_t].detach().clone().float()))(l)) for l in layers]
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return out, {l: cap[l] for l in layers}
        _, mb = capture(batch, nb_t, [SRC_L]); _, md = capture(db, nd_t, [SRC_L])
        dm = (md[SRC_L] - mb[SRC_L]).to(backend.device)
        u = dm.mean(0); u = u / u.norm()
        _, wt = capture(batch, sem_t, [SRC_L])
        write_norm_t = round(float(wt[SRC_L].norm(dim=1).mean()), 1)
        downstream = list(range(SRC_L + 1, 12))
        def push(mult, layers=(), u=None):
            """add mult*DOSE*u to mlp:01's write at t (u passed explicitly -- the parity's own u is NOT the dosed direction);
            return whole-model recovery (and downstream Down outputs at t)."""
            delta = (mult * DOSE) * u[None, :].expand(rows, -1)
            def hk(m_, a, o):
                o = o.clone(); o[ar, sem_t] += delta.to(o.dtype); return o
            h_ = model.transformer.h[SRC_L].mlp.Down.register_forward_hook(hk)
            try:
                out, caps = capture(batch, sem_t, list(layers))
            finally:
                h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 4), caps
        return {"prep": prep, "rows": rows, "u": u, "push": push, "downstream": downstream, "write_norm_t": write_norm_t,
                "cue_positions": {"base": nb, "donor": nd}, "exact_hub": round(g.recovery(prep, [-(float(x) - float(f)) for x, f in g.forward_units(backend, batch, units=[HUB], donor_cache=prep.donor_cache, base_cache=prep.base_cache).tolist()]), 3)}

    for par in ("p0", "p1"):
        R[par] = setup(par)
    u_pl = R["p1"]["u"]
    u0 = R["p0"]["u"]; cos0 = float(u0 @ u_pl); u0 = u0 * (1.0 if cos0 >= 0 else -1.0)     # p0's own estimate, sign-aligned to u_pl
    gen = torch.Generator(device="cpu").manual_seed(0)
    u_rand = torch.randn(u_pl.shape[0], generator=gen).to(backend.device); u_rand = u_rand / u_rand.norm()
    for par in ("p0", "p1"):
        L = R[par]; L["u"] = u_pl
        f = {}
        for mm in MULTS:
            for sgn in (+1.0, -1.0):
                f[sgn * mm], _ = L["push"](sgn * mm, u=u_pl)
        E = {mm: round((f[mm] + f[-mm]) / 2, 4) for mm in MULTS}
        O = {mm: round((f[mm] - f[-mm]) / 2, 4) for mm in MULTS}
        ds = torch.tensor(sorted(f), dtype=torch.float64); ys = torch.tensor([f[float(d)] for d in ds.tolist()], dtype=torch.float64)
        X = torch.stack([ds, ds ** 2], 1)
        coef = torch.linalg.lstsq(X, ys[:, None]).solution[:, 0]
        pred = X @ coef
        r2 = round(float(1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()), 4)
        # where it enters: downstream Down outputs at t under +-1x vs base
        _, base_caps = L["push"](0.0, L["downstream"], u=u_pl)
        _, plus = L["push"](+1.0, L["downstream"], u=u_pl); _, minus = L["push"](-1.0, L["downstream"], u=u_pl)
        layer_split = {}
        for l in L["downstream"]:
            dp, dn = plus[l] - base_caps[l], minus[l] - base_caps[l]
            ev, od = (dp + dn) / 2, (dp - dn) / 2
            layer_split[f"mlp:{l:02d}"] = {"even_norm": round(float(ev.norm(dim=1).mean()), 1), "odd_norm": round(float(od.norm(dim=1).mean()), 1),
                                          "base_write_norm": round(float(base_caps[l].norm(dim=1).mean()), 1)}
        f0 = {sgn * mm: L["push"](sgn * mm, u=u0)[0] for mm in (1.0, 2.0) for sgn in (+1.0, -1.0)}
        fr = {sgn * mm: L["push"](sgn * mm, u=u_rand)[0] for mm in (0.5, 2.0) for sgn in (+1.0, -1.0)}
        S = {"rows": L["rows"], "exact_hub": L["exact_hub"], "dose": DOSE,
             "second_direction": {"cos_to_u": round(abs(cos0), 3), "f": {str(k): v for k, v in sorted(f0.items())},
                                  "even_2x": round((f0[2.0] + f0[-2.0]) / 2, 4), "odd_1x": round((f0[1.0] - f0[-1.0]) / 2, 4)},
             "random_direction": {"f": {str(k): v for k, v in sorted(fr.items())}}, "push_over_mlp01_write_norm_at_t": round(DOSE / L["write_norm_t"], 2),
             "f": {str(k): v for k, v in sorted(f.items())}, "even": {str(k): v for k, v in E.items()}, "odd": {str(k): v for k, v in O.items()},
             "fit": {"b_linear": round(float(coef[0]), 4), "a_quadratic": round(float(coef[1]), 4), "r2": r2},
             "downstream_split_at_1x": layer_split, "cue_positions": L["cue_positions"]}
        R[par] = S
        print(par, json.dumps(S), flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_hub_number_dir_dose_curve_v169", "candidate_id": "corpus.unit_tier5_hub_number_dir_dose_curve_v169",
              "bars": BARS, "source_unit": SRC, "direction": "lexical p1 diff-in-means of mlp:01 noun write (singular->plural), dosed at t",
              "sets": {"lex_seven": {"family": SETS["lex_seven"][0], "units": list(SETS["lex_seven"][1])}}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    L = [R["p0"], R["p1"]]
    pred_a = all(S["even"]["0.25"] <= B["even_small_max"] and S["even"]["0.5"] <= B["even_small_max"] and S["even"]["2.0"] >= B["even_min"] and S["even"]["3.0"] >= B["even_min"] for S in L)
    pred_b = all(S["even"]["1.0"] >= B["even_ratio_floor"] and B["even_ratio_band"][0] <= S["even"]["2.0"] / S["even"]["1.0"] <= B["even_ratio_band"][1] for S in L)
    pred_c = all(S["odd"][k] <= B["odd_max"] for S in L for k in ("1.0", "2.0", "3.0"))
    pred_d = all(S["fit"]["r2"] >= B["r2_min"] and S["fit"]["a_quadratic"] > 0 for S in L)
    pred_e = all(S["second_direction"]["even_2x"] >= B["second_even_min"] and S["second_direction"]["odd_1x"] <= B["second_odd_max"] for S in L)
    pred_f = all(abs(v) <= B["random_abs_max"] for S in L for v in S["random_direction"]["f"].values())
    return {"pred_a_even_part_changes_sign": pred_a, "pred_b_even_grows": pred_b, "pred_c_odd_context_signed": pred_c, "pred_d_quadratic_fit": pred_d,
            "pred_e_second_direction_same_curve": pred_e, "pred_f_random_direction_inert": pred_f}


if __name__ == "__main__":
    main()
