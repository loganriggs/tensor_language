#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; units mlp:08/15/16/17 and doses 1W/2W fixed; removal = clamp the unit's write at t to base.
"""v181: is mlp:17 a margin DAMPER? v181 found that clamping mlp:17's write at the answer position t to its base value INCREASES the
effect of 07:08's cue write on 7 of 14 cells (cost -0.12..-0.52; mlp:15/16 negative on several more) — a late unit writes AGAINST the
answer the write is pushing toward. Two readings: (i) mlp:17 reads the answer margin at t and damps it whatever its source (a calibration
unit), or (ii) the opposition is specific to 07:08's write. This rung keeps v181's instrument (same 14 cells, same 16 rows, cue writes base,
07:08's write W added at layer 7) and adds (1) a DOSE arm, add = 2W, and (2) a FULL-INTERCHANGE arm: the donor batch forwarded with one or
more late MLP writes at t clamped to the base batch's value (cost = 1 - recovery, recovery is 1.000 unclamped by construction).

Magnitudes printed before writing (v181 receipt): mlp:17 cost under 1W = -0.33 (both/either p1), -0.35 (both/neither p1), -0.12 / -0.23
(either/neither p0/p1), -0.52 (degree p0, whole 0.021), -0.39 (finiteness p1), -0.12 (polarity p0); |cost| <= 0.07 on the other 7. mlp:08
cost 0.08-0.45 (top-3 on 11/14).

Smoke (CPU, coordination p0 + coordination p1, 3 rows, 45 s): whole 0.063 / 0.102 (v179 0.058 / 0.100); donor unclamped 1.000 / 1.000
(after fixing the donor batch's axis sign — the first smoke read 0.12 / -0.16 with the base batch's negated margin); full-interchange
mlp:17 cost -0.015 / -0.057, trio -0.056 / -0.078, mlp:08 +0.40 / +0.24; 1W mlp:17 cost -0.03 / +0.08 (v180 -0.02 / +0.03). Coordination is
not a dose cell; no bar was contradicted (the count bars cannot pass on 2 cells).

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar has a lower AND an upper bound where a fraction is compared):
  pred_a_instrument     whole (add = W) equals v179's whole within 0.02 on 14/14 AND the donor batch unclamped recovers 0.99-1.01 on 14/14
  pred_b_damper_full    under the full interchange, clamping mlp:17 at t to base gives cost <= -0.05 on >= 9 of 14 (mlp:17 opposes the
                        answer margin whatever its source; prior 0.5)
  pred_c_dose           on the 7 cells listed above (mlp:17 cost <= -0.1 under 1W in v181), cost(2W) / cost(1W) lies in 1.0-2.5 on >= 5 of 7
                        (the opposition grows super-linearly with dose, as a Bilinear MLP's quadratic term would; prior 0.5)
  pred_d_late_trio_full under the full interchange, clamping mlp:15 + mlp:16 + mlp:17 jointly at t gives cost <= -0.1 on >= 8 of 14 (prior 0.4)
  pred_e_mlp08_full     under the full interchange, clamping mlp:08 at t to base gives cost >= +0.1 on >= 8 of 14 (the write's first decoder
                        is a decoder of the whole cue interchange too; the full-interchange instrument can show a POSITIVE cost; prior 0.6)
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_mlp17_damper_v181_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V180 = ROOT / "circuits/followups/unit_tier5_hub0708_write_decoders_v180_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LATE = ("mlp:15", "mlp:16", "mlp:17")
DOSE_CELLS = ("correlative_both_either:p1", "correlative_both_neither:p1", "correlative_either_neither:p0", "correlative_either_neither:p1",
              "degree_frame:p0", "finiteness_selection:p1", "polarity_state:p0")
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "donor_band": [0.99, 1.01], "damper_max": -0.05, "damper_min_cells": 9, "dose_band": [1.0, 2.5], "dose_min_cells": 5,
        "trio_max": -0.1, "trio_min_cells": 8, "mlp08_min": 0.1, "mlp08_min_cells": 8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_mlp17_damper_v181", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V181_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V181_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V181_SMOKE_CELLS", "2"))]
    cells = {}

    def setup(fam, par):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1)
        fb, fd = g.cue_positions(prep.base_batch, prep.donor_batch, which="first")
        lb, ld = g.cue_positions(prep.base_batch, prep.donor_batch, which="last")
        assert all(fb[i] == lb[i] and fd[i] == ld[i] for i in range(len(a1))), "multi-token cue"
        batch, db = prep.base_batch, prep.donor_batch
        rows = len(batch.row_ids)
        ar = torch.arange(rows, device=backend.device)
        cb_t, cd_t = torch.tensor(lb, device=backend.device), torch.tensor(ld, device=backend.device)
        def capture(bt, pos_t):
            cap = {}
            hs = []
            for l in range(N_LAYERS):
                hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(f"attn:{l:02d}", a[0][ar, pos_t].detach().clone())))
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: cap.__setitem__(f"mlp:{l:02d}", o[ar, pos_t].detach().clone())))
            hs.append(model.transformer.h[1].attn.register_forward_pre_hook(lambda m_, a: cap.__setitem__("v1", a[1][ar, pos_t].detach().clone())))
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cap
        cap, capb = capture(db, cd_t), capture(batch, cb_t)
        t_t0 = torch.tensor(list(batch.semantic_positions), device=backend.device)
        capt = capture(batch, t_t0)
        delta = (cap["v1"][:, CH].float() - capb["v1"][:, CH].float()).mean(0)
        t_t = torch.tensor(list(batch.semantic_positions), device=backend.device)
        sl = slice(CH * HD, (CH + 1) * HD)
        def run(vec=None, add=None, grab=None, remove=()):
            """cue-position writes base; base x0; if vec: slice CH of the cue's v1 = base + vec at layers 1-17; if add: (B,T,128) added to
            head CH's c_proj input at layer HUB; if grab: dict receiving that c_proj input (full tensor)."""
            hs = []
            for u in UNITS:
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u, l=l):
                        v = a[0].clone(); v[ar, cb_t] = capb[u].to(v.dtype)
                        if l == HUB and grab is not None: grab["x"] = v.detach().clone()
                        if l == HUB and add is not None: v[:, :, sl] = v[:, :, sl] + add.to(v.dtype)
                        if u in remove: v[ar, t_t] = capt[u].to(v.dtype)
                        return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype)
                        if u in remove: o[ar, t_t] = capt[u].to(o.dtype)
                        return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            if vec is not None:
                tgt = (capb["v1"][:, CH].float() + vec[None, :]).to(capb["v1"].dtype)
                for l in range(1, N_LAYERS):
                    def vh(m_, a):
                        t = a[1].clone(); t[ar, cb_t, CH] = tgt.to(t.dtype); return (a[0], t)
                    hs.append(model.transformer.h[l].attn.register_forward_pre_hook(vh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        td_t = torch.tensor(list(db.semantic_positions), device=backend.device)
        def run_donor(remove=()):
            """the donor batch itself (the full interchange), with the writes in `remove` clamped at t to the BASE batch's value."""
            hs = []
            for u in remove:
                l = int(u.split(":")[1])
                def mh(m_, a, o, u=u):
                    o = o.clone(); o[ar, td_t] = capt[u].to(o.dtype); return o
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, db, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [float(x) - float(f) for x, f in out.tolist()]), 3)   # donor batch: its own answer axis
        gs, gb = {}, {}
        full_swap = run(vec=delta, grab=gs); run(grab=gb)
        W = (gs["x"][:, :, sl].float() - gb["x"][:, :, sl].float())          # (B, T, 128) 07:08's write change under the slice-8 swap
        wt = W[ar, t_t]                                                        # (rows, 128) at t
        direction = wt.mean(0); direction = direction / direction.norm()
        sv = torch.linalg.svdvals(wt)
        return {"rows": rows, "delta": delta, "run": run, "run_donor": run_donor, "W": W, "direction": direction, "full_swap": full_swap,
                "rank1_frac": round(float(sv[0] ** 2 / (sv ** 2).sum()), 3), "wt_norm": round(float(wt.norm(dim=1).mean()), 3),
                "t_share_of_norm": round(float(wt.norm() ** 2 / W.norm() ** 2), 3)}


    for fam, par in cells_run:
        cells[(fam, par)] = setup(fam, par)

    R = {}
    v180 = json.loads(V180.read_text())["measures"] if V180.exists() else {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        whole = C["run"](add=C["W"]); zero = C["run"]()
        whole2 = C["run"](add=2 * C["W"])
        fr = lambda x, w: round((x - zero) / (w - zero), 3) if abs(w - zero) > 1e-6 else None
        cost1 = {u: round(1 - fr(C["run"](add=C["W"], remove=(u,)), whole), 3) for u in LATE}
        cost2 = {u: round(1 - fr(C["run"](add=2 * C["W"], remove=(u,)), whole2), 3) for u in LATE}
        donor = C["run_donor"]()
        full = {u: round(1 - C["run_donor"](remove=(u,)), 3) for u in LATE + ("mlp:08",)}
        full["trio"] = round(1 - C["run_donor"](remove=LATE), 3)
        c1, c2 = cost1["mlp:17"], cost2["mlp:17"]
        S = {"rows": C["rows"], "whole_write": whole, "zero": zero, "whole_2W": whole2, "v179_whole": v179.get(f"{fam}:{par}", {}).get("whole_write"),
             "v180_cost17": v180.get(f"{fam}:{par}", {}).get("cost", {}).get("mlp:17"), "cost_1W": cost1, "cost_2W": cost2,
             "dose_ratio_17": round(c2 / c1, 3) if c1 and abs(c1) > 1e-6 else None, "donor_recovery": donor, "full_cost": full}
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_mlp17_damper_v181", "candidate_id": "corpus.unit_tier5_mlp17_damper_v181",
              "bars": BARS, "families": list(FAMILIES), "channel": CH,
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def pearson(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5; sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return round(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb), 3) if sa > 0 and sb > 0 else None



def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    pred_a = all(S["v179_whole"] is not None and abs(S["whole_write"] - S["v179_whole"]) <= B["reproduce_tol"] for S in cells) \
        and all(inb(S["donor_recovery"], B["donor_band"]) for S in cells)
    pred_b = sum(S["full_cost"]["mlp:17"] <= B["damper_max"] for S in cells) >= B["damper_min_cells"]
    pred_c = sum(inb(R[k]["dose_ratio_17"], B["dose_band"]) for k in DOSE_CELLS if k in R) >= B["dose_min_cells"]
    pred_d = sum(S["full_cost"]["trio"] <= B["trio_max"] for S in cells) >= B["trio_min_cells"]
    pred_e = sum(S["full_cost"]["mlp:08"] >= B["mlp08_min"] for S in cells) >= B["mlp08_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_damper_full": pred_b, "pred_c_dose": pred_c, "pred_d_late_trio_full": pred_d, "pred_e_mlp08_full": pred_e}


if __name__ == "__main__":
    main()
