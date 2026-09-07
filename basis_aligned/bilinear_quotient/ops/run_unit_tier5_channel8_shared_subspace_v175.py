#!/usr/bin/env python3
# BQGATE: five frozen predictions; families fixed (v170-v174's seven function-word cells); subspace ranks fixed (1, 2, 6) before the run; seeded random controls.
"""Tier-5: do the seven function-word cues share a causal subspace of the v1 channel 8?

v174: the cue's block-0 value residual reaches the hub heads through head-index channels; slice 8 (07:08 / 14:08 / 16:08) is the largest
on 13/14 cells (0.29-0.60 of the value side). Each family's slice-8 delta is a FIXED 128-d vector (v1 = c_v(rms_norm(12.19 x0)) depends on
the token only): and/or, both/either, both/neither, either/neither, than/as, to/that, any/some. The user's standing question is whether
circuits share subspaces. This rung projects each family's slice-8 delta onto (i) each other family's delta (rank 1), (ii) the span of the
other six deltas of the same parity (rank 6, leave-one-out), (iii) for the correlatives, the span of the other two correlative deltas
(rank 2: neither-either = (neither-both) - (either-both) EXACTLY, so this is a by-construction positive control of the projection path),
(iv) a seeded random 6-dim subspace and a seeded random direction (controls that can fail), and swaps only the projected component into
the cue's slice 8 for layers 1-17 (every layer clamped, v174's instrument). Whole-model recovery as a fraction of the full slice-8 swap.
Smoke (CPU, 3 rows per parity, 131 s, run BEFORE enqueue with the bars above already written): pred_a false only through the 3-row vs
16-row receipt comparison (|full - v174| up to 0.031; row_dev 0.0 on 14/14); positive control 1.000 on 6/6; leave-one-out rank-6 span on the
non-correlative cells 0.09/0.09 (coordination), 0.00/0.03 (degree), 0.06/0.05 (finiteness) but 0.47/0.46 on polarity (6/8 in band, exactly the
bar); random rank-6 -0.05..0.19 (14/14); cross-family pairs 69/72 in band, the three out being polarity <-> both/neither (0.32-0.38) and
polarity -> either/neither (0.29); polarity -> either/neither 0.287/0.269 vs polarity -> both/either 0.011/0.019 (pred_e holds at both parities,
and the both/neither delta carries even more of polarity: 0.376/0.382; cos(any-some, neither-both) = -0.485, cos to either-both = -0.103).
The bars are unchanged from the pre-smoke registration.

Registered before the run:
  pred_a_instrument         the full delta reproduces v174's slice-8 swap within 0.02 on every cell, and the per-row delta deviates from its
                            mean by <= 0.01 of its norm on every cell (the delta is a token function)
  pred_b_positive_control   on the six correlative cells the rank-2 span of the other two correlative deltas carries 0.95-1.05 x the full swap
  pred_c_no_shared_subspace on the eight non-correlative cells the leave-one-out rank-6 span carries -0.2..0.4 x the full swap on >= 6 of 8, and the
                            random rank-6 subspace carries -0.2..0.2 x on >= 12 of 14 cells
  pred_d_pairs_disjoint     rank-1 projection onto another family's delta carries -0.3..0.3 x the full swap on >= 80% of the cross-family pairs
                            (within-correlative pairs excluded), and the random direction -0.15..0.15 on >= 12 of 14
  pred_e_polarity_npi       polarity (any/some) projected onto either/neither's delta exceeds polarity projected onto both/either's delta by >= 0.05
                            (x full) on both parities (the negative-polarity cue shares an axis with 'neither', not with 'both')
Reported, unregistered: the 7 x 7 projection matrix per parity, cosines between deltas, norm fractions of every projection.
Smoke: V175_SMOKE=<out.json> -> CPU, V175_SMOKE_ROWS rows per parity (default 3).
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
OUT = ROOT / "circuits/followups/unit_tier5_channel8_shared_subspace_v175_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
CORR = ("correlative_both_either", "correlative_both_neither", "correlative_either_neither")
N_LAYERS, N_HEADS, CH = 18, 9, 8
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "row_uniform_max": 0.01, "positive_band": [0.95, 1.05], "loo_band": [-0.2, 0.4], "loo_min_cells": 6,
        "random6_band": [-0.2, 0.2], "random_min_cells": 12, "pair_band": [-0.3, 0.3], "pair_min_frac": 0.8, "random1_band": [-0.15, 0.15],
        "npi_margin": 0.05, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_channel8_shared_subspace_v175", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V175_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V175_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
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
        d8 = (cap["v1"][:, CH].float() - capb["v1"][:, CH].float())            # (rows, 128)
        delta = d8.mean(0)
        row_dev = float((d8 - delta).norm(dim=1).max() / delta.norm())
        def run(vec):
            """cue-position writes base; base x0; slice CH of the cue's v1 = base + vec at every layer 1-17."""
            hs = []
            for u in UNITS:
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u):
                        v = a[0].clone(); v[ar, cb_t] = capb[u].to(v.dtype); return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype); return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
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
        return {"rows": rows, "delta": delta, "row_dev": row_dev, "run": run}

    for fam in FAMILIES:
        for par in ("p0", "p1"):
            cells[(fam, par)] = setup(fam, par)

    def proj(vec, basis):
        """projection of vec onto the span of the columns of basis (128, k) via QR."""
        Q, _ = torch.linalg.qr(basis)
        return Q @ (Q.T @ vec)

    gen = torch.Generator(device="cpu").manual_seed(BARS["seed"])
    R = {}
    for par in ("p0", "p1"):
        deltas = {fam: cells[(fam, par)]["delta"] for fam in FAMILIES}
        for fam in FAMILIES:
            C = cells[(fam, par)]
            dl = deltas[fam]
            full = C["run"](dl)
            zero = C["run"](torch.zeros_like(dl))
            ref = v174.get(f"{fam}:{par}", {}).get("single_slice", {}).get(str(CH))
            fr = lambda x: round((x - zero) / (full - zero), 3) if abs(full - zero) > 1e-6 else None
            pair, cos, pair_norm = {}, {}, {}
            for other in FAMILIES:
                if other == fam: continue
                u = deltas[other] / deltas[other].norm()
                pv = (dl @ u) * u
                pair[other] = fr(C["run"](pv)); cos[other] = round(float(dl @ u / dl.norm()), 3); pair_norm[other] = round(float(pv.norm() / dl.norm()), 3)
            others = torch.stack([deltas[o] for o in FAMILIES if o != fam], 1)
            loo_v = proj(dl, others); loo = fr(C["run"](loo_v))
            pos = None
            if fam in CORR:
                pos_v = proj(dl, torch.stack([deltas[o] for o in CORR if o != fam], 1)); pos = fr(C["run"](pos_v))
            rb = torch.randn(dl.shape[0], 6, generator=gen).to(dl.device, dl.dtype)
            r6_v = proj(dl, rb); r6 = fr(C["run"](r6_v))
            r1 = torch.randn(dl.shape[0], generator=gen).to(dl.device, dl.dtype); r1 = r1 / r1.norm()
            r1_v = (dl @ r1) * r1; rnd1 = fr(C["run"](r1_v))
            S = {"rows": C["rows"], "full": full, "zero": zero, "v174_slice8": ref, "row_dev": round(C["row_dev"], 4), "delta_norm": round(float(dl.norm()), 3),
                 "pair": pair, "pair_cos": cos, "pair_norm_frac": pair_norm, "loo6": loo, "loo6_norm_frac": round(float(loo_v.norm() / dl.norm()), 3),
                 "positive_control": pos, "random6": r6, "random6_norm_frac": round(float(r6_v.norm() / dl.norm()), 3), "random1": rnd1}
            R[f"{fam}:{par}"] = S
            print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_channel8_shared_subspace_v175", "candidate_id": "corpus.unit_tier5_channel8_shared_subspace_v175",
              "bars": BARS, "families": list(FAMILIES), "channel": CH,
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, band: x is not None and band[0] <= x <= band[1]
    pred_a = all(S["v174_slice8"] is not None and abs(S["full"] - S["v174_slice8"]) <= B["reproduce_tol"] and S["row_dev"] <= B["row_uniform_max"] for S in cells)
    corr = [S for k, S in R.items() if k.split(":")[0] in CORR]
    noncorr = [S for k, S in R.items() if k.split(":")[0] not in CORR]
    pred_b = len(corr) == 6 and all(inb(S["positive_control"], B["positive_band"]) for S in corr)
    pred_c = (sum(inb(S["loo6"], B["loo_band"]) for S in noncorr) >= B["loo_min_cells"]
              and sum(inb(S["random6"], B["random6_band"]) for S in cells) >= B["random_min_cells"])
    pairs = [(inb(v, B["pair_band"])) for k, S in R.items() for o, v in S["pair"].items() if not (k.split(":")[0] in CORR and o in CORR)]
    pred_d = (sum(pairs) / len(pairs) >= B["pair_min_frac"]) and sum(inb(S["random1"], B["random1_band"]) for S in cells) >= B["random_min_cells"]
    pred_e = all(R[f"polarity_state:{p}"]["pair"]["correlative_either_neither"] is not None and R[f"polarity_state:{p}"]["pair"]["correlative_both_either"] is not None
                 and R[f"polarity_state:{p}"]["pair"]["correlative_either_neither"] - R[f"polarity_state:{p}"]["pair"]["correlative_both_either"] >= B["npi_margin"]
                 for p in ("p0", "p1"))
    return {"pred_a_instrument": pred_a, "pred_b_positive_control": pred_b, "pred_c_no_shared_subspace": pred_c,
            "pred_d_pairs_disjoint": pred_d, "pred_e_polarity_npi": pred_e}


if __name__ == "__main__":
    main()
