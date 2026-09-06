"""v55: WHICH hidden units of mlp 12-17 carry the polarity gate (write x converted-write product)? Tier-4 residual.

v42/v43: 78% of the polarity gate b is formed as bilinear products inside mlp 12-17 (additive over layers). The Bilinear MLP is
exactly quadratic, so for live input u = u_B + d:  out(u) = out(u_B) + bil(u_B, d) + Down[(L d) * (R d)]  (identity, no
approximation). Masking the last term to a unit subset S in EVERY run of the four-run design (B, D, V, DV) leaves the linear
part intact and lets only S form products; the interaction I then measures the gate carried by S (plus downstream cascade).
S = all units reproduces the unmasked I (executable identity), S = {} reproduces v43's all-linearised floor.
Ranking (FIT, even A1 rows): score_j = mean_rows[ ((L p)_j (R q)_j + (L q)_j (R p)_j) * (Down[:, j] . (w_ans - w_foil)) ] with
p = u_D - u_B, q = u_V - u_B from the D and V runs and the row's donor-axis unembedding difference as the readout proxy.
FINAL (odd A1 rows, one-shot): arms unmasked / all / none / top-256 / top-1024 / random-1024 / top-256 ranked on the odd rows
themselves (ranking-stability control). 27,648 units in total; 256 = 0.9%, 1024 = 3.7%.
Shares are of the gated part G = I_unmasked - I_none on the FINAL rows.

REGISTERED BEFORE THE RUN
    pred_a_identity        |I_all - I_unmasked| <= 1e-3 |I_unmasked| and |I_none - I_v43floor| <= 0.05 |I_unmasked| (floor = all-linear
                           MLP 12-17 via v43._linearised on the same rows). Worked: 1e-6 True.
    pred_b_top256          top-256 (fit ranking) share of G >= 0.50 on the odd rows. Worked: 0.62 True; 0.30 False.
    pred_c_top1024         top-1024 share of G >= 0.80. Worked: 0.86 True; 0.60 False.
    pred_d_random          random-1024 share of G <= 0.15 (uniform expectation 0.037). Worked: 0.05 True; 0.25 False.
    pred_e_ranking_stable  top-256 (fit ranking) >= 0.80 x top-256 (odd-row ranking) on the odd rows. Worked: 0.62/0.70 True; 0.40/0.70 False.
    Reading rule. b and c True: the gate is a small named set of Left/Right factor pairs -- the Tier-4 object for this behaviour;
    report the top units per layer. b False: the product is distributed over thousands of units (a dense bilinear form), and the
    tensor-native statement stays at the layer level. d False: the masking perturbs through cascade -- report, do not re-rank.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_downstream_linearisation_v43 as v43

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_gate_hidden_units_v55_result.json"
NAME, N_LAYERS, D, HID = "polarity_licensing", 18, 1152, 4608
ID_TOL, FLOOR_TOL, TOP256_MIN, TOP1024_MIN, RAND_MAX, STABLE_FRAC = 1e-3, 0.05, 0.50, 0.80, 0.15, 0.80
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 1000


def _plan():
    return {"candidate_id": "corpus.unit_gate_hidden_units_v55", "set": v35.SETS[NAME][1], "stack": v35.STACK[NAME],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


@contextmanager
def _masked_quadratic(torch, model, layers, positions, device, u_base, out_base, masks):
    """out = out_B + bil(u_B, d) + Down[mask * (L d * R d)], d = live u - u_B, at the rows' positions; mask=None -> no hook."""
    idx = torch.arange(len(positions), device=device)
    pos = torch.tensor(positions, device=device)
    handles = []

    def mk(l):
        mlp = model.transformer.h[l].mlp
        WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()

        def hook(m, a, o):
            d = a[0][idx, pos].float() - u_base[l]
            new = out_base[l] + v43.v42._bil(mlp, u_base[l], d) + (((d @ WL.T) * (d @ WR.T)) * masks[l]) @ WD.T
            y = o.clone(); y[idx, pos] = new.to(y.dtype); return y
        return hook
    for l in layers:
        handles.append(model.transformer.h[l].mlp.register_forward_hook(mk(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def setup(backend, prep, units, stack_mlps):
    """Four-run ingredients for one prep (v43/v48 recipe): cfg(), RUNS, u_base/out_base for the downstream layers, ins of D/V."""
    torch, model = backend.torch, backend.model
    layers = [g.unit_layer(m) for m in stack_mlps]
    first, last = layers[0], layers[-1]
    down = list(range(last + 1, N_LAYERS))
    cache = v25._merged_cache(prep, units)
    rids = list(prep.base_batch.row_ids)
    resid_b, resid_p = {}, {}
    _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
    v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
    delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
    _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
    v = v38._cross(model.transformer.h[first].mlp, ins_b[first], ins_p1[first] - ins_b[first])

    def cfg(write, vec):
        c = dict(cache)
        for l, m in zip(layers, stack_mlps):
            for i, rid in enumerate(rids):
                c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
        kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
        if write is not None:
            kw["resid_add"] = {first: write}
        return kw
    RUNS = {"B": (None, None), "D": (delta1, None), "V": (None, v), "DV": (delta1, v)}
    ins = {}
    outs = {}
    for k, (w_, vec) in RUNS.items():
        _, ins[k], outs[k] = v30._capture(backend, prep, down, **cfg(w_, vec))
    return {"down": down, "cfg": cfg, "RUNS": RUNS, "u_base": ins["B"], "out_base": outs["B"], "ins": ins,
            "positions": list(prep.base_batch.semantic_positions)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    module, units = v35.SETS[NAME]
    stack_mlps = [m for m in v35.STACK[NAME] if m.startswith("mlp:")]
    rows = g.rows_of(module, "A1")
    fit, fin = g.prepare(backend, rows[0::2], valid_only=True), g.prepare(backend, rows[1::2], valid_only=True)
    W = model.lm_head.weight.float()

    def scores(prep, S):
        """Per-layer (HID,) ranking scores from the D/V runs of this prep."""
        wa = W[torch.tensor(prep.base_batch.answer_ids, device=W.device)]
        wf = W[torch.tensor(prep.base_batch.foil_ids, device=W.device)]
        # the four-run I is signed toward the donor: donor axis = -(base answer - base foil) on answer-changing rows
        axis = -(wa - wf)
        out = {}
        for l in S["down"]:
            mlp = model.transformer.h[l].mlp
            WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()
            p, q = S["ins"]["D"][l] - S["u_base"][l], S["ins"]["V"][l] - S["u_base"][l]
            h = (p @ WL.T) * (q @ WR.T) + (q @ WL.T) * (p @ WR.T)          # (n, HID)
            read = axis @ WD                                                  # (n, HID): Down[:, j] . axis per row
            out[l] = (h * read).mean(0)
        return out

    def masks_top(sc, k, down):
        flat = torch.cat([sc[l] for l in down])
        keep = torch.zeros_like(flat)
        keep[torch.topk(flat, k).indices] = 1.0
        return {l: keep[i * HID:(i + 1) * HID] for i, l in enumerate(down)}, [(down[j // HID], int(j % HID)) for j in torch.topk(flat, min(k, 24)).indices.tolist()]

    def masks_rand(k, down, seed=1):
        gen = torch.Generator().manual_seed(seed)
        flat = torch.zeros(HID * len(down))
        flat[torch.randperm(HID * len(down), generator=gen)[:k]] = 1.0
        flat = flat.to(backend.device)
        return {l: flat[i * HID:(i + 1) * HID] for i, l in enumerate(down)}

    def four(prep, S, masks, floor=False):
        recs = {}
        for k, (w_, vec) in S["RUNS"].items():
            ctxs = []
            if floor:
                ctxs.append(v43._linearised(torch, model, S["down"], S["positions"], backend.device, S["u_base"], S["out_base"]))
            elif masks is not None:
                ctxs.append(_masked_quadratic(torch, model, S["down"], S["positions"], backend.device, S["u_base"], S["out_base"], masks))
            with ExitStack() as es:
                for c in ctxs:
                    es.enter_context(c)
                rec, _, _ = v30._capture(backend, prep, S["down"], **S["cfg"](w_, vec))
            recs[k] = rec
        return recs["DV"] - recs["D"] - recs["V"] + recs["B"]

    S_fit, S_fin = setup(backend, fit, units, stack_mlps), setup(backend, fin, units, stack_mlps)
    down = S_fin["down"]
    sc_fit, sc_fin = scores(fit, S_fit), scores(fin, S_fin)
    ones = {l: torch.ones(HID, device=backend.device) for l in down}
    zeros = {l: torch.zeros(HID, device=backend.device) for l in down}
    m256, top_units = masks_top(sc_fit, 256, down)
    m1024, _ = masks_top(sc_fit, 1024, down)
    m256_own, _ = masks_top(sc_fin, 256, down)
    arms = {"unmasked": four(fin, S_fin, None), "all": four(fin, S_fin, ones), "none": four(fin, S_fin, zeros), "v43_floor": four(fin, S_fin, None, floor=True),
            "top256_fit": four(fin, S_fin, m256), "top1024_fit": four(fin, S_fin, m1024), "rand1024": four(fin, S_fin, masks_rand(1024, down)),
            "top256_own": four(fin, S_fin, m256_own)}
    G = arms["unmasked"] - arms["none"]
    share = {k: (v_ - arms["none"]) / G if G else None for k, v_ in arms.items()}
    per_layer_top256 = {f"mlp:{l:02d}": int(m256[l].sum().item()) for l in down}
    predictions = {
        'pred_a_identity': abs(arms["all"] - arms["unmasked"]) <= ID_TOL * abs(arms["unmasked"]) and abs(arms["none"] - arms["v43_floor"]) <= FLOOR_TOL * abs(arms["unmasked"]),
        'pred_b_top256': share["top256_fit"] is not None and share["top256_fit"] >= TOP256_MIN,
        'pred_c_top1024': share["top1024_fit"] is not None and share["top1024_fit"] >= TOP1024_MIN,
        'pred_d_random': share["rand1024"] is not None and share["rand1024"] <= RAND_MAX,
        'pred_e_ranking_stable': share["top256_fit"] is not None and share["top256_own"] and share["top256_fit"] >= STABLE_FRAC * share["top256_own"],
    }
    result = {"predictions": predictions, "schema": "circuit_unit_gate_hidden_units_result_v1", "candidate_id": "corpus.unit_gate_hidden_units_v55",
              "set": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows_fit": len(fit.rows), "rows_final": len(fin.rows),
              "bars": {"id_tol": ID_TOL, "floor_tol": FLOOR_TOL, "top256_min": TOP256_MIN, "top1024_min": TOP1024_MIN, "rand_max": RAND_MAX, "stable_frac": STABLE_FRAC},
              "interactions": arms, "gated_part_G": G, "shares_of_G": share, "top256_units_per_layer": per_layer_top256,
              "top24_units_fit": [f"mlp:{l:02d}:neuron:{j}" for l, j in top_units],
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "interactions": {k: round(v_, 5) for k, v_ in arms.items()}, "shares_of_G": {k: (round(v_, 3) if v_ is not None else None) for k, v_ in share.items()},
                      "top256_per_layer": per_layer_top256, "top24": result["top24_units_fit"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
