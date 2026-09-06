#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the 3x3 grid and the bilinear surface fit fixed before the run.
"""v32: is the quadratic conversion a CROSS-LAYER product of the write with the stack's linear response?

v29: stack conversion ~ alpha^2. v30: within each stack MLP the response to the write is 72-79% cross-term,
i.e. LINEAR in the write. v31: the margin reads the stack's output linearly (beta slope 1.01-1.04) and the
write reaching the first stack layer is linear (polarity 1.08, quantifier 1.00; voice 1.30). The remaining
place for alpha^2 is a product ACROSS layers: after the stack, the residual carries both the write (alpha)
and the stack's response (~alpha); a later bilinear reader multiplies them. v31's beta arm scaled only one
factor and was linear -- exactly what a product predicts.

Design. Write scaled by alpha in {0, 0.5, 1} (set blocks, q = sqrt(alpha) I); stack MLP outputs replaced by
m_b + beta DeltaM_l(alpha=1) for beta in {0, 0.5, 1} (static replay of the alpha=1 deltas). 9 runs per set.
rec(alpha, beta) fitted by least squares as a + b alpha + c beta + d alpha beta. d is the interaction.
Conversion at alpha=1 = rec(1,1) - rec(1,0). Also rec_frozen(alpha) := rec(alpha, 0) per alpha (repairs v31 arm X).
Polarity's 09:07 stays live in every arm.

REGISTERED BEFORE THE RUN
    pred_a_instrument       rec(1,1) matches the recorded exact-set recovery (quantifier 0.635, polarity 0.585, voice 0.645)
                            within 0.005 and rec(0,0) = 0 within 0.005.
    pred_b_product          interaction share d / conversion >= 0.50 for polarity AND voice. Worked polarity: conv 0.318,
                            d 0.20 -> 0.63 True; d 0.10 -> 0.31 False.
    pred_c_delta_alone_small  rec(0,1) <= 0.30 * conversion for polarity AND voice: the stack's alpha=1 response written
                            on the BASE run (no write) barely moves the margin. Worked: 0.05 / 0.318 = 0.16 True.
    pred_d_surface_bilinear R^2 of the 4-parameter fit over the 9 points >= 0.98 on all three sets. Worked: 0.995 True.
    pred_e_quantifier_additive  quantifier interaction share d / conversion <= 0.50 (late set, v29 slope 1.55, mostly
                            additive). Worked: 0.35 True; 0.6 False.
    Reading rule. b,c True: the converter stack's response is inert on its own and gains its effect by multiplying the
    write downstream -- the "conversion" is a coincidence detector between the write and its own echo; Tier 4 must
    name the downstream multiplier (next: per-layer freeze of mlp 12-17 / final rms_norm under (1,1) vs (1,0)).
    b False: the product is inside the stack's own layer-to-layer cascade (w_last superlinear), expand self-consistently.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_damper_law_v27 as v27
import run_unit_selfterm_sufficiency_v30 as v30

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_cross_layer_product_v32_result.json"
SETS, STACK, EARLY = v30.SETS, v30.STACK, v30.EARLY
GRID = (0.0, 0.5, 1.0)
EXPECTED = {"quantifier_number": 0.635, "polarity_licensing": 0.585, "voice_frame": 0.645}
INSTR_TOL, PRODUCT_BAR, ALONE_BAR, R2_BAR = 0.005, 0.50, 0.30, 0.98
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 1500


def _plan():
    return {"candidate_id": "corpus.unit_cross_layer_product_v32", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _fit(torch, surface):
    pts = [(a, b, r) for (a, b), r in surface.items()]
    X = torch.tensor([[1.0, a, b, a * b] for a, b, _ in pts], dtype=torch.float64)
    y = torch.tensor([r for _, _, r in pts], dtype=torch.float64)
    coef = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return [float(c) for c in coef], (1 - float((resid ** 2).sum()) / ss_tot) if ss_tot else None


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        _, _, outs_b = v30._capture(backend, prep, layers)
        _, _, outs_p1 = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache)
        all_units = list(units) + stack_mlps
        surface = {}
        for a in GRID:
            q = v27._scaled_q(backend, units, a)
            for key in g.blocks_of(stack_mlps):
                q[key] = torch.eye(g.N_EMBD, device=backend.device)
            for b in GRID:
                c = dict(cache)
                for l, m in zip(layers, stack_mlps):
                    for i, rid in enumerate(rids):
                        c[(rid, m)] = outs_b[l][i] + b * (outs_p1[l][i] - outs_b[l][i])
                surface[(a, b)] = v30._capture(backend, prep, layers, units=all_units, donor_cache=c, base_cache=prep.base_cache, q=q)[0]
        coef, r2 = _fit(torch, surface)
        conv = surface[(1.0, 1.0)] - surface[(1.0, 0.0)]
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows),
                        "surface": {f"{a}|{b}": r for (a, b), r in surface.items()},
                        "coef": dict(zip(("a", "b_alpha", "c_beta", "d_alpha_beta"), coef)), "r2": r2,
                        "conversion": conv, "interaction_share": coef[3] / conv if conv else None,
                        "delta_alone_over_conversion": surface[(0.0, 1.0)] / conv if conv else None,
                        "rec_frozen_by_alpha": {str(a): surface[(a, 0.0)] for a in GRID},
                        "conversion_by_alpha": {str(a): surface[(a, 1.0)] - surface[(a, 0.0)] for a in GRID}}
        print(name, {k: round(v, 3) for k, v in report[name]["surface"].items()}, "coef", {k: round(v, 3) for k, v in report[name]["coef"].items()},
              "r2 %.3f share %.2f alone %.2f" % (r2, report[name]["interaction_share"], report[name]["delta_alone_over_conversion"]), flush=True)

    predictions = {
        'pred_a_instrument': all(abs(report[n]["surface"]["1.0|1.0"] - EXPECTED[n]) <= INSTR_TOL and abs(report[n]["surface"]["0.0|0.0"]) <= INSTR_TOL for n in SETS),
        'pred_b_product': all(report[n]["interaction_share"] is not None and report[n]["interaction_share"] >= PRODUCT_BAR for n in EARLY),
        'pred_c_delta_alone_small': all(report[n]["delta_alone_over_conversion"] is not None and report[n]["delta_alone_over_conversion"] <= ALONE_BAR for n in EARLY),
        'pred_d_surface_bilinear': all(report[n]["r2"] is not None and report[n]["r2"] >= R2_BAR for n in SETS),
        'pred_e_quantifier_additive': report["quantifier_number"]["interaction_share"] is not None and report["quantifier_number"]["interaction_share"] <= PRODUCT_BAR,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_cross_layer_product_result_v1",
              "candidate_id": "corpus.unit_cross_layer_product_v32", "grid": list(GRID),
              "bars": {"instrument": INSTR_TOL, "product": PRODUCT_BAR, "alone": ALONE_BAR, "r2": R2_BAR},
              "shares": {n: {"interaction": report[n]["interaction_share"], "alone": report[n]["delta_alone_over_conversion"], "r2": report[n]["r2"]} for n in SETS},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "shares": result["shares"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
