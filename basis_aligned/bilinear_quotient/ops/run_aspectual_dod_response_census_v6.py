#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_direct_term_share_in_band pred_d_downstream_amplifies_in_every_construction pred_e_largest_responder_is_a_named_mlp
"""Aspectual has/had definition-of-done battery, step 6: RESPONSE CENSUS of the attention9 removal.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v4 (weight-only attention9
H1/H4 readout removal, 0.70 logits on the discovery shape) and v5 (transfers to three templates).

WHY. better_circuits §3.3: "no suffix is proposed without a forward response census (one edited
forward, decomposed into module responses) run first." The v4/v5 damage is measured through the
native suffix (blocks 9-17). This run splits it exactly: the removed write's own contribution to
resid18 (the DIRECT term, attn:09) and every downstream module's response (mlp:09, attn:10 ...
mlp:17), using the model's residual recurrence x_{l+1} = lambda0_l x_l + lambda1_l x_0 + attn_l
+ mlp_l, so the resid18 change is EXACTLY the lambda-weighted sum of module output changes. The
margin is then attributed linearly (gradient of the scored contrast at the native resid18) with
the nonlinear remainder reported, not hidden. Evidence tag: response (opened rows).

ROWS: the 64 discovery-shaped rows (v1 lexicon) + the 96 template rows (v5) = 160 rows, four
constructions; all opened for removal, none used for selection.

PREDICTIONS (scored as written; failures preserved)
    pred_a_recurrence_closure          lambda-weighted sum of module deltas equals the resid18
                                       delta within relative L2 1e-3 on every row
    pred_b_linearization_remainder_small
                                       |mean exact margin change - mean linear attribution| <=
                                       0.10 x |mean exact margin change|, pooled
    pred_c_direct_term_share_in_band   the direct attn:09 term is between 0.20 and 0.80 of the
                                       pooled linear attribution. Prior: unsure; a share near 1
                                       would mean the suffix is inert for this edit, near 0 that
                                       the head's write acts only through later readers.
    pred_d_downstream_amplifies_in_every_construction
                                       in each of the four constructions the net downstream
                                       response has the same sign as the direct term
    pred_e_largest_responder_is_a_named_mlp
                                       the largest |downstream| responder (pooled) is one of
                                       mlp:12, mlp:14, mlp:17 (the path's suffix MLPs / the
                                       calibrator). Prior: unsure.

PRICE (registered maximum): 5 batches x (native trace + edited trace) = 10 forwards; margin
gradients on captured 1152-d vectors only; 0 fits. Bar <= 20 forwards.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_response_census_v6_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_response_census_v6"
EXPECTED_DISCOVERY_SHA256 = "5ec7d32f3f7b6bbc9ebc31de3411074b24a998430f7f929bd11d0813e40dfcd2"
EXPECTED_TEMPLATE_SHA256 = "dca4aa137b6add050fd694d496b40414b241ea86fe58086b8007bae3375917d3"
COMPONENT = next(c for c in L.COMPONENTS if c.name == "attn9_h1_h4_final")
TOKENS = {"has": 468, "had": 550}
BATCH = 32
CLOSURE_TOL, REMAINDER_RATIO, DIRECT_BAND, NAMED_MLPS = 1e-3, 0.10, (0.20, 0.80), ("mlp:12", "mlp:14", "mlp:17")
FORWARDS_MAX = 20
MODULES = [f"{kind}:{layer:02d}" for layer in range(9, 18) for kind in ("attn", "mlp")]


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "component": COMPONENT.name,
            "modules": MODULES, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "remainder_ratio": REMAINDER_RATIO,
                     "direct_band": list(DIRECT_BAND), "named_mlps": list(NAMED_MLPS)}}


def main() -> None:
    discovery, templates = L.build_rows(), L.build_template_rows()
    if L.rows_sha256(discovery) != EXPECTED_DISCOVERY_SHA256 or L.rows_sha256(templates) != EXPECTED_TEMPLATE_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    rows = discovery + templates
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMPONENT,), TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, edited = [], []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        native.extend(L.forward_trace(fw, chunk, from_layer=9)); forwards += 1
        edited.extend(L.forward_trace(fw, chunk, components=(COMPONENT,), mode="project", from_layer=9)); forwards += 1

    def margin(x, row):
        return L.final_margin(model, F, x, row.answer_id, row.foil_id) * (1.0 if row.present else 1.0)

    per_row = []
    for row, nat, ed in zip(rows, native, edited):
        weights = {}
        for layer in range(9, 18):
            w = 1.0
            for later in range(layer + 1, 18):
                w *= nat[f"lambda0:{later:02d}"]
            weights[f"attn:{layer:02d}"] = w
            weights[f"mlp:{layer:02d}"] = w
        deltas = {m: ed[m] - nat[m] for m in MODULES}
        recon = sum(weights[m] * deltas[m] for m in MODULES)
        true_delta = ed["resid18"] - nat["resid18"]
        closure = float((recon - true_delta).norm() / true_delta.norm())
        # gradient of the scored contrast at the native resid18 (oriented toward the native answer)
        x = nat["resid18"].clone().requires_grad_(True)
        normed = F.rms_norm(x, (x.shape[-1],))
        l = 30.0 * (model.lm_head(normed) / 30.0).tanh()
        contrast = l[row.answer_id] - l[row.foil_id]
        contrast.backward()
        g = x.grad.detach()
        exact_change = margin(ed["resid18"], row) - margin(nat["resid18"], row)
        attribution = {m: float(g @ (weights[m] * deltas[m])) for m in MODULES}
        per_row.append({"row_id": row.row_id, "construction": row.construction, "closure": closure,
                        "exact_change": exact_change, "linear_total": sum(attribution.values()),
                        "attribution": attribution})

    constructions = sorted({r.construction for r in rows})
    def pooled(items):
        n = len(items)
        out = {"n": n, "exact_change_mean": sum(i["exact_change"] for i in items) / n,
               "linear_total_mean": sum(i["linear_total"] for i in items) / n,
               "attribution_mean": {m: sum(i["attribution"][m] for i in items) / n for m in MODULES}}
        out["direct_term_mean"] = out["attribution_mean"]["attn:09"]
        out["downstream_net_mean"] = out["linear_total_mean"] - out["direct_term_mean"]
        out["direct_share_of_linear"] = out["direct_term_mean"] / out["linear_total_mean"] if out["linear_total_mean"] else None
        down = {m: v for m, v in out["attribution_mean"].items() if m != "attn:09"}
        out["largest_downstream_responder"] = max(down, key=lambda m: abs(down[m]))
        out["remainder"] = out["exact_change_mean"] - out["linear_total_mean"]
        return out
    summary = {"pooled": pooled(per_row), "by_construction": {c: pooled([i for i in per_row if i["construction"] == c]) for c in constructions}}
    P = summary["pooled"]
    max_closure = max(i["closure"] for i in per_row)
    predictions = {
        "pred_a_recurrence_closure": max_closure <= CLOSURE_TOL,
        "pred_b_linearization_remainder_small": abs(P["remainder"]) <= REMAINDER_RATIO * abs(P["exact_change_mean"]),
        "pred_c_direct_term_share_in_band": P["direct_share_of_linear"] is not None and DIRECT_BAND[0] <= P["direct_share_of_linear"] <= DIRECT_BAND[1],
        "pred_d_downstream_amplifies_in_every_construction": all(
            (s["downstream_net_mean"] > 0) == (s["direct_term_mean"] > 0) for s in summary["by_construction"].values()),
        "pred_e_largest_responder_is_a_named_mlp": P["largest_downstream_responder"] in NAMED_MLPS,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_response_census_result_v6", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "max_closure_relative_error": max_closure, "summary": summary,
              "per_row": per_row, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "max_closure": max_closure,
                      "pooled": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P.items() if k != "attribution_mean"},
                      "attribution_mean": {m: round(v, 4) for m, v in P["attribution_mean"].items()}}, indent=2))


if __name__ == "__main__":
    main()
