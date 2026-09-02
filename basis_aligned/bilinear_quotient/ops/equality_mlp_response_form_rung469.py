#!/usr/bin/env python3
"""RUNG469 -- factorization-invariant MLP response/state transfer.

Registered before opening rung-469 gradients, forms, or removals:
  pred_a: frozen inputs, exact replay/factors/calls, and algebraic accounting.
  pred_b: full quadratic downstream-reader forms transfer across registers.
  pred_c: reader, state, or reader/state coupling localizes the transfer failure.
  pred_d: code reader plus target state predicts target local response.
  pred_e: the same frozen object predicts exact complete-MLP removal effects.
Strong null: invalid instrument, unresolved localization, or no natural predictive gain.
Literal deployed price: zero parameters saved and zero added.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_mlp_product_term_group_rung467 as code_parent
import equality_mlp_product_term_natural_transfer_rung468 as natural_parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_MLP_RESPONSE_FORM_RUNG469_PREREGISTRATION.md"
DERIVATION = POLY / "EQUALITY_MLP_RESPONSE_FORM_RUNG469_DERIVATION.md"
CODE_RESULT = ROOT / "equality_mlp_product_term_group_rung467_results.json"
NATURAL_RESULT = ROOT / "equality_mlp_product_term_natural_transfer_rung468_results.json"
CODE_SOURCE = ROOT / "ops/equality_mlp_product_term_group_rung467.py"
NATURAL_SOURCE = ROOT / "ops/equality_mlp_product_term_natural_transfer_rung468.py"
OUT = ROOT / "equality_mlp_response_form_rung469_results.json"
MODULES = code_parent.MODULES
SITES = code_parent.SITES
SOURCES = code_parent.SOURCES
CONTEXT_CELLS = code_parent.CONTEXT_CELLS
TARGETS = (*SITES, "union")
D = 1152
HIDDEN = code_parent.HIDDEN
BATCH = 4
WINDOWS = (
    ("code_discovery", "code", 0, 96),
    ("code_validation", "code", 96, 192),
    ("natural_wave0", "natural", 0, 96),
    ("natural_wave1", "natural", 96, 192),
)
FORWARDS_PER_BATCH = 2 + 1 + len(SOURCES) * (2 + len(TARGETS))
EXPECTED_FORWARDS = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS) \
    * FORWARDS_PER_BATCH
EXPECTED_BACKWARDS = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS) \
    * len(SOURCES) * len(CONTEXT_CELLS)
HASHES = {
    PREREG: "e767c28f8c81316f0d448ea1cee3bd4ad7ff4c1d7226a0df06838a6058b17bac",
    DERIVATION: "9a0d3eee456119a9abe3bee8ced712b9e9ec1127e2a32ef4b208dee4059c04cf",
    CODE_RESULT: "cc0480fc260c81b0fe512ec694413178de181b767f1dbfec43c56804b1ee5015",
    NATURAL_RESULT: "115024d73722b2906eed0ce0739012a874c845664ba4be3e2dd9690980a37c6c",
    CODE_SOURCE: "3665fc1b33ebb7bff78f78a9548d75219a43e3a0593e79bed6075a42a821bc8b",
    NATURAL_SOURCE: "2006eafbf9ceef7b882fc88338da1f32afc0f073e7644ae5e5bb938d27bf66c7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    code_result = json.loads(CODE_RESULT.read_text())
    natural_result = json.loads(NATURAL_RESULT.read_text())
    if code_result.get("rung") != 467 or code_result.get("strong_null") is not False:
        raise RuntimeError("rung467 identity changed")
    if natural_result.get("rung") != 468 \
            or natural_result.get("pred_a_instrument") is not True \
            or natural_result.get("strong_null") is not True:
        raise RuntimeError("rung468 registered null identity changed")
    # Rung467 intentionally hash-pinned the pre-result dossier.  The dossier was
    # later append-only updated with 467/468, so recover the unchanged code rows
    # and source scale through rung466 while independently pinning 467 above.
    code_payload, code_masks, code_scale, code_meta, _ = code_parent.parent.validate_inputs()
    natural_payload, natural_masks, natural_scale, _, _, natural_meta, _ = \
        natural_parent.validate_inputs()
    if code_scale != natural_scale:
        raise RuntimeError("source scale changed across roles")
    metadata = {
        "windows": [list(row) for row in WINDOWS],
        "modules": list(MODULES), "sources": list(SOURCES),
        "context_cells": list(CONTEXT_CELLS),
        "code": code_meta, "natural": natural_meta,
        "rung467_result_sha256": sha256(CODE_RESULT),
        "rung468_result_sha256": sha256(NATURAL_RESULT),
    }
    return {
        "code": (code_payload, code_masks),
        "natural": (natural_payload, natural_masks),
    }, code_scale, metadata


def _cosine(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    return float(torch.dot(left, right) / max(
        float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)), 1e-30,
    ))


def _metrics(target, prediction):
    target = torch.as_tensor(target, dtype=torch.float64)
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target_norm = float(torch.linalg.vector_norm(target))
    prediction_norm = float(torch.linalg.vector_norm(prediction))
    return {
        "cosine": _cosine(target, prediction),
        "projection_on_target": float(torch.dot(target, prediction)
                                      / max(float(torch.dot(target, target)), 1e-30)),
        "target_norm": target_norm, "prediction_norm": prediction_norm,
        "normalized_l2_error": float(torch.linalg.vector_norm(prediction - target)
                                     / max(target_norm, 1e-30)),
    }


def _fit_scale(local, exact):
    local = torch.as_tensor(local, dtype=torch.float64)
    exact = torch.as_tensor(exact, dtype=torch.float64)
    return float(torch.dot(local, exact) / max(float(torch.dot(local, local)), 1e-30))


def quadratic_reader(left, right, down, output_reader):
    """Return the unique symmetric matrix for g^T F(x)."""
    coefficient = down.T @ output_reader
    raw = left.T @ (coefficient[:, None] * right)
    return (raw + raw.T) / 2


def state_form(source_state, absent_state):
    """Sum symmetric state forms over the leading positions."""
    plus = (source_state + absent_state).reshape(-1, source_state.shape[-1])
    minus = (source_state - absent_state).reshape(-1, source_state.shape[-1])
    raw = plus.T @ minus
    return (raw + raw.T) / 2


def _capture_states(model, tokens, **kwargs):
    captured = {}
    handles = []
    for layer, site in zip(MODULES, SITES):
        module = model.transformer.h[layer].mlp.Left

        def hook(_module, inputs, name=site):
            if name in captured:
                raise RuntimeError(f"duplicate state capture at {name}")
            captured[name] = inputs[0].detach().clone()

        handles.append(module.register_forward_pre_hook(hook))
    try:
        output = code_parent.run_term_forward(model, tokens, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(SITES):
        raise RuntimeError("not every MLP input state was captured")
    return (*output, captured)


def _record(audit_totals, key, audit):
    row = audit_totals.setdefault(key, {"forwards": 0, "product_captures": 0,
                                        "product_patches": 0})
    row["forwards"] += 1
    row["product_captures"] += audit["product_captures"]
    row["product_patches"] += audit["product_patches"]


def _new_raw():
    return {
        "counts": torch.zeros(len(CONTEXT_CELLS), dtype=torch.float64),
        "positions": 0,
        "gradient_sum": torch.zeros(len(SOURCES), len(MODULES), len(CONTEXT_CELLS), D,
                                    dtype=torch.float64),
        "state_sum": torch.zeros(len(SOURCES), len(MODULES), D, D, dtype=torch.float32),
        "product_delta_sum": torch.zeros(len(SOURCES), len(MODULES), HIDDEN,
                                         dtype=torch.float64),
        "local_sum": torch.zeros(len(SOURCES), len(MODULES), len(CONTEXT_CELLS),
                                 dtype=torch.float64),
        "exact_sum": torch.zeros(len(SOURCES), len(TARGETS), len(CONTEXT_CELLS),
                                 dtype=torch.float64),
        "empty_patch_max_abs": 0.0,
    }


def _nll(logits, rows):
    targets = rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)


def collect_window(model, payload, masks, scale, start, stop, audit_totals, replay):
    raw = _new_raw()
    rows = payload["rows"]
    device = next(model.parameters()).device
    reconstruction = 0.0
    for global_start in range(start, stop, BATCH):
        batch_rows = rows[global_start:global_start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        with torch.no_grad():
            native, _, audit, _ = code_parent.source_parent.run_forward(model, tokens, arm="native")
            audit_parent._record_audit(
                audit_totals, "rung469:native", audit, analytical=False, captures=0, patches=0,
            )
            replay_logits, _, audit, error = code_parent.source_parent.run_forward(
                model, tokens, arm="replay",
            )
            audit_parent._record_audit(
                audit_totals, "rung469:replay", audit, analytical=True, captures=0, patches=0,
            )
            difference = replay_logits - native
            replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
            replay["relative_squared"] = max(
                replay["relative_squared"],
                float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
            )
            reconstruction = max(reconstruction, error)
            absent_logits, absent_products, _, audit, error, absent_states = _capture_states(
                model, tokens, arm="base", capture_products=True,
            )
            _record(audit_totals, "rung469:absent", audit)
            reconstruction = max(reconstruction, error)
        raw["positions"] += int(tokens.shape[0] * tokens.shape[1])
        batch_masks = [masks[cell][global_start:global_start + BATCH].to(device)
                       for cell in CONTEXT_CELLS]
        for ci, selected in enumerate(batch_masks):
            raw["counts"][ci] += int(selected.sum())
        del absent_logits

        for si, source in enumerate(SOURCES):
            arm = code_parent.source_parent.SOURCE_ARMS[source]
            with torch.enable_grad():
                logits, products, writes, audit, error, states = _capture_states(
                    model, tokens, arm=arm, scale=scale, capture_products=True,
                    gradient_writes=True,
                )
                _record(audit_totals, f"rung469:source:{source}", audit)
                reconstruction = max(reconstruction, error)
                losses = _nll(logits, batch_rows)
                delta_writes = {}
                for mi, site in enumerate(SITES):
                    module = model.transformer.h[MODULES[mi]].mlp
                    delta_product = (products[site] - absent_products[site]).float()
                    delta_writes[site] = F.linear(delta_product, module.Down.weight.float())
                    raw["product_delta_sum"][si, mi] += \
                        delta_product.sum((0, 1)).double().cpu()
                    raw["state_sum"][si, mi] += state_form(
                        states[site].float(), absent_states[site].float(),
                    ).cpu()
                active = [(ci, losses[selected].sum()) for ci, selected in enumerate(batch_masks)
                          if int(selected.sum())]
                for active_i, (ci, loss) in enumerate(active):
                    gradients = torch.autograd.grad(
                        loss, tuple(writes[site] for site in SITES),
                        retain_graph=active_i + 1 < len(active), allow_unused=False,
                    )
                    for mi, (site, gradient) in enumerate(zip(SITES, gradients)):
                        removal_reader = -gradient.float()
                        raw["gradient_sum"][si, mi, ci] += \
                            removal_reader.sum((0, 1)).double().cpu()
                        raw["local_sum"][si, mi, ci] += \
                            (removal_reader * delta_writes[site]).sum().double().cpu()
                source_nll = losses.detach()
            del writes, losses, delta_writes

            with torch.no_grad():
                empty_logits, _, _, audit, error = code_parent.run_term_forward(
                    model, tokens, arm=arm, scale=scale,
                    baseline_products={}, term_groups={},
                )
                _record(audit_totals, f"rung469:empty:{source}", audit)
                reconstruction = max(reconstruction, error)
                raw["empty_patch_max_abs"] = max(
                    raw["empty_patch_max_abs"],
                    float((empty_logits - logits.detach()).abs().max()),
                )
                for ti, target in enumerate(TARGETS):
                    chosen = SITES if target == "union" else (target,)
                    term_groups = {site: range(HIDDEN) for site in chosen}
                    baselines = {site: absent_products[site] for site in chosen}
                    patched, _, _, audit, error = code_parent.run_term_forward(
                        model, tokens, arm=arm, scale=scale,
                        baseline_products=baselines, term_groups=term_groups,
                    )
                    _record(audit_totals, f"rung469:patch:{source}:{target}", audit)
                    reconstruction = max(reconstruction, error)
                    damage = _nll(patched, batch_rows) - source_nll
                    for ci, selected in enumerate(batch_masks):
                        raw["exact_sum"][si, ti, ci] += damage[selected].sum().double().cpu()
            del logits, products, states, source_nll
        del absent_products, absent_states
    return raw, reconstruction


@torch.no_grad()
def finalize_window(raw, model):
    counts = raw["counts"].clamp_min(1)
    positions = max(int(raw["positions"]), 1)
    q_forms = torch.empty(len(SOURCES), len(MODULES), len(CONTEXT_CELLS), D, D,
                          dtype=torch.float32)
    s_forms = raw["state_sum"] / positions
    local = raw["local_sum"] / counts[None, None, :]
    exact = raw["exact_sum"] / counts[None, None, :]
    mean_part = torch.zeros_like(local)
    term_mean_part = torch.zeros_like(local)
    device = next(model.parameters()).device
    for mi, layer in enumerate(MODULES):
        module = model.transformer.h[layer].mlp
        left = module.Left.weight.float()
        right = module.Right.weight.float()
        down = module.Down.weight.float()
        for si in range(len(SOURCES)):
            state = s_forms[si, mi].to(device)
            mean_delta_product = (raw["product_delta_sum"][si, mi] / positions).to(device).float()
            for ci in range(len(CONTEXT_CELLS)):
                reader = (raw["gradient_sum"][si, mi, ci] / counts[ci]).to(device).float()
                q = quadratic_reader(left, right, down, reader)
                q_forms[si, mi, ci] = q.cpu()
                mean_part[si, mi, ci] = (q.double() * state.double()).sum().cpu()
                coefficient = down.T @ reader
                term_mean_part[si, mi, ci] = \
                    torch.dot(coefficient.double(), mean_delta_product.double()).cpu()
    covariance = local - mean_part
    identity_error = float((mean_part - term_mean_part).abs().max())
    accounting_error = float((mean_part + covariance - local).abs().max())
    return {
        "counts": raw["counts"], "positions": positions,
        "empty_patch_max_abs": raw["empty_patch_max_abs"],
        "q_forms": q_forms, "s_forms": s_forms,
        "local": local, "exact": exact, "mean_part": mean_part,
        "covariance": covariance, "matrix_term_identity_max_abs": identity_error,
        "mean_covariance_accounting_max_abs": accounting_error,
    }


def _stack_form_cosines(discovery, target):
    q_cos, s_cos = {}, {}
    for si, source in enumerate(SOURCES):
        q_cos[source], s_cos[source] = {}, {}
        for mi, site in enumerate(SITES):
            q_cos[source][site] = _cosine(
                discovery["q_forms"][si, mi], target["q_forms"][si, mi],
            )
            s_cos[source][site] = _cosine(
                discovery["s_forms"][si, mi], target["s_forms"][si, mi],
            )
    return q_cos, s_cos


def _covariance_fraction(window, si):
    values = []
    for mi in range(len(MODULES)):
        for ci in range(len(CONTEXT_CELLS)):
            values.append(abs(float(window["covariance"][si, mi, ci])) /
                          max(abs(float(window["local"][si, mi, ci])), 1e-12))
    return float(torch.tensor(values, dtype=torch.float64).median())


def analyze(windows):
    discovery = windows["code_discovery"]
    targets = ("code_validation", "natural_wave0", "natural_wave1")
    form_cosines = {}
    for name in targets:
        q_cos, s_cos = _stack_form_cosines(discovery, windows[name])
        form_cosines[name] = {"reader_q": q_cos, "state_s": s_cos}

    qualifying_pairs = []
    for source in SOURCES:
        for site in SITES:
            if form_cosines["code_validation"]["reader_q"][source][site] >= .90 \
                    and all(form_cosines[name]["reader_q"][source][site] >= .75
                            for name in ("natural_wave0", "natural_wave1")):
                qualifying_pairs.append(f"{source}:{site}")
    pred_b = len(qualifying_pairs) >= 4

    localization = {}
    labels = []
    for name in ("natural_wave0", "natural_wave1"):
        localization[name] = {}
        for si, source in enumerate(SOURCES):
            q_values = [form_cosines[name]["reader_q"][source][site] for site in SITES]
            s_values = [form_cosines[name]["state_s"][source][site] for site in SITES]
            q_median = float(torch.tensor(q_values, dtype=torch.float64).median())
            s_median = float(torch.tensor(s_values, dtype=torch.float64).median())
            code_cov = _covariance_fraction(discovery, si)
            target_cov = _covariance_fraction(windows[name], si)
            if q_median - s_median >= .15:
                label = "state_shift"
            elif s_median - q_median >= .15:
                label = "reader_shift"
            elif q_median >= .75 and s_median >= .75 and abs(target_cov - code_cov) >= .20:
                label = "coupling_shift"
            else:
                label = "mixed_or_unresolved"
            localization[name][source] = {
                "label": label, "reader_q_median_cosine": q_median,
                "state_s_median_cosine": s_median,
                "code_covariance_fraction_median": code_cov,
                "target_covariance_fraction_median": target_cov,
            }
            labels.append(label)
    pred_c = len(set(labels)) == 1 and labels[0] != "mixed_or_unresolved"

    scales = {source: {} for source in SOURCES}
    for si, source in enumerate(SOURCES):
        for mi, site in enumerate(SITES):
            scales[source][site] = _fit_scale(
                discovery["local"][si, mi], discovery["exact"][si, mi],
            )
        scales[source]["union"] = _fit_scale(
            discovery["local"][si].sum(0), discovery["exact"][si, -1],
        )

    prediction_report = {}
    pred_d = True
    pred_e = True
    natural_d_or_e = False
    for name in targets:
        target = windows[name]
        prediction_report[name] = {}
        for si, source in enumerate(SOURCES):
            cross_modules = torch.zeros(len(MODULES), len(CONTEXT_CELLS), dtype=torch.float64)
            for mi in range(len(MODULES)):
                for ci in range(len(CONTEXT_CELLS)):
                    cross_modules[mi, ci] = (
                        discovery["q_forms"][si, mi, ci].double()
                        * target["s_forms"][si, mi].double()
                    ).sum()
            cross_union = cross_modules.sum(0)
            target_local_union = target["local"][si].sum(0)
            local_metrics = _metrics(target_local_union, cross_union)
            naive_local_metrics = _metrics(target_local_union, discovery["local"][si].sum(0))
            local_improvement = 1 - local_metrics["normalized_l2_error"] / max(
                naive_local_metrics["normalized_l2_error"], 1e-30,
            )
            d_ok = bool(local_metrics["cosine"] >= .80
                        and .25 <= local_metrics["projection_on_target"] <= 1.75
                        and local_improvement >= .10)
            pred_d &= d_ok

            causal_prediction = cross_union * scales[source]["union"]
            target_exact_union = target["exact"][si, -1]
            causal_metrics = _metrics(target_exact_union, causal_prediction)
            naive_causal_metrics = _metrics(target_exact_union, discovery["exact"][si, -1])
            causal_improvement = 1 - causal_metrics["normalized_l2_error"] / max(
                naive_causal_metrics["normalized_l2_error"], 1e-30,
            )
            individual = {}
            qualifying = 0
            for mi, site in enumerate(SITES):
                prediction = cross_modules[mi] * scales[source][site]
                metrics = _metrics(target["exact"][si, mi], prediction)
                naive = _metrics(target["exact"][si, mi], discovery["exact"][si, mi])
                improvement = 1 - metrics["normalized_l2_error"] / max(
                    naive["normalized_l2_error"], 1e-30,
                )
                okay = metrics["cosine"] >= .75 and improvement >= .10
                qualifying += int(okay)
                individual[site] = {
                    "prediction": prediction.tolist(), "metrics": metrics,
                    "naive_metrics": naive, "error_improvement": improvement,
                    "qualifies": bool(okay),
                }
            e_ok = bool(causal_metrics["cosine"] >= .75
                        and .25 <= causal_metrics["projection_on_target"] <= 1.75
                        and causal_improvement >= .10 and qualifying >= 2)
            pred_e &= e_ok
            if name.startswith("natural"):
                natural_d_or_e |= d_ok or e_ok
            prediction_report[name][source] = {
                "cross_local_prediction": cross_union.tolist(),
                "target_local": target_local_union.tolist(),
                "local_metrics": local_metrics, "naive_local_metrics": naive_local_metrics,
                "local_error_improvement": local_improvement, "pred_d_window": d_ok,
                "causal_prediction": causal_prediction.tolist(),
                "target_exact_causal": target_exact_union.tolist(),
                "causal_metrics": causal_metrics, "naive_causal_metrics": naive_causal_metrics,
                "causal_error_improvement": causal_improvement,
                "qualifying_individual_count": qualifying, "individual": individual,
                "pred_e_window": e_ok,
            }
    return {
        "form_cosines": form_cosines, "qualifying_reader_pairs": qualifying_pairs,
        "localization": localization, "frozen_code_scales": scales,
        "predictions": prediction_report,
        "pred_b_reader_stability": bool(pred_b),
        "pred_c_failure_localization": bool(pred_c),
        "pred_d_local_response_prediction": bool(pred_d),
        "pred_e_exact_causal_prediction": bool(pred_e),
        "natural_any_d_or_e": bool(natural_d_or_e),
    }


def _window_summary(window):
    return {
        "counts": window["counts"].tolist(), "positions": window["positions"],
        "local_first_order": window["local"].tolist(),
        "exact_causal": window["exact"].tolist(),
        "mean_form_part": window["mean_part"].tolist(),
        "reader_state_covariance": window["covariance"].tolist(),
        "empty_patch_max_abs": window["empty_patch_max_abs"],
        "matrix_term_identity_max_abs": window["matrix_term_identity_max_abs"],
        "mean_covariance_accounting_max_abs": window["mean_covariance_accounting_max_abs"],
        "raw_forms_or_states_included": False,
    }


def main():
    started = time.time()
    roles, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 469, "model_loaded": False,
            "new_gradients_forms_or_removals_opened": False, "sealed_opened": False,
            "windows": [list(row) for row in WINDOWS],
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_backwards": EXPECTED_BACKWARDS, "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung469 result namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    raw_windows = {}
    reconstruction = 0.0
    for name, role, start, stop in WINDOWS:
        payload, masks = roles[role]
        raw, error = collect_window(
            model, payload, masks, scale, start, stop, audit_totals, replay,
        )
        reconstruction = max(reconstruction, error)
        raw_windows[name] = raw
    windows = {name: finalize_window(raw, model) for name, raw in raw_windows.items()}
    del raw_windows
    analysis = analyze(windows)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    identity_error = max(window["matrix_term_identity_max_abs"] for window in windows.values())
    accounting_error = max(
        window["mean_covariance_accounting_max_abs"] for window in windows.values()
    )
    empty_error = max(window["empty_patch_max_abs"] for window in windows.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and forwards == EXPECTED_FORWARDS and identity_error <= 1e-5
        and accounting_error <= 1e-9
        and empty_error == 0
    )
    strong_null = bool(
        not pred_a or not analysis["pred_c_failure_localization"]
        or not analysis["natural_any_d_or_e"]
    )
    result = {
        "status": "complete", "rung": 469,
        "claim_level": "gauge_invariant_response_form_cross_register_test",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "window_summaries": {name: _window_summary(window) for name, window in windows.items()},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "matrix_term_identity_max_abs": identity_error,
        "mean_covariance_accounting_max_abs": accounting_error,
        "empty_patch_max_abs": empty_error,
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": forwards, "backwards": EXPECTED_BACKWARDS,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_reader_stability': analysis["pred_b_reader_stability"],
        'pred_c_failure_localization': analysis["pred_c_failure_localization"],
        'pred_d_local_response_prediction': analysis["pred_d_local_response_prediction"],
        'pred_e_exact_causal_prediction': analysis["pred_e_exact_causal_prediction"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "derive_exact_intervention_from_response_state_interface"
            if pred_a and all(analysis[key] for key in (
                "pred_b_reader_stability", "pred_c_failure_localization",
                "pred_d_local_response_prediction", "pred_e_exact_causal_prediction",
            )) else "context_conditioned_state_level_causal_quotient_no_product_or_rank_tuning"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 469,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "localization": analysis["localization"],
        "qualifying_reader_pairs": analysis["qualifying_reader_pairs"],
        "prediction_summary": analysis["predictions"],
        "instrument": {
            "replay": replay, "factor_error": reconstruction,
            "matrix_term_identity_error": identity_error,
            "accounting_error": accounting_error, "empty_patch_error": empty_error,
        },
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
