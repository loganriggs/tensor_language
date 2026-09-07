#!/usr/bin/env python3
"""Single-construction rank-one oracle fits for the projective-bisector falsifier."""
from __future__ import annotations

import math

from head_response_projector_contract import orthonormal_basis
from multi_environment_projector_contract import score_checkpoint, select_checkpoint
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent


TARGET_PANELS = ("A1", "A2")
CONTROL_PANELS = ("P", "C")
PROJECTION_MIN, DIRECTION_MIN = 0.75, 0.875
TARGET_MATCH_WEIGHT, BARRIER, TAU = 0.10, 100.0, 0.05
STEPS, CHECKPOINTS, LEARNING_RATE = 8, (0, 4, 8), 0.03


class ConstructionOracleFitError(ValueError):
    pass


def _panel(panel: str) -> str:
    if panel not in TARGET_PANELS:
        raise ConstructionOracleFitError(f"unknown target panel {panel}")
    return panel


def training_loss(backend, context, output, target_panel):
    """Guide on one construction; exact held-parity metrics license checkpoints."""
    panel = _panel(target_panel)
    torch, F = backend.torch, backend.F
    logits = output["logits"]
    margin = logits[context["index"], context["answer"]] - logits[context["index"], context["foil"]]
    delta, target = margin - context["base_margin"], context["target_margin"]
    rows = context["panel_indices"][panel]
    ratio = delta[rows] * target[rows].sign() / target[rows].abs().clamp_min(1e-6)
    projection = (delta[rows] @ target[rows]) / target[rows].square().sum().clamp_min(1e-12)
    soft_direction = torch.sigmoid(20.0 * ratio).mean()
    violation = (torch.relu(PROJECTION_MIN - projection).square()
                 + torch.relu(DIRECTION_MIN - soft_direction).square())
    controls = []
    for control in CONTROL_PANELS:
        selected = context["panel_indices"][control]
        log_patch, log_base = F.log_softmax(logits[selected], -1), context["base_log_probs"][selected]
        kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
        controls.append(TAU * torch.logsumexp(kl / TAU, 0) - TAU * math.log(len(kl)))
    worst_control = TAU * torch.logsumexp(torch.stack(controls) / TAU, 0) - TAU * math.log(len(controls))
    return worst_control + TARGET_MATCH_WEIGHT * (ratio - 1.0).square().mean() + BARRIER * violation


def checkpoint_report(backend, context, raw_by_site, step, counters, target_panel):
    panel, torch = _panel(target_panel), backend.torch
    with torch.no_grad():
        bases = {site: orthonormal_basis(torch, raw).detach().clone()
                 for site, raw in raw_by_site.items()}
        output = parent.manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases, complete15=True)
        metrics = parent.output_metrics(backend, context, output)
    checkpoint = {
        "step": step,
        "targets": {panel: {
            "signed_projection": metrics["targets"][panel]["behavior"]["signed_projection"],
            "direction_fraction": metrics["targets"][panel]["behavior"]["direction_fraction"],
        }},
        "controls": {control: {
            key: metrics["controls"][control][key]
            for key in ("median_kl", "mean_kl", "max_kl", "top1_flip_count")
        } for control in CONTROL_PANELS},
        "metrics": metrics,
    }
    return checkpoint, score_checkpoint(
        checkpoint, target_panels=(panel,), control_panels=CONTROL_PANELS,
        projection_min=PROJECTION_MIN, direction_min=DIRECTION_MIN), bases


def fit_one(backend, train, select, initial, target_panel, fold, counters):
    """Fit one target construction on one parity and select on the opposite parity."""
    panel, torch = _panel(target_panel), backend.torch
    raw = {site: value.detach().clone().requires_grad_(True) for site, value in initial.items()}
    optimizer = torch.optim.Adam(tuple(raw.values()), lr=LEARNING_RATE)
    trace, snapshots, gradient_max = [], {}, 0.0

    def checkpoint(step):
        report, score, bases = checkpoint_report(
            backend, select, raw, step, counters, panel)
        trace.append({"report": report, "score": score})
        snapshots[step] = bases

    checkpoint(0)
    for update in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        output = parent.manual_forward(
            backend, train["base_batch"], counters, context=train,
            raw_by_site=raw, complete15=True, grad=True)
        training_loss(backend, train, output, panel).backward()
        gradient_max = max(gradient_max, max(
            float(value.grad.abs().max()) if value.grad is not None else 0.0
            for value in raw.values()))
        optimizer.step()
        counters["model_updates"] += 1
        if update in CHECKPOINTS:
            checkpoint(update)
    chosen_report, chosen_score = select_checkpoint(
        [item["report"] for item in trace], target_panels=(panel,),
        control_panels=CONTROL_PANELS, projection_min=PROJECTION_MIN,
        direction_min=DIRECTION_MIN)
    chosen = next(item for item in trace if item["report"] is chosen_report)
    assert chosen["score"] == chosen_score
    return {
        "target_panel": panel, "fold": fold, "trace": trace, "best": chosen,
        "best_bases": snapshots[chosen_report["step"]],
        "initial_bases": snapshots[0], "gradient_max_abs": gradient_max,
    }


def strip_fit(value):
    return {key: item for key, item in value.items()
            if key not in {"best_bases", "initial_bases"}}
