#!/usr/bin/env python3
"""RUNG467 -- held-out equality-correction groups inside MLP8/9/12.

Registered before opening product fingerprints:
  pred_a: exact replay/factor/parent/no-op/census identities.
  pred_b: the fixed alignment rule finds a stable nonempty split in at least two MLPs.
  pred_c: exact held-out removal carries the parent correction direction.
  pred_d: the task-conditioned group beats matched-count amplitude and random controls.
  pred_e: at least two MLP groups are causal members and composition is resolved.
Strong null: invalid instrument, empty/tiny/source-opposed group, or loss to both controls.
Literal deployed price: zero parameters saved and zero added; identification probe only.
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
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_correction_group_factorial_rung466 as parent


source_parent = parent.parent.parent
PREREG = POLY / "EQUALITY_MLP_PRODUCT_TERM_GROUP_RUNG467_PREREGISTRATION.md"
DOSSIER = POLY / "explanations/MLP8_MLP9_MLP12_CURRENT_UNDERSTANDING.md"
PARENT_RESULT = ROOT / "equality_correction_group_factorial_rung466_results.json"
PARENT_SOURCE = ROOT / "ops/equality_correction_group_factorial_rung466.py"
OUT = ROOT / "equality_mlp_product_term_group_rung467_results.json"
SOURCES = ("N", "H")
ALL_SOURCES = ("0", "N", "H")
MODULES = (8, 9, 12)
SITES = tuple(f"m{layer}" for layer in MODULES)
SUBSETS = tuple(range(8))
CONTROL_MASKS = (1, 2, 4, 7)
CONTROL_TYPES = ("amplitude", "random")
CELLS = parent.CELLS
CONTEXT_CELLS = parent.CONTEXT_CELLS
DOCUMENTS = parent.DOCUMENTS
BATCH = parent.BATCH
DISCOVERY_STOP = 96
VALIDATION_START = 96
HIDDEN = 4608
TARGET = torch.tensor((-1.0, 1.0, 1.0, -1.0), dtype=torch.float64) / 2.0
EXPECTED_FORWARDS = (
    (DOCUMENTS // BATCH) * 2
    + (DISCOVERY_STOP // BATCH) * len(ALL_SOURCES)
    + ((DOCUMENTS - VALIDATION_START) // BATCH)
    * (len(ALL_SOURCES) + len(SOURCES) * (
        len(SUBSETS) + len(SUBSETS) + len(CONTROL_TYPES) * len(CONTROL_MASKS)
    ))
)
HASHES = {
    PREREG: "09414307fe2d96f4f9547a11e78aa6b3c2345573e9b35b9f5bd8fa1fed06d37e",
    DOSSIER: "8008c50bcf6d398fe14bba2c41f45bfa2627c083e61495f144d6403c47774ce2",
    PARENT_RESULT: "d04acf3637834830f8ee7bd73eaa8a6c435386816ef54fce1d8451b0597132fe",
    PARENT_SOURCE: "48fd463c04981b601a969e8fa9f1020180c75b0bff68092e11d2d233e54ddd72",
    source_parent.path_parent.ROW_RECEIPT:
        "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    source_parent.path_parent.ROWS:
        "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
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
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 466 or not all(result.get(key) is True for key in (
        "pred_a_instrument", "pred_b_task_group_context",
        "pred_c_broad_suppressor_role", "pred_d_cross_group_interaction",
        "pred_e_five_site_extraction",
    )) or result.get("strong_null") is not False:
        raise RuntimeError("rung466 full-pass identity changed")
    payload, masks, scale, metadata, _ = parent.validate_inputs()
    metadata = {
        **metadata, "rung466_result_sha256": sha256(PARENT_RESULT),
        "rung466_source_sha256": sha256(PARENT_SOURCE),
        "dossier_sha256": sha256(DOSSIER), "modules": list(MODULES),
        "discovery_documents": [0, DISCOVERY_STOP],
        "validation_documents": [VALIDATION_START, DOCUMENTS],
        "context_cells": list(CONTEXT_CELLS),
        "selection_cosine_threshold": .70,
    }
    return payload, masks, scale, metadata, result


def _cosine(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return float(torch.dot(left, right) / max(
        float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)), 1e-30,
    ))


def _metrics(parent_vector, vector):
    parent_vector = torch.as_tensor(parent_vector, dtype=torch.float64)
    vector = torch.as_tensor(vector, dtype=torch.float64)
    pn = float(torch.linalg.vector_norm(parent_vector))
    vn = float(torch.linalg.vector_norm(vector))
    return {
        "cosine": float(torch.dot(parent_vector, vector) / max(pn * vn, 1e-30)),
        "parent_norm": pn, "vector_norm": vn,
        "projection_on_parent": float(
            torch.dot(parent_vector, vector) / max(float(torch.dot(parent_vector, parent_vector)), 1e-30)
        ),
    }


def _sign_pattern(vector):
    return bool(vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0)


def _jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _quantiles(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        return {}
    return {str(q): float(torch.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, .99, 1)}


def _selection_one(pooled, halves):
    """Return fixed selected indices and half-only selections from [2,H,4] fingerprints."""
    selected = []
    for j in range(pooled.shape[1]):
        n, h = pooled[0, j], pooled[1, j]
        if _cosine(n, TARGET) < .70 or _cosine(h, TARGET) < .70 or _cosine(n, h) < .70:
            continue
        stable = True
        for half in range(2):
            for source in range(2):
                q = halves[half, source, j]
                if float(torch.dot(q, TARGET)) <= 0:
                    stable = False
            if _cosine(halves[half, 0, j], halves[half, 1, j]) <= 0:
                stable = False
        for source in range(2):
            if _cosine(halves[0, source, j], halves[1, source, j]) <= 0:
                stable = False
        if stable:
            selected.append(j)
    half_selected = []
    for half in range(2):
        indices = []
        for j in range(pooled.shape[1]):
            n, h = halves[half, 0, j], halves[half, 1, j]
            if _cosine(n, TARGET) >= .70 and _cosine(h, TARGET) >= .70 \
                    and _cosine(n, h) >= .70:
                indices.append(j)
        half_selected.append(indices)
    return selected, half_selected


def select_groups(fingerprint_numerators, fingerprint_counts, amplitude):
    """Freeze groups after discovery; no validation tensors enter this function."""
    halves = fingerprint_numerators / fingerprint_counts[:, :, None, None, :].clamp_min(1)
    pooled_num = fingerprint_numerators.sum(0)
    pooled_counts = fingerprint_counts.sum(0)
    pooled = pooled_num / pooled_counts[:, None, None, :].clamp_min(1)
    groups, controls, report = {}, {name: {} for name in CONTROL_TYPES}, {}
    for mi, site in enumerate(SITES):
        chosen, split = _selection_one(pooled[:, mi], halves[:, :, mi])
        groups[site] = chosen
        n = len(chosen)
        chosen_set = set(chosen)
        remaining = [j for j in range(HIDDEN) if j not in chosen_set]
        amplitude_order = torch.argsort(amplitude[mi], descending=True).tolist()
        controls["amplitude"][site] = (
            [j for j in amplitude_order if j not in chosen_set]
            + [j for j in amplitude_order if j in chosen_set]
        )[:n]
        seed = int(hashlib.sha256(f"rung467:{site}".encode()).hexdigest()[:16], 16)
        generator = torch.Generator().manual_seed(seed)
        random_order = torch.randperm(HIDDEN, generator=generator).tolist()
        controls["random"][site] = (
            [j for j in random_order if j not in chosen_set]
            + [j for j in random_order if j in chosen_set]
        )[:n]
        target_cos = {
            SOURCES[si]: [_cosine(pooled[si, mi, j], TARGET) for j in range(HIDDEN)]
            for si in range(2)
        }
        source_cos = [_cosine(pooled[0, mi, j], pooled[1, mi, j]) for j in range(HIDDEN)]
        report[site] = {
            "selected_count": n, "selected_indices": chosen,
            "half_selection_counts": [len(x) for x in split],
            "half_selection_jaccard": _jaccard(*split),
            "selected_indices_by_discovery_half": split,
            "target_cosine_quantiles": {s: _quantiles(v) for s, v in target_cos.items()},
            "source_cosine_quantiles": _quantiles(source_cos),
            "amplitude_control_indices": controls["amplitude"][site],
            "random_control_indices": controls["random"][site],
        }
    return groups, controls, report, pooled, halves


def _record(audit_totals, key, audit):
    row = audit_totals.setdefault(key, {"forwards": 0, "product_captures": 0,
                                        "product_patches": 0})
    row["forwards"] += 1
    row["product_captures"] += audit["product_captures"]
    row["product_patches"] += audit["product_patches"]


def run_term_forward(
    model, tokens, *, arm, scale=None, capture_products=False,
    baseline_products: Mapping[str, torch.Tensor] | None = None,
    term_groups: Mapping[str, Sequence[int] | torch.Tensor] | None = None,
    gradient_writes=False,
):
    """The rung-464 source path plus exact product capture/replacement."""
    if arm not in ("base", "reference", "score"):
        raise ValueError("term forward accepts only source trajectories")
    if arm == "score" and scale is None:
        raise ValueError("score arm requires frozen scale")
    analytical = True
    baseline_products = {} if baseline_products is None else dict(baseline_products)
    term_groups = {} if term_groups is None else dict(term_groups)
    if set(baseline_products) != set(term_groups):
        raise ValueError("every term group needs one absent-product baseline")
    if set(term_groups) - set(SITES):
        raise ValueError("unregistered product-term site")
    cached_early, products, writes = {}, {}, {}
    audit = {"product_captures": 0, "product_patches": 0}
    max_reconstruction = 0.0

    def attention(event):
        nonlocal max_reconstruction
        if event.site in source_parent.path_parent.parent.stage1.SITE_HEADS:
            write, factors, support, reconstruction = source_parent.path_parent.parent.factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens,
            )
            max_reconstruction = max(max_reconstruction, reconstruction)
            if arm != "replay":
                early, late = source_parent.path_parent.parent.PAIR
                early_site = source_parent.path_parent.parent.factor_parent.TERMS[early][1]
                late_site = source_parent.path_parent.parent.factor_parent.TERMS[late][1]
                if event.site == early_site:
                    cached_early.update(factors[early])
                    write = write - factors[early]["native_term"]
                if event.site == late_site:
                    if not cached_early:
                        raise RuntimeError("early factors missing")
                    late_factor = factors[late]
                    if arm != "reference":
                        write = write - late_factor["native_term"]
                        if arm == "score":
                            score = cached_early["p"] * scale["score_ratio"]
                            write = write + torch.bmm(score * support, late_factor["u"]).to(write.dtype)
            next_value = event.first_value
        else:
            write, next_value = event.block.attn(event.state, event.first_value)
        return write, next_value

    def mlp(event):
        site = f"m{event.site}"
        module = event.block.mlp
        if site in SITES:
            z = module.Left(event.state) * module.Right(event.state)
            if capture_products:
                products[site] = z.detach().clone()
                audit["product_captures"] += 1
            if site in term_groups:
                index = torch.as_tensor(term_groups[site], dtype=torch.long, device=z.device)
                baseline = baseline_products[site]
                if baseline.shape != z.shape or baseline.dtype != z.dtype or baseline.device != z.device:
                    raise RuntimeError(f"malformed absent product baseline at {site}")
                if index.numel():
                    z = z.clone()
                    z[..., index] = baseline[..., index]
                audit["product_patches"] += 1
            write = module.Down(z) + module.Down_bias
            if gradient_writes:
                if event.site == MODULES[0]:
                    write = write.detach().requires_grad_(True)
                if not write.requires_grad:
                    raise RuntimeError("gradient write graph did not start at MLP8")
                writes[site] = write
            return write
        return module(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    if capture_products and set(products) != set(SITES):
        raise RuntimeError("not every target product vector was captured")
    if audit["product_patches"] != len(term_groups):
        raise RuntimeError("not every declared product patch fired")
    return logits, products, writes, audit, max_reconstruction


def _discovery(model, rows, masks, scale, audit_totals):
    numerators = torch.zeros(2, 2, len(MODULES), HIDDEN, len(CONTEXT_CELLS),
                             dtype=torch.float64)
    counts = torch.zeros(2, 2, len(CONTEXT_CELLS), dtype=torch.float64)
    amplitude = torch.zeros(len(MODULES), HIDDEN, dtype=torch.float64)
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DISCOVERY_STOP, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        with torch.no_grad():
            _, base_products, _, audit, error = run_term_forward(
                model, tokens, arm="base", capture_products=True,
            )
        _record(audit_totals, "discovery:0", audit)
        reconstruction = max(reconstruction, error)
        half = start // 48
        for si, source in enumerate(SOURCES):
            with torch.enable_grad():
                logits, products, writes, audit, error = run_term_forward(
                    model, tokens, arm=source_parent.SOURCE_ARMS[source], scale=scale,
                    capture_products=True, gradient_writes=True,
                )
                _record(audit_totals, f"discovery:{source}", audit)
                reconstruction = max(reconstruction, error)
                targets = batch_rows[:, 1:].to(device)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
                ).view(len(batch_rows), -1)
                active = []
                for ci, cell in enumerate(CONTEXT_CELLS):
                    selected = masks[cell][start:start + BATCH].to(device)
                    observed = int(selected.sum())
                    counts[half, si, ci] += observed
                    if observed:
                        active.append((ci, nll[selected].sum()))
                for active_i, (ci, loss) in enumerate(active):
                    gradients = torch.autograd.grad(
                        loss, tuple(writes[site] for site in SITES),
                        retain_graph=active_i + 1 < len(active), allow_unused=False,
                    )
                    for mi, (site, gradient) in enumerate(zip(SITES, gradients)):
                        module = model.transformer.h[MODULES[mi]].mlp
                        reader = torch.matmul(gradient.float(), module.Down.weight.float())
                        delta = (products[site] - base_products[site]).float()
                        score = -(reader * delta).sum((0, 1)).double().cpu()
                        numerators[half, si, mi, :, ci] += score
                for mi, site in enumerate(SITES):
                    delta = (products[site] - base_products[site]).float()
                    down2 = model.transformer.h[MODULES[mi]].mlp.Down.weight.float().square().sum(0)
                    amplitude[mi] += (delta.square().sum((0, 1)) * down2).double().cpu()
                del logits, products, writes, nll
        del base_products
    return numerators, counts, amplitude, reconstruction


def _ce_sums(logits, rows, masks, global_start):
    return source_parent.path_parent.parent._ce_sums(logits, rows, masks, global_start)


def _loss_report(base, other, counts, start, stop):
    return source_parent.path_parent.parent.effect_report(base, other, counts, start, stop)


@torch.no_grad()
def _validation(model, rows, masks, scale, groups, controls, audit_totals):
    n = DOCUMENTS - VALIDATION_START
    full = torch.zeros(len(ALL_SOURCES), n, len(CELLS), dtype=torch.float64)
    proposed = torch.zeros(len(SOURCES), len(SUBSETS), n, len(CELLS), dtype=torch.float64)
    complete = torch.zeros_like(proposed)
    control = torch.zeros(len(SOURCES), len(CONTROL_TYPES), len(CONTROL_MASKS), n,
                          len(CELLS), dtype=torch.float64)
    counts = torch.zeros(n, len(CELLS), dtype=torch.float64)
    reconstruction = 0.0
    device = next(model.parameters()).device
    for global_start in range(VALIDATION_START, DOCUMENTS, BATCH):
        local = global_start - VALIDATION_START
        batch_rows = rows[global_start:global_start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        products_by_source = {}
        for ai, source in enumerate(ALL_SOURCES):
            logits, products, _, audit, error = run_term_forward(
                model, tokens, arm=source_parent.SOURCE_ARMS[source], scale=scale,
                capture_products=True,
            )
            _record(audit_totals, f"validation:capture:{source}", audit)
            reconstruction = max(reconstruction, error)
            sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
            full[ai, local:local + BATCH] = sums
            if ai == 0:
                counts[local:local + BATCH] = observed
            elif not torch.equal(observed, counts[local:local + BATCH]):
                raise RuntimeError("validation cell support changed")
            products_by_source[source] = products
        absent = products_by_source["0"]
        for si, source in enumerate(SOURCES):
            arm = source_parent.SOURCE_ARMS[source]
            for mask in SUBSETS:
                term_groups = {site: groups[site] for bit, site in enumerate(SITES)
                               if mask & (1 << bit)}
                baselines = {site: absent[site] for site in term_groups}
                logits, _, _, audit, error = run_term_forward(
                    model, tokens, arm=arm, scale=scale,
                    baseline_products=baselines, term_groups=term_groups,
                )
                _record(audit_totals, f"validation:proposed:{source}:{mask}", audit)
                reconstruction = max(reconstruction, error)
                sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
                if not torch.equal(observed, counts[local:local + BATCH]):
                    raise RuntimeError("proposed support changed")
                proposed[si, mask, local:local + BATCH] = sums
                full_groups = {site: range(HIDDEN) for bit, site in enumerate(SITES)
                               if mask & (1 << bit)}
                full_baselines = {site: absent[site] for site in full_groups}
                logits, _, _, audit, error = run_term_forward(
                    model, tokens, arm=arm, scale=scale,
                    baseline_products=full_baselines, term_groups=full_groups,
                )
                _record(audit_totals, f"validation:complete:{source}:{mask}", audit)
                reconstruction = max(reconstruction, error)
                sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
                complete[si, mask, local:local + BATCH] = sums
            for ti, control_type in enumerate(CONTROL_TYPES):
                for ci, mask in enumerate(CONTROL_MASKS):
                    term_groups = {site: controls[control_type][site]
                                   for bit, site in enumerate(SITES) if mask & (1 << bit)}
                    baselines = {site: absent[site] for site in term_groups}
                    logits, _, _, audit, error = run_term_forward(
                        model, tokens, arm=arm, scale=scale,
                        baseline_products=baselines, term_groups=term_groups,
                    )
                    _record(audit_totals, f"validation:{control_type}:{source}:{mask}", audit)
                    reconstruction = max(reconstruction, error)
                    sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
                    control[si, ti, ci, local:local + BATCH] = sums
        del products_by_source
    return full, proposed, complete, control, counts, reconstruction


def _window(full, proposed, complete, control, counts, start, stop):
    base = full[0]
    full_reports = {source: _loss_report(base, full[ALL_SOURCES.index(source)], counts, start, stop)
                    for source in SOURCES}

    def vectors_for(losses):
        output = {source: {} for source in SOURCES}
        for si, source in enumerate(SOURCES):
            for mask in SUBSETS:
                report = _loss_report(base, losses[si, mask], counts, start, stop)
                output[source][mask] = [
                    full_reports[source][cell]["effect_nat"] - report[cell]["effect_nat"]
                    for cell in CONTEXT_CELLS
                ]
        return output

    proposed_vectors = vectors_for(proposed)
    complete_vectors = vectors_for(complete)
    control_vectors = {source: {name: {} for name in CONTROL_TYPES} for source in SOURCES}
    control_off = {source: {} for source in SOURCES}
    for si, source in enumerate(SOURCES):
        for ti, name in enumerate(CONTROL_TYPES):
            for ci, mask in enumerate(CONTROL_MASKS):
                report = _loss_report(base, control[si, ti, ci], counts, start, stop)
                control_vectors[source][name][mask] = [
                    full_reports[source][cell]["effect_nat"] - report[cell]["effect_nat"]
                    for cell in CONTEXT_CELLS
                ]
                control_off[source][f"{name}:{mask}"] = (
                    full_reports[source]["off_target"]["effect_nat"]
                    - report["off_target"]["effect_nat"]
                )
    proposed_off = {}
    for si, source in enumerate(SOURCES):
        report = _loss_report(base, proposed[si, 7], counts, start, stop)
        proposed_off[source] = (full_reports[source]["off_target"]["effect_nat"]
                                - report["off_target"]["effect_nat"])
    return {
        "full_reports": full_reports, "proposed_vectors": proposed_vectors,
        "complete_vectors": complete_vectors, "control_vectors": control_vectors,
        "proposed_off_target": proposed_off, "control_off_target": control_off,
    }


def analyze(full, proposed, complete, control, counts, selection_report):
    pooled = _window(full, proposed, complete, control, counts, 0, len(counts))
    halves = [_window(full, proposed, complete, control, counts, start, start + 48)
              for start in (0, 48)]
    counts_by = {site: selection_report[site]["selected_count"] for site in SITES}
    total = sum(counts_by.values())
    stable_modules = sum(selection_report[site]["half_selection_jaccard"] >= .20
                         for site in SITES)
    pred_b = bool(sum(value >= 4 for value in counts_by.values()) >= 2
                  and 12 <= total <= 3456 and stable_modules >= 2)

    union_metrics, source_agreement = {}, _metrics(
        pooled["proposed_vectors"]["N"][7], pooled["proposed_vectors"]["H"][7]
    )
    for source in SOURCES:
        union_metrics[source] = _metrics(
            pooled["complete_vectors"][source][7], pooled["proposed_vectors"][source][7]
        )
    pred_c = bool(
        all(_sign_pattern(pooled["proposed_vectors"][source][7])
            and union_metrics[source]["cosine"] >= .80
            and .20 <= union_metrics[source]["projection_on_parent"] <= 1.25
            and union_metrics[source]["vector_norm"] >= .01
            and union_metrics[source]["vector_norm"]
            >= 2 * abs(pooled["proposed_off_target"][source]) for source in SOURCES)
        and source_agreement["cosine"] >= .80
        and all(_metrics(half["complete_vectors"][source][7],
                         half["proposed_vectors"][source][7])["cosine"] > 0
                for half in halves for source in SOURCES)
    )

    control_comparison, control_branches = {}, {}
    pred_d = True
    for source in SOURCES:
        selected = union_metrics[source]
        control_metrics = {name: _metrics(
            pooled["complete_vectors"][source][7], pooled["control_vectors"][source][name][7]
        ) for name in CONTROL_TYPES}
        max_cos = max(row["cosine"] for row in control_metrics.values())
        max_proj = max(0.0, *(row["projection_on_parent"] for row in control_metrics.values()))
        cosine_branch = selected["cosine"] >= max_cos + .15
        projection_branch = (selected["cosine"] >= .75
                             and selected["projection_on_parent"] >= 2 * max_proj)
        branch = "cosine" if cosine_branch else "projection" if projection_branch else "failed"
        half_wins = True
        for half in halves:
            sm = _metrics(half["complete_vectors"][source][7],
                          half["proposed_vectors"][source][7])
            cms = [_metrics(half["complete_vectors"][source][7],
                            half["control_vectors"][source][name][7]) for name in CONTROL_TYPES]
            if branch == "cosine":
                half_wins &= sm["cosine"] > max(row["cosine"] for row in cms)
            elif branch == "projection":
                half_wins &= sm["projection_on_parent"] > max(
                    0.0, *(row["projection_on_parent"] for row in cms)
                )
            else:
                half_wins = False
        pred_d &= (branch != "failed" and half_wins)
        control_comparison[source] = {"selected": selected, "controls": control_metrics,
                                      "winning_branch": branch, "half_wins": half_wins}
        control_branches[source] = branch

    individual = {source: {} for source in SOURCES}
    qualifying = []
    for bit, site in enumerate(SITES):
        mask = 1 << bit
        okay = True
        source_vectors = []
        for source in SOURCES:
            row = _metrics(pooled["complete_vectors"][source][mask],
                           pooled["proposed_vectors"][source][mask])
            individual[source][site] = row
            source_vectors.append(pooled["proposed_vectors"][source][mask])
            okay &= row["cosine"] >= .60
            okay &= all(_metrics(half["complete_vectors"][source][mask],
                                 half["proposed_vectors"][source][mask])["cosine"] > 0
                          for half in halves)
        okay &= _cosine(*source_vectors) >= .70
        if okay:
            qualifying.append(site)
    interaction = {}
    for source in SOURCES:
        union = torch.tensor(pooled["proposed_vectors"][source][7])
        summed = sum(torch.tensor(pooled["proposed_vectors"][source][1 << bit])
                     for bit in range(3))
        residual = union - summed
        interaction[source] = {
            "vector": residual.tolist(), "norm": float(torch.linalg.vector_norm(residual)),
            "relative_to_union": float(torch.linalg.vector_norm(residual)
                                       / max(float(torch.linalg.vector_norm(union)), 1e-30)),
        }
    additive = all(interaction[source]["relative_to_union"] <= .25 for source in SOURCES)
    interaction_cosine = _cosine(interaction["N"]["vector"], interaction["H"]["vector"])
    interactive = bool(all(interaction[source]["norm"] >= .01 for source in SOURCES)
                       and interaction_cosine >= .80)
    regime = "approximately_additive" if additive else "stably_interactive" if interactive else "unresolved"
    pred_e = bool(len(qualifying) >= 2 and regime != "unresolved")
    all_empty = total == 0
    both_tiny = all(union_metrics[source]["vector_norm"] < .005 for source in SOURCES)
    loses_both = all(
        union_metrics[source]["cosine"]
        <= min(_metrics(pooled["complete_vectors"][source][7],
                        pooled["control_vectors"][source][name][7])["cosine"]
               for name in CONTROL_TYPES)
        and union_metrics[source]["projection_on_parent"]
        <= min(_metrics(pooled["complete_vectors"][source][7],
                        pooled["control_vectors"][source][name][7])["projection_on_parent"]
               for name in CONTROL_TYPES)
        for source in SOURCES
    )
    strong_science_null = bool(all_empty or both_tiny or source_agreement["cosine"] <= 0 or loses_both)
    return {
        "pooled": pooled, "halves": halves, "selected_counts": counts_by,
        "selected_total": total, "stable_module_count": stable_modules,
        "union_metrics": union_metrics, "union_source_agreement": source_agreement,
        "control_comparison": control_comparison,
        "individual_module_metrics": individual, "qualifying_modules": qualifying,
        "composition": {"regime": regime, "approximately_additive": additive,
                        "stably_interactive": interactive,
                        "source_interaction_cosine": interaction_cosine,
                        "source_interactions": interaction},
        "pred_b_stable_split": pred_b, "pred_c_exact_heldout_correction": pred_c,
        "pred_d_beats_matched_controls": bool(pred_d),
        "pred_e_cross_module_composition": pred_e,
        "strong_science_null": strong_science_null,
    }


def _parent_validation_error(analysis, parent_result):
    expected = parent_result["analysis"]["halves"][1]["subset_vectors"]
    maximum = 0.0
    for source in SOURCES:
        observed = analysis["pooled"]["complete_vectors"][source][7]
        reference = expected[source]["7"]
        maximum = max(maximum, max(abs(a - b) for a, b in zip(observed, reference)))
    return maximum


def main():
    started = time.time()
    payload, masks, scale, metadata, parent_result = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 467, "model_loaded": False,
            "new_product_fingerprints_opened": False,
            "new_product_interventions_opened": False, "sealed_opened": False,
            "modules": MODULES, "sources": SOURCES,
            "discovery_documents": DISCOVERY_STOP,
            "validation_documents": DOCUMENTS - VALIDATION_START,
            "expected_forwards": EXPECTED_FORWARDS, "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung467 result namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    rows = payload["rows"]
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, DOCUMENTS, BATCH):
            tokens = rows[start:start + BATCH, :-1].to(device)
            native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
            parent.parent.path_parent.parent._record_audit(
                audit_totals, "rung467:native", audit, analytical=False, captures=0, patches=0,
            )
            replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
            parent.parent.path_parent.parent._record_audit(
                audit_totals, "rung467:replay", audit, analytical=True, captures=0, patches=0,
            )
            difference = replay_logits - native
            replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
            replay["relative_squared"] = max(
                replay["relative_squared"],
                float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
            )
            reconstruction = max(reconstruction, error)
    numerators, discovery_counts, amplitude, discovery_error = _discovery(
        model, rows, masks, scale, audit_totals,
    )
    reconstruction = max(reconstruction, discovery_error)
    groups, controls, selection_report, pooled_fingerprints, half_fingerprints = select_groups(
        numerators, discovery_counts, amplitude,
    )
    full, proposed, complete, control, counts, validation_error = _validation(
        model, rows, masks, scale, groups, controls, audit_totals,
    )
    reconstruction = max(reconstruction, validation_error)
    analysis = analyze(full, proposed, complete, control, counts, selection_report)
    parent_error = _parent_validation_error(analysis, parent_result)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    empty_error = max(abs(
        analysis["pooled"]["proposed_vectors"][source][0][ci]
    ) for source in SOURCES for ci in range(len(CONTEXT_CELLS)))
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and parent_error <= 1e-10 and empty_error <= 1e-12
        and forwards == EXPECTED_FORWARDS
    )
    strong_null = bool(not pred_a or analysis["strong_science_null"])
    result = {
        "status": "complete", "rung": 467,
        "claim_level": "heldout_within_code_product_term_causal_group_test",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "selection": selection_report,
        "fingerprint_summary": {
            "pooled_shape": list(pooled_fingerprints.shape),
            "half_shape": list(half_fingerprints.shape),
            "discovery_cell_counts": discovery_counts,
            "raw_rows_tokens_logits_or_hidden_states_included": False,
        },
        "analysis": analysis,
        "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "parent_validation_task_group_max_abs_error_nat": parent_error,
        "empty_group_max_abs_effect_nat": empty_error,
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": forwards, "discovery_backwards": (DISCOVERY_STOP // BATCH) * 8,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_stable_split': analysis["pred_b_stable_split"],
        'pred_c_exact_heldout_correction': analysis["pred_c_exact_heldout_correction"],
        'pred_d_beats_matched_controls': analysis["pred_d_beats_matched_controls"],
        'pred_e_cross_module_composition': analysis["pred_e_cross_module_composition"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "fresh_corpus_confirmation_without_refitting"
            if pred_a and all(analysis[key] for key in (
                "pred_b_stable_split", "pred_c_exact_heldout_correction",
                "pred_d_beats_matched_controls", "pred_e_cross_module_composition",
            )) else "class_projected_full_bilinear_form_or_state_causal_quotient_no_topk_rescue"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 467,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "selected_counts": analysis["selected_counts"],
        "selected_total": analysis["selected_total"],
        "union_metrics": analysis["union_metrics"],
        "source_agreement": analysis["union_source_agreement"],
        "controls": analysis["control_comparison"],
        "qualifying_modules": analysis["qualifying_modules"],
        "composition": analysis["composition"],
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "parent_error": parent_error, "empty_error": empty_error},
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
