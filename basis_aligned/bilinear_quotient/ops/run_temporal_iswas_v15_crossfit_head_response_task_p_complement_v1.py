#!/usr/bin/env python3
"""Cross-fitted linear task/P-nuisance splits inside four v15 attention heads."""

# BQGATE: EXPERIMENT pred_a_authority_crossfit_projectors_closure_finiteness_and_exact_price pred_b_p_complement_svd8_reduces_aligned_p_collateral pred_c_p_complement_svd8_preserves_target_transfer pred_d_p_complement_svd8_is_selective pred_e_analytic_rank1_is_sufficient pred_f_control_complement_improves_matched_rank_tradeoff
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1 as greedy


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_crossfit_head_response_task_p_complement_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
LATTICE_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_aligned_control_attention_lattice_v1_result.json"
LATTICE_RUNNER = ROOT / "ops/run_temporal_iswas_v15_aligned_control_attention_lattice_v1.py"
HEAD_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1_result.json"
HEAD_RUNNER = ROOT / "ops/run_temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_crossfit_head_response_task_p_complement_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_crossfit_head_response_task_p_complement_v1"
ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
HEADS = ("L8H1", "L9H1", "L9H4", "L11H3")
COMPLETE_BRANCH = "attn:15"
MODES = ("dim1", "p_complement_dim1", "task_svd8", "p_complement_svd8")
TARGET_PANELS = ("A1", "A2")
CONTROL_PANELS = ("P", "C")
EXACT_FORWARDS = 17
RANK = 8
RELATIVE_SINGULAR_TOLERANCE = 1e-6
EXPECTED = {
    "prior": "5154367054f754d47c79b026cbdd75be09ecb593c5abe1a9a85390e341d7e7a6",
    "builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "lattice_result": "8eee3c031f9e7bae8bac578bd6fdd368f19fc5f027e1f184d58c7f325c4fee43",
    "lattice_runner": "bb1f9d75de5433a508604e3e8cbc9d527b7ec8280c12dedb36b377ab9ce2a5ae",
    "head_result": "cf2427059f698eac94a9e733ff873e1892a49882f60bc587c3ac83c681507025",
    "head_runner": "eb89c0948b03f9df15ce35cdbf07c8483c87a3404b8dd36efb6ebc1a873b2282",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def top_basis(torch, matrix, maximum_rank):
    matrix = matrix.float()
    if matrix.numel() == 0:
        return matrix.new_zeros((matrix.shape[-1], 0)), []
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or float(singular[0]) == 0.0:
        return matrix.new_zeros((matrix.shape[-1], 0)), [float(value) for value in singular]
    live = int((singular > singular[0] * RELATIVE_SINGULAR_TOLERANCE).sum())
    rank = min(maximum_rank, live)
    return vh[:rank].T.contiguous(), [float(value) for value in singular]


def main() -> None:
    paths = {
        "prior": PRIOR, "builder": BUILDER, "lattice_result": LATTICE_RESULT,
        "lattice_runner": LATTICE_RUNNER, "head_result": HEAD_RESULT, "head_runner": HEAD_RUNNER,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    lattice = json.loads(LATTICE_RESULT.read_text())
    rows = fresh.build_rows()
    alignment = derive_full_sequence_alignment_contract(
        rows, required_panels=TARGET_PANELS + CONTROL_PANELS
    )
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and fresh.validate_rows(rows) == ROWS_SHA256
        and alignment["panel_counts"] == {panel: 16 for panel in TARGET_PANELS + CONTROL_PANELS}
        and lattice.get("terminal") == "aligned_no_selective_attention_subset"
        and lattice.get("components") == list(HEADS + (COMPLETE_BRANCH,))
        and len(rows) == len({row["row_id"] for row in rows}) == 64
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "crossfit_folds": 2, "modes": list(MODES), "maximum_rank": RANK,
        "model_forwards_exact": EXACT_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    native, forwards = backend.native, 0

    def counted(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return native(*args, **kwargs)

    backend.native = counted
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_cache = greedy.capture(backend, base_batch)
    donor_output, donor_cache = greedy.capture(backend, donor_batch)
    base_state = greedy.module_impl.states(torch, backend, base_output, rows)
    donor_state = greedy.module_impl.states(torch, backend, donor_output, rows)
    base_logits = das.head_logits(backend, base_state).float()
    donor_logits = das.head_logits(backend, donor_state).float()
    width = int(backend.model.config.n_embd // backend.model.config.n_head)

    def response_delta(site):
        layer, head = greedy.parse_head(site)
        left, right = head * width, (head + 1) * width
        key = f"head_layer:{layer}"
        return donor_cache[key][..., left:right] - base_cache[key][..., left:right]

    projectors, projector_receipts = {}, []
    for evaluation_parity in (0, 1):
        training_parity = 1 - evaluation_parity
        target_indices = [
            index for index, row in enumerate(rows)
            if row["transform_id"] == "A1" and row["group_number"] % 2 == training_parity
        ]
        nuisance_indices = [
            index for index, row in enumerate(rows)
            if row["transform_id"] == "P" and row["group_number"] % 2 == training_parity
        ]
        evaluation_ids = {
            row["row_id"] for row in rows if row["group_number"] % 2 == evaluation_parity
        }
        fit_ids = {rows[index]["row_id"] for index in target_indices + nuisance_indices}
        if fit_ids & evaluation_ids:
            raise RuntimeError("crossfit leakage")
        for site in HEADS:
            delta = response_delta(site)

            def stack(indices, oriented=False):
                values = []
                for index in indices:
                    stop = int(base_batch.semantic_positions[index]) + 1
                    value = delta[index, :stop].float()
                    if oriented:
                        sign = 1.0 if rows[index]["direction_id"] == "present_to_past" else -1.0
                        value = value * sign
                    values.append(value)
                return torch.cat(values, dim=0)

            target = stack(target_indices)
            oriented_target = stack(target_indices, oriented=True)
            nuisance = stack(nuisance_indices)
            nuisance_basis, nuisance_singular = top_basis(torch, nuisance, RANK)
            task_basis, task_singular = top_basis(torch, target, RANK)
            dim = oriented_target.mean(0)
            dim_norm = float(torch.linalg.vector_norm(dim))
            dim_basis = dim[:, None] / dim_norm if dim_norm > 0 else dim.new_zeros((width, 0))
            residual_target = target - (target @ nuisance_basis) @ nuisance_basis.T
            complement_basis, complement_singular = top_basis(torch, residual_target, RANK)
            complement_dim = dim - nuisance_basis @ (nuisance_basis.T @ dim)
            complement_dim_norm = float(torch.linalg.vector_norm(complement_dim))
            complement_dim_basis = (
                complement_dim[:, None] / complement_dim_norm
                if complement_dim_norm > dim_norm * RELATIVE_SINGULAR_TOLERANCE
                else dim.new_zeros((width, 0))
            )
            bases = {
                "dim1": dim_basis, "p_complement_dim1": complement_dim_basis,
                "task_svd8": task_basis, "p_complement_svd8": complement_basis,
            }
            for mode, basis in bases.items():
                projectors[(evaluation_parity, site, mode)] = basis
            projector_receipts.append({
                "evaluation_parity": evaluation_parity, "training_parity": training_parity,
                "site": site, "fit_row_ids": sorted(fit_ids), "evaluation_row_ids": sorted(evaluation_ids),
                "ranks": {mode: int(basis.shape[1]) for mode, basis in bases.items()},
                "dim_norm": dim_norm, "complement_dim_norm": complement_dim_norm,
                "nuisance_singular_values": nuisance_singular,
                "task_singular_values": task_singular,
                "complement_singular_values": complement_singular,
            })

    def projected_patch(source_cache, mode, sites, include_complete_branch):
        sites = set(sites)
        handles = []
        by_layer = {}
        for site in sites:
            layer, head = greedy.parse_head(site)
            by_layer.setdefault(layer, []).append((site, head * width, (head + 1) * width))
        for layer, entries in by_layer.items():
            key = f"head_layer:{layer}"

            def hook(_module, arguments, entries=entries, key=key):
                changed = arguments[0].clone()
                for index, row in enumerate(rows):
                    stop = int(base_batch.semantic_positions[index]) + 1
                    parity = int(row["group_number"]) % 2
                    for site, left, right in entries:
                        basis = projectors[(parity, site, mode)].to(changed)
                        delta = source_cache[key][index, :stop, left:right].to(changed) - base_cache[key][index, :stop, left:right].to(changed)
                        projected = (delta @ basis) @ basis.T if basis.shape[1] else torch.zeros_like(delta)
                        changed[index, :stop, left:right] = base_cache[key][index, :stop, left:right].to(changed) + projected
                return (changed,) + tuple(arguments[1:])

            handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
        if include_complete_branch:
            handles.append(greedy.site_module(backend, COMPLETE_BRANCH).register_forward_pre_hook(
                greedy.prefix_patch(base_batch, source_cache[COMPLETE_BRANCH])
            ))
        try:
            return backend.native(base_batch, capture=True)
        finally:
            for handle in handles:
                handle.remove()

    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)
    panel_indices = {
        panel: torch.as_tensor([i for i, row in enumerate(rows) if row["transform_id"] == panel], device=backend.device)
        for panel in TARGET_PANELS + CONTROL_PANELS
    }

    def margins(logits):
        return logits[all_index, answer] - logits[all_index, foil]

    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def control_report(logits, panel):
        index = panel_indices[panel]
        log_base, log_patch = F.log_softmax(base_logits[index], -1), F.log_softmax(logits[index], -1)
        kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
        flips = base_logits[index].argmax(-1) != logits[index].argmax(-1)
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        return {"median_kl": float(kl.median()), "max_kl": float(kl.max()),
                "top1_flip_count": int(flips.sum()), "top1_flip_fraction": float(flips.float().mean()),
                "flipped_row_ids": [panel_rows[i]["row_id"] for i in range(len(panel_rows)) if bool(flips[i])]}

    def report(output):
        state = greedy.module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets = {}
        for panel in TARGET_PANELS:
            index = panel_indices[panel]
            behavior = greedy.module_impl.vector_metrics(torch, delta_margin[index], target_margin[index])
            behavior["direction_fraction"] = float(((delta_margin[index] * target_margin[index]) > 0).float().mean())
            targets[panel] = {"behavior": behavior,
                "final_residual": greedy.module_impl.vector_metrics(torch, delta_state[index], target_state[index])}
        return {"targets": targets, "controls": {panel: control_report(logits, panel) for panel in CONTROL_PANELS}}

    def selective(value):
        return bool(all(value["targets"][panel]["behavior"]["signed_projection"] >= .75
                        and value["targets"][panel]["behavior"]["direction_fraction"] >= .875
                        for panel in TARGET_PANELS)
                    and all(value["controls"][panel]["top1_flip_count"] == 0 for panel in CONTROL_PANELS)
                    and max(value["controls"][panel]["median_kl"] for panel in CONTROL_PANELS) <= .02)

    self_output = projected_patch(base_cache, "p_complement_svd8", HEADS, True)
    self_state = greedy.module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(float((self_state - base_state).abs().max()), float((self_logits - base_logits).abs().max()))
    arms = {
        "complete_parent": report(greedy.run_patch(backend, base_batch, donor_cache, HEADS + (COMPLETE_BRANCH,))),
        "complete_attn15_only": report(greedy.run_patch(backend, base_batch, donor_cache, (COMPLETE_BRANCH,))),
    }
    for mode in MODES:
        arms[f"composed_{mode}"] = report(projected_patch(donor_cache, mode, HEADS, True))
    for mode in ("task_svd8", "p_complement_svd8"):
        for site in HEADS:
            arms[f"singleton_{site}_{mode}"] = report(projected_patch(donor_cache, mode, (site,), False))
    selective_arms = {name: selective(value) for name, value in arms.items() if name.startswith("composed_")}
    ranks_ok = all(0 <= rank <= RANK for receipt in projector_receipts for rank in receipt["ranks"].values())
    orthogonality_error = max((
        float(((basis.T @ basis) - torch.eye(basis.shape[1], device=basis.device)).abs().max())
        for basis in projectors.values() if basis.shape[1]
    ), default=0.0)
    full, complement, task = arms["complete_parent"], arms["composed_p_complement_svd8"], arms["composed_task_svd8"]
    pred_a = bool(authority_ok and ranks_ok and orthogonality_error <= 1e-4 and self_error <= 1e-4
                  and finite(arms) and forwards == EXACT_FORWARDS)
    pred_b = bool(complement["controls"]["P"]["median_kl"] <= .5 * full["controls"]["P"]["median_kl"]
                  and complement["controls"]["P"]["top1_flip_count"] <= 1)
    pred_c = all(complement["targets"][panel]["behavior"]["signed_projection"] >= .75
                 and complement["targets"][panel]["behavior"]["direction_fraction"] >= .875
                 for panel in TARGET_PANELS)
    pred_d = selective_arms["composed_p_complement_svd8"]
    pred_e = any(selective_arms[f"composed_{mode}"] for mode in ("dim1", "p_complement_dim1"))
    pred_f = bool(min(complement["targets"][panel]["behavior"]["signed_projection"] for panel in TARGET_PANELS)
                  >= min(task["targets"][panel]["behavior"]["signed_projection"] for panel in TARGET_PANELS) - .05
                  and complement["controls"]["P"]["median_kl"] < task["controls"]["P"]["median_kl"])
    predictions = {
        "pred_a_authority_crossfit_projectors_closure_finiteness_and_exact_price": pred_a,
        "pred_b_p_complement_svd8_reduces_aligned_p_collateral": pred_b,
        "pred_c_p_complement_svd8_preserves_target_transfer": pred_c,
        "pred_d_p_complement_svd8_is_selective": pred_d,
        "pred_e_analytic_rank1_is_sufficient": pred_e,
        "pred_f_control_complement_improves_matched_rank_tradeoff": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif pred_e:
        terminal = "crossfit_rank1_selective_response_split"
    elif pred_b and pred_c and pred_d and pred_f:
        terminal = "crossfit_p_complement_selective_response_split"
    elif selective_arms["composed_task_svd8"]:
        terminal = "crossfit_task_span_selective_without_complement"
    elif not any(selective_arms.values()):
        terminal = "linear_response_split_insufficient"
    else:
        terminal = "partial"
    result = {
        "schema": "temporal_iswas_v15_crossfit_head_response_task_p_complement_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256, "alignment_contract": alignment,
        "projector_spec": {"maximum_rank": RANK, "relative_singular_tolerance": RELATIVE_SINGULAR_TOLERANCE,
                           "modes": list(MODES), "fit_panel": "A1", "nuisance_panel": "P"},
        "projector_receipts": projector_receipts,
        "instrument": {"base_self_max_abs_error": self_error, "projector_orthogonality_max_abs_error": orthogonality_error},
        "arms": arms, "selective_arms": selective_arms, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_exact": EXACT_FORWARDS, "model_forwards_observed": forwards,
                  "example_evaluations": forwards * len(rows), "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0}, "dryrun": dryrun,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "instrument", "selective_arms", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
