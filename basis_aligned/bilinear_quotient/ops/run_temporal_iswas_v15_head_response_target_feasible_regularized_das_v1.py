#!/usr/bin/env python3
"""Cross-fitted target-feasible DAS inside the four material v15 attention heads."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_absolute_clamp_gradient_closure_finiteness_and_price pred_b_target_feasible_regularized_checkpoint_exists pred_c_regularization_beats_dim_on_opposite_parity pred_d_projector_is_fold_stable pred_e_sealed_a2_transfers pred_f_sealed_controls_are_selective pred_g_learned_projector_beats_exact_factor_baselines
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
import attention_source_destination_eval as attention_eval
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
from head_response_projector_contract import (
    absolute_projected_head_response,
    orthonormal_basis,
    principal_angle_cosines,
    projector_frobenius_distance,
)
import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factor_parent
import run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1 as greedy


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.json"
DUAL_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_head_factor_dual_greedy_v1_result.json"
DUAL_RUNNER = ROOT / "ops/run_temporal_iswas_v15_head_factor_dual_greedy_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
CLAMP_CONTRACT = ROOT / "ops/absolute_head_response_clamp_contract.py"
PROJECTOR_CONTRACT = ROOT / "ops/head_response_projector_contract.py"
FACTOR_RUNNER = ROOT / "ops/run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1.py"
GREEDY_RUNNER = ROOT / "ops/run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_head_response_target_feasible_regularized_das_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_head_response_target_feasible_regularized_das_v1"
ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
EXPECTED = {
    "prior": "e1f4031ec8445b2fb97c76d2f89348320fc16a1ed7a27c404a3b8e0e5ee27ac4",
    "dual_result": "51c705281cc4fa10aa7d5c3c54bc3b1ee1e9d6d3af3e74f3b98cef3a34ad0285",
    "dual_runner": "8c02ee2a04faad82c3341667f457c30537f351c9147a26dadd7b19736a5d3a48",
    "builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "clamp_contract": "b16b86b23d50b1284eecaa01855301d50ced82e2112bf44167b9596c0b7fd0f3",
    "projector_contract": "23dbdd6380b745683f502ab0678b6e65e9168b5485eb488d759fa6eebbdc23ef",
    "factor_runner": "3424f394930cd9a0c25f05b5620f17fb37cfb852ce4e07154c8f4a8f979d64ab",
    "greedy_runner": "de7906f02a1739c0ef01d08210a68140bfcac3a496a3d62780bfc024423f6954",
}
HEADS = ((8, 1), (9, 1), (9, 4), (11, 3))
HEAD_NAMES = tuple(f"L{layer}H{head}" for layer, head in HEADS)
HEADS_BY_LAYER = {8: (1,), 9: (1, 4), 11: (3,)}
TARGET_PANELS = ("A1", "A2")
CONTROL_PANELS = ("P", "C")
RANKS = (1, 2, 4)
INITIALIZATIONS = ("dim_task_svd", "factor_svd")
NOISE_JACOBIAN = ((0.0, 0.0), (0.05, 0.25), (0.05, 1.0), (0.10, 0.25), (0.10, 1.0))
STABILITY_WEIGHT = 0.25
STEPS, CHECKPOINTS, LR = 8, (0, 4, 8), 0.03
BARRIER, TARGET_MATCH_WEIGHT, TAU = 100.0, 0.10, 0.05
TARGET_PROJECTION_BAR, DIRECTION_FRACTION_BAR = 0.75, 0.875
CONTROL_MEDIAN_KL_BAR = 0.02
STABILITY_MIN_COSINE, STABILITY_MAX_NORMALIZED_FROBENIUS = 0.70, 0.70
PRICE_MAX = {
    "native_capture_forwards": 10,
    "differentiable_transformer_forwards": 1800,
    "transformer_backward_forwards": 1250,
    "model_updates": 480,
    "example_evaluations": 30000,
    "fit_parameters": 2048,
}


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def configuration_grid():
    return tuple(
        {"rank": rank, "initialization": initialization,
         "noise_sigma": sigma, "jacobian_weight": weight,
         "name": f"k{rank}_{initialization}_noise{sigma:.2f}_jac{weight:.2f}"}
        for rank in RANKS
        for initialization in INITIALIZATIONS
        for sigma, weight in NOISE_JACOBIAN
    )


def row_indices(rows, *, panels=None, parity=None):
    allowed = None if panels is None else set(panels)
    return tuple(
        index for index, row in enumerate(rows)
        if (allowed is None or row["transform_id"] in allowed)
        and (parity is None or int(row["group_number"]) % 2 == parity)
    )


def _subbatch(batch, indices):
    cls = producer.ModelBatch
    return cls(
        row_ids=tuple(batch.row_ids[i] for i in indices), side=batch.side,
        token_rows=tuple(batch.token_rows[i] for i in indices),
        answer_ids=tuple(batch.answer_ids[i] for i in indices),
        foil_ids=tuple(batch.foil_ids[i] for i in indices),
        semantic_positions=tuple(batch.semantic_positions[i] for i in indices),
    )


def _slice_tensor_map(values, indices, length):
    return {key: value[list(indices), :length].detach() for key, value in values.items()}


def _top_right_basis(torch, matrix, rank):
    matrix = matrix.float()
    if matrix.ndim != 2 or matrix.shape[1] != 128:
        raise ExperimentError("basis matrix must have native head width")
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    live = int((singular > singular[0].clamp_min(1e-30) * 1e-7).sum()) if len(singular) else 0
    columns = vh[:min(rank, live)].T.contiguous()
    if columns.shape[1] < rank:
        eye = torch.eye(matrix.shape[1], device=matrix.device)
        candidates = eye - columns @ (columns.T @ eye) if columns.shape[1] else eye
        _u2, _s2, vh2 = torch.linalg.svd(candidates.T, full_matrices=False)
        columns = torch.cat((columns, vh2[:rank - columns.shape[1]].T), dim=1)
    basis = orthonormal_basis(torch, columns[:, :rank])
    return basis, [float(value) for value in singular.detach().cpu()]


def _dim_task_basis(torch, matrix, rank):
    mean = matrix.float().mean(0)
    if float(mean.norm()) <= 1e-12:
        return _top_right_basis(torch, matrix, rank)
    first = mean[:, None] / mean.norm()
    residual = matrix.float() - (matrix.float() @ first) @ first.T
    if rank == 1:
        _u, singular, _vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        return first, [float(value) for value in singular.detach().cpu()]
    rest, singular = _top_right_basis(torch, residual, rank - 1)
    return orthonormal_basis(torch, torch.cat((first, rest), dim=1)), singular


def stack_head_prefix(torch, tensor, rows, indices, head):
    pieces = [tensor[index, :int(rows[index]["base_semantic_position"]) + 1, head].float()
              for index in indices]
    if not pieces:
        raise ExperimentError("empty response stack")
    return torch.cat(pieces, dim=0)


def factor_tensors(torch, base_capture, donor_capture):
    dp = donor_capture["pattern"].float() - base_capture["pattern"].float()
    dv = donor_capture["value"].float() - base_capture["value"].float()
    return {
        "pattern_on_base_value": torch.einsum(
            "bhqk,bkhd->bqhd", dp, base_capture["value"].float()),
        "base_pattern_on_value_change": torch.einsum(
            "bhqk,bkhd->bqhd", base_capture["pattern"].float(), dv),
        "pattern_value_interaction": torch.einsum("bhqk,bkhd->bqhd", dp, dv),
    }


def capture_bank(backend, rows, counters, *, factors):
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_cache = greedy.capture(backend, base_batch)
    donor_output, donor_cache = greedy.capture(backend, donor_batch)
    bank = {
        "rows": list(rows), "base_batch": base_batch, "donor_batch": donor_batch,
        "base_output": base_output, "donor_output": donor_output,
        "base_cache": base_cache, "donor_cache": donor_cache, "factors": {},
        "attention_reconstruction_max_abs_error": 0.0,
        "factor_closure_max_abs_error": 0.0,
    }
    if factors:
        captures = {"base": {}, "donor": {}}
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            for layer in HEADS_BY_LAYER:
                _output, capture = attention_eval.capture_layer_attention(backend, batch, layer)
                captures[side][layer] = capture
                bank["attention_reconstruction_max_abs_error"] = max(
                    bank["attention_reconstruction_max_abs_error"],
                    float(capture["reconstruction_max_abs"]))
        bank["factors"] = {
            layer: factor_tensors(backend.torch, captures["base"][layer], captures["donor"][layer])
            for layer in HEADS_BY_LAYER
        }
        for layer, heads in HEADS_BY_LAYER.items():
            total = sum(bank["factors"][layer].values(),
                        backend.torch.zeros_like(captures["base"][layer]["head_output"]).float())
            actual = (captures["donor"][layer]["head_output"].float()
                      - captures["base"][layer]["head_output"].float())
            for index, row in enumerate(rows):
                stop = int(row["base_semantic_position"]) + 1
                for head in heads:
                    bank["factor_closure_max_abs_error"] = max(
                        bank["factor_closure_max_abs_error"],
                        float((total[index, :stop, head] - actual[index, :stop, head]).abs().max()))
    return bank


def subset_context(bank, indices):
    indices = tuple(indices)
    base_batch, donor_batch = _subbatch(bank["base_batch"], indices), _subbatch(bank["donor_batch"], indices)
    length = max(len(row) for row in base_batch.token_rows)
    rows = [bank["rows"][index] for index in indices]
    return {
        "rows": rows, "base_batch": base_batch, "donor_batch": donor_batch,
        "base_cache": _slice_tensor_map(bank["base_cache"], indices, length),
        "donor_cache": _slice_tensor_map(bank["donor_cache"], indices, length),
        "base_answer_foil": tuple(bank["base_output"].answer_foil[index] for index in indices),
        "donor_answer_foil": tuple(bank["donor_output"].answer_foil[index] for index in indices),
        "source_indices": indices,
    }


def manual_forward(backend, batch, counters, *, context=None, raw_by_site=None,
                   noise_by_site=None, complete15=False, grad=False):
    torch, F, model = backend.torch, backend.F, backend.model
    counters["differentiable_transformer_forwards"] += 1
    counters["example_evaluations"] += len(batch.row_ids)
    if grad:
        counters["transformer_backward_forwards"] += 1
    tokens, lengths = backend._tensor_batch(batch)
    n_head, width = int(model.config.n_head), int(model.config.n_embd // model.config.n_head)
    selected = raw_by_site or {}
    noise = noise_by_site or {}
    with torch.set_grad_enabled(grad):
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            layer_sites = tuple((head, f"L{layer}H{head}") for head in HEADS_BY_LAYER.get(layer, ())
                                if f"L{layer}H{head}" in selected)

            def patch(_module, arguments, layer=layer, layer_sites=layer_sites):
                flattened = arguments[0]
                if context is None:
                    raise ExperimentError("intervention lacks a capture context")
                changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], n_head, width)
                if layer_sites:
                    base = context["base_cache"][f"head_layer:{layer}"].view_as(changed).to(changed)
                    donor = context["donor_cache"][f"head_layer:{layer}"].view_as(changed).to(changed).clone()
                    raw_heads = {}
                    for head, site in layer_sites:
                        raw_heads[head] = selected[site]
                        if site in noise:
                            donor[..., head, :] = donor[..., head, :] + noise[site].to(donor)
                    absolute = absolute_projected_head_response(torch, base, donor, raw_heads)
                    for index, stop in enumerate(batch.semantic_positions):
                        for head, _site in layer_sites:
                            changed[index, :int(stop) + 1, head] = absolute[index, :int(stop) + 1, head].to(changed)
                if complete15 and layer == 15:
                    donor15 = context["donor_cache"]["attn:15"].view_as(changed).to(changed)
                    for index, stop in enumerate(batch.semantic_positions):
                        changed[index, :int(stop) + 1] = donor15[index, :int(stop) + 1]
                return (changed.reshape_as(flattened),) + tuple(arguments[1:])

            use_hook = bool(layer_sites or (complete15 and layer == 15))
            handle = block.attn.c_proj.register_forward_pre_hook(patch) if use_hook else None
            try:
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            finally:
                if handle is not None:
                    handle.remove()
            x = live + attention
            x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
        index = torch.arange(len(lengths), device=backend.device)
        position = torch.tensor([length - 1 for length in lengths], device=backend.device)
        state = x[index, position].float()
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(state, (model.config.n_embd,))) / 30.0)
        if not grad:
            state, logits = state.detach(), logits.detach()
        return {"state": state, "logits": logits}


def attach_references(backend, context, counters):
    torch, F = backend.torch, backend.F
    base = manual_forward(backend, context["base_batch"], counters)
    donor = manual_forward(backend, context["donor_batch"], counters)
    answer = torch.tensor([row["donor_answer_id"] for row in context["rows"]], device=backend.device)
    foil = torch.tensor([row["donor_foil_id"] for row in context["rows"]], device=backend.device)
    index = torch.arange(len(context["rows"]), device=backend.device)
    margin = lambda logits: logits[index, answer] - logits[index, foil]
    panel_indices = {
        panel: torch.tensor([i for i, row in enumerate(context["rows"])
                             if row["transform_id"] == panel], device=backend.device)
        for panel in TARGET_PANELS + CONTROL_PANELS
    }
    native_base = torch.tensor(context["base_answer_foil"], device=backend.device)
    native_donor = torch.tensor(context["donor_answer_foil"], device=backend.device)
    closure = max(
        float((base["logits"][index, torch.tensor(context["base_batch"].answer_ids, device=backend.device)]
               - native_base[:, 0]).abs().max()),
        float((base["logits"][index, torch.tensor(context["base_batch"].foil_ids, device=backend.device)]
               - native_base[:, 1]).abs().max()),
        float((donor["logits"][index, torch.tensor(context["donor_batch"].answer_ids, device=backend.device)]
               - native_donor[:, 0]).abs().max()),
        float((donor["logits"][index, torch.tensor(context["donor_batch"].foil_ids, device=backend.device)]
               - native_donor[:, 1]).abs().max()),
    )
    center = lambda value: value - value.mean(-1, keepdim=True)
    a1 = panel_indices["A1"]
    scale = ((center(donor["logits"][a1]) - center(base["logits"][a1])).square().mean()
             if len(a1) else torch.tensor(1.0, device=backend.device)).clamp_min(1e-8)
    context.update({
        "base": base, "donor": donor, "base_log_probs": F.log_softmax(base["logits"], -1).detach(),
        "base_margin": margin(base["logits"]).detach(), "donor_margin": margin(donor["logits"]).detach(),
        "target_margin": (margin(donor["logits"]) - margin(base["logits"])).detach(),
        "target_state": (donor["state"] - base["state"]).detach(),
        "answer": answer, "foil": foil, "index": index, "panel_indices": panel_indices,
        "logit_response_scale": scale.detach(), "manual_native_max_abs_error": closure,
    })
    return context


def initialization_for(backend, bank, context, train_parity, rank, name):
    torch = backend.torch
    bank_rows = bank["rows"]
    indices = row_indices(bank_rows, panels=("A1",), parity=train_parity)
    receipts, bases = {}, {}
    for layer, head in HEADS:
        site, key = f"L{layer}H{head}", f"head_layer:{layer}"
        full = (bank["donor_cache"][key].view(len(bank_rows), -1, 9, 128)
                - bank["base_cache"][key].view(len(bank_rows), -1, 9, 128))
        task_matrix = stack_head_prefix(torch, full, bank_rows, indices, head)
        factor_matrix = torch.cat(tuple(
            stack_head_prefix(torch, bank["factors"][layer][factor], bank_rows, indices, head)
            for factor in attention_eval.RESPONSE_FACTORS
        ), dim=0)
        if name == "dim_task_svd":
            basis, singular = _dim_task_basis(torch, task_matrix, rank)
        elif name == "factor_svd":
            basis, singular = _top_right_basis(torch, factor_matrix, rank)
        else:
            raise ExperimentError("unknown initialization")
        bases[site] = basis.detach().clone()
        gap = (float(singular[rank - 1] - singular[rank])
               if len(singular) > rank else float(singular[rank - 1]) if len(singular) >= rank else 0.0)
        scale = float(singular[0]) if singular else 0.0
        receipts[site] = {"singular_values_first_12": singular[:12],
                          "relative_retained_gap": gap / max(scale, 1e-30)}
    return bases, receipts


def make_noise(backend, context, raw_by_site, sigma, seed):
    torch = backend.torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for layer, head in HEADS:
        site, key = f"L{layer}H{head}", f"head_layer:{layer}"
        if site not in raw_by_site:
            continue
        base = context["base_cache"][key].view(len(context["rows"]), -1, 9, 128)[..., head, :]
        donor = context["donor_cache"][key].view(
            len(context["rows"]), -1, 9, 128)[..., head, :]
        delta = donor.float() - base.float()
        rms = delta.square().mean().sqrt().clamp_min(1e-8)
        result[site] = torch.randn(delta.shape, generator=generator).to(backend.device) * rms * sigma
    return result


def output_metrics(backend, context, output):
    torch, F = backend.torch, backend.F
    logits = output["logits"]
    margin = logits[context["index"], context["answer"]] - logits[context["index"], context["foil"]]
    delta_margin = margin - context["base_margin"]
    values = {}
    for panel in TARGET_PANELS:
        selected = context["panel_indices"][panel]
        if not len(selected):
            continue
        behavior = greedy.module_impl.vector_metrics(
            torch, delta_margin[selected], context["target_margin"][selected])
        behavior["direction_fraction"] = float(
            ((delta_margin[selected] * context["target_margin"][selected]) > 0).float().mean())
        values[panel] = {
            "behavior": behavior,
            "final_residual": greedy.module_impl.vector_metrics(
                torch, output["state"][selected] - context["base"]["state"][selected],
                context["target_state"][selected]),
        }
    controls = {}
    for panel in CONTROL_PANELS:
        selected = context["panel_indices"][panel]
        if not len(selected):
            continue
        log_patch = F.log_softmax(logits[selected], -1)
        log_base = context["base_log_probs"][selected]
        kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
        flips = context["base"]["logits"][selected].argmax(-1) != logits[selected].argmax(-1)
        panel_rows = [context["rows"][int(i)] for i in selected]
        controls[panel] = {
            "mean_kl": float(kl.mean()), "median_kl": float(kl.median()), "max_kl": float(kl.max()),
            "top1_flip_count": int(flips.sum()), "top1_flip_fraction": float(flips.float().mean()),
            "flipped_row_ids": [panel_rows[i]["row_id"] for i in range(len(panel_rows)) if bool(flips[i])],
            "per_row_kl": [float(value) for value in kl.detach().cpu()],
        }
    return {"targets": values, "controls": controls}


def training_loss(backend, context, output):
    torch, F = backend.torch, backend.F
    logits = output["logits"]
    margin = logits[context["index"], context["answer"]] - logits[context["index"], context["foil"]]
    delta = margin - context["base_margin"]
    target = context["target_margin"]
    a1, p = context["panel_indices"]["A1"], context["panel_indices"]["P"]
    projection = (delta[a1] @ target[a1]) / target[a1].square().sum().clamp_min(1e-12)
    signed_ratio = delta[a1] * target[a1].sign() / target[a1].abs().clamp_min(1e-6)
    ratio = signed_ratio
    soft_direction = torch.sigmoid(20.0 * signed_ratio).mean()
    target_match = (ratio - 1.0).square().mean()
    log_patch = F.log_softmax(logits[p], -1)
    log_base = context["base_log_probs"][p]
    kl_rows = (log_base.exp() * (log_base - log_patch)).sum(-1)
    worst_p = TAU * torch.logsumexp(kl_rows / TAU, 0) - TAU * math.log(len(kl_rows))
    violation = torch.relu(TARGET_PROJECTION_BAR - projection).square()
    violation = violation + torch.relu(DIRECTION_FRACTION_BAR - soft_direction).square()
    return worst_p + TARGET_MATCH_WEIGHT * target_match + BARRIER * violation


def sensitivity_variant(backend, context, clean_logits, raw_by_site, noise, sigma, sign,
                        counters, *, grad):
    torch = backend.torch
    center = lambda value: value - value.mean(-1, keepdim=True)
    signed = {site: sign * value for site, value in noise.items()}
    output = manual_forward(
        backend, context["base_batch"], counters, context=context,
        raw_by_site=raw_by_site, noise_by_site=signed, complete15=True, grad=grad)
    return ((center(output["logits"]) - center(clean_logits)).square().mean()
            / (context["logit_response_scale"] * sigma * sigma))


def sensitivity_loss(backend, context, clean_logits, raw_by_site, sigma, seed, counters, *, grad):
    torch = backend.torch
    if sigma == 0.0:
        return torch.tensor(0.0, device=backend.device)
    noise = make_noise(backend, context, raw_by_site, sigma, seed)
    return sum(sensitivity_variant(
        backend, context, clean_logits, raw_by_site, noise, sigma, sign, counters, grad=grad)
               for sign in (-1.0, 1.0)) / 2.0


def checkpoint_report(backend, context, raw_by_site, configuration, step, fold, counters):
    torch = backend.torch
    with torch.no_grad():
        bases = {site: orthonormal_basis(torch, raw).detach().clone()
                 for site, raw in raw_by_site.items()}
        output = manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases, complete15=True)
        metrics = output_metrics(backend, context, output)
        a1, p = metrics["targets"]["A1"]["behavior"], metrics["controls"]["P"]
        sensitivity = float(sensitivity_loss(
            backend, context, output["logits"], bases, configuration["noise_sigma"],
            910000 + 1000 * fold + 10 * step + configuration_grid().index(configuration), counters,
            grad=False)) if configuration["jacobian_weight"] else 0.0
        feasible = bool(a1["signed_projection"] >= TARGET_PROJECTION_BAR
                        and a1["direction_fraction"] >= DIRECTION_FRACTION_BAR)
        violation = (max(0.0, TARGET_PROJECTION_BAR - a1["signed_projection"])
                     + max(0.0, DIRECTION_FRACTION_BAR - a1["direction_fraction"]))
        report = {
            "step": step, "feasible": feasible, "target_violation": violation,
            "a1_signed_projection": a1["signed_projection"],
            "a1_direction_fraction": a1["direction_fraction"],
            "p_mean_kl": p["mean_kl"], "p_median_kl": p["median_kl"],
            "p_max_kl": p["max_kl"], "p_top1_flip_count": p["top1_flip_count"],
            "jacobian_sensitivity": sensitivity,
            "secondary": p["mean_kl"] + configuration["jacobian_weight"] * sensitivity,
        }
        return report, bases


def fit_one(backend, train, select, initial, init_receipt, configuration, fold, counters):
    torch = backend.torch
    raw = {site: value.detach().clone().requires_grad_(True) for site, value in initial.items()}
    optimizer = torch.optim.Adam(tuple(raw.values()), lr=LR)
    trace, snapshots, gradient_max = [], {}, 0.0

    def checkpoint(step):
        report, bases = checkpoint_report(backend, select, raw, configuration, step, fold, counters)
        trace.append(report)
        snapshots[step] = bases

    checkpoint(0)
    for update in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        clean = manual_forward(
            backend, train["base_batch"], counters, context=train,
            raw_by_site=raw, complete15=True, grad=True)
        training_loss(backend, train, clean).backward()
        if configuration["jacobian_weight"]:
            noise = make_noise(
                backend, train, raw, configuration["noise_sigma"],
                810000 + 10000 * fold + 100 * configuration_grid().index(configuration) + update)
            for sign in (-1.0, 1.0):
                sensitivity = sensitivity_variant(
                    backend, train, clean["logits"].detach(), raw, noise,
                    configuration["noise_sigma"], sign, counters, grad=True)
                (0.5 * configuration["jacobian_weight"] * sensitivity).backward()
        gradient_max = max(gradient_max, max(
            float(value.grad.abs().max()) if value.grad is not None else 0.0
            for value in raw.values()))
        optimizer.step()
        counters["model_updates"] += 1
        if update in CHECKPOINTS:
            checkpoint(update)
    eligible = [item for item in trace if item["feasible"]]
    best_report = (min(eligible, key=lambda item: (item["secondary"], item["step"]))
                   if eligible else min(trace, key=lambda item: (
                       item["target_violation"], item["secondary"], item["step"])))
    return {
        "configuration": configuration, "fold": fold, "trace": trace,
        "best": best_report, "best_bases": snapshots[best_report["step"]],
        "initial_bases": snapshots[0], "initialization_receipt": init_receipt,
        "gradient_max_abs": gradient_max,
    }


def stability_report(backend, left, right, receipts, rank):
    torch = backend.torch
    by_head = {}
    for site in HEAD_NAMES:
        cosine = principal_angle_cosines(torch, left[site], right[site])
        distance = projector_frobenius_distance(torch, left[site], right[site])
        normalized = float(distance / math.sqrt(2.0 * rank))
        by_head[site] = {
            "principal_angle_cosines": [float(value) for value in cosine.detach().cpu()],
            "minimum_principal_cosine": float(cosine.min()),
            "projector_frobenius": float(distance), "normalized_projector_frobenius": normalized,
            "relative_retained_gaps": [receipt[site]["relative_retained_gap"] for receipt in receipts],
        }
    return {
        "by_head": by_head,
        "minimum_principal_cosine": min(item["minimum_principal_cosine"] for item in by_head.values()),
        "mean_normalized_projector_frobenius": sum(
            item["normalized_projector_frobenius"] for item in by_head.values()) / len(by_head),
        "maximum_normalized_projector_frobenius": max(
            item["normalized_projector_frobenius"] for item in by_head.values()),
    }


def strip_fit(value):
    return {key: item for key, item in value.items()
            if key not in {"best_bases", "initial_bases"}}


def combine_reports(backend, contexts_outputs):
    torch, F = backend.torch, backend.F
    target_parts = {panel: [] for panel in TARGET_PANELS}
    control_parts = {panel: [] for panel in CONTROL_PANELS}
    for context, output in contexts_outputs:
        logits = output["logits"]
        margin = logits[context["index"], context["answer"]] - logits[context["index"], context["foil"]]
        for panel in TARGET_PANELS:
            selected = context["panel_indices"][panel]
            if len(selected):
                target_parts[panel].append((
                    margin[selected] - context["base_margin"][selected],
                    context["target_margin"][selected],
                    output["state"][selected] - context["base"]["state"][selected],
                    context["target_state"][selected],
                ))
        for panel in CONTROL_PANELS:
            selected = context["panel_indices"][panel]
            if len(selected):
                log_patch, log_base = F.log_softmax(logits[selected], -1), context["base_log_probs"][selected]
                kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
                flips = context["base"]["logits"][selected].argmax(-1) != logits[selected].argmax(-1)
                rows = [context["rows"][int(i)] for i in selected]
                control_parts[panel].extend((int(rows[i]["group_number"]), float(kl[i]),
                                             bool(flips[i]), rows[i]["row_id"])
                                            for i in range(len(rows)))
    targets = {}
    for panel, parts in target_parts.items():
        if not parts:
            continue
        predicted, target, predicted_state, target_state = (torch.cat(items) for items in zip(*parts))
        behavior = greedy.module_impl.vector_metrics(torch, predicted, target)
        behavior["direction_fraction"] = float(((predicted * target) > 0).float().mean())
        targets[panel] = {"behavior": behavior,
                          "final_residual": greedy.module_impl.vector_metrics(
                              torch, predicted_state, target_state)}
    controls = {}
    for panel, parts in control_parts.items():
        if not parts:
            continue
        parts = sorted(parts)
        ordered = [value for _group, value, _flip, _row in parts]
        ordered_sorted = sorted(ordered)
        median = ordered_sorted[(len(ordered_sorted) - 1) // 2]
        flipped = [row for _group, _value, flip, row in parts if flip]
        controls[panel] = {
            "mean_kl": sum(ordered) / len(ordered), "median_kl": median, "max_kl": max(ordered),
            "top1_flip_count": len(flipped), "top1_flip_fraction": len(flipped) / len(parts),
            "flipped_row_ids": flipped,
        }
    return {"targets": targets, "controls": controls}


def registered_report(value):
    """Drop diagnostic control means so replay matches the frozen parent schema."""
    return {
        "targets": value["targets"],
        "controls": {
            panel: {key: item for key, item in value["controls"][panel].items()
                    if key != "mean_kl"}
            for panel in value["controls"]
        },
    }


def run_crossfit(backend, contexts, bases_by_evaluation_parity, counters):
    outputs = []
    for parity, context in sorted(contexts.items()):
        output = manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases_by_evaluation_parity[parity], complete15=True)
        outputs.append((context, output))
    return outputs, combine_reports(backend, outputs)


def main():
    paths = {
        "prior": PRIOR, "dual_result": DUAL_RESULT, "dual_runner": DUAL_RUNNER,
        "builder": BUILDER, "clamp_contract": CLAMP_CONTRACT,
        "projector_contract": PROJECTOR_CONTRACT, "factor_runner": FACTOR_RUNNER,
        "greedy_runner": GREEDY_RUNNER,
    }
    observed = {key: sha256(path) for key, path in paths.items()}
    prior, dual = json.loads(PRIOR.read_text()), json.loads(DUAL_RESULT.read_text())
    rows = fresh.build_rows()
    alignment = derive_full_sequence_alignment_contract(
        rows, required_panels=TARGET_PANELS + CONTROL_PANELS)
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and fresh.validate_rows(rows) == ROWS_SHA256
        and alignment["panel_counts"] == {panel: 16 for panel in TARGET_PANELS + CONTROL_PANELS}
        and dual.get("terminal") == "no_selective_dual_greedy_program"
        and dual.get("predictions", {}).get(
            "pred_a_authority_absolute_clamp_factor_closure_self_full_replay_finiteness_and_price")
        and len(configuration_grid()) == 30
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "heads": list(HEAD_NAMES), "folds": 2, "configurations": list(configuration_grid()),
        "steps": STEPS, "checkpoints": list(CHECKPOINTS), "price_max": PRICE_MAX,
        "sealed_until_selection": ["A2", "C"],
    }
    if not authority_ok:
        raise ExperimentError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    counters = {key: 0 for key in PRICE_MAX}
    native = backend.native

    def counted_native(batch, *, capture):
        counters["native_capture_forwards"] += 1
        counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)

    backend.native = counted_native
    fit_rows = [row for row in rows if row["transform_id"] in ("A1", "P")]
    fit_bank = capture_bank(backend, fit_rows, counters, factors=True)
    fit_contexts = {
        parity: attach_references(
            backend, subset_context(fit_bank, row_indices(fit_rows, parity=parity)), counters)
        for parity in (0, 1)
    }
    if max(context["manual_native_max_abs_error"] for context in fit_contexts.values()) > 1e-4:
        raise ExperimentError("manual reader does not reproduce native A1/P logits")
    identity = {site: backend.torch.eye(128, device=backend.device) for site in HEAD_NAMES}
    early_identity_outputs = [
        (context, manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=identity, complete15=True))
        for context in fit_contexts.values()
    ]
    early_identity_report = combine_reports(backend, early_identity_outputs)
    early_parent = {
        "targets": {"A1": dual["reports"]["4095"]["report"]["targets"]["A1"]},
        "controls": {"P": dual["reports"]["4095"]["report"]["controls"]["P"]},
    }
    early_replay = factor_parent.replay_comparison(
        registered_report(early_identity_report), early_parent)
    if (not early_replay["numeric_schema_match"] or not early_replay["categorical_match"]
            or early_replay["numeric_max_abs_error"] > 1e-4
            or fit_bank["attention_reconstruction_max_abs_error"] > 1e-4
            or fit_bank["factor_closure_max_abs_error"] > 1e-4):
        raise ExperimentError("A1/P positive-control replay or factor capture failed")

    fit_pairs = []
    for configuration in configuration_grid():
        fits = {}
        for training_parity in (0, 1):
            initial, receipt = initialization_for(
                backend, fit_bank, fit_contexts[training_parity], training_parity,
                configuration["rank"], configuration["initialization"])
            fits[training_parity] = fit_one(
                backend, fit_contexts[training_parity], fit_contexts[1 - training_parity],
                initial, receipt, configuration, training_parity, counters)
        stability = stability_report(
            backend, fits[0]["best_bases"], fits[1]["best_bases"],
            [fits[0]["initialization_receipt"], fits[1]["initialization_receipt"]],
            configuration["rank"])
        both_feasible = all(value["best"]["feasible"] for value in fits.values())
        score = (max(value["best"]["secondary"] for value in fits.values())
                 + STABILITY_WEIGHT * stability["mean_normalized_projector_frobenius"])
        fit_pairs.append({
            "configuration": configuration, "fits": fits, "stability": stability,
            "both_feasible": both_feasible, "selection_score": score,
            "maximum_target_violation": max(value["best"]["target_violation"] for value in fits.values()),
        })
    eligible_pairs = [pair for pair in fit_pairs if pair["both_feasible"]]
    selected = (min(eligible_pairs, key=lambda pair: (
                    pair["selection_score"], pair["configuration"]["rank"],
                    pair["configuration"]["name"]))
                if eligible_pairs else min(fit_pairs, key=lambda pair: (
                    pair["maximum_target_violation"], pair["selection_score"],
                    pair["configuration"]["rank"], pair["configuration"]["name"])))
    selection_finished_utc = utc_now()

    sealed_rows = [row for row in rows if row["transform_id"] in ("A2", "C")]
    sealed_bank = capture_bank(backend, sealed_rows, counters, factors=False)
    sealed_contexts = {
        parity: attach_references(
            backend, subset_context(sealed_bank, row_indices(sealed_rows, parity=parity)), counters)
        for parity in (0, 1)
    }
    evaluation_contexts = {
        0: fit_contexts[0], 1: fit_contexts[1],
        2: sealed_contexts[0], 3: sealed_contexts[1],
    }
    selected_bases = {
        0: selected["fits"][1]["best_bases"], 1: selected["fits"][0]["best_bases"],
        2: selected["fits"][1]["best_bases"], 3: selected["fits"][0]["best_bases"],
    }
    selected_outputs, selected_report = run_crossfit(
        backend, evaluation_contexts, selected_bases, counters)

    rank = selected["configuration"]["rank"]
    baseline_fits = {}
    for training_parity in (0, 1):
        candidates = [pair["fits"][training_parity] for pair in fit_pairs
                      if pair["configuration"]["rank"] == rank
                      and pair["configuration"]["initialization"] == "dim_task_svd"]
        baseline_fits[training_parity] = candidates[0]
    baseline_bases = {
        0: baseline_fits[1]["initial_bases"], 1: baseline_fits[0]["initial_bases"],
        2: baseline_fits[1]["initial_bases"], 3: baseline_fits[0]["initial_bases"],
    }
    _baseline_outputs, baseline_report = run_crossfit(
        backend, evaluation_contexts, baseline_bases, counters)

    identity_bases = {parity: identity for parity in evaluation_contexts}
    _identity_outputs, identity_report = run_crossfit(
        backend, evaluation_contexts, identity_bases, counters)
    parent_report = factor_parent.registered_parent_report(dual["reports"]["4095"]["report"])
    replay = factor_parent.replay_comparison(registered_report(identity_report), parent_report)
    closure = max(context["manual_native_max_abs_error"] for context in evaluation_contexts.values())
    orthogonality = max(
        float((basis.T @ basis - backend.torch.eye(basis.shape[1], device=basis.device)).abs().max())
        for bases in selected_bases.values() for basis in bases.values())
    counters["fit_parameters"] = 4 * 128 * max(RANKS)
    price_ok = all(counters[key] <= PRICE_MAX[key] for key in PRICE_MAX)

    baseline_fold_reports = {}
    for training in (0, 1):
        fit = baseline_fits[training]
        baseline_fold_reports[str(training)] = fit["trace"][0]
    exact_zero_flip = max(
        (item for item in dual["reports"].values()
         if item["report"]["controls"]["P"]["top1_flip_count"] == 0
         and item["report"]["controls"]["C"]["top1_flip_count"] == 0
         and max(item["report"]["controls"][panel]["median_kl"] for panel in CONTROL_PANELS)
             <= CONTROL_MEDIAN_KL_BAR),
        key=lambda item: item["report"]["targets"]["A1"]["behavior"]["signed_projection"])
    exact_baselines = {
        "best_zero_flip": exact_zero_flip,
        "full_factor": dual["reports"]["4095"],
    }

    gradient_min_fit_max = min(
        value["gradient_max_abs"] for pair in fit_pairs for value in pair["fits"].values())
    pred_a = bool(
        authority_ok and closure <= 1e-4 and replay["numeric_schema_match"]
        and replay["categorical_match"] and replay["numeric_max_abs_error"] <= 1e-4
        and orthogonality <= 1e-4 and gradient_min_fit_max > 1e-10
        and finite({"fit_pairs": [{"score": pair["selection_score"],
                                    "violation": pair["maximum_target_violation"]} for pair in fit_pairs],
                    "selected_report": selected_report, "baseline_report": baseline_report,
                    "identity_report": identity_report}) and price_ok)
    pred_b = bool(selected["both_feasible"] and all(
        value["best"]["step"] > 0 for value in selected["fits"].values()))
    pred_c = bool(pred_b and all(
        selected["fits"][training]["best"]["p_mean_kl"]
        < baseline_fold_reports[str(training)]["p_mean_kl"]
        and baseline_fold_reports[str(training)]["feasible"]
        for training in (0, 1)))
    pred_d = bool(
        selected["stability"]["minimum_principal_cosine"] >= STABILITY_MIN_COSINE
        and selected["stability"]["maximum_normalized_projector_frobenius"]
            <= STABILITY_MAX_NORMALIZED_FROBENIUS)
    pred_e = bool(
        selected_report["targets"]["A2"]["behavior"]["signed_projection"] >= TARGET_PROJECTION_BAR
        and selected_report["targets"]["A2"]["behavior"]["direction_fraction"] >= DIRECTION_FRACTION_BAR)
    pred_f = bool(all(
        selected_report["controls"][panel]["top1_flip_count"] == 0
        and selected_report["controls"][panel]["median_kl"] <= CONTROL_MEDIAN_KL_BAR
        for panel in CONTROL_PANELS))
    exact_failures = {
        "best_zero_flip_target_fails": exact_baselines["best_zero_flip"]["report"]["targets"]["A1"]["behavior"]["signed_projection"] < TARGET_PROJECTION_BAR,
        "full_factor_controls_fail": any(
            exact_baselines["full_factor"]["report"]["controls"][panel]["top1_flip_count"] > 0
            or exact_baselines["full_factor"]["report"]["controls"][panel]["median_kl"] > CONTROL_MEDIAN_KL_BAR
            for panel in CONTROL_PANELS),
    }
    pred_g = bool(pred_b and pred_e and pred_f and all(exact_failures.values()))
    predictions = {
        "pred_a_authority_alignment_absolute_clamp_gradient_closure_finiteness_and_price": pred_a,
        "pred_b_target_feasible_regularized_checkpoint_exists": pred_b,
        "pred_c_regularization_beats_dim_on_opposite_parity": pred_c,
        "pred_d_projector_is_fold_stable": pred_d,
        "pred_e_sealed_a2_transfers": pred_e,
        "pred_f_sealed_controls_are_selective": pred_f,
        "pred_g_learned_projector_beats_exact_factor_baselines": pred_g,
    }
    if not pred_a:
        terminal = "invalid"
    elif not selected["both_feasible"]:
        terminal = "head_projector_family_infeasible"
    elif not pred_b or not pred_c:
        terminal = "regularization_does_not_beat_dim"
    elif not pred_d:
        terminal = "projector_instability"
    elif not pred_e:
        terminal = "construction_memorization"
    elif not pred_f:
        terminal = "control_nonselective"
    elif not pred_g:
        terminal = "regularization_does_not_beat_dim"
    else:
        terminal = "target_feasible_regularized_das_candidate"

    result = {
        "schema": "temporal_iswas_v15_head_response_target_feasible_regularized_das_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "selection_finished_utc": selection_finished_utc,
        "sealed_opened_after_selection": True, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256,
        "alignment_contract": alignment, "dryrun": dryrun,
        "fit_spec": {
            "heads": list(HEAD_NAMES), "ranks": list(RANKS),
            "initializations": list(INITIALIZATIONS), "noise_jacobian": list(NOISE_JACOBIAN),
            "stability_weight": STABILITY_WEIGHT, "steps": STEPS,
            "checkpoints": list(CHECKPOINTS), "learning_rate": LR,
            "target_projection_bar": TARGET_PROJECTION_BAR,
            "direction_fraction_bar": DIRECTION_FRACTION_BAR,
        },
        "fit_pairs": [{
            "configuration": pair["configuration"], "both_feasible": pair["both_feasible"],
            "selection_score": pair["selection_score"],
            "maximum_target_violation": pair["maximum_target_violation"],
            "stability": pair["stability"],
            "fits": {str(key): strip_fit(value) for key, value in pair["fits"].items()},
        } for pair in fit_pairs],
        "selected": {
            "configuration": selected["configuration"], "selection_score": selected["selection_score"],
            "both_feasible": selected["both_feasible"], "stability": selected["stability"],
            "folds": {str(key): strip_fit(value) for key, value in selected["fits"].items()},
            "projectors": {
                str(training): {site: basis.detach().cpu().tolist()
                                for site, basis in selected["fits"][training]["best_bases"].items()}
                for training in (0, 1)
            },
        },
        "reports": {"selected_crossfit": selected_report, "matched_dim_step_zero": baseline_report,
                    "identity_full_parent": identity_report},
        "exact_factor_baselines": exact_baselines, "exact_factor_failure_modes": exact_failures,
        "instrument": {
            "manual_native_max_abs_error": closure, "full_parent_replay": replay,
            "early_a1_p_full_parent_replay": early_replay,
            "attention_reconstruction_max_abs_error": fit_bank["attention_reconstruction_max_abs_error"],
            "factor_closure_max_abs_error": fit_bank["factor_closure_max_abs_error"],
            "projector_orthogonality_max_abs_error": orthogonality,
            "gradient_min_fit_max_abs": gradient_min_fit_max,
            "gradient_max_abs": max(value["gradient_max_abs"] for pair in fit_pairs
                                      for value in pair["fits"].values()),
        },
        "predictions": predictions, "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "selected_configuration": selected["configuration"],
        "selected_fold_best": {str(key): value["best"] for key, value in selected["fits"].items()},
        "stability": selected["stability"], "selected_report": selected_report,
        "instrument": result["instrument"], "predictions": predictions,
        "terminal": terminal, "price": result["price"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
