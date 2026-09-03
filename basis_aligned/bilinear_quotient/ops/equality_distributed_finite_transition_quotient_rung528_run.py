#!/usr/bin/env python3
"""RUNG 528 -- continuation-defined equality finite-transition quotient.

pred_a: raw post-MLP12 capture/insertion is exact, live, and call-accounted
pred_b: one or more correct-gauge transitions share a discovery response
pred_c: a discovery relation passes physical insertion and new documents
pred_d: a fixed relation predicts 30 unopened circuits and document halves
pred_e: at least one validated relation is task-selective

Strong null: any of A--E fails.  Later phases remain fail-closed.
Literal price: at most 11,330 forwards, zero backwards, zero deployed values.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
POLY = REPO / "basis_aligned/polynomial_causal"
for path in (OPS, POLY, BQ, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade  # noqa: E402
import equality_distributed_finite_transition_quotient_rung528_math as qm  # noqa: E402
import equality_score_gauged_downstream_program_rung505 as r505  # noqa: E402
import natural_action_conditioned_later_write_state_atlas_rung506 as r506  # noqa: E402


RUNNER = Path(__file__).resolve()
PREREG = POLY / "EQUALITY_DISTRIBUTED_FINITE_TRANSITION_QUOTIENT_RUNG528_PREREGISTRATION.md"
MATH = OPS / "equality_distributed_finite_transition_quotient_rung528_math.py"
R505_SOURCE = OPS / "equality_score_gauged_downstream_program_rung505.py"
R505_RESULT = BQ / "equality_score_gauged_downstream_program_rung505_results.json"
R506_SOURCE = OPS / "natural_action_conditioned_later_write_state_atlas_rung506.py"
R506_RESULT = BQ / "natural_action_conditioned_later_write_state_atlas_rung506_results.json"
R501_SOURCE = OPS / "equality_score_directed_action_graph_rung501.py"
R501_RESULT = BQ / "equality_score_directed_action_graph_rung501_results.json"
R510_SOURCE = OPS / "mlp10_observable_predictive_state_quotient_rung510.py"
R510_RESULT = BQ / "mlp10_observable_predictive_state_quotient_rung510_results.json"
OUT = BQ / "equality_distributed_finite_transition_quotient_rung528_results.json"
SMOKE_OUT = BQ / "equality_distributed_finite_transition_quotient_rung528_gpu_smoke_results.json"

FROZEN_SHA256 = {
    PREREG: "96b62e3265698467be05848bf239dc49fb4daecbdea7145e4955c550ade5ea2d",
    MATH: "1363371d5df5a8e8e14682907a172ccb65b39df7c5c5e9ffb3ce0dfe72ae5728",
    R505_SOURCE: "0c5f6679ec40cb02bd6af1e28b0b41ca2ad7967fd4b6c9d73a4f388153f3e4de",
    R505_RESULT: "3720a2feb24fc5ec4554d858a00a576a1fcd44f0e789d2b728e66483d7d8d1a1",
    R506_SOURCE: "9a17e28312a0e7214e5fc587123e3267e2650b382f3a40daf12ad1a380b1d004",
    R506_RESULT: "f86e5f0303ab0616ea14e3141fd09886ca54d326e8d83ea6c8c13a62f66db75e",
    R501_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    R501_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    R510_SOURCE: "7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a",
    R510_RESULT: "16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177",
}

D = 1152
TOKENS = 256
BOUNDARY_SITE = 12
CONTINUATION_PATCHES = {
    "native": (),
    "without_A14": ("a14",),
    "without_M17": ("m17",),
    "without_both": ("a14", "m17"),
}
SOURCES = r505.SOURCES
WRONG_SIGNS = r505.WRONG_SIGNS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_dependencies() -> tuple[dict[str, str], tuple]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {actual} != {expected}")
        observed[str(path.relative_to(REPO))] = actual
    r506_result = json.loads(R506_RESULT.read_text())
    r510_result = json.loads(R510_RESULT.read_text())
    if not (
        r506_result.get("pred_a_exact_live_conditional_instrument") is True
        and r506_result.get("pred_b_score_actions_recalibrate_new_documents") is True
        and r506_result.get("pred_c_at_least_one_whole_write_edge_confirms") is False
        and r506_result.get("strong_null") is True
        and r506_result.get("next_step") == "split_fixed_writes_into_exact_attention_or_bilinear_terms"
    ):
        raise RuntimeError("rung506 whole-write null route changed")
    if not (
        r510_result.get("pred_a_exact_live_singleton_and_substitution_instrument") is True
        and r510_result.get("pred_b_one_to_sixteen_discovery_equivalence_pairs") is False
        and r510_result.get("strong_null") is True
    ):
        raise RuntimeError("rung510 exact-term null route changed")
    population = r506.validate_inputs()
    return observed, population


def scaled_boundary(absent_boundary: torch.Tensor, donor_delta: torch.Tensor, gamma: float) -> torch.Tensor:
    """Construct a deployed boundary in float32, then round once to model dtype."""
    if absent_boundary.shape != donor_delta.shape:
        raise ValueError("boundary and transition shapes differ")
    if not math.isfinite(gamma):
        raise ValueError("boundary scale is nonfinite")
    result = absent_boundary.float() + float(gamma) * donor_delta.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("scaled boundary is nonfinite")
    return result.to(absent_boundary.dtype)


@torch.no_grad()
def boundary_forward(
    model,
    tokens: torch.Tensor,
    *,
    action: str = "N",
    absent: bool = False,
    scales: Mapping[str, Mapping[str, float]] | None = None,
    direct: bool = False,
    boundary_override: torch.Tensor | None = None,
    capture_writes: Sequence[str] = (),
    patch_writes: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], dict[str, int]]:
    """Run the production recurrence while exposing one exact raw residual boundary."""
    if action not in r505.SOURCE_ACTIONS:
        raise ValueError("unregistered score action")
    if direct and (action != "N" or absent or boundary_override is not None or capture_writes or patch_writes):
        raise ValueError("direct native arm cannot carry analytical edits")
    pair = r505.SOURCE_ACTIONS[action]["pair"]
    if (pair is not None or absent) and scales is None:
        raise ValueError("score edit requires frozen scales")
    capture_set = set(capture_writes)
    patch_writes = {} if patch_writes is None else dict(patch_writes)
    allowed_writes = {"a14", "m17"}
    if len(capture_set) != len(capture_writes) or not capture_set <= allowed_writes:
        raise ValueError("capture identity changed")
    if not set(patch_writes) <= allowed_writes:
        raise ValueError("continuation patch identity changed")

    cached: dict[str, torch.Tensor] = {}
    captures: dict[str, torch.Tensor] = {}
    states: dict[str, torch.Tensor] = {}
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "score_edit_rms": 0.0,
        "boundary_override_rms": 0.0,
        "continuation_patch_rms_max": 0.0,
    }
    audit = {
        "native_attention": 0,
        "replayed_attention": 0,
        "native_mlp": 0,
        "boundary_captures": 0,
        "boundary_overrides": 0,
        "write_captures": 0,
        "write_patches": 0,
    }
    factor_parent = r505.action_parent.factor_parent

    def maybe_patch(key: str, write: torch.Tensor) -> torch.Tensor:
        if key in patch_writes:
            replacement = patch_writes[key]
            if replacement.shape != write.shape or replacement.dtype != write.dtype \
                    or replacement.device != write.device or not bool(torch.isfinite(replacement).all()):
                raise RuntimeError(f"malformed continuation patch at {key}")
            diagnostics["continuation_patch_rms_max"] = max(
                diagnostics["continuation_patch_rms_max"],
                float((replacement.float() - write.float()).square().mean().sqrt()),
            )
            write = replacement
            audit["write_patches"] += 1
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["write_captures"] += 1
        return write

    def attention(site: int, block, state: torch.Tensor, first_value: torch.Tensor | None):
        if direct or site not in factor_parent.stage1.SITE_HEADS:
            write, next_value = block.attn(state, first_value)
            audit["native_attention"] += 1
        else:
            write, factors, support, error = factor_parent._factor_site(
                state, first_value, block.attn, site, tokens)
            audit["replayed_attention"] += 1
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], error)
            edit_pair = (0, 3) if absent else pair
            if edit_pair is not None:
                donor, recipient = edit_pair
                if site == factor_parent.TERMS[donor][1]:
                    cached.update(factors[donor])
                if site == factor_parent.TERMS[recipient][1]:
                    if not cached:
                        raise RuntimeError("donor factors unavailable at recipient")
                    target = factors[recipient]
                    replacement = torch.zeros_like(target["factor_term"])
                    if not absent:
                        row = r505.signed_scales(scales, action)
                        if row is None:
                            raise RuntimeError("edited score action lacks scale")
                        replacement = torch.bmm(cached["p"] * row["score_ratio"] * support, target["u"])
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["score_edit_rms"] = float(edit.float().square().mean().sqrt())
            next_value = first_value
        return maybe_patch(f"a{site}", write), next_value

    facade.validate_production_model(model)
    facade.validate_tokens(tokens, production_shape=True)
    x = F.rms_norm(model.transformer.wte(tokens), (D,))
    x0 = x
    first_value = None
    prior_writes: list[torch.Tensor] = []
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (D,))
        attention_write, first_value = attention(site, block, attention_state, first_value)
        x = x + attention_write
        mlp_state = F.rms_norm(x, (D,))
        mlp_write = maybe_patch(f"m{site}", block.mlp(mlp_state))
        audit["native_mlp"] += 1
        prior_writes.append(mlp_write)
        x = x + mlp_write
        if site == BOUNDARY_SITE:
            states["native_boundary"] = x.detach().clone()
            states["embedding_skip"] = x0.detach().clone()
            states["first_value"] = first_value.detach().clone()
            audit["boundary_captures"] += 1
            if boundary_override is not None:
                if boundary_override.shape != x.shape or boundary_override.dtype != x.dtype \
                        or boundary_override.device != x.device or not bool(torch.isfinite(boundary_override).all()):
                    raise RuntimeError("malformed boundary override")
                diagnostics["boundary_override_rms"] = float(
                    (boundary_override.float() - x.float()).square().mean().sqrt())
                x = boundary_override
                audit["boundary_overrides"] += 1
            states["effective_boundary"] = x.detach().clone()

    logits = model.lm_head(F.rms_norm(x, (D,)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if tuple(logits.shape) != (*tokens.shape, facade.LOGIT_VOCAB) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("rung528 logits malformed")
    expected_attention = {"native_attention": 18, "replayed_attention": 0} if direct else {
        "native_attention": 15, "replayed_attention": 3}
    if any(audit[key] != value for key, value in expected_attention.items()) or audit["native_mlp"] != 18:
        raise RuntimeError(f"forward call audit changed: {audit}")
    if audit["boundary_captures"] != 1 or audit["boundary_overrides"] != int(boundary_override is not None):
        raise RuntimeError(f"boundary audit changed: {audit}")
    if audit["write_captures"] != len(capture_set) or set(captures) != capture_set \
            or audit["write_patches"] != len(patch_writes):
        raise RuntimeError(f"continuation audit changed: {audit}")
    states.update(captures)
    return logits, states, diagnostics, audit


def maximum_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def relative_squared(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = left.double() - right.double()
    return float(difference.square().sum() / left.double().square().sum().clamp_min(1e-300))


@torch.no_grad()
def gpu_smoke() -> dict[str, object]:
    if SMOKE_OUT.exists():
        raise FileExistsError(f"refusing to overwrite result: {SMOKE_OUT}")
    dependencies, population = validate_dependencies()
    rows, _task_masks, _circuit_masks, scales, discovery_tags, _validation_tags, metadata = population
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    tokens = rows[:4, :-1].to("cuda")

    direct_logits, direct_state, direct_diag, direct_audit = boundary_forward(
        model, tokens, direct=True)
    absent_logits, absent_state, absent_diag, absent_audit = boundary_forward(
        model, tokens, action="P", absent=True, scales=scales,
        capture_writes=("a14", "m17"))
    action_runs = {}
    injections = {}
    total_patches = 0
    minimum_transition_rms = math.inf
    minimum_continuation_patch_rms = math.inf
    maximum_self_logit_error = 0.0
    maximum_self_logit_relative_squared = 0.0
    maximum_effective_boundary_error = 0.0
    maximum_skip_state_error = 0.0
    maximum_first_value_error = 0.0
    factor_reconstruction_max = absent_diag["factor_reconstruction_max"]
    minimum_score_edit_rms = absent_diag["score_edit_rms"]
    for action in SOURCES:
        logits, state, diagnostics, audit = boundary_forward(
            model, tokens, action=action, scales=scales)
        action_runs[action] = {"logits": logits, "state": state, "diagnostics": diagnostics, "audit": audit}
        factor_reconstruction_max = max(factor_reconstruction_max, diagnostics["factor_reconstruction_max"])
        if diagnostics["score_edit_rms"] > 0:
            minimum_score_edit_rms = min(minimum_score_edit_rms, diagnostics["score_edit_rms"])
        delta = state["native_boundary"].float() - absent_state["native_boundary"].float()
        transition_rms = float(delta.square().mean().sqrt())
        minimum_transition_rms = min(minimum_transition_rms, transition_rms)
        injections[action] = {}
        for continuation, sites in CONTINUATION_PATCHES.items():
            patches = {site: absent_state[site] for site in sites}
            replacement = scaled_boundary(absent_state["native_boundary"], delta, 1.0)
            injected_logits, injected_state, injected_diag, injected_audit = boundary_forward(
                model, tokens, action="P", absent=True, scales=scales,
                boundary_override=replacement, patch_writes=patches)
            injections[action][continuation] = injected_logits
            factor_reconstruction_max = max(
                factor_reconstruction_max, injected_diag["factor_reconstruction_max"])
            total_patches += injected_audit["write_patches"]
            if sites:
                minimum_continuation_patch_rms = min(
                    minimum_continuation_patch_rms,
                    injected_diag["continuation_patch_rms_max"])
            maximum_effective_boundary_error = max(
                maximum_effective_boundary_error,
                maximum_abs(injected_state["effective_boundary"], state["native_boundary"]))
            if continuation == "native":
                maximum_self_logit_error = max(
                    maximum_self_logit_error, maximum_abs(injected_logits, logits))
                maximum_self_logit_relative_squared = max(
                    maximum_self_logit_relative_squared, relative_squared(logits, injected_logits))
                maximum_skip_state_error = max(
                    maximum_skip_state_error,
                    maximum_abs(injected_state["embedding_skip"], state["embedding_skip"]))
                maximum_first_value_error = max(
                    maximum_first_value_error,
                    maximum_abs(injected_state["first_value"], state["first_value"]))

    native_replay_logit_max_abs = maximum_abs(direct_logits, action_runs["N"]["logits"])
    native_replay_boundary_max_abs = maximum_abs(
        direct_state["native_boundary"], action_runs["N"]["state"]["native_boundary"])
    diagnostics = {
        "full_model_forwards": 22,
        "direct_forwards": 1,
        "analytical_forwards": 21,
        "boundary_captures": 22,
        "boundary_overrides": 16,
        "continuation_write_captures": absent_audit["write_captures"],
        "continuation_write_patches": total_patches,
        "native_replay_logit_max_abs": native_replay_logit_max_abs,
        "native_replay_boundary_max_abs": native_replay_boundary_max_abs,
        "maximum_self_insertion_logit_abs": maximum_self_logit_error,
        "maximum_self_insertion_logit_relative_squared": maximum_self_logit_relative_squared,
        "maximum_effective_boundary_abs": maximum_effective_boundary_error,
        "maximum_embedding_skip_abs": maximum_skip_state_error,
        "maximum_first_value_abs": maximum_first_value_error,
        "factor_reconstruction_max": factor_reconstruction_max,
        "minimum_score_edit_rms": minimum_score_edit_rms,
        "minimum_transition_rms": minimum_transition_rms,
        "minimum_continuation_patch_rms": minimum_continuation_patch_rms,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "calls_exact": bool(total_patches == 16 and absent_audit["write_captures"] == 2),
    }
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and native_replay_logit_max_abs == 0.0
        and native_replay_boundary_max_abs == 0.0
        and maximum_self_logit_error == 0.0
        and maximum_effective_boundary_error == 0.0
        and maximum_skip_state_error == 0.0
        and maximum_first_value_error == 0.0
        and factor_reconstruction_max <= 1e-10
        and diagnostics["calls_exact"])
    pred_b = bool(minimum_score_edit_rms > 0 and minimum_transition_rms > 0)
    pred_c = bool(minimum_continuation_patch_rms > 0)
    result = {
        "status": "complete",
        "rung": "528_gpu_smoke",
        "claim_level": "raw_boundary_operational_smoke_no_scientific_outcomes_retained",
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "dependency_sha256": dependencies,
        "input_identity": {
            "documents_exercised": [0, 4],
            "circuit_tags_named_but_not_scored": discovery_tags[:2],
            "population": metadata,
        },
        "diagnostics": diagnostics,
        "planted_suite": qm.planted_suite(),
        "check_a_exact_boundary_self_replay": pred_a,
        "check_b_score_transitions_are_live": pred_b,
        "check_c_continuation_patches_are_live": pred_c,
        "strong_null": not (pred_a and pred_b and pred_c),
        "scientific_task_or_circuit_effects_retained": False,
        "next_step": "implement_and_enqueue_full_rung528" if pred_a and pred_b and pred_c else "repair_instrument_only",
    }
    atomic_json(SMOKE_OUT, result)
    print(json.dumps({
        "status": result["status"],
        "rung": result["rung"],
        "checks": {key: value for key, value in result.items() if key.startswith("check_")},
        "strong_null": result["strong_null"],
        "diagnostics": diagnostics,
        "scientific_outcomes_retained": False,
    }, indent=2, sort_keys=True))
    return result


def dry_run() -> dict[str, object]:
    dependencies, population = validate_dependencies()
    rows, _task_masks, _circuit_masks, _scales, discovery_tags, validation_tags, _metadata = population
    planted = qm.planted_suite()
    assert tuple(rows.shape) == (1000, 257)
    assert len(discovery_tags) == 32 and len(validation_tags) == 30
    assert tuple(CONTINUATION_PATCHES) == qm.CONTINUATIONS
    assert sum(len(value) for value in CONTINUATION_PATCHES.values()) == 4
    assert 62 * 32 == 1984
    assert 1984 + 31 * 8 * 3 + 62 * (22 + 8 * 3) + 125 * (22 + 8 * 3) == 11330
    assert planted["passes"]
    return {
        "status": "dry_run_passed",
        "rung": 528,
        "model_loaded": False,
        "outcomes_opened": False,
        "dependencies": dependencies,
        "discovery_documents": [0, 248, 124],
        "confirmation_documents": [248, 496, 372],
        "validation_documents": [500, 1000, 750],
        "discovery_circuits": len(discovery_tags),
        "validation_circuits": len(validation_tags),
        "unconditional_discovery_forwards": 1984,
        "maximum_conditional_forwards": 11330,
        "pred_a_exact_boundary_self_replay": None,
        "pred_b_shared_discovery_transition": None,
        "pred_c_physical_new_document_prediction": None,
        "pred_d_heldout_circuits_documents": None,
        "pred_e_selective_distributed_state": None,
        "planted_suite_passes": planted["passes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu_smoke:
        gpu_smoke()
        return
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return
    raise RuntimeError("full rung528 collection is not implemented; GPU smoke must pass first")


if __name__ == "__main__":
    main()
