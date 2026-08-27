#!/usr/bin/env python3
"""Numerical site-1, strata, and program-freeze stage for compiler-v2.1.

This stage is legal only after the last-written site0 training receipt.  It
loads fit plus mapped-validation rows, independently regenerates the true,
shuffle, and mean autoregressive site1 contexts, freezes both complete 108-cell
preselector ledgers, derives the validation-only collateral strata, and writes
the final unlock only after the outer model invocation and hook restoration.
It never requests or deserializes compiler_final_v21.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import early_mlp_state_complete_compiler_v2_site0 as old_site0  # noqa: E402
import early_mlp_state_complete_compiler_v21 as lifecycle  # noqa: E402
import early_mlp_state_complete_compiler_v21_site0 as site0  # noqa: E402
import prepare_state_complete_compiler_rows_v21 as authority  # noqa: E402


MANIFEST = authority.SITE1_MANIFEST
FIT_SEED = 271828
SCORER = "CUDA float32 per-token; float64 row/aggregate"


def _capture_sha256(value: Mapping[str, torch.Tensor]) -> str:
    return site0._capture_sha256(value)


def _validation_identity(row_receipt: Mapping[str, Any]) -> str:
    return site0._validation_identity(row_receipt)


def _programs_cpu() -> dict[str, dict[int, Mapping[str, Any]]]:
    programs0 = lifecycle._selected_site0_programs()
    return {
        arm: {0: programs0[arm]} for arm in ("true", "shuffle", "mean")
    }


def _configure_upstream(
    hook: Any, arm: str, state0: Mapping[str, Any], *, device: Any,
) -> None:
    hook.programs = {arm: {0: old_site0._device_state(state0, device)}}


def capture_site1_fit(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int], *, arm: str, state0: Mapping[str, Any],
    capture_adjoint: bool,
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    """Capture site1 under exactly one frozen upstream program."""

    _configure_upstream(hook, arm, state0, device=sa.DEV)
    hook.configure(
        {0: "Q"}, program_name=arm, capture_site=1,
        capture_adjoint=capture_adjoint,
    )
    with old_site0.FrozenParameters(sa.m), old_site0.runtime.OriginalMLPCallGuard(
        sa.H, {1}
    ) as guard:
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = sa.fwd_arm(
                idx, all_attention, twall, frozenset(range(18))
            ).float()
            if capture_adjoint:
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                ce[site0.old_site0._valid_mask(targets)].mean().backward()
                hook.collect_pending_adjoint()
    guard.assert_contract(require_allowed_calls=True)
    captured = hook.captured()
    expected = len(rows) * 64
    keys = {"z", "p", "mo", "c"} | ({"adjoint"} if capture_adjoint else set())
    shapes = {
        "z": (expected, authority.compiler.D_MODEL),
        "p": (expected, authority.compiler.COEFFICIENT_DIM),
        "mo": (expected, authority.compiler.COEFFICIENT_DIM),
        "c": (expected, authority.compiler.COEFFICIENT_DIM),
        **({"adjoint": (expected, authority.compiler.COEFFICIENT_DIM)}
           if capture_adjoint else {}),
    }
    if set(captured) != keys or {
        key: tuple(value.shape) for key, value in captured.items()
    } != shapes:
        raise RuntimeError(f"v2.1 {arm} site1 fit capture shape changed")
    return captured, dict(guard.counts)


@torch.no_grad()
def capture_site1_validation(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int], *, arm: str, state0: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    _configure_upstream(hook, arm, state0, device=sa.DEV)
    hook.configure({0: "Q"}, program_name=arm, capture_site=1)
    with old_site0.runtime.OriginalMLPCallGuard(sa.H, {1}) as guard:
        for start in range(0, len(rows), 8):
            idx = rows[start:start + 8, :-1].to(sa.DEV).contiguous()
            sa.fwd_arm(idx, all_attention, twall, frozenset(range(18)))
    guard.assert_contract(require_allowed_calls=True)
    return hook.captured(), dict(guard.counts)


def build_site1_candidates(
    captured: Mapping[str, torch.Tensor], block: Any, basis: torch.Tensor,
    device: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The registered A--E grammar is site-invariant."""

    return old_site0.build_site0_candidates(captured, block, basis, device)


@torch.no_grad()
def teacher_bank(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int], *, arm: str, state0: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, dict[int, int]]]:
    """Return Q0-O1 teacher and Q0-N1 baseline in one context."""

    _configure_upstream(hook, arm, state0, device=sa.DEV)
    parts: dict[str, list[torch.Tensor]] = {"teacher": [], "baseline": []}
    targets_parts: list[torch.Tensor] = []
    idx_parts: list[torch.Tensor] = []
    counters: dict[str, dict[int, int]] = {}
    for name, states, allowed in (
        ("teacher", {0: "Q", 1: "O"}, {1}),
        ("baseline", {0: "Q"}, set()),
    ):
        hook.configure(states, program_name=arm)
        with old_site0.runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8].to(sa.DEV)
                idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
                logits = sa.fwd_arm(
                    idx, all_attention, twall, frozenset(range(18))
                ).float()[:, 64:]
                parts[name].append(logits.cpu().contiguous())
                if name == "teacher":
                    targets_parts.append(targets.cpu())
                    idx_parts.append(idx.cpu())
        guard.assert_contract(require_allowed_calls=bool(allowed))
        counters[name] = dict(guard.counts)

    teacher, baseline = torch.cat(parts["teacher"]), torch.cat(parts["baseline"])
    targets_all, idx_all = torch.cat(targets_parts), torch.cat(idx_parts)
    copy_all = old_site0._copy_mask(idx_all, targets_all)[:, 64:]
    teacher_kl_sum = 0.0
    copy_ce_sum = 0.0
    teacher_token_count = 0
    copy_token_count = 0
    for start in range(0, len(rows), 8):
        stop = min(start + 8, len(rows))
        target = targets_all[start:stop, 64:].to(sa.DEV)
        teacher_batch = teacher[start:stop].to(sa.DEV)
        baseline_batch = baseline[start:stop].to(sa.DEV)
        teacher_logp = F.log_softmax(teacher_batch, dim=-1)
        baseline_logp = F.log_softmax(baseline_batch, dim=-1)
        kl = (
            teacher_logp.exp() * (teacher_logp - baseline_logp)
        ).sum(dim=-1)
        teacher_kl_sum += float(kl.double().sum())
        teacher_token_count += int(kl.numel())
        baseline_ce = F.cross_entropy(
            baseline_batch.reshape(-1, baseline_batch.shape[-1]),
            target.reshape(-1), reduction="none",
        ).view(stop - start, -1)
        copy = copy_all[start:stop].to(sa.DEV)
        copy_ce_sum += float(baseline_ce[copy].double().sum())
        copy_token_count += int(copy.sum())
    if teacher_token_count != authority.VALIDATION_TOKEN_COUNT or not (
        teacher_kl_sum > 0 and copy_token_count > 0
    ):
        raise RuntimeError(f"v2.1 {arm} site1 teacher support changed")
    return teacher, {
        "teacher_denominator": teacher_kl_sum / teacher_token_count,
        "teacher_kl_sum": teacher_kl_sum,
        "teacher_token_count": teacher_token_count,
        "copy_baseline": copy_ce_sum / copy_token_count,
        "copy_ce_sum": copy_ce_sum,
        "copy_token_count": copy_token_count,
    }, counters


@torch.no_grad()
def score_candidate(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int], name: str, state1: Mapping[str, Any],
    teacher: torch.Tensor, context: Mapping[str, Any], *, arm: str,
    state0: Mapping[str, Any], price: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, int]]:
    hook.programs = {arm: {
        0: old_site0._device_state(state0, sa.DEV),
        1: old_site0._device_state(state1, sa.DEV),
    }}
    hook.configure({0: "Q", 1: "Q"}, program_name=arm)
    kl_sum = 0.0
    global_ce_sum = 0.0
    copy_ce_sum = 0.0
    token_count = 0
    copy_count = 0
    with old_site0.runtime.OriginalMLPCallGuard(sa.H, set()) as guard:
        row_offset = 0
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            candidate = sa.fwd_arm(
                idx, all_attention, twall, frozenset(range(18))
            ).float()[:, 64:]
            teacher_batch = teacher[row_offset:row_offset + len(batch)].to(sa.DEV)
            teacher_logp = F.log_softmax(teacher_batch, dim=-1)
            candidate_logp = F.log_softmax(candidate, dim=-1)
            kl = (
                teacher_logp.exp() * (teacher_logp - candidate_logp)
            ).sum(dim=-1)
            kl_sum += float(kl.double().sum())
            token_count += int(kl.numel())
            target = targets[:, 64:]
            ce = F.cross_entropy(
                candidate.reshape(-1, candidate.shape[-1]),
                target.reshape(-1), reduction="none",
            ).view(len(batch), -1)
            global_ce_sum += float(ce.double().sum())
            copy = old_site0._copy_mask(idx, targets)[:, 64:]
            copy_ce_sum += float(ce[copy].double().sum())
            copy_count += int(copy.sum())
            row_offset += len(batch)
    guard.assert_contract(require_allowed_calls=False)
    if token_count != authority.VALIDATION_TOKEN_COUNT or copy_count != context[
        "copy_token_count"
    ]:
        raise RuntimeError(f"v2.1 {arm}:{name} site1 scorer support changed")
    raw = {
        "candidate_teacher_kl_sum": kl_sum,
        "candidate_teacher_kl_count": token_count,
        "global_ce_sum": global_ce_sum,
        "global_ce_count": token_count,
        "copy_ce_sum": copy_ce_sum,
        "copy_ce_count": copy_count,
    }
    metrics = site0.metrics_from_sufficient_statistics(
        state1, context, raw, price=price,
    )
    return metrics, dict(guard.counts)


def full_native_control(
    state: Mapping[str, Any], gate: Mapping[str, Any],
    validation_calls: Mapping[int, int], *, context: str,
    upstream_state_sha256: str, validation_identity: str,
    physical_reference_scale: float,
) -> dict[str, Any]:
    if gate.get("passed") is not True:
        raise RuntimeError(f"v2.1 site1 {context} full-native gate did not pass")
    tolerance = float(gate["physical_tolerance"])
    if tolerance != 4e-6 * max(1.0, physical_reference_scale):
        raise RuntimeError("v2.1 site1 full-native scale binding changed")
    observed = {
        "physical_max_abs_error": float(gate["physical_max_abs_error"]),
        "physical_reference_scale": float(physical_reference_scale),
        "physical_tolerance": tolerance,
        "target_original_mlp_calls": 0,
        "capture_call_counters": dict(validation_calls),
        "scored_arm_call_counters": dict(gate["poison_calls"]),
        "max_row_ce_abs_error": float(gate["row_ce_max_abs_drift"]),
    }
    gates = {
        "algebra_identity": True,
        "physical_identity": True,
        "poison_zero_original_calls": True,
        "row_ce_identity": True,
    }
    measurement = {
        "context": context,
        "upstream_state_sha256": upstream_state_sha256,
        "validation_document_ids_sha256": validation_identity,
        "scorer": SCORER,
        "state_sha256": authority.state_logical_sha256(state),
        "integrity_gates": gates,
        "observed": observed,
    }
    return {
        "state": dict(state),
        "context": context,
        "upstream_state_sha256": upstream_state_sha256,
        "validation_document_ids_sha256": validation_identity,
        "scorer": SCORER,
        "integrity_gates": gates,
        "observed": observed,
        "measurement_sha256": authority.logical_json_sha256(measurement),
    }


@torch.no_grad()
def full_native_validation_gate(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int], state0: Mapping[str, Any],
    full_state1: Mapping[str, Any], validation_capture: Mapping[str, torch.Tensor],
    teacher: torch.Tensor, context: Mapping[str, Any], basis1: torch.Tensor,
    *, arm: str,
) -> dict[str, Any]:
    device_state = old_site0._device_state(full_state1, sa.DEV)
    predicted = []
    for start in range(0, validation_capture["z"].shape[0], 1024):
        predicted.append(old_site0.runtime.runtime_projected_output(
            validation_capture["z"][start:start + 1024].to(sa.DEV), device_state
        ).cpu())
    coefficient_error = torch.cat(predicted) - validation_capture["p"]
    physical = []
    basis_device = basis1.to(sa.DEV).float()
    for start in range(0, coefficient_error.shape[0], 1024):
        physical.append((
            coefficient_error[start:start + 1024].to(sa.DEV) @ basis_device.T
        ).cpu())
    physical_max = float(torch.cat(physical).abs().max())
    physical_scale = max(1.0, float(
        (validation_capture["c"] @ basis1.cpu().float().T).abs().max()
    ))
    physical_tolerance = 4e-6 * physical_scale
    if physical_max > physical_tolerance:
        raise RuntimeError("v2.1 site1 full-native physical gate failed")
    metrics, calls = score_candidate(
        sa, hook, rows, twall, all_attention, "full_native_site1", full_state1,
        teacher, context, arm=arm, state0=state0,
        price=authority.selection.state_price(full_state1),
    )
    teacher_rows = _row_ce_from_logits(teacher, rows, device=sa.DEV)
    candidate_rows = _score_rows(
        sa, hook, rows, twall, all_attention, programs={arm: {
            0: state0, 1: full_state1,
        }}, program_name=arm, states={0: "Q", 1: "Q"}, allowed=set(),
    )[0]
    row_drift = float((candidate_rows - teacher_rows).abs().max())
    if row_drift > 2e-6:
        raise RuntimeError("v2.1 site1 full-native row-CE gate failed")
    return {
        "passed": True,
        "physical_max_abs_error": physical_max,
        "physical_tolerance": physical_tolerance,
        "physical_reference_scale": physical_scale,
        "row_ce_max_abs_drift": row_drift,
        "candidate_metrics": metrics,
        "poison_calls": calls,
    }


def _row_ce_from_logits(
    logits: torch.Tensor, rows: torch.Tensor, *, device: Any,
) -> torch.Tensor:
    output = []
    for start in range(0, len(rows), 8):
        stop = min(start + 8, len(rows))
        target = rows[start:stop, 65:257].to(device).contiguous()
        batch = logits[start:stop].to(device)
        ce = F.cross_entropy(
            batch.reshape(-1, batch.shape[-1]), target.reshape(-1), reduction="none",
        ).view(stop - start, -1)
        output.append(ce.double().mean(dim=1).cpu())
    return torch.cat(output).contiguous()


@torch.no_grad()
def _score_rows(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int], *, programs: Mapping[str, Mapping[int, Any]],
    program_name: str, states: Mapping[int, str], allowed: set[int],
) -> tuple[torch.Tensor, dict[int, int]]:
    hook.programs = {
        name: {site: old_site0._device_state(state, sa.DEV)
               for site, state in sites.items()}
        for name, sites in programs.items()
    }
    hook.configure(states, program_name=program_name)
    rows_ce = []
    with old_site0.runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = sa.fwd_arm(
                idx, all_attention, twall, frozenset(range(18))
            ).float()[:, 64:]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets[:, 64:].reshape(-1), reduction="none",
            ).view(len(batch), -1)
            rows_ce.append(ce.double().mean(dim=1).cpu())
    guard.assert_contract(require_allowed_calls=bool(allowed))
    return torch.cat(rows_ce).contiguous(), dict(guard.counts)


class OmissionAuditHook(old_site0.runtime.StateCompleteCorrectionHook):
    """Full-native Q0/Q1 with an optional post-live-state site1 mask."""

    def __init__(self, bases: Mapping[int, torch.Tensor], programs: Mapping[str, Any]):
        super().__init__(bases, programs)
        self.omitted_direction: int | None = None

    def __call__(self, site: int, block: Any, z: torch.Tensor, mo: torch.Tensor) -> torch.Tensor:
        if site != 1 or self.site_states.get(1, "N") != "Q":
            return super().__call__(site, block, z, mo)
        self.calls[site] = self.calls.get(site, 0) + 1
        basis = self.bases[1].to(z.device)
        state = self.programs[self.program_name][1]
        coefficients = old_site0.runtime.runtime_coefficients(z, mo, basis, state)
        if self.omitted_direction is not None:
            coefficients = coefficients.clone()
            coefficients[:, self.omitted_direction] = 0.0
        delta = (coefficients @ basis.T).view_as(mo)
        return mo + delta.to(mo.dtype)


@torch.no_grad()
def causal_omission_audit(
    sa: Any, bases: Mapping[int, torch.Tensor], rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int], *,
    true_state0: Mapping[str, Any], full_state1: Mapping[str, Any],
    predictor_state1: Mapping[str, Any],
    validation_capture: Mapping[str, torch.Tensor], validation_identity: str,
) -> dict[str, Any]:
    programs = {"audit": {
        0: old_site0._device_state(true_state0, sa.DEV),
        1: old_site0._device_state(full_state1, sa.DEV),
    }}
    hook = OmissionAuditHook(bases, programs)
    prior_hook = sa.add_oracle_correction
    try:
        sa.add_oracle_correction = hook
        hook.omitted_direction = None
        full_rows, full_calls = _score_rows(
            sa, hook, rows, twall, all_attention, programs=programs,
            program_name="audit", states={0: "Q", 1: "Q"}, allowed=set(),
        )
        omit_rows = []
        omission_calls = {0: 0, 1: 0, 2: 0}
        for direction in range(authority.compiler.COEFFICIENT_DIM):
            hook.omitted_direction = direction
            row_ce, calls = _score_rows(
                sa, hook, rows, twall, all_attention, programs=programs,
                program_name="audit", states={0: "Q", 1: "Q"}, allowed=set(),
            )
            if calls != {0: 0, 1: 0, 2: 0}:
                raise RuntimeError("v2.1 causal omission original-call poison failed")
            omit_rows.append(row_ce)
        omission_calls = {0: 0, 1: 0, 2: 0}
    finally:
        hook.clear()
        sa.add_oracle_correction = prior_hook
        old_site0.exact_runner.require_inert_correction_state(sa)
    full_rows = full_rows.double().contiguous()
    omit = torch.stack(omit_rows).double().contiguous()
    square_sums = validation_capture["p"].double().square().sum(dim=0).contiguous()
    derived = authority.derive_causal_audit(
        full_rows, omit, square_sums, int(validation_capture["p"].numel() // 64),
    )
    predictor = old_site0._device_state(predictor_state1, sa.DEV)
    predictor_error_square_sums = torch.zeros(
        authority.compiler.COEFFICIENT_DIM, dtype=torch.float64,
    )
    for start in range(0, validation_capture["z"].shape[0], 1024):
        stop = min(start + 1024, validation_capture["z"].shape[0])
        predicted = old_site0.runtime.runtime_projected_output(
            validation_capture["z"][start:stop].to(sa.DEV), predictor,
        ).cpu()
        target_key = (
            "c" if predictor_state1.get("interface") == "z_only_c" else "p"
        )
        error = predicted.double() - validation_capture[target_key][start:stop].double()
        predictor_error_square_sums += error.square().sum(dim=0)
    direction_prediction = authority.derive_direction_prediction(
        derived["omission_losses"], derived["target_second_moments"],
        predictor_error_square_sums, int(validation_capture["p"].numel() // 64),
    )
    return {
        "context": "true_site0",
        "upstream_state_sha256": authority.state_logical_sha256(true_state0),
        "validation_document_ids_sha256": validation_identity,
        "scorer": SCORER,
        "quantile_currency": "torch.float64 q=0.05 interpolation=linear",
        "rule": "abs(loss)/max(second_moment,1e-12); positive 5pct floor; mean-one",
        "full_oracle_row_ce": full_rows,
        "omit_row_ce": omit,
        "full_oracle_row_ce_sha256": authority.tensor_sha256(full_rows),
        "omit_row_ce_sha256": authority.tensor_sha256(omit),
        "target_p_square_sums": square_sums,
        "target_p_square_sums_sha256": authority.tensor_sha256(square_sums),
        "target_p_count": int(validation_capture["p"].numel() // 64),
        **derived,
        "direction_prediction": {
            "predictor_family": "A_v1_like_z_only_affine_euclidean",
            "predictor_state_sha256": authority.state_logical_sha256(
                predictor_state1
            ),
            "predictor_error_square_sums": predictor_error_square_sums,
            "predictor_error_square_sums_sha256": authority.tensor_sha256(
                predictor_error_square_sums
            ),
            "predictor_error_count": int(validation_capture["p"].numel() // 64),
            **direction_prediction,
        },
        "call_counters": {
            "full_oracle": full_calls,
            "omissions": omission_calls,
        },
    }


def _shuffle_sensitivity(bank: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "status": "selected",
            "selection": authority.selection.freeze_control_selection(bank),
        }
    except RuntimeError as error:
        return {"status": "empty", "failure_message": str(error)}


def freeze_and_select_site1(
    ledgers: Mapping[str, Mapping[str, Any]], controls: Mapping[str, Any],
    diagnostics: Mapping[str, Any], *, launch_state: lifecycle.LaunchState,
) -> dict[str, dict[str, Any]]:
    if set(ledgers) != {"true_site1", "shuffle_site1"} or any(
        len(bank) != 108 for bank in ledgers.values()
    ):
        raise RuntimeError("v2.1 site1 numerical banks are incomplete")
    lifecycle.freeze_preselector_stage(
        "site1", ledgers, controls, diagnostics, launch_state=launch_state,
    )
    return lifecycle.select_frozen_stage("site1")


def _run_numerical(launch_state: lifecycle.LaunchState) -> None:
    before = authority.protected_snapshot()
    manifest = {
        "status": "running_v21_site1_before_role_deserialization",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "source_commit": launch_state.source_commit,
        "source_hashes": dict(launch_state.source_hashes),
        "site0_training_receipt_sha256": authority.file_sha256(
            authority.SITE0_TRAINING_RECEIPT
        ),
        "requested_roles": ["compiler_fit_v21", "compiler_validation_v21"],
        "forbidden_roles": ["compiler_final_v21"],
        "protected_before": before,
    }
    authority.write_json_atomic(manifest, MANIFEST)
    site0_authorization = lifecycle.load_site0_training_authorization()
    row_receipt, row_tensors = authority.load_roles_and_validate((
        "compiler_fit_v21", "compiler_validation_v21",
    ))
    rows = {name: value[:, :257].contiguous() for name, value in row_tensors.items()}
    fit_rows = rows["compiler_fit_v21"]
    validation_rows = rows["compiler_validation_v21"]
    fit_document_ids = [
        record["document_id"]
        for record in row_receipt["document_provenance"]["sets"]["compiler_fit_v21"]
    ]
    validation_identity = _validation_identity(row_receipt)
    row_permutation = old_site0.fit.document_block_permutation(
        fit_document_ids, FIT_SEED
    )
    permutation = old_site0.fit.expand_capture_permutation(row_permutation)
    if authority.tensor_sha256(permutation.long()) != (
        authority.expected_fit_permutation_sha256(row_receipt)
    ):
        raise RuntimeError("v2.1 site1 fit permutation changed")
    old_receipt, _ = old_site0.old_rows.validate_receipt()
    code_rows, _ = old_site0.code_oracle.load_frozen_corpus()
    old_site0.frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = old_site0.v3.validate_basis_pair()
    torch.manual_seed(old_site0.exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(old_site0.exact_runner.SHIP_SEED)
    sys.path.insert(0, str(authority.BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    started = time.time()
    callback_state: dict[str, Any] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        realization, _ = old_site0.frozen.restore_ship_realization(
            sa, twall, all_attention, old_receipt, code_rows
        )
        if realization != old_site0.SHIP_HASH:
            raise RuntimeError("v2.1 frozen ship realization changed")
        component_before = old_site0.exact_runner.component_tree_sha256(
            sa, twall, all_attention
        )
        bases = {
            site: bases_payload["sites"][site]["basis"].to(sa.DEV).float()
            for site in (0, 1)
        }
        programs0 = _programs_cpu()
        hook = old_site0.runtime.StateCompleteCorrectionHook(bases, {})
        prior_hook = sa.add_oracle_correction
        sa.add_oracle_correction = hook
        try:
            captures: dict[str, dict[str, torch.Tensor]] = {}
            fit_calls: dict[str, dict[int, int]] = {}
            validation_captures: dict[str, dict[str, torch.Tensor]] = {}
            validation_calls: dict[str, dict[int, int]] = {}
            candidates: dict[str, dict[str, Mapping[str, Any]]] = {}
            for arm in ("true", "shuffle"):
                capture, calls = capture_site1_fit(
                    sa, hook, fit_rows, twall, all_attention, arm=arm,
                    state0=programs0[arm][0], capture_adjoint=True,
                )
                fit_calls[arm] = calls
                if arm == "shuffle":
                    capture = old_site0.shuffled_fit_capture(capture, permutation)
                captures[arm] = capture
                candidates[arm], _ = build_site1_candidates(
                    capture, sa.H[1], bases[1], sa.DEV
                )
                validation_captures[arm], validation_calls[arm] = (
                    capture_site1_validation(
                        sa, hook, validation_rows, twall, all_attention, arm=arm,
                        state0=programs0[arm][0],
                    )
                )
            mean_capture, mean_capture_calls = capture_site1_fit(
                sa, hook, fit_rows, twall, all_attention, arm="mean",
                state0=programs0["mean"][0], capture_adjoint=False,
            )
            mean_sum = mean_capture["p"].double().sum(dim=0).contiguous()
            mean_state1 = old_site0._cpu_state({
                "grammar": "constant",
                "interface": "state_complete_p",
                "family": "fit_mean_control",
                "bias": (mean_sum / authority.FIT_CAPTURE_COUNT).float().contiguous(),
            })
            full_state1 = old_site0.full_native_state(sa.H[1], bases[1])
            ledgers = {"true_site1": {}, "shuffle_site1": {}}
            candidate_calls: dict[str, dict[str, dict[int, int]]] = {
                "true_site1": {}, "shuffle_site1": {},
            }
            contexts = {}
            full_controls = {}
            teachers = {}
            for arm in ("true", "shuffle"):
                ledger_name = f"{arm}_site1"
                teacher, context, teacher_calls = teacher_bank(
                    sa, hook, validation_rows, twall, all_attention, arm=arm,
                    state0=programs0[arm][0],
                )
                teachers[arm] = teacher
                for index, (name, state) in enumerate(sorted(candidates[arm].items())):
                    metrics, calls = score_candidate(
                        sa, hook, validation_rows, twall, all_attention, name, state,
                        teacher, context, arm=arm, state0=programs0[arm][0],
                    )
                    ledgers[ledger_name][name] = {"state": state, "metrics": metrics}
                    candidate_calls[ledger_name][name] = calls
                    print(
                        f"compiler-v2.1 {ledger_name} {index + 1}/108 {name}",
                        flush=True,
                    )
                contexts[ledger_name] = {
                    "upstream_state_sha256": authority.state_logical_sha256(
                        programs0[arm][0]
                    ),
                    "scorer": SCORER,
                    **context,
                    "call_counters": {
                        "fit_capture": fit_calls[arm],
                        "validation_capture": validation_calls[arm],
                        "teacher": teacher_calls["teacher"],
                        "copy_baseline": teacher_calls["baseline"],
                        "candidates": candidate_calls[ledger_name],
                    },
                }
                gate = full_native_validation_gate(
                    sa, hook, validation_rows, twall, all_attention,
                    programs0[arm][0], full_state1, validation_captures[arm],
                    teacher, context, bases[1], arm=arm,
                )
                full_controls[arm] = full_native_control(
                    full_state1, gate, validation_calls[arm],
                    context=f"{arm}_site0",
                    upstream_state_sha256=authority.state_logical_sha256(
                        programs0[arm][0]
                    ), validation_identity=validation_identity,
                    physical_reference_scale=gate["physical_reference_scale"],
                )

            mean_teacher, mean_context, mean_teacher_calls = teacher_bank(
                sa, hook, validation_rows, twall, all_attention, arm="mean",
                state0=programs0["mean"][0],
            )
            mean_metrics, mean_score_calls = score_candidate(
                sa, hook, validation_rows, twall, all_attention, "mean_site1",
                mean_state1, mean_teacher, mean_context, arm="mean",
                state0=programs0["mean"][0], price=authority._constant_price(),
            )
            controls = {
                "mean_site1": mean_state1,
                "full_native_site1_true_context": full_controls["true"],
                "full_native_site1_shuffle_context": full_controls["shuffle"],
            }
            diagnostics = {
                "fit_permutation_sha256": authority.tensor_sha256(permutation.long()),
                "capture_hashes": {
                    "true_fit_site1": _capture_sha256(captures["true"]),
                    "shuffle_fit_site1": _capture_sha256(captures["shuffle"]),
                    "true_validation_site1": _capture_sha256(
                        validation_captures["true"]
                    ),
                    "shuffle_validation_site1": _capture_sha256(
                        validation_captures["shuffle"]
                    ),
                    "mean_fit_site1": _capture_sha256(mean_capture),
                },
                "contexts": contexts,
                "mean_control": {
                    "context": "mean_site0",
                    "upstream_state_sha256": authority.state_logical_sha256(
                        programs0["mean"][0]
                    ),
                    "scorer": "CUDA float32 capture; float64 coefficient sums",
                    "p_sum": mean_sum,
                    "p_sum_sha256": authority.tensor_sha256(mean_sum),
                    "p_count": authority.FIT_CAPTURE_COUNT,
                    "capture_call_counter": mean_capture_calls,
                },
                "mean_context": {
                    "upstream_state_sha256": authority.state_logical_sha256(
                        programs0["mean"][0]
                    ),
                    "scorer": SCORER,
                    **mean_context,
                },
                "mean_score": {
                    "context": "mean_site0",
                    "upstream_state_sha256": authority.state_logical_sha256(
                        programs0["mean"][0]
                    ),
                    "metrics": mean_metrics,
                    "call_counter": mean_score_calls,
                    "teacher_call_counter": mean_teacher_calls["teacher"],
                    "baseline_call_counter": mean_teacher_calls["baseline"],
                },
            }
            selections = freeze_and_select_site1(
                ledgers, controls, diagnostics, launch_state=launch_state,
            )
            callback_state["selected_fit_numerics"] = {
                arm: site0.selected_fit_numerics(
                    captures[arm]["z"],
                    ledgers[f"{arm}_site1"][
                        selections[f"{arm}_site1"]["selected"]
                    ]["state"],
                    selections[f"{arm}_site1"]["selected"],
                ) for arm in ("true", "shuffle")
            }
            callback_state["selected_fit_numerics"]["mean"] = (
                site0.selected_fit_numerics(
                    mean_capture["z"], mean_state1, "mean_site1",
                )
            )
            causal = causal_omission_audit(
                sa, bases, validation_rows, twall, all_attention,
                true_state0=programs0["true"][0], full_state1=full_state1,
                predictor_state1=ledgers["true_site1"][
                    selections["true_site1"]["family_representatives"][
                        "A_v1_like_z_only_affine_euclidean"
                    ]
                ]["state"],
                validation_capture=validation_captures["true"],
                validation_identity=validation_identity,
            )
            token_frequency = authority.derive_token_frequency_strata(
                row_tensors["compiler_fit_v21"],
                row_tensors["compiler_validation_v21"],
                authority.TOKEN_FREQUENCY_BOUNDARIES,
            )
            strata = {
                "source": "compiler_validation_v21",
                "validation_document_ids_sha256": validation_identity,
                "token_frequency": token_frequency,
                "causal_omission_audit": causal,
            }
            site0_payload, _ = lifecycle.load_frozen_stage("site0")
            bundle_controls = {
                "full_native_site0": site0_payload["controls"]["full_native_site0"],
                "full_native_site1_true_context": full_controls["true"],
                "full_native_site1_shuffle_context": full_controls["shuffle"],
                "copy_constrained_shuffle_sensitivity_site0": _shuffle_sensitivity(
                    site0_payload["candidate_ledgers"]["shuffle_site0"]
                ),
                "copy_constrained_shuffle_sensitivity_site1": _shuffle_sensitivity(
                    ledgers["shuffle_site1"]
                ),
            }
            lifecycle.freeze_program_bundle(
                mean_programs={0: programs0["mean"][0], 1: mean_state1},
                controls=bundle_controls, strata=strata, launch_state=launch_state,
            )
            component_after = old_site0.exact_runner.component_tree_sha256(
                sa, twall, all_attention
            )
            if component_after != component_before:
                raise RuntimeError("v2.1 component tree changed during site1")
            callback_state.update({
                "component_before": component_before,
                "component_after": component_after,
                "selections": selections,
            })
        finally:
            hook.clear()
            sa.add_oracle_correction = prior_hook
            old_site0.exact_runner.require_inert_correction_state(sa)
            callback_state["hook_restored"] = True

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)
    closure = lifecycle.close_execution(
        sa, outer_model_returned=True,
        component_tree_before=callback_state.get("component_before", ""),
        component_tree_after=callback_state.get("component_after", ""),
    )
    after = authority.protected_snapshot()
    if after != before:
        raise RuntimeError("v2.1 site1 changed protected historical artifacts")
    manifest.update({
        "status": "completed_v21_site1_program_bundle_pending_final_unlock",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "site0_training_authority": site0_authorization["authority"],
        "site1_ledger_sha256": authority.file_sha256(
            authority.SITE1_LEDGER_ARTIFACT
        ),
        "site1_receipt_sha256": authority.file_sha256(
            authority.SITE1_LEDGER_RECEIPT
        ),
        "programs_artifact_sha256": authority.file_sha256(
            authority.PROGRAMS_ARTIFACT
        ),
        "selected": {
            arm: callback_state["selections"][f"{arm}_site1"]["selected"]
            for arm in ("true", "shuffle")
        },
        "component_tree_unchanged": True,
        "hook_restored_and_inert": callback_state.get("hook_restored") is True,
        "outer_model_returned": True,
        "protected_after": after,
        "runtime_s": round(time.time() - started, 1),
        "peak_cpu_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "peak_gpu_allocated_bytes": int(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "selected_fit_numerics": callback_state["selected_fit_numerics"],
    })
    authority.write_json_atomic(manifest, MANIFEST)
    lifecycle.write_final_unlock_after_outer_return(
        launch_state=launch_state, execution_closure=closure,
    )


def main() -> None:
    if MANIFEST.exists():
        raise RuntimeError(f"refusing to overwrite v2.1 site1 manifest: {MANIFEST}")
    with lifecycle.exclusive_run_claim() as nonce:
        launch_state = lifecycle.resume_after_site0(lock_nonce=nonce)
        try:
            _run_numerical(launch_state)
        except Exception as error:
            if authority.PROGRAMS_RECEIPT.is_file():
                raise
            failure = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
            failure.update({
                "status": "failed_v21_site1_or_program_freeze",
                "authorized_for_training": False,
                "authorized_for_final_scoring": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "preserved_outputs": {
                    str(path): authority.file_sha256(path)
                    for path in (
                        authority.SITE1_LEDGER_ARTIFACT,
                        authority.SITE1_LEDGER_RECEIPT,
                        authority.PROGRAMS_ARTIFACT,
                        authority.PROGRAMS_RECEIPT,
                    ) if path.is_file()
                },
            })
            authority.write_json_atomic(failure, MANIFEST)
            raise


if __name__ == "__main__":
    main()
