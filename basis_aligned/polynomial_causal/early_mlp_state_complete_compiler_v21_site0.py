#!/usr/bin/env python3
"""Numerical site-0 stage for the preregistered compiler-v2.1 pipeline.

This stage loads fit plus mapped-validation rows only.  It captures and fits the
two exact 108-cell banks, scores every candidate in the registered CUDA
currency, freezes the external preselector ledger, and writes the site1
training authorization only after the outer model invocation returns.
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
import prepare_state_complete_compiler_rows_v21 as authority  # noqa: E402


MANIFEST = authority.SITE0_MANIFEST
FIT_SEED = 271828


def ridge_condition_numbers(z: torch.Tensor) -> dict[str, Any]:
    normalized, _, _ = old_site0.fit._normalized_fit(z)
    eigenvalues = torch.linalg.eigvalsh(
        normalized.T @ normalized / normalized.shape[0]
    ).double()
    minimum, maximum = float(eigenvalues[0]), float(eigenvalues[-1])
    by_lambda = {}
    for ridge in old_site0.affine_v1.LAMBDA_GRID:
        denominator = minimum + float(ridge)
        by_lambda[str(float(ridge))] = {
            "status": "singular_or_indefinite" if denominator <= 0 else "evaluated",
            "value": None if denominator <= 0 else (
                (maximum + float(ridge)) / denominator
            ),
        }
    return {
        "matrix": "normalized fit Gram plus lambda I",
        "rows": int(normalized.shape[0]), "columns": int(normalized.shape[1]),
        "minimum_gram_eigenvalue": minimum,
        "maximum_gram_eigenvalue": maximum,
        "condition_number_by_lambda": by_lambda,
    }


def float32_replay(z: torch.Tensor, state: Mapping[str, Any]) -> dict[str, Any]:
    sample = z[:64].contiguous()
    deployed = old_site0.runtime.runtime_projected_output(sample, state).double()
    x = sample.double().reshape(-1, old_site0.compiler.D_MODEL)
    if state["grammar"] == "affine":
        normalized = (x - state["mean"].double()) / state["scale"].double()
        reference = (
            (normalized @ state["left"].double()) @ state["right"].double()
            + state["bias"].double()
        )
    elif state["grammar"] == "native":
        reference = old_site0.compiler.native_projected_output(x, state)
    elif state["grammar"] == "constant":
        reference = state["bias"].double().expand(x.shape[0], -1)
    else:
        raise RuntimeError("v2.1 precision replay encountered unknown grammar")
    error = deployed - reference
    return {
        "status": "evaluated_serialized_float32_parameters",
        "support_positions": int(sample.shape[0]),
        "reference": "float64 accumulation", "deployed": "float32 accumulation",
        "max_abs_coefficient_drift": float(error.abs().max()),
        "rms_coefficient_drift": float(error.square().mean().sqrt()),
    }


def selected_fit_numerics(
    z: torch.Tensor, state: Mapping[str, Any], name: str,
    *, condition_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "selected": name, "grammar": state["grammar"],
        "ridge_condition_numbers": dict(
            condition_report if condition_report is not None
            else ridge_condition_numbers(z)
        ),
        "float64_to_float32_replay": float32_replay(z, state),
        "quantization_status": (
            "none; all floating parameter tensors float32; native indices int64"
        ),
    }


def _capture_sha256(value: Mapping[str, torch.Tensor]) -> str:
    return authority.logical_json_sha256({
        key: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": authority.tensor_sha256(tensor),
        }
        for key, tensor in sorted(value.items())
    })


def _validation_identity(row_receipt: Mapping[str, Any]) -> str:
    records = row_receipt["document_provenance"]["sets"]["compiler_validation_v21"]
    return authority.logical_json_sha256([record["document_id"] for record in records])


def metrics_from_sufficient_statistics(
    state: Mapping[str, Any], context: Mapping[str, Any], raw: Mapping[str, Any],
    *, price: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the complete selector metric record from immutable raw aggregates."""

    token_count = raw["candidate_teacher_kl_count"]
    copy_count = raw["copy_ce_count"]
    if token_count != raw["global_ce_count"] or token_count <= 0 or copy_count <= 0:
        raise RuntimeError("v2.1 candidate scorer counts are invalid")
    candidate_kl = float(raw["candidate_teacher_kl_sum"]) / token_count
    denominator = float(context["teacher_denominator"])
    global_ce = float(raw["global_ce_sum"]) / token_count
    copy_ce = float(raw["copy_ce_sum"]) / copy_count
    return {
        "candidate_teacher_kl": candidate_kl,
        "oracle_denominator_kl": denominator,
        "remaining_kl_ratio": candidate_kl / denominator,
        "recovery": 1.0 - candidate_kl / denominator,
        "global_ce": global_ce,
        "copy_ce": copy_ce,
        "copy_count": copy_count,
        "copy_worsening": copy_ce - float(context["copy_baseline"]),
        "price": (
            authority.selection.state_price(state) if price is None else dict(price)
        ),
        "raw_sufficient_statistics": dict(raw),
    }


@torch.no_grad()
def teacher_bank(
    sa: Any, hook: Any, rows: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, dict[int, int]]]:
    """Return OON logits and exact OON-vs-NON/copy sufficient statistics."""

    parts: dict[str, list[torch.Tensor]] = {"OON": [], "NON": []}
    targets_parts: list[torch.Tensor] = []
    idx_parts: list[torch.Tensor] = []
    counters: dict[str, dict[int, int]] = {}
    runtime = old_site0.runtime
    for name, states, allowed in (
        ("OON", {0: "O", 1: "O"}, {0, 1}),
        ("NON", {1: "O"}, {1}),
    ):
        hook.configure(states)
        with runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8].to(sa.DEV)
                idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
                logits = sa.fwd_arm(
                    idx, all_attention, twall, frozenset(range(18))
                ).float()[:, 64:]
                parts[name].append(logits.cpu().contiguous())
                if name == "OON":
                    targets_parts.append(targets.cpu())
                    idx_parts.append(idx.cpu())
        guard.assert_contract(require_allowed_calls=True)
        counters[name] = dict(guard.counts)

    teacher, baseline = torch.cat(parts["OON"]), torch.cat(parts["NON"])
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
        kl = (teacher_logp.exp() * (teacher_logp - baseline_logp)).sum(dim=-1)
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
        raise RuntimeError("v2.1 site0 teacher/copy sufficient statistics changed")
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
    all_attention: frozenset[int], name: str, state: Mapping[str, Any],
    teacher: torch.Tensor, context: Mapping[str, Any],
    *, price: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Score one QON candidate and retain the literal aggregate sums/counts."""

    runtime = old_site0.runtime
    hook.programs = {name: {0: old_site0._device_state(state, sa.DEV)}}
    hook.configure({0: "Q", 1: "O"}, program_name=name)
    kl_sum = 0.0
    global_ce_sum = 0.0
    copy_ce_sum = 0.0
    token_count = 0
    copy_count = 0
    with runtime.OriginalMLPCallGuard(sa.H, {1}) as guard:
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
    guard.assert_contract(require_allowed_calls=True)
    if token_count != authority.VALIDATION_TOKEN_COUNT or copy_count != context[
        "copy_token_count"
    ]:
        raise RuntimeError(f"v2.1 {name} scorer support changed")
    raw = {
        "candidate_teacher_kl_sum": kl_sum,
        "candidate_teacher_kl_count": token_count,
        "global_ce_sum": global_ce_sum,
        "global_ce_count": token_count,
        "copy_ce_sum": copy_ce_sum,
        "copy_ce_count": copy_count,
    }
    metrics = metrics_from_sufficient_statistics(state, context, raw, price=price)
    return metrics, dict(guard.counts)


def full_native_control(
    state: Mapping[str, Any], gate: Mapping[str, Any], validation_calls: Mapping[int, int],
    *, validation_identity: str, physical_reference_scale: float,
) -> dict[str, Any]:
    """Convert the measured v2 identity gate to the exact v2.1 control schema."""

    if gate.get("passed") is not True:
        raise RuntimeError("v2.1 site0 full-native gate did not pass")
    physical_tolerance = float(gate["physical_tolerance"])
    if physical_tolerance != 4e-6 * max(1.0, physical_reference_scale):
        raise RuntimeError("v2.1 site0 full-native scale binding changed")
    observed = {
        "physical_max_abs_error": float(gate["physical_max_abs_error"]),
        "physical_reference_scale": float(physical_reference_scale),
        "physical_tolerance": physical_tolerance,
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
        "context": "baseline",
        "upstream_state_sha256": "baseline",
        "validation_document_ids_sha256": validation_identity,
        "scorer": "CUDA float32 per-token; float64 row/aggregate",
        "state_sha256": authority.state_logical_sha256(state),
        "integrity_gates": gates,
        "observed": observed,
    }
    return {
        "state": dict(state),
        "context": "baseline",
        "upstream_state_sha256": "baseline",
        "validation_document_ids_sha256": validation_identity,
        "scorer": measurement["scorer"],
        "integrity_gates": gates,
        "observed": observed,
        "measurement_sha256": authority.logical_json_sha256(measurement),
    }


def freeze_and_select_site0(
    ledgers: Mapping[str, Mapping[str, Any]], controls: Mapping[str, Any],
    diagnostics: Mapping[str, Any], *, launch_state: lifecycle.LaunchState,
) -> dict[str, dict[str, Any]]:
    """Expose one testable ordering boundary: both banks freeze before selection."""

    if set(ledgers) != {"true_site0", "shuffle_site0"} or any(
        len(bank) != 108 for bank in ledgers.values()
    ):
        raise RuntimeError("v2.1 site0 numerical banks are incomplete")
    lifecycle.freeze_preselector_stage(
        "site0", ledgers, controls, diagnostics, launch_state=launch_state,
    )
    return lifecycle.select_frozen_stage("site0")


def _run_numerical(launch_state: lifecycle.LaunchState) -> None:
    before = authority.protected_snapshot()
    manifest = {
        "status": "running_v21_site0_before_role_deserialization",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "source_commit": launch_state.source_commit,
        "source_hashes": dict(launch_state.source_hashes),
        "rows_receipt_sha256": launch_state.rows_receipt_sha256,
        "rows_manifest_sha256": launch_state.rows_manifest_sha256,
        "requested_roles": ["compiler_fit_v21", "compiler_validation_v21"],
        "forbidden_roles": ["compiler_final_v21"],
        "protected_before": before,
    }
    authority.write_json_atomic(manifest, MANIFEST)
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
    old_receipt, _ = old_site0.old_rows.validate_receipt()
    code_rows, _ = old_site0.code_oracle.load_frozen_corpus()
    old_site0.frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = old_site0.v3.validate_basis_pair()
    manifest["status"] = "running_v21_site0_numerical_stage"
    authority.write_json_atomic(manifest, MANIFEST)
    torch.manual_seed(old_site0.exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(old_site0.exact_runner.SHIP_SEED)
    sys.path.insert(0, str(authority.BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    callback_state: dict[str, Any] = {}
    started = time.time()
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
        basis = bases_payload["sites"][0]["basis"].to(sa.DEV).float()
        hook = old_site0.runtime.StateCompleteCorrectionHook({
            0: basis,
            1: bases_payload["sites"][1]["basis"].to(sa.DEV).float(),
        }, {})
        prior_hook = sa.add_oracle_correction
        sa.add_oracle_correction = hook
        try:
            captured, fit_calls = old_site0.capture_site0_fit(
                sa, hook, fit_rows, twall, all_attention
            )
            candidates, _ = old_site0.build_site0_candidates(
                captured, sa.H[0], basis, sa.DEV
            )
            row_permutation = old_site0.fit.document_block_permutation(
                fit_document_ids, FIT_SEED
            )
            permutation = old_site0.fit.expand_capture_permutation(row_permutation)
            shuffled_capture = old_site0.shuffled_fit_capture(captured, permutation)
            shuffled_candidates, _ = old_site0.build_site0_candidates(
                shuffled_capture, sa.H[0], basis, sa.DEV
            )
            validation_capture, validation_calls = old_site0.capture_site0_validation(
                sa, hook, validation_rows, twall, all_attention
            )
            teacher, context_base, teacher_calls = teacher_bank(
                sa, hook, validation_rows, twall, all_attention
            )
            mean_state = old_site0._cpu_state(old_site0.fit.constant_state(captured["p"]))
            full_state = old_site0.full_native_state(sa.H[0], basis)
            retry = __import__("early_mlp_state_complete_compiler_v2_site0_retry1")
            gate = retry.full_native_gate(
                sa, hook, validation_rows, twall, all_attention, full_state,
                validation_capture, teacher, context_base["copy_baseline"], basis,
            )
            full_control = full_native_control(
                full_state, gate, validation_calls,
                validation_identity=validation_identity,
                physical_reference_scale=max(1.0, float(
                    (validation_capture["c"] @ basis.cpu().float().T).abs().max()
                )),
            )
            ledgers: dict[str, dict[str, Any]] = {
                "true_site0": {}, "shuffle_site0": {},
            }
            candidate_calls: dict[str, dict[str, dict[int, int]]] = {
                "true_site0": {}, "shuffle_site0": {},
            }
            for ledger_name, states in (
                ("true_site0", candidates), ("shuffle_site0", shuffled_candidates),
            ):
                for index, (name, state) in enumerate(sorted(states.items())):
                    metrics, calls = score_candidate(
                        sa, hook, validation_rows, twall, all_attention, name, state,
                        teacher, context_base,
                    )
                    ledgers[ledger_name][name] = {"state": state, "metrics": metrics}
                    candidate_calls[ledger_name][name] = calls
                    print(
                        f"compiler-v2.1 {ledger_name} {index + 1}/{len(states)} {name}",
                        flush=True,
                    )
            mean_metrics, mean_calls = score_candidate(
                sa, hook, validation_rows, twall, all_attention, "mean_site0",
                mean_state, teacher, context_base, price=authority._constant_price(),
            )
            contexts = {}
            for ledger_name in ("true_site0", "shuffle_site0"):
                contexts[ledger_name] = {
                    "upstream_state_sha256": "baseline",
                    "scorer": "CUDA float32 per-token; float64 row/aggregate",
                    **context_base,
                    "call_counters": {
                        "fit_capture": fit_calls,
                        "validation_capture": validation_calls,
                        "teacher": teacher_calls["OON"],
                        "copy_baseline": teacher_calls["NON"],
                        "candidates": candidate_calls[ledger_name],
                    },
                }
            controls = {
                "mean_site0": mean_state,
                "full_native_site0": full_control,
            }
            diagnostics = {
                "fit_permutation_sha256": authority.tensor_sha256(permutation.long()),
                "capture_hashes": {
                    "fit_original": _capture_sha256(captured),
                    "fit_shuffled": _capture_sha256(shuffled_capture),
                    "validation_site0": _capture_sha256(validation_capture),
                },
                "contexts": contexts,
                "mean_score": {
                    "context": "baseline",
                    "upstream_state_sha256": "baseline",
                    "metrics": mean_metrics,
                    "call_counter": mean_calls,
                },
            }
            selections = freeze_and_select_site0(
                ledgers, controls, diagnostics, launch_state=launch_state,
            )
            condition_report = ridge_condition_numbers(captured["z"])
            callback_state["selected_fit_numerics"] = {
                arm: selected_fit_numerics(
                    captured["z"],
                    ledgers[f"{arm}_site0"][
                        selections[f"{arm}_site0"]["selected"]
                    ]["state"],
                    selections[f"{arm}_site0"]["selected"],
                    condition_report=condition_report,
                ) for arm in ("true", "shuffle")
            }
            callback_state["selected_fit_numerics"]["mean"] = selected_fit_numerics(
                captured["z"], mean_state, "mean_site0",
                condition_report=condition_report,
            )
            component_after = old_site0.exact_runner.component_tree_sha256(
                sa, twall, all_attention
            )
            if component_after != component_before:
                raise RuntimeError("v2.1 component tree changed during site0")
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
        raise RuntimeError("v2.1 site0 changed protected historical artifacts")
    manifest.update({
        "status": "completed_v21_site0_pending_last_training_authority",
        "authorized_for_training": False,
        "training_license_sites": [],
        "authorized_for_final_scoring": False,
        "site0_ledger_sha256": authority.file_sha256(authority.SITE0_LEDGER_ARTIFACT),
        "site0_receipt_sha256": authority.file_sha256(authority.SITE0_LEDGER_RECEIPT),
        "selected": {
            arm: callback_state["selections"][f"{arm}_site0"]["selected"]
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
    lifecycle.write_site0_training_authorization_after_outer_return(
        launch_state=launch_state, execution_closure=closure,
    )


def main() -> None:
    if MANIFEST.exists():
        raise RuntimeError(f"refusing to overwrite v2.1 site0 manifest: {MANIFEST}")
    with lifecycle.exclusive_run_claim() as nonce:
        launch_state = lifecycle.verify_launch(lock_nonce=nonce)
        try:
            _run_numerical(launch_state)
        except Exception as error:
            if authority.SITE0_TRAINING_RECEIPT.is_file():
                raise
            failure = (
                json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
            )
            failure.update({
                "status": "failed_v21_site0_numerical_stage",
                "authorized_for_training": False,
                "training_license_sites": [],
                "authorized_for_final_scoring": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "preserved_outputs": {
                    str(path): authority.file_sha256(path)
                    for path in (
                        authority.SITE0_LEDGER_ARTIFACT,
                        authority.SITE0_LEDGER_RECEIPT,
                        authority.SITE0_TRAINING_RECEIPT,
                    )
                    if path.is_file()
                },
            })
            authority.write_json_atomic(failure, MANIFEST)
            raise


if __name__ == "__main__":
    main()
