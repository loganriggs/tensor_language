#!/usr/bin/env python3
"""Create-only production collector for the frozen MLP1 physical-gate assay."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

import freeze_mlp1_global_gate_plan as frozen_plan
import mlp1_global_gate_analysis as analysis
import mlp_global_gate_response as gate_math
import tensor_bilin18_global_gate_intervention as intervention
import tensor_bilin18_tangent_authority as authority_helpers
import tensor_bilin18_tangent_pilot as parent_pilot


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
OUTPUT = HERE / "tensor_bilin18_mlp1_global_gate_results.json"
BUNDLE = HERE / "tensor_bilin18_mlp1_global_gate_bundle.pt"
AUTHORITY_RECEIPT = HERE / "tensor_bilin18_mlp1_global_gate_authority_receipt.json"
FAILURE = HERE / "tensor_bilin18_mlp1_global_gate_failure.json"
RUN_LOCK = HERE / ".tensor_bilin18_mlp1_global_gate.lock"
PLAN = HERE / "mlp1_global_gate_plan.json"
PREREG = HERE / "MLP1_GLOBAL_GATE_RESPONSE_PREREGISTRATION.md"
ROWS = BQ / ".rowcache_mlp1_global_gate_v1/fineweb_32_source_documents.pt"
ROWS_RECEIPT = BQ / "mlp1_global_gate_v1_rows_receipt.json"
ROW_USE_AUTHORITY = HERE / "mlp1_global_gate_row_use_authority.json"
ROOT_ROW_AUTHORITY = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
RANK640_PARENT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
PARENT_PROGRAM_AUTHORITY = HERE / "tensor_bilin18_tangent_authority_receipt.json"

EXPECTED_PLAN_SHA256 = "4eefcc28ec3ed9fda09b047bb122aa47bc314e29f6a3857bc1da541bf7f5f8b1"
EXPECTED_PLAN_FINGERPRINT = "404f4ca0ee4362c343eab4fd8ea43866a6edf02eb6a80a996d8c23101a726b07"
EXPECTED_ROWS_SHA256 = "bdb34db40fffcfcbb22c88cf958ad7d6894cdce770399f48d7669acec684bd04"
EXPECTED_ROWS_RAW_SHA256 = "a9b79378e7660cac7965870563d4ef1c49e2a6dc148b037d9f063c6b0efec132"
EXPECTED_ROWS_RECEIPT_SHA256 = "63d35040a22c5da69a889cd94ece37cf7c6d353c41ebda3fdbaa12114303b3cd"
EXPECTED_ROW_USE_SHA256 = "9177ca13727e268d4d7ea492d832296b4853ead5b6b4764c4a11444ef3f3b40f"
EXPECTED_ROOT_ROW_AUTHORITY_SHA256 = "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
EXPECTED_RANK640_SHA256 = "639fb8480efee790403113079333100bd63bb61426f6fd6e4dcebd89b21c337d"
EXPECTED_CAUSAL_SHA256 = "73bd18ee81067775680b7d579036e6ec8c04b41116cd3e516b8460a7e7c7ab20"
EXPECTED_PROGRAM_AUTHORITY_SHA256 = "1dc6fa711803e6d7ac1c7958e8507fec66c8dab983c7562c605331ee46adaadd"
EXPECTED_ROWS_IMPLEMENTATION_HASHES = {
    "basis_aligned/polynomial_causal/prepare_mlp1_global_gate_rows.py":
        "84cab37b9750145aa8a5243f83152fbc141a7e299924c22b616f861a1f6b9174",
    "basis_aligned/polynomial_causal/MLP1_GLOBAL_GATE_RESPONSE_PREREGISTRATION.md":
        "aa7fa4067e2232f504eff6a4103b040613c3205f12b1824c2e1d163d667d7914",
    "basis_aligned/polynomial_causal/prepare_mlp0_c512_mlp2_compensation_v1_rows.py":
        "0885ac954ae7f2dce2f035d760443f79d9ff3afc434f6b1335940024ef84988e",
    "basis_aligned/polynomial_causal/prepare_mlp0_native_down_hierarchy_v1_rows.py":
        "f7eee860ef40a6301d78461c5d677394757014b6096ecbb853b8e8d48990ab83",
    "basis_aligned/polynomial_causal/local_fineweb_harvest.py":
        "87d9abeaf1182811650c35bcae25b0373687d2e87aede895bc9f2bc440b90b04",
}

SOURCE_SITE = 1
SCORE_START = 128
SCORE_STOP = 256
DOCUMENTS_PER_COHORT = 16
PROBES_PER_HALF = 32
PRODUCTION_BATCH = 4


def _deduplicate(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


SOURCES = _deduplicate((
    Path(__file__), HERE / "test_tensor_bilin18_mlp1_global_gate_collector.py",
    PREREG, PLAN, HERE / "freeze_mlp1_global_gate_plan.py",
    HERE / "test_freeze_mlp1_global_gate_plan.py",
    ROW_USE_AUTHORITY, HERE / "freeze_mlp1_global_gate_row_use.py",
    HERE / "test_freeze_mlp1_global_gate_row_use.py",
    HERE / "mlp_global_gate_response.py",
    HERE / "test_mlp_global_gate_response.py",
    HERE / "mlp1_global_gate_analysis.py",
    HERE / "test_mlp1_global_gate_analysis.py",
    HERE / "tensor_bilin18_global_gate_intervention.py",
    HERE / "test_tensor_bilin18_global_gate_intervention.py",
    *parent_pilot.SOURCES,
))


def file_sha256(path: Path) -> str:
    return authority_helpers.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return authority_helpers.canonical_sha256(value)


def protected_snapshot() -> dict[str, Any]:
    authority_helpers.require_committed_sources(SOURCES)
    sources = {str(path): file_sha256(path) for path in SOURCES}
    immutable = {
        "plan": file_sha256(PLAN),
        "rows": file_sha256(ROWS),
        "rows_receipt": file_sha256(ROWS_RECEIPT),
        "row_use_authority": file_sha256(ROW_USE_AUTHORITY),
        "root_row_authority": file_sha256(ROOT_ROW_AUTHORITY),
        "rank640_predictive": file_sha256(RANK640_PARENT),
        "rank640_causal": file_sha256(CAUSAL_PARENT),
        "program_authority": file_sha256(PARENT_PROGRAM_AUTHORITY),
    }
    expected = {
        "plan": EXPECTED_PLAN_SHA256,
        "rows": EXPECTED_ROWS_SHA256,
        "rows_receipt": EXPECTED_ROWS_RECEIPT_SHA256,
        "row_use_authority": EXPECTED_ROW_USE_SHA256,
        "root_row_authority": EXPECTED_ROOT_ROW_AUTHORITY_SHA256,
        "rank640_predictive": EXPECTED_RANK640_SHA256,
        "rank640_causal": EXPECTED_CAUSAL_SHA256,
        "program_authority": EXPECTED_PROGRAM_AUTHORITY_SHA256,
    }
    if immutable != expected:
        raise RuntimeError("MLP1 global-gate immutable parent identity changed")
    snapshot = {
        "source_closure": sources,
        "immutable_inputs": immutable,
        "git": authority_helpers.git_identity(SOURCES),
    }
    snapshot["fingerprint"] = canonical_sha256(snapshot)
    return snapshot


def _namespace_paths() -> tuple[Path, ...]:
    return AUTHORITY_RECEIPT, BUNDLE, OUTPUT, FAILURE


def _namespace_contract() -> dict[str, str]:
    return {
        "authority_receipt": str(AUTHORITY_RECEIPT.resolve().relative_to(ROOT)),
        "bundle": str(BUNDLE.resolve().relative_to(ROOT)),
        "result": str(OUTPUT.resolve().relative_to(ROOT)),
        "failure": str(FAILURE.resolve().relative_to(ROOT)),
        "run_lock": str(RUN_LOCK.resolve().relative_to(ROOT)),
    }


def require_authority_namespace_empty() -> None:
    if any(path.exists() for path in _namespace_paths()):
        raise RuntimeError("global-gate authority requires every dedicated namespace absent")


def authority_publication_guard(
    run_lock: authority_helpers.RunLock, expected_snapshot: Mapping[str, Any],
) -> None:
    run_lock.assert_owned()
    if any(path.exists() for path in _namespace_paths()):
        raise RuntimeError("global-gate namespace changed before authority publication")
    if protected_snapshot() != dict(expected_snapshot):
        raise RuntimeError("global-gate protected state changed before authority publication")


def _plan_and_rows() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if file_sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("serialized global-gate plan changed")
    plan = json.loads(PLAN.read_text())
    if plan != frozen_plan.build_plan() or plan.get("plan_fingerprint") != (
        EXPECTED_PLAN_FINGERPRINT
    ) or plan.get("status") != "frozen_cpu_plan_no_gpu_authority" or plan[
        "decision"
    ].get("consequence_stage_authorized") is not False:
        raise RuntimeError("global-gate plan semantics changed")
    rows_receipt = json.loads(ROWS_RECEIPT.read_text())
    row_use = json.loads(ROW_USE_AUTHORITY.read_text())
    predictive = json.loads(RANK640_PARENT.read_text())
    causal = json.loads(CAUSAL_PARENT.read_text())
    if (
        rows_receipt.get("status") != "frozen_before_any_global_gate_model_forward"
        or rows_receipt.get("authorized_for_scored_experiments") is not True
        or rows_receipt.get("authorized_for_training") is not False
        or not all(rows_receipt.get("disjointness_gates", {}).values())
        or row_use.get("status") != "mlp1_global_gate_row_use_frozen_no_model_outcomes"
        or row_use.get("model_training_forbidden") is not True
        or row_use.get("finite_gate_scaling_forbidden") is not True
        or row_use["wave_B_evaluation_only"].get("fit_or_modify_coefficients") is not False
        or predictive.get("status") != "pass" or predictive.get("rank") != 640
        or causal.get("status") != "rank640_robust_pass"
        or rows_receipt.get("implementation_hashes") != EXPECTED_ROWS_IMPLEMENTATION_HASHES
    ):
        raise RuntimeError("global-gate parent semantics changed")
    before_file = file_sha256(ROWS)
    rows = torch.load(ROWS, map_location="cpu", weights_only=True)
    after_file = file_sha256(ROWS)
    if (
        before_file != EXPECTED_ROWS_SHA256 or after_file != before_file
        or tuple(rows.shape) != (32, 513) or rows.dtype != torch.int64
        or frozen_plan.tensor_raw_sha256(rows) != EXPECTED_ROWS_RAW_SHA256
    ):
        raise RuntimeError("global-gate row tensor changed")
    fit = rows[:DOCUMENTS_PER_COHORT, :SCORE_STOP].contiguous()
    validation = rows[DOCUMENTS_PER_COHORT:, :SCORE_STOP].contiguous()
    for name, value in (("fit", fit), ("validation", validation)):
        cohort = plan["cohorts"][name]
        subset = rows[cohort["row_indices"]].contiguous()
        if (
            frozen_plan.tensor_raw_sha256(subset) != cohort["subset_tensor_raw_sha256"]
            or frozen_plan.tensor_raw_sha256(value) != cohort[
                "model_input_256_raw_sha256"
            ] or len(set(cohort["document_ids"])) != DOCUMENTS_PER_COHORT
        ):
            raise RuntimeError(f"global-gate {name} cohort identity changed")
    return plan, fit, validation


def _validate_program_against_parent(program, receipt: Mapping[str, Any]) -> dict[str, Any]:
    authority_helpers.validate_program_receipt(receipt)
    manifest = authority_helpers.program_buffer_manifest(program)
    parent = json.loads(PARENT_PROGRAM_AUTHORITY.read_text())
    if (
        dict(receipt) != parent.get("program_receipt")
        or manifest != parent.get("program_buffers")
    ):
        raise RuntimeError("rebuilt rank640 program differs from parent authority")
    return manifest


def freeze_authority(
    run_lock: authority_helpers.RunLock, runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    require_authority_namespace_empty()
    run_lock.assert_owned()
    before = protected_snapshot()
    plan, _, _ = _plan_and_rows()
    program, program_receipt = parent_pilot.build_rank640_program(torch.device("cuda"))
    manifest = _validate_program_against_parent(program, program_receipt)
    if protected_snapshot() != before:
        raise RuntimeError("protected state changed while freezing global-gate authority")
    result = {
        "status": "mlp1_global_gate_authority_frozen_no_outcomes",
        "protected_snapshot": before,
        "plan_fingerprint": plan["plan_fingerprint"],
        "plan_sha256": file_sha256(PLAN),
        "program_receipt": program_receipt,
        "program_buffers": manifest,
        "runtime_environment": dict(runtime_environment),
        "namespace": _namespace_contract(),
        "namespace_absent_before_authority": {
            name: True for name in ("authority_receipt", "bundle", "result", "failure")
        },
        "product_activations_computed": False,
        "score_targets_sampled": False,
        "score_gradients_computed": False,
        "bundle_computed": False,
        "validation_opened": False,
        "result_computed": False,
    }
    authority_helpers.publish_json_create_only(
        AUTHORITY_RECEIPT, result,
        ownership_check=lambda: authority_publication_guard(run_lock, before),
    )
    return result


def validate_authority(
    value: Any, *, snapshot: Mapping[str, Any], runtime_environment: Mapping[str, Any],
) -> None:
    required = {
        "status", "protected_snapshot", "plan_fingerprint", "plan_sha256",
        "program_receipt", "program_buffers", "runtime_environment",
        "namespace", "namespace_absent_before_authority",
        "product_activations_computed", "score_targets_sampled",
        "score_gradients_computed", "bundle_computed", "validation_opened",
        "result_computed",
    }
    false_fields = required - {
        "status", "protected_snapshot", "plan_fingerprint", "plan_sha256",
        "program_receipt", "program_buffers", "runtime_environment",
        "namespace", "namespace_absent_before_authority",
    }
    if (
        not isinstance(value, dict) or set(value) != required
        or value["status"] != "mlp1_global_gate_authority_frozen_no_outcomes"
        or value["protected_snapshot"] != dict(snapshot)
        or value["plan_fingerprint"] != EXPECTED_PLAN_FINGERPRINT
        or value["plan_sha256"] != EXPECTED_PLAN_SHA256
        or value["runtime_environment"] != dict(runtime_environment)
        or value["namespace"] != _namespace_contract()
        or value["namespace_absent_before_authority"] != {
            name: True for name in ("authority_receipt", "bundle", "result", "failure")
        }
        or any(value[name] is not False for name in false_fields)
    ):
        raise RuntimeError("global-gate authority schema changed")
    authority_helpers.validate_program_receipt(value["program_receipt"])
    parent = json.loads(PARENT_PROGRAM_AUTHORITY.read_text())
    if (
        value["program_receipt"] != parent.get("program_receipt")
        or value["program_buffers"] != parent.get("program_buffers")
    ):
        raise RuntimeError("global-gate authority program buffers differ from parent")


def _publish_torch_create_only(
    path: Path, value: Any, *, ownership_check,
) -> None:
    authority_helpers.publish_torch_create_only(
        path, value, ownership_check=ownership_check,
    )


def _bundle_publication_guard(
    run_lock: authority_helpers.RunLock, expected_snapshot: Mapping[str, Any],
    authority_hash: str,
) -> None:
    run_lock.assert_owned()
    if BUNDLE.exists() or OUTPUT.exists() or FAILURE.exists():
        raise RuntimeError("global-gate outcome namespace changed before bundle publication")
    if file_sha256(AUTHORITY_RECEIPT) != authority_hash or protected_snapshot() != dict(
        expected_snapshot
    ):
        raise RuntimeError("global-gate protected state changed before bundle publication")


def _result_publication_guard(
    run_lock: authority_helpers.RunLock, expected_snapshot: Mapping[str, Any],
    authority_hash: str, bundle_hash: str,
) -> None:
    run_lock.assert_owned()
    if not BUNDLE.exists() or OUTPUT.exists() or FAILURE.exists():
        raise RuntimeError("global-gate result namespace changed before publication")
    if (
        file_sha256(AUTHORITY_RECEIPT) != authority_hash
        or file_sha256(BUNDLE) != bundle_hash
        or protected_snapshot() != dict(expected_snapshot)
    ):
        raise RuntimeError("global-gate protected state changed before result publication")


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{label} receipt schema changed")


def _validate_product_receipt(
    receipt: Mapping[str, Any], tokens: torch.Tensor, product: torch.Tensor,
) -> None:
    _require_exact_keys(receipt, {
        "status", "source_site", "attention_calls", "completed_prior_mlp_calls",
        "product_shape", "tokens_sha256", "products_sha256",
        "raw_logits_returned", "suffix_executed",
    }, "product")
    if (
        receipt["status"] != "complete" or receipt["source_site"] != SOURCE_SITE
        or receipt["attention_calls"] != list(range(SOURCE_SITE + 1))
        or receipt["completed_prior_mlp_calls"] != list(range(SOURCE_SITE))
        or receipt["product_shape"] != list(product.shape)
        or receipt["tokens_sha256"] != intervention.tensor_sha256(tokens)
        or receipt["products_sha256"] != intervention.tensor_sha256(product)
        or receipt["raw_logits_returned"] is not False
        or receipt["suffix_executed"] is not False
    ):
        raise RuntimeError("product receipt did not replay returned tensors and calls")


def _validate_response_receipt(
    receipt: Mapping[str, Any], batch: Any, *, tokens: torch.Tensor,
    row_ids: Sequence[str], first_seeds: Sequence[int], second_seeds: Sequence[int],
    dual: bool, rms: torch.Tensor | None, orientation: torch.Tensor | None,
    permutation: Sequence[int] | None,
) -> int:
    common = {
        "status", "row_ids", "first_probe_seeds", "second_probe_seeds",
        "probe_halves_disjoint", "tokens_sha256", "first_target_ids_sha256",
        "second_target_ids_sha256", "response_shape_per_half", "source_site",
        "score_support", "forward", "raw_logits_returned", "raw_targets_returned",
        "raw_residual_vjps_returned", "graph_aliases_revoked",
    }
    dual_only = {
        "canonical_rms_sha256", "canonical_orientation_sha256",
        "derangement_sha256", "first_native_response_sha256",
        "second_native_response_sha256", "first_deranged_response_sha256",
        "second_deranged_response_sha256",
        "real_and_control_measured_in_same_backward",
    }
    native_only = {
        "first_response_sha256", "second_response_sha256",
        "all_token_positions_share_each_gate_scale",
        "contexts_have_independent_gate_scale_leaves",
    }
    _require_exact_keys(receipt, common | (dual_only if dual else native_only), "response")
    first = batch.first_native if dual else batch.first
    second = batch.second_native if dual else batch.second
    if (
        receipt["status"] != "complete" or receipt["row_ids"] != list(row_ids)
        or receipt["first_probe_seeds"] != list(first_seeds)
        or receipt["second_probe_seeds"] != list(second_seeds)
        or receipt["probe_halves_disjoint"] is not True
        or receipt["tokens_sha256"] != intervention.tensor_sha256(tokens)
        or receipt["response_shape_per_half"] != list(first.shape)
        or receipt["source_site"] != SOURCE_SITE
        or receipt["score_support"] != [SCORE_START, SCORE_STOP]
        or receipt["raw_logits_returned"] is not False
        or receipt["raw_targets_returned"] is not False
        or receipt["raw_residual_vjps_returned"] is not False
        or receipt["graph_aliases_revoked"] is not True
    ):
        raise RuntimeError("response receipt identity or privacy contract changed")
    forward = receipt["forward"]
    expected_forward = {
        "attention_calls": tuple(range(18)), "mlp_calls": tuple(range(18)),
        "source_site": SOURCE_SITE, "scale_shared_across_positions": True,
        "context_scales_independent": True,
    }
    if dual:
        expected_forward |= {
            "native_scale_baseline": 1.0, "deranged_auxiliary_baseline": 0.0,
            "canonical_derangement_fixed_point_free": True,
            "complete_suffix_executed": True,
        }
        assert rms is not None and orientation is not None and permutation is not None
        tensor_hashes = {
            "canonical_rms_sha256": intervention.tensor_sha256(rms),
            "canonical_orientation_sha256": intervention.tensor_sha256(orientation),
            "derangement_sha256": canonical_sha256(list(permutation)),
            "first_native_response_sha256": intervention.tensor_sha256(batch.first_native),
            "second_native_response_sha256": intervention.tensor_sha256(batch.second_native),
            "first_deranged_response_sha256": intervention.tensor_sha256(batch.first_deranged),
            "second_deranged_response_sha256": intervention.tensor_sha256(batch.second_deranged),
        }
        if (
            receipt["real_and_control_measured_in_same_backward"] is not True
            or any(receipt[name] != digest for name, digest in tensor_hashes.items())
        ):
            raise RuntimeError("dual response tensor or shared-backward receipt changed")
    else:
        tensor_hashes = {
            "first_response_sha256": intervention.tensor_sha256(batch.first),
            "second_response_sha256": intervention.tensor_sha256(batch.second),
        }
        if (
            receipt["all_token_positions_share_each_gate_scale"] is not True
            or receipt["contexts_have_independent_gate_scale_leaves"] is not True
            or any(receipt[name] != digest for name, digest in tensor_hashes.items())
        ):
            raise RuntimeError("native response tensor receipt changed")
    if forward != expected_forward:
        raise RuntimeError("response forward call ledger changed")
    return len(first_seeds) + len(second_seeds)


def _collect_products(program, fit_rows: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    chunks = []
    receipts = []
    for start in range(0, DOCUMENTS_PER_COHORT, PRODUCTION_BATCH):
        stop = start + PRODUCTION_BATCH
        tokens = fit_rows[start:stop].to("cuda").contiguous()
        product, receipt = intervention.collect_mlp_product_activations(
            program, tokens,
            source_site=SOURCE_SITE, production=True,
        )
        _validate_product_receipt(receipt, tokens, product)
        chunks.append(product)
        receipts.append(receipt)
        print(f"MLP1 product batch {stop // PRODUCTION_BATCH}/4", flush=True)
    result = torch.cat(chunks, dim=0).contiguous()
    if tuple(result.shape) != (DOCUMENTS_PER_COHORT, SCORE_STOP, 4608):
        raise RuntimeError("global-gate fit product bank is incomplete")
    return result, receipts


def _control_gauge(
    products: torch.Tensor, down: torch.Tensor, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], dict[str, Any]]:
    rms, orientation, pivots = gate_math.factor_product_canonical_gauge(products, down)
    permutation = gate_math.canonical_factor_product_derangement(products, down, seed)
    canonical_h, canonical_d, _ = gate_math.canonicalize_factor_product_gates(
        products, down,
    )
    pattern = torch.tensor([2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125])
    repeats = (products.shape[2] + pattern.numel() - 1) // pattern.numel()
    gauge = pattern.repeat(repeats)[:products.shape[2]].double()
    replay_h, replay_d, _ = gate_math.canonicalize_factor_product_gates(
        products * gauge, down / gauge,
    )
    replay_permutation = gate_math.canonical_factor_product_derangement(
        products * gauge, down / gauge, seed,
    )
    exact_replay = (
        torch.equal(canonical_h, replay_h) and torch.equal(canonical_d, replay_d)
        and permutation == replay_permutation
    )
    if not exact_replay:
        raise RuntimeError("factor-product canonical gauge replay failed")
    receipt = {
        "status": "complete",
        "fit_products": intervention.tensor_sha256(products),
        "down": intervention.tensor_sha256(down),
        "canonical_rms": intervention.tensor_sha256(rms),
        "canonical_orientation": intervention.tensor_sha256(orientation),
        "sign_pivots": intervention.tensor_sha256(pivots),
        "derangement_sha256": canonical_sha256(list(permutation)),
        "derangement_seed": seed,
        "derangement_fixed_point_free": all(
            source != target for source, target in enumerate(permutation)
        ),
        "scale_sign_gauge_replay_exact": exact_replay,
        "permutation_equivariance_source_test_bound": True,
    }
    del canonical_h, canonical_d, replay_h, replay_d
    return rms, orientation, permutation, receipt


def _collect_cohort_responses(
    program, rows: torch.Tensor, row_ids: Sequence[str], *,
    first_seeds: Sequence[int], second_seeds: Sequence[int],
    rms: torch.Tensor | None = None, orientation: torch.Tensor | None = None,
    permutation: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[dict[str, Any]], set[str]]:
    first_chunks, second_chunks, deranged_chunks = [], [], []
    receipts = []
    target_hashes: set[str] = set()
    dual = rms is not None
    if dual != (orientation is not None and permutation is not None):
        raise ValueError("dual cohort controls must be provided together")
    for start in range(0, DOCUMENTS_PER_COHORT, PRODUCTION_BATCH):
        stop = start + PRODUCTION_BATCH
        common = {
            "program": program,
            "tokens": rows[start:stop].to("cuda").contiguous(),
            "row_ids": tuple(row_ids[start:stop]),
            "first_probe_seeds": tuple(first_seeds),
            "second_probe_seeds": tuple(second_seeds),
            "score_start": SCORE_START,
            "score_stop": SCORE_STOP,
            "source_site": SOURCE_SITE,
            "production": True,
        }
        if dual:
            transaction = intervention.DualGlobalGateResponseTransaction(
                **common, canonical_rms=rms, canonical_orientation=orientation,
                derangement=permutation,
            )
            batch = transaction.consume()
            _validate_response_receipt(
                batch.receipt, batch, tokens=common["tokens"],
                row_ids=common["row_ids"], first_seeds=first_seeds,
                second_seeds=second_seeds, dual=True, rms=rms,
                orientation=orientation, permutation=permutation,
            )
            first_chunks.append(batch.first_native)
            second_chunks.append(batch.second_native)
            deranged_chunks.append(batch.first_deranged)
        else:
            transaction = intervention.GlobalGateResponseTransaction(**common)
            batch = transaction.consume()
            _validate_response_receipt(
                batch.receipt, batch, tokens=common["tokens"],
                row_ids=common["row_ids"], first_seeds=first_seeds,
                second_seeds=second_seeds, dual=False, rms=None,
                orientation=None, permutation=None,
            )
            first_chunks.append(batch.first)
            second_chunks.append(batch.second)
        if not transaction.aliases_revoked:
            raise RuntimeError("global-gate response transaction retained graph aliases")
        for name in ("first_target_ids_sha256", "second_target_ids_sha256"):
            target_hash = str(batch.receipt[name])
            if target_hash in target_hashes:
                raise RuntimeError("global-gate target bank replayed within cohort")
            target_hashes.add(target_hash)
        receipts.append(dict(batch.receipt))
        del transaction, batch
        torch.cuda.empty_cache()
        print(f"MLP1 {'fit-dual' if dual else 'validation'} batch {stop // 4}/4", flush=True)
    first = torch.cat(first_chunks, dim=0).contiguous()
    second = torch.cat(second_chunks, dim=0).contiguous()
    deranged = torch.cat(deranged_chunks, dim=0).contiguous() if dual else None
    if (
        tuple(first.shape) != (DOCUMENTS_PER_COHORT, PROBES_PER_HALF, 4608)
        or second.shape != first.shape or len(target_hashes) != 8
        or (dual and (deranged is None or deranged.shape != first.shape))
    ):
        raise RuntimeError("global-gate response cohort is incomplete")
    return first, second, deranged, receipts, target_hashes


def run(
    run_lock: authority_helpers.RunLock, runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if OUTPUT.exists() or BUNDLE.exists() or FAILURE.exists():
        raise RuntimeError("global-gate outcome namespace is create-only and already spent")
    if not AUTHORITY_RECEIPT.exists():
        raise RuntimeError("freeze global-gate authority before measurement")
    started = time.time()
    run_lock.assert_owned()
    before = protected_snapshot()
    authority_hash = file_sha256(AUTHORITY_RECEIPT)
    frozen_authority = json.loads(AUTHORITY_RECEIPT.read_text())
    validate_authority(
        frozen_authority, snapshot=before, runtime_environment=runtime_environment,
    )
    plan, fit_rows, validation_rows = _plan_and_rows()
    program, program_receipt = parent_pilot.build_rank640_program(torch.device("cuda"))
    manifest = _validate_program_against_parent(program, program_receipt)
    if manifest != frozen_authority["program_buffers"]:
        raise RuntimeError("measurement program differs from frozen global-gate authority")
    first_seeds = tuple(plan["probe_halves"]["first"]["probe_seeds"])
    second_seeds = tuple(plan["probe_halves"]["second"]["probe_seeds"])
    fit_ids = tuple(plan["cohorts"]["fit"]["row_ids"])
    validation_ids = tuple(plan["cohorts"]["validation"]["row_ids"])

    product_started = time.time()
    products, product_receipts = _collect_products(program, fit_rows)
    mlp1 = program.mlp_bank.programs[SOURCE_SITE]
    if mlp1.down.weight is None:
        raise RuntimeError("production MLP1 Down unexpectedly is not dense")
    down = mlp1.down.weight.detach().cpu().double().contiguous()
    rms, orientation, permutation, control_receipt = _control_gauge(
        products, down, int(plan["selectors"]["derangement_seed"]),
    )
    product_seconds = time.time() - product_started

    fit_started = time.time()
    fit_first, fit_second, deranged_fit_first, fit_receipts, fit_target_hashes = (
        _collect_cohort_responses(
            program, fit_rows, fit_ids, first_seeds=first_seeds,
            second_seeds=second_seeds, rms=rms, orientation=orientation,
            permutation=permutation,
        )
    )
    assert deranged_fit_first is not None
    fit_summary, core_bundle = analysis.build_fit_gate_bundle(
        fit_first, fit_second, deranged_fit_first=deranged_fit_first,
        activation_rms=rms, down=down, plan=plan,
    )
    analysis.validate_fit_gate_bundle(
        fit_summary, core_bundle, plan, replay_inputs={
            "fit_first": fit_first, "fit_second": fit_second,
            "deranged_fit_first": deranged_fit_first,
            "activation_rms": rms, "down": down,
        },
    )
    bundle_value = {
        "status": "mlp1_global_gate_fit_bundle_frozen_before_validation",
        "plan_fingerprint": plan["plan_fingerprint"],
        "authority_sha256": authority_hash,
        "protected_snapshot_fingerprint": before["fingerprint"],
        "program_buffer_manifest_sha256": manifest["manifest_sha256"],
        "fit_rows_raw_sha256": frozen_plan.tensor_raw_sha256(fit_rows),
        "fit_summary": fit_summary,
        "control_receipt": control_receipt,
        "analysis_bundle": core_bundle,
    }
    _publish_torch_create_only(
        BUNDLE, bundle_value,
        ownership_check=lambda: _bundle_publication_guard(
            run_lock, before, authority_hash,
        ),
    )
    bundle_hash = file_sha256(BUNDLE)
    frozen_bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    if not analysis.tensor_tree_equal(frozen_bundle, bundle_value):
        raise RuntimeError("serialized fit bundle does not exactly replay memory")
    fit_seconds = time.time() - fit_started

    # Validation is opened only after the immutable fit support/coefficient artifact exists.
    bundle_frozen_before_validation = (
        BUNDLE.exists() and file_sha256(BUNDLE) == bundle_hash
    )
    if not bundle_frozen_before_validation:
        raise RuntimeError("fit bundle was not frozen before validation opened")
    validation_started = time.time()
    validation_first, validation_second, _, validation_receipts, validation_target_hashes = (
        _collect_cohort_responses(
            program, validation_rows, validation_ids, first_seeds=first_seeds,
            second_seeds=second_seeds,
        )
    )
    if fit_target_hashes & validation_target_hashes or len(
        fit_target_hashes | validation_target_hashes
    ) != 16:
        raise RuntimeError("fit and validation target banks overlap or are incomplete")
    cells = {
        "fit_first": fit_first,
        "fit_second": fit_second,
        "validation_first": validation_first,
        "validation_second": validation_second,
    }
    scientific, rebuilt_bundle = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged_fit_first,
        activation_rms=rms, down=down, plan=plan,
    )
    analysis.validate_gate_analysis_result(
        scientific, plan, replay_inputs={
            "cells": cells, "deranged_fit_first": deranged_fit_first,
            "activation_rms": rms, "down": down, "bundle": core_bundle,
        },
    )
    validation_did_not_alter_bundle = analysis.tensor_tree_equal(
        core_bundle, rebuilt_bundle,
    ) and file_sha256(BUNDLE) == bundle_hash
    if not validation_did_not_alter_bundle:
        raise RuntimeError("validation altered the frozen fit bundle")
    validation_seconds = time.time() - validation_started
    response_receipts = fit_receipts + validation_receipts
    backward_passes = sum(
        len(receipt["first_probe_seeds"]) + len(receipt["second_probe_seeds"])
        for receipt in response_receipts
    )
    unique_target_hashes = len(fit_target_hashes | validation_target_hashes)
    if (
        len(product_receipts) != DOCUMENTS_PER_COHORT // PRODUCTION_BATCH
        or len(fit_receipts) != DOCUMENTS_PER_COHORT // PRODUCTION_BATCH
        or len(validation_receipts) != DOCUMENTS_PER_COHORT // PRODUCTION_BATCH
        or backward_passes != 2 * DOCUMENTS_PER_COHORT // PRODUCTION_BATCH * (
            len(first_seeds) + len(second_seeds)
        )
        or unique_target_hashes != 2 * len(response_receipts)
    ):
        raise RuntimeError("derived global-gate execution ledger is incomplete")
    response_hashes = {
        name: intervention.tensor_sha256(value) for name, value in cells.items()
    }
    result = {
        "status": scientific["status"],
        "scope": scientific["scope"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "scientific": scientific,
        "fit_phase": fit_summary,
        "control": control_receipt,
        "program": program_receipt,
        "program_buffers": manifest,
        "execution": {
            "product_batches": len(product_receipts),
            "fit_response_batches": len(fit_receipts),
            "validation_response_batches": len(validation_receipts),
            "backward_passes": backward_passes,
            "unique_target_hashes": unique_target_hashes,
            "product_receipts": product_receipts,
            "fit_response_receipts": fit_receipts,
            "validation_response_receipts": validation_receipts,
            "response_hashes": response_hashes,
            "bundle_frozen_before_validation": bundle_frozen_before_validation,
            "validation_did_not_alter_bundle": validation_did_not_alter_bundle,
            "raw_logits_published": False,
            "raw_targets_published": False,
            "raw_vjps_published": False,
            "raw_responses_published": False,
        },
        "provenance": {
            "protected_snapshot": before,
            "authority_sha256": authority_hash,
            "bundle_sha256": bundle_hash,
            "rows_sha256": file_sha256(ROWS),
            "rows_raw_sha256": EXPECTED_ROWS_RAW_SHA256,
            "fit_row_ids": list(fit_ids),
            "validation_row_ids": list(validation_ids),
        },
        "runtime_environment": dict(runtime_environment),
        "runtime_s": {
            "product_and_canonical_control": product_seconds,
            "fit_response_analysis_and_bundle": fit_seconds,
            "validation_response_and_analysis": validation_seconds,
            "total": time.time() - started,
        },
    }
    after = protected_snapshot()
    if (
        after != before or file_sha256(AUTHORITY_RECEIPT) != authority_hash
        or file_sha256(BUNDLE) != bundle_hash
        or authority_helpers.program_buffer_manifest(program) != manifest
        or frozen_plan.tensor_raw_sha256(fit_rows) != plan["cohorts"]["fit"][
            "model_input_256_raw_sha256"
        ] or frozen_plan.tensor_raw_sha256(validation_rows) != plan[
            "cohorts"
        ]["validation"]["model_input_256_raw_sha256"]
    ):
        raise RuntimeError("global-gate protected state changed before result publication")
    authority_helpers.publish_json_create_only(
        OUTPUT, result,
        ownership_check=lambda: _result_publication_guard(
            run_lock, before, authority_hash, bundle_hash,
        ),
    )
    del products, fit_first, fit_second, deranged_fit_first
    del validation_first, validation_second, cells
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _failure_publication_guard(
    run_lock: authority_helpers.RunLock, expected_snapshot: Mapping[str, Any],
    authority_hash: str, bundle_hash: str | None,
) -> None:
    run_lock.assert_owned()
    if FAILURE.exists() or OUTPUT.exists() or not AUTHORITY_RECEIPT.exists():
        raise RuntimeError("global-gate failure namespace changed before publication")
    if (
        file_sha256(AUTHORITY_RECEIPT) != authority_hash
        or protected_snapshot() != dict(expected_snapshot)
        or (bundle_hash is None and BUNDLE.exists())
        or (bundle_hash is not None and (
            not BUNDLE.exists() or file_sha256(BUNDLE) != bundle_hash
        ))
    ):
        raise RuntimeError("global-gate protected state changed before failure publication")


def _publish_failure(run_lock: authority_helpers.RunLock, error: BaseException) -> None:
    if FAILURE.exists() or OUTPUT.exists() or not AUTHORITY_RECEIPT.exists():
        return
    authority_hash = file_sha256(AUTHORITY_RECEIPT)
    authority = json.loads(AUTHORITY_RECEIPT.read_text())
    expected_snapshot = authority.get("protected_snapshot")
    if not isinstance(expected_snapshot, dict):
        return
    bundle_hash = file_sha256(BUNDLE) if BUNDLE.exists() else None
    authority_helpers.publish_json_create_only(FAILURE, {
        "status": "mlp1_global_gate_execution_failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "authority_sha256": authority_hash,
        "protected_snapshot_fingerprint": expected_snapshot.get("fingerprint"),
        "bundle_exists": bundle_hash is not None,
        "bundle_sha256": bundle_hash,
        "timestamp_unix": time.time(),
    }, ownership_check=lambda: _failure_publication_guard(
        run_lock, expected_snapshot, authority_hash, bundle_hash,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-authority", action="store_true")
    arguments = parser.parse_args()
    runtime = authority_helpers.configure_production_runtime()
    with authority_helpers.exclusive_run_lock(RUN_LOCK) as run_lock:
        try:
            result = (
                freeze_authority(run_lock, runtime)
                if arguments.freeze_authority else run(run_lock, runtime)
            )
        except BaseException as error:
            if not arguments.freeze_authority:
                _publish_failure(run_lock, error)
            raise
    print(json.dumps({
        "status": result["status"],
        "plan_fingerprint": result.get("plan_fingerprint"),
        "result": str(AUTHORITY_RECEIPT if arguments.freeze_authority else OUTPUT),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
