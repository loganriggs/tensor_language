#!/usr/bin/env python3
"""Freeze the outcome-blind 2x2 MLP1 global-gate response assay.

This builder reads only committed provenance and prior authorities.  It partitions 32
registry-wide fresh source documents into fit and validation waves and crosses both
with two new categorical-Fisher probe halves.  It does not run the model, compute a
response, or grant GPU authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
ROWS_RECEIPT = BQ / "mlp1_global_gate_v1_rows_receipt.json"
ROWS = BQ / ".rowcache_mlp1_global_gate_v1/fineweb_32_source_documents.pt"
ROW_USE_AUTHORITY = HERE / "mlp1_global_gate_row_use_authority.json"
RANK640_PREDICTIVE = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
RANK640_CAUSAL = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
PROGRAM_AUTHORITY = HERE / "tensor_bilin18_tangent_authority_receipt.json"
OUT = HERE / "mlp1_global_gate_plan.json"

EXPECTED_ROWS_RECEIPT_SHA256 = "63d35040a22c5da69a889cd94ece37cf7c6d353c41ebda3fdbaa12114303b3cd"
EXPECTED_ROWS_FILE_SHA256 = "bdb34db40fffcfcbb22c88cf958ad7d6894cdce770399f48d7669acec684bd04"
EXPECTED_ROWS_RAW_SHA256 = "a9b79378e7660cac7965870563d4ef1c49e2a6dc148b037d9f063c6b0efec132"
EXPECTED_ROW_USE_AUTHORITY_SHA256 = "9177ca13727e268d4d7ea492d832296b4853ead5b6b4764c4a11444ef3f3b40f"
EXPECTED_RANK640_PREDICTIVE_SHA256 = "639fb8480efee790403113079333100bd63bb61426f6fd6e4dcebd89b21c337d"
EXPECTED_RANK640_CAUSAL_SHA256 = "73bd18ee81067775680b7d579036e6ec8c04b41116cd3e516b8460a7e7c7ab20"
EXPECTED_PROGRAM_AUTHORITY_SHA256 = "1dc6fa711803e6d7ac1c7958e8507fec66c8dab983c7562c605331ee46adaadd"

DERANGEMENT_SEED = 2026082806
RANDOM_CONTROL_SEED = 2026082807
BOOTSTRAP_SEED = 2026082808
BOOTSTRAP_REPETITIONS = 20_000
FIRST_PROBE_SEED = 2026083101
SECOND_PROBE_SEED = 2026090101
DOCUMENTS_PER_COHORT = 16
PROBES_PER_HALF = 32
SCORE_START = 128
SCORE_STOP = 256
SOURCE_SITE = 1
GATES = 4608
BUDGETS = (32, 128, 512)
TARGET_RANK_BY_BUDGET = ((32, 16), (128, 64), (512, 256))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_raw_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _validate_parents() -> dict[str, str]:
    protected = {
        "rows_receipt": (ROWS_RECEIPT, EXPECTED_ROWS_RECEIPT_SHA256),
        "rows_file": (ROWS, EXPECTED_ROWS_FILE_SHA256),
        "row_use_authority": (ROW_USE_AUTHORITY, EXPECTED_ROW_USE_AUTHORITY_SHA256),
        "rank640_predictive": (RANK640_PREDICTIVE, EXPECTED_RANK640_PREDICTIVE_SHA256),
        "rank640_causal": (RANK640_CAUSAL, EXPECTED_RANK640_CAUSAL_SHA256),
        "program_authority": (PROGRAM_AUTHORITY, EXPECTED_PROGRAM_AUTHORITY_SHA256),
    }
    observed = {name: file_sha256(path) for name, (path, _) in protected.items()}
    if any(observed[name] != expected for name, (_, expected) in protected.items()):
        raise RuntimeError("global-gate protected parent identity changed")
    predictive = json.loads(RANK640_PREDICTIVE.read_text())
    causal = json.loads(RANK640_CAUSAL.read_text())
    program = json.loads(PROGRAM_AUTHORITY.read_text())
    row_use = json.loads(ROW_USE_AUTHORITY.read_text())
    if (
        predictive.get("status") != "pass" or predictive.get("rank") != 640
        or causal.get("status") != "rank640_robust_pass"
        or program.get("status") != "rank640_program_authority_frozen_no_outcomes"
        or row_use.get("status") != "mlp1_global_gate_row_use_frozen_no_model_outcomes"
        or row_use.get("parent_rows_receipt_sha256") != EXPECTED_ROWS_RECEIPT_SHA256
        or row_use.get("model_training_forbidden") is not True
        or row_use.get("wave_B_evaluation_only", {}).get("select_or_modify_support")
        is not False
    ):
        raise RuntimeError("rank640 predictive, causal, or program parent is not admitted")
    return observed


def _cohort_record(
    name: str, indices: list[int], documents: list[str], rows: torch.Tensor,
) -> dict[str, Any]:
    subset = rows[indices].contiguous()
    inputs = subset[:, :SCORE_STOP].contiguous()
    return {
        "name": name,
        "contexts": len(indices),
        "one_context_per_document": True,
        "row_indices": indices,
        "row_ids": [f"mlp1_global_gate_v1:{index}" for index in indices],
        "document_ids": documents,
        "subset_shape": list(subset.shape),
        "model_input_shape": list(inputs.shape),
        "subset_tensor_raw_sha256": tensor_raw_sha256(subset),
        "model_input_256_raw_sha256": tensor_raw_sha256(inputs),
    }


def build_plan() -> dict[str, Any]:
    parent_hashes = _validate_parents()
    rows = torch.load(ROWS, map_location="cpu", weights_only=True)
    if tuple(rows.shape) != (32, 513) or rows.dtype != torch.int64 or (
        tensor_raw_sha256(rows) != EXPECTED_ROWS_RAW_SHA256
    ):
        raise RuntimeError("global-gate row tensor changed")
    authority = json.loads(ROWS_RECEIPT.read_text())
    if (
        authority.get("status") != "frozen_before_any_global_gate_model_forward"
        or authority.get("authorized_for_scored_experiments") is not True
        or authority.get("authorized_for_training") is not False
        or not all(authority.get("disjointness_gates", {}).values())
        or authority.get("entries", {}).get("all", {}).get("tensor_raw_sha256")
        != EXPECTED_ROWS_RAW_SHA256
        or authority.get("entries", {}).get("all", {}).get("file_sha256")
        != EXPECTED_ROWS_FILE_SHA256
    ):
        raise RuntimeError("global-gate row authority is not active")
    provenance = authority["document_provenance"]["sets"]["all"]
    if len(provenance) != len(rows) or any(
        not isinstance(record.get("document_id"), str)
        or type(record.get("dataset_document_index")) is not int
        for record in provenance
    ):
        raise RuntimeError("global-gate row provenance schema changed")
    fit_indices = list(range(DOCUMENTS_PER_COHORT))
    validation_indices = list(range(DOCUMENTS_PER_COHORT, 2 * DOCUMENTS_PER_COHORT))
    fit_documents = [provenance[index]["document_id"] for index in fit_indices]
    validation_documents = [
        provenance[index]["document_id"] for index in validation_indices
    ]
    if set(fit_documents) & set(validation_documents) or set(fit_indices) & set(
        validation_indices
    ):
        raise RuntimeError("fit and validation response cohorts overlap")

    first_seeds = tuple(FIRST_PROBE_SEED + index for index in range(PROBES_PER_HALF))
    second_seeds = tuple(SECOND_PROBE_SEED + index for index in range(PROBES_PER_HALF))
    if set(first_seeds) & set(second_seeds):
        raise RuntimeError("global-gate probe halves overlap")

    plan: dict[str, Any] = {
        "status": "frozen_cpu_plan_no_gpu_authority",
        "claim": (
            "checkpoint-relative MLP1 physical-gate tangent response in the admitted "
            "rank640 shell; registry-fresh FineWeb fit/validation, not cross-corpus OOD, "
            "finite deletion, intrinsic tensor rank, or autoregressive-rollout Fisher"
        ),
        "operator": {
            "source_site": SOURCE_SITE,
            "gates": GATES,
            "response_layout": ["context", "probe", "gate"],
            "one_independent_gate_scale_per_context": True,
            "each_gate_scale_shared_across_all_input_positions": [0, SCORE_STOP],
            "score_positions": [SCORE_START, SCORE_STOP],
            "score": "sum of sampled categorical log probabilities over positions 128:256",
            "probes_per_half": PROBES_PER_HALF,
            "backward_passes_at_batch4": (
                2 * 2 * (DOCUMENTS_PER_COHORT // 4) * PROBES_PER_HALF
            ),
        },
        "cohorts": {
            "fit": _cohort_record("registry_fresh_fit_wave_A", fit_indices, fit_documents, rows),
            "validation": _cohort_record(
                "registry_fresh_validation_wave_B", validation_indices,
                validation_documents, rows,
            ),
            "selection_rule": (
                "the outcome-blind row freezer orders 32 one-row source documents; "
                "indices 0:16 are fit wave A and indices 16:32 are validation wave B"
            ),
            "document_disjoint": True,
            "registry_wide_disjoint_from_every_prior_role": True,
            "validation_is_fresh_fineweb_not_cross_corpus_ood": True,
        },
        "probe_halves": {
            "first": {
                "base_seed": FIRST_PROBE_SEED,
                "probe_seeds": list(first_seeds),
            },
            "second": {
                "base_seed": SECOND_PROBE_SEED,
                "probe_seeds": list(second_seeds),
            },
            "new_relative_to_historical_probe_halves": True,
            "disjoint": True,
            "stateless_uniform": (
                "(uint64_be(sha256(f'{seed}:{row_id}:{absolute_position}')[:8])+0.5)/2**64"
            ),
        },
        "selectors": {
            "budgets": list(BUDGETS),
            "primary": "context-balanced deterministic top ridge-leverage score",
            "target_rank_by_budget": {str(k): rank for k, rank in TARGET_RANK_BY_BUDGET},
            "ridge_lambda": "squared rank-r tail Frobenius norm divided by r",
            "candidate_bundle": (
                "support and all-on coefficients fit on fit documents/probe-half first "
                "only; every other cell is evaluation-only and cannot alter the bundle"
            ),
            "controls": {
                "response_energy": "top column squared norm on the same fit response",
                "activation_down": (
                    "top sqrt(mean h_n(z_q)^2 over fit documents and q=0:256) "
                    "times Euclidean norm of Down column n"
                ),
                "factor_product_derangement": (
                    "on fit-wave A, normalize each h_n trace to unit RMS and absorb its "
                    "inverse scale into d_n; orient the joint sign so the first "
                    "maximum-absolute d_n coordinate is positive; order gates by "
                    "sha256(derangement_seed || canonical float64 h_n trace || "
                    "canonical d_n bytes); "
                    "bytes; define pi(order[i])=order[(i+1) mod 4608], with exact "
                    "content then smaller native index as collision tie-breaks; at the "
                    "unchanged native baseline measure auxiliary zero-leaf columns "
                    "sum_q hcanon_n(z_q)*dcanon_pi(n)^T*g_q, then apply the identical "
                    "balanced ridge selector and target-rank rule"
                ),
                "hash_random": "smallest sha256(random_control_seed:gate_id)",
            },
            "derangement_seed": DERANGEMENT_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "tie_rule": "higher score first, then smaller physical gate index",
            "raw_weighted_currency_reported": True,
            "context_balanced_currency_reported": True,
            "only_context_balanced_primary_can_promote": True,
            "identical_context_balancing_for_response_energy_and_deranged_response": True,
            "factor_product_derangement_shift": "+1 in ascending canonical-content hash order",
            "factor_product_derangement_fixed_point_free": True,
            "factor_product_derangement_scale_sign_gauge_invariant": True,
            "factor_product_derangement_gauge_replay_required": True,
            "activation_down_support_is_fit_wave_A_only": True,
            "all_control_transfer_metrics_use_the_same_balanced_evaluation_responses": True,
        },
        "linear_solver": {
            "precision": "CPU torch.float64",
            "algorithm": "torch.linalg.svd with full_matrices=False",
            "intercept": False,
            "relative_singular_cutoff": 1e-10,
            "relative_tikhonov_ridge_to_largest_squared_singular_value": 1e-6,
            "solution": (
                "V*diag(s/(s^2+lambda))*U^T*target after zeroing singular values "
                "below relative_singular_cutoff*s_max"
            ),
            "maximum_unregularized_retained_condition_number": 1e6,
            "maximum_css_frobenius_norm_divided_by_sqrt_number_of_gates": 10.0,
            "maximum_all_on_l2_norm_divided_by_sqrt_budget": 10.0,
            "failure_rule": (
                "a solver diagnostic breach rejects that budget for the primary or "
                "control independently; no fallback solver, ridge, or cutoff"
            ),
            "serialization": (
                "support indices in score order and float64 coefficients with dtype, "
                "shape, raw-byte SHA-256, solver diagnostics, and plan fingerprint"
            ),
            "same_solver_for_primary_and_every_control": True,
        },
        "metrics": {
            "promotive_css": (
                "fit column interpolant on fit/first and transfer identical matrix to "
                "fit/second, validation/first, and validation/second"
            ),
            "promotive_all_on": (
                "fit coefficients for E@1 on fit/first and transfer identical support "
                "and coefficients to all three evaluation cells"
            ),
            "nonpromotive_span": "support span recomputed inside each cell",
            "support_stability": "Jaccard of fit/first versus fit/second selections",
            "score_rank_stability": (
                "nonpromotive Spearman correlation of all 4608 fit/first versus "
                "fit/second selector scores, using average ranks for exact ties"
            ),
            "document_unit": "one document/context; never a token or probe",
            "per_document_css_loss": (
                "||E_eval[d,S] X_fit - E_eval[d]||_F^2 / "
                "max(||E_eval[d]||_F^2,1e-30)"
            ),
            "per_document_all_on_loss": (
                "||E_eval[d,S] beta_fit - E_eval[d] 1||_2^2 / "
                "max(||E_eval[d] 1||_2^2,1e-30)"
            ),
            "pooled_loss": (
                "sum document loss numerators divided by sum matching denominators; "
                "never mean of per-document ratios"
            ),
            "relative_improvement": (
                "(pooled_control_loss-pooled_primary_loss)/"
                "max(pooled_control_loss,1e-30); positive favors primary"
            ),
            "bootstrap": {
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": BOOTSTRAP_SEED,
                "simultaneous_family_size": 48,
                "family": "3 budgets x 4 controls x 2 metrics x 2 validation probe halves",
                "simultaneous_confidence": 0.95,
                "method": "shared-document nonstudentized basic max-error bootstrap",
                "paired_recompute_selectors": False,
                "rule": (
                    "for each replicate resample 16 validation document indices with "
                    "replacement and use that same index vector for every primary, "
                    "control, metric, half, and budget; recompute all 48 pooled-loss "
                    "improvements with supports and coefficients fixed; take the "
                    "maximum over j of observed_improvement_j-bootstrap_improvement_j"
                ),
                "critical_value": (
                    "ascending max-error order statistic at one-indexed rank "
                    "ceil(0.95*20000)=19000; no interpolation"
                ),
                "simultaneous_lcb": "observed_improvement_j minus the shared critical value",
            },
        },
        "decision": {
            "support_jaccard_minimum": 0.5,
            "relative_improvement_lcb_over_every_control_minimum": 0.05,
            "maximum_per_document_primary_minus_each_control_loss": 0.02,
            "best_control_selection": "none; primary is compared separately with every control",
            "both_validation_probe_halves_must_pass": True,
            "all_three_budgets_observed_improvement_must_be_positive_for_every_comparison": True,
            "full_numeric_pass_required_at_least_one_budget": True,
            "promoted_budget": "smallest budget with a full numeric pass",
            "css_and_all_on_must_both_pass": True,
            "bootstrap_unit": "validation document",
            "multiplicity": (
                "one shared 95% max-error bootstrap band across all 48 promotive "
                "comparisons; within a budget every control, metric, and validation "
                "half must pass, so no best-arm selection"
            ),
            "failure": (
                "no stable physical-gate support at registered budgets; do not run "
                "finite gate scaling and advance to the native quadratic-form Gram audit"
            ),
            "consequence_stage_authorized": False,
        },
        "finite_followup_if_promoted": {
            "candidate_path": "alpha(epsilon)=1+epsilon*(beta_tilde-1)",
            "epsilon": 0.1,
            "selected_only_alpha_0_9_is_sensitivity_control": True,
            "fisher_to_kl_normalization": "0.5*epsilon^2",
            "authorized_by_this_plan": False,
        },
        "prices": {
            "standalone_native_support_values_per_site": "3456*K+1152",
            "bilinear_multiplies_per_token": "K",
            "support_precision_and_metadata_required": True,
            "response_rank_is_not_a_storage_price": True,
        },
        "parents": {
            "hashes": parent_hashes,
            "rows_raw_sha256": EXPECTED_ROWS_RAW_SHA256,
        },
        "publication": {
            "fit_bundle_support_and_coefficients_are_outcomes": True,
            "raw_logits_published": False,
            "raw_targets_published": False,
            "raw_residual_vjps_published": False,
            "response_tensor_published": False,
            "aggregate_selector_ledgers_published": True,
            "create_only_result_and_bundle_namespaces_required": True,
        },
        "remaining_authority": (
            "A committed create-only collector must bind this exact plan, complete "
            "scientific source closure, row bytes, rank640 program buffers, result and "
            "bundle namespaces, run lock, and final-write snapshot before GPU execution."
        ),
    }
    plan["plan_fingerprint"] = canonical_sha256(plan)
    return plan


def main() -> None:
    if OUT.exists():
        raise RuntimeError("global-gate plan is create-only and already exists")
    OUT.write_text(json.dumps(build_plan(), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "frozen", "path": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
