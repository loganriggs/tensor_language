#!/usr/bin/env python3
"""Freeze the outcome-blind MLP1 same-context split-probe assay.

This reads only frozen CPU provenance and existing tangent authorities.  It selects one
row per document by stateless hashing, fixes a common injection position, and creates
two disjoint categorical-Fisher probe plans.  It performs no model forward, gradient,
geometry fit, or outcome analysis and grants no GPU authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
ROWS = BQ / ".rowcache/fineweb_n96_skip80.pt"
ROW_AUTHORITY = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
PARENT_PLAN = HERE / "finite_horizon_tangent_plan.json"
PARENT_RESULT = HERE / "tensor_bilin18_tangent_pilot_results.json"
PARENT_PROGRAM_AUTHORITY = HERE / "tensor_bilin18_tangent_authority_receipt.json"
PARENT_GEOMETRY = HERE / "tensor_bilin18_tangent_geometry.pt"
PARENT_GEOMETRY_AUTHORITY = HERE / "tensor_bilin18_tangent_geometry_receipt.json"
OUT = HERE / "mlp1_split_probe_plan.json"

EXPECTED_ROWS_FILE_SHA256 = "94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda"
EXPECTED_ROWS_RAW_SHA256 = "a703cadb1a5e27497cba43d21bca889a1d765b861c3da311a1dc4dfeb28b21cc"
EXPECTED_ROW_AUTHORITY_SHA256 = "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
EXPECTED_PARENT_PLAN_SHA256 = "0e1e0aa35b70d6761658a055d308728cd172196369f2ec5e9d79b543d87d74e9"
EXPECTED_PARENT_RESULT_SHA256 = "efd788fa0089008c4a2b0767244f1759453f02dd6e98b31aceae3847b26bc9d4"
EXPECTED_PROGRAM_AUTHORITY_SHA256 = "1dc6fa711803e6d7ac1c7958e8507fec66c8dab983c7562c605331ee46adaadd"
EXPECTED_GEOMETRY_SHA256 = "5f8aeac18fef087b9217eedfde4fff254275e94f2b1b9716c03a3a1bcd5a40be"
EXPECTED_GEOMETRY_AUTHORITY_SHA256 = "2b96c001db6053934dd1aa8f33a5cbbcac3e81b59b2525e089baf8e89e7f0e1b"
EXPECTED_MLP1_DIRECTIONS_SHA256 = "efa4a5d3956bfe3a52e561adb6565f54a850e28095f4b9b59a8726f59f918e7f"

SELECTION_SEED = 2026082804
DIRECTION_SEED = 2026082801
FIRST_PROBE_SEED = 2026082901
SECOND_PROBE_SEED = 2026083001
CONTEXTS = 16
PROBES_PER_HALF = 32
INJECTION_POSITION = 128

SPLIT_PROBE_PROTOCOL = {
    "claim": (
        "MLP1 final-output response geometry only; H_c=D_cE_c does not identify "
        "encoder versus decoder variation; conditional follow-up on historical rows "
        "already exposed by the parent tangent result, not fresh-document confirmation"
    ),
    "direction_rule": (
        "reuse only the parent site's frozen 32x1152 MLP1 direction matrix after "
        "validating the complete parent authority chain; do not refit directions"
    ),
    "fisher_rule": (
        "two disjoint halves of 32 independent full-50304-way categorical score "
        "probes from baseline float32 probabilities; stateless sha256 inverse-CDF"
    ),
    "causal_score_rule": (
        "for every selected context inject at absolute position 128 and sum scores "
        "over every output t=128,...,255"
    ),
    "physical_geometry_rule": (
        "direction_matrix^T=Q R; H_physical=H R^-1; compare Q times right singular "
        "frames of H_physical at fixed ranks 8,16,24"
    ),
    "replication_rule": (
        "same 16 ordered one-per-document contexts in both independent probe halves; "
        "fixed first 12 hash-ordered contexts form the promotion cohort and remaining "
        "4 are diagnostics; compare same-context noise with cross-context variation"
    ),
}


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


def selection_digest(label: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{label}".encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


SPLIT_PROBE_PROTOCOL_SHA256 = canonical_sha256(SPLIT_PROBE_PROTOCOL)


def half_plan_fingerprint(
    *, experiment_id: str, probe_seed: int, row_ids: tuple[str, ...],
    document_ids: tuple[str, ...], splits: tuple[str, ...],
) -> str:
    return canonical_sha256({
        "experiment_id": experiment_id,
        "row_artifact_sha256": EXPECTED_ROWS_RAW_SHA256,
        "row_ids": row_ids,
        "document_ids": document_ids,
        "splits": splits,
        "scored_positions": (INJECTION_POSITION,) * CONTEXTS,
        "input_dims": ((1, 32),),
        "target_site": 3,
        "probes_per_row": PROBES_PER_HALF,
        "direction_seed": DIRECTION_SEED,
        "probe_seed": probe_seed,
        "selection_seed": SELECTION_SEED,
        "protocol_sha256": SPLIT_PROBE_PROTOCOL_SHA256,
    })


def build_plan() -> dict[str, Any]:
    protected = {
        "rows_file": (ROWS, EXPECTED_ROWS_FILE_SHA256),
        "row_authority": (ROW_AUTHORITY, EXPECTED_ROW_AUTHORITY_SHA256),
        "parent_plan": (PARENT_PLAN, EXPECTED_PARENT_PLAN_SHA256),
        "parent_result": (PARENT_RESULT, EXPECTED_PARENT_RESULT_SHA256),
        "program_authority": (
            PARENT_PROGRAM_AUTHORITY, EXPECTED_PROGRAM_AUTHORITY_SHA256,
        ),
        "geometry": (PARENT_GEOMETRY, EXPECTED_GEOMETRY_SHA256),
        "geometry_authority": (
            PARENT_GEOMETRY_AUTHORITY, EXPECTED_GEOMETRY_AUTHORITY_SHA256,
        ),
    }
    observed_hashes = {}
    for name, (path, expected) in protected.items():
        observed_hashes[name] = file_sha256(path)
        if observed_hashes[name] != expected:
            raise RuntimeError(f"protected {name} identity changed")

    rows = torch.load(ROWS, map_location="cpu", weights_only=True)
    if tuple(rows.shape) != (96, 513) or rows.dtype != torch.int64:
        raise RuntimeError("frozen row tensor shape or dtype changed")
    if tensor_raw_sha256(rows) != EXPECTED_ROWS_RAW_SHA256:
        raise RuntimeError("frozen row tensor content changed")
    authority = json.loads(ROW_AUTHORITY.read_text())
    parent_plan = json.loads(PARENT_PLAN.read_text())
    if (
        authority.get("authority") != "pinned_local_ordered_manifest"
        or authority.get("authorized_for_scored_experiments") is not True
        or parent_plan.get("status") != "frozen_cpu_plan_no_gpu_authority"
        or parent_plan.get("plan_fingerprint") != (
            "b9caa7ce2ecbd63a197262098931541c32dce27ed31b35454753b773f8cf4e20"
        )
    ):
        raise RuntimeError("row authority or parent tangent plan semantics changed")
    provenance = authority["document_provenance"]["sets"]["n96_skip80"]
    expected_provenance_keys = {
        "document_id", "dataset_document_index", "chunk_id", "token_start",
    }
    if len(provenance) != len(rows) or any(
        set(record) != expected_provenance_keys
        or not isinstance(record["document_id"], str) or not record["document_id"]
        or any(type(record[key]) is not int or record[key] < 0 for key in (
            "dataset_document_index", "chunk_id", "token_start",
        )) for record in provenance
    ) or authority["entries"]["n96_skip80"]["tensor_raw_sha256"] != (
        EXPECTED_ROWS_RAW_SHA256
    ):
        raise RuntimeError("row provenance schema or tensor binding changed")

    rows_by_document: dict[str, list[int]] = {}
    for index, record in enumerate(provenance):
        rows_by_document.setdefault(record["document_id"], []).append(index)
    ranked_documents = sorted(rows_by_document, key=selection_digest)
    selected_documents = ranked_documents[:CONTEXTS]
    selected_indices = []
    for document in selected_documents:
        selected_indices.append(min(
            rows_by_document[document],
            key=lambda index: selection_digest(f"{document}:n96_skip80:{index}"),
        ))
    row_ids = tuple(f"n96_skip80:{index}" for index in selected_indices)
    document_ids = tuple(selected_documents)
    if len(set(row_ids)) != CONTEXTS or len(set(document_ids)) != CONTEXTS:
        raise RuntimeError("selected MLP1 contexts are not row/document unique")
    selected_rows = rows[selected_indices].contiguous()
    selected_inputs = selected_rows[:, :256].contiguous()
    splits = tuple("primary" if index < CONTEXTS // 2 else "replication"
                   for index in range(CONTEXTS))

    first_fingerprint = half_plan_fingerprint(
        experiment_id="bilin18-mlp1-same-context-probe-half-a-v1",
        probe_seed=FIRST_PROBE_SEED, row_ids=row_ids,
        document_ids=document_ids, splits=splits,
    )
    second_fingerprint = half_plan_fingerprint(
        experiment_id="bilin18-mlp1-same-context-probe-half-b-v1",
        probe_seed=SECOND_PROBE_SEED, row_ids=row_ids,
        document_ids=document_ids, splits=splits,
    )
    first_seeds = tuple(FIRST_PROBE_SEED + i for i in range(PROBES_PER_HALF))
    second_seeds = tuple(SECOND_PROBE_SEED + i for i in range(PROBES_PER_HALF))
    if set(first_seeds) & set(second_seeds):
        raise RuntimeError("probe halves overlap")

    parent_result = json.loads(PARENT_RESULT.read_text())
    geometry_authority = json.loads(PARENT_GEOMETRY_AUTHORITY.read_text())
    if (
        parent_result["geometry"]["sites"]["1"]["directions_sha256"]
        != EXPECTED_MLP1_DIRECTIONS_SHA256
        or geometry_authority["geometry_receipt"]["geometry_manifest_sha256"]
        != parent_result["geometry"]["geometry_manifest_sha256"]
    ):
        raise RuntimeError("parent MLP1 direction identity changed")

    result = {
        "status": "frozen_cpu_plan_no_gpu_authority",
        "claim": (
            "MLP1 final-output response geometry only; H_c=D_cE_c does not identify "
            "encoder versus decoder variation"
        ),
        "protocol": SPLIT_PROBE_PROTOCOL,
        "protocol_sha256": SPLIT_PROBE_PROTOCOL_SHA256,
        "selection": {
            "rule": (
                "take the 16 documents with smallest sha256(selection_seed:document_id); "
                "within each take the row with smallest "
                "sha256(selection_seed:document_id:row_id)"
            ),
            "seed": SELECTION_SEED,
            "contexts": CONTEXTS,
            "one_context_per_document": True,
            "row_indices": selected_indices,
            "row_ids": list(row_ids),
            "document_ids": list(document_ids),
            "subset_tensor_raw_sha256": tensor_raw_sha256(selected_rows),
            "model_input_256_raw_sha256": tensor_raw_sha256(selected_inputs),
            "subset_shape": list(selected_rows.shape),
            "model_input_shape": list(selected_inputs.shape),
            "analysis_splits": list(splits),
            "promotion_context_indices": list(range(12)),
            "diagnostic_context_indices": list(range(12, 16)),
            "promotion_cohort_rule": (
                "first 12 contexts in the frozen document-hash order; fixed before outcomes"
            ),
            "common_injection_position": INJECTION_POSITION,
            "future_output_positions_per_context": 256 - INJECTION_POSITION,
        },
        "operator": {
            "source_site": 1,
            "directions": 32,
            "physical_write_width": 1152,
            "target": "sum of categorical scores at every output t=128,...,255",
            "probes_per_half": PROBES_PER_HALF,
            "backward_passes_at_batch4": 2 * (CONTEXTS // 4) * PROBES_PER_HALF,
            "fixed_physical_projector_ranks": [8, 16, 24],
        },
        "probe_halves": {
            "first": {
                "plan_fingerprint": first_fingerprint,
                "probe_seed": FIRST_PROBE_SEED,
                "probe_seeds": list(first_seeds),
            },
            "second": {
                "plan_fingerprint": second_fingerprint,
                "probe_seed": SECOND_PROBE_SEED,
                "probe_seeds": list(second_seeds),
            },
            "stateless_uniform": (
                "(uint64_be(sha256(f'{seed}:{row_id}:{absolute_position}')[:8])+0.5)/2**64"
            ),
            "disjoint": True,
            "same_ordered_contexts": True,
        },
        "parent_authority": {
            "hashes": observed_hashes,
            "row_authority_sha256": file_sha256(ROW_AUTHORITY),
            "parent_plan_sha256": file_sha256(PARENT_PLAN),
            "mlp1_directions_sha256": EXPECTED_MLP1_DIRECTIONS_SHA256,
            "geometry_manifest_sha256": parent_result[
                "geometry"
            ]["geometry_manifest_sha256"],
            "reuse_rule": (
                "load only parent site-1 directions after validating the complete parent "
                "program/geometry/result authority chain; do not refit directions"
            ),
        },
        "analysis": {
            "module": "finite_horizon_tangent_bundle.analyze_repeated_probe_physical_bundle",
            "energy_fraction": 0.95,
            "gap_ratio": 2.0,
            "support_rtol_relative_to_context_leader": 1e-12,
            "local_rank_limit": 16,
            "maximum_half_rank_difference": 2,
            "maximum_same_context_physical_projector_distance": 0.15,
            "minimum_context_fraction": 0.75,
            "minimum_cross_minus_same_bootstrap_lcb_95": 0.05,
            "bootstrap_repetitions": 1000,
            "bootstrap_seed": 20260828,
            "physical_frame": (
                "direction_matrix^T=Q R; H_physical=H R^-1; "
                "U_cr=Q V_cr(H_physical)"
            ),
            "bundle_contrast_population": (
                "the fixed first-12 hash-ordered promotion cohort; all 12 must pass "
                "support, selected-rank, rank-agreement, and same-context rank-16 stability"
            ),
        },
        "decision": {
            "probe_limited_high_rank": (
                "both halves support rank >=24 and r95>16 in at least 75% of contexts; "
                "prune any <=16 local tangent-state story"
            ),
            "stable_context_varying_response_bundle": (
                "both halves have numerical support >=16, admitted local rank <=16, "
                "and same-context rank-16 physical distance <=0.15 in all 12 members "
                "of the fixed promotion cohort; within that same fixed cohort, the "
                "rank-16 cross-minus-same physical distance has bootstrap LCB >=0.05"
            ),
            "no_admitted_local_bundle": (
                "same-context halves are unstable or neither registered alternative passes; "
                "do not fit context-to-chart transport"
            ),
            "consequence_stage_authorized": False,
        },
        "prohibitions": {
            "raw_logits_published": False,
            "raw_responses_published": False,
            "physical_frames_published": False,
            "projectors_published": False,
            "finite_replacement_claim": False,
            "encoder_gauge_claim": False,
        },
        "remaining_authority": (
            "A create-only paired collector must validate source/program/geometry/row "
            "closure and pass independent lifecycle audit before GPU execution."
        ),
    }
    result["plan_fingerprint"] = canonical_sha256(result)
    return result


def main() -> None:
    result = build_plan()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "plan_fingerprint": result["plan_fingerprint"],
        "row_indices": result["selection"]["row_indices"],
        "probe_fingerprints": {
            key: value["plan_fingerprint"]
            for key, value in result["probe_halves"].items()
            if isinstance(value, dict)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
