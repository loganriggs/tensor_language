#!/usr/bin/env python3
# BQGATE: frozen A-E query-index instrument diagnostic; CUDA is managed-queue only.
"""Diagnose the query/full-sequence projection audit mismatch in source compression v1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_fast_screen_producer as producer
import run_aspectual_anchor_attention11h3_15h5_source_compression_split_v1 as invalid_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1.json"
INVALID_RESULT = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_split_v1_result.json"
INVALID_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11h3_15h5_source_compression_split_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.attention11h3_15h5_source_projection_query_diagnostic_v1"
EXPECTED_PRIOR_SHA256 = "fc9a4c39adaef15c6dbc13da297a6184c25be16e49c2e2f06ce98d5269605863"
EXPECTED_INVALID_RESULT_SHA256 = "7b4ba19260f4d311ca959331e60f29c64136f416566582d510591c96d5243adb"
EXPECTED_INVALID_RUNNER_SHA256 = "419c0be9a1c9cebf594225c492b9402af7240d7ffd95d7987320d3e2ecb18a30"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
MODEL_FORWARDS_MAX = 8
EXAMPLE_EVALUATIONS_MAX = 64


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        INVALID_RESULT: EXPECTED_INVALID_RESULT_SHA256,
        INVALID_RUNNER: EXPECTED_INVALID_RUNNER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    invalid = json.loads(INVALID_RESULT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID or invalid.get("terminal") != "invalid":
        raise ExperimentError("prior or invalid-result authority changed")
    _selection, confirmation, spec, _parent = invalid_runner.validate_static()
    if invalid_runner.suffix.ids_sha256(confirmation) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation split changed")
    return confirmation, spec


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "row_count": len(rows), "boundaries": list(invalid_runner.BOUNDARIES),
        "head_by_boundary": invalid_runner.HEAD_BY_BOUNDARY,
        "source_roles": list(invalid_runner.ROLES),
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = invalid_runner.SourceCompressionBackend.load("cuda")
    term_error = {boundary: 0.0 for boundary in invalid_runner.BOUNDARIES}
    query_error = {boundary: 0.0 for boundary in invalid_runner.BOUNDARIES}
    off_query_error = {boundary: 0.0 for boundary in invalid_runner.BOUNDARIES}
    full_error = {boundary: 0.0 for boundary in invalid_runner.BOUNDARIES}
    partition_rows = 0
    records = []
    forward_calls = evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            role_banks = backend.role_positions(base_batch, donor_batch)
            partition_rows += len(role_banks)
            _base_native, base_bilinear = backend.capture_bilinear(base_batch)
            _donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            _base_manual, base_capture = backend.capture_suffix_heads(base_batch)
            _writer_output, hybrid_capture, _writer_error = backend.capture_writer_suffix_heads(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            for boundary in invalid_runner.BOUNDARIES:
                base_pattern, base_value, base_reconstruction = backend.attention_terms(
                    base_batch, base_capture, boundary
                )
                hybrid_pattern, hybrid_value, hybrid_reconstruction = backend.attention_terms(
                    base_batch, hybrid_capture, boundary
                )
                term_error[boundary] = max(
                    term_error[boundary], base_reconstruction, hybrid_reconstruction
                )
                projected_sources = backend.projected_source_delta(
                    base_batch, role_banks, (base_pattern, base_value),
                    (hybrid_pattern, hybrid_value), boundary, invalid_runner.ROLES
                )
                projected_head = backend.projected_head_delta(
                    base_capture, hybrid_capture, boundary,
                    (invalid_runner.HEAD_BY_BOUNDARY[boundary],),
                )
                difference = (projected_sources - projected_head).abs().float()
                full_error[boundary] = max(full_error[boundary], float(difference.max()))
                for i, (row, query) in enumerate(zip(chunk, base_batch.semantic_positions)):
                    row_query = float(difference[i, query].max())
                    row_off = max(
                        float(difference[i, :query].max()) if query else 0.0,
                        float(difference[i, query + 1:].max()) if query + 1 < difference.shape[1] else 0.0,
                    )
                    query_error[boundary] = max(query_error[boundary], row_query)
                    off_query_error[boundary] = max(off_query_error[boundary], row_off)
                    if not math.isfinite(row_query) or not math.isfinite(row_off):
                        raise ExperimentError("diagnostic error is nonfinite")
                    records.append({
                        "boundary": boundary, "head": invalid_runner.HEAD_BY_BOUNDARY[boundary],
                        "family": family, "row_id": str(row["row_id"]),
                        "query_projection_max_abs": row_query,
                        "off_query_projection_max_abs": row_off,
                    })

    pred_a = all(value <= 1.0e-4 for value in term_error.values())
    pred_b = all(value <= 0.04 for value in query_error.values())
    pred_c = all(
        full_error[boundary] > 1.0
        and abs(full_error[boundary] - off_query_error[boundary]) <= 1.0e-6
        for boundary in invalid_runner.BOUNDARIES
    )
    pred_d = partition_rows == len(rows)
    pred_e = (
        len(records) == 32 and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else (
        "null" if pred_a and pred_d and pred_e else "invalid"
    )
    reason = {
        "screen": "v1_invalidity_localized_to_full_sequence_audit_index",
        "null": "query_projection_does_not_close",
        "invalid": "authority_reconstruction_partition_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "invalid_result_sha256": EXPECTED_INVALID_RESULT_SHA256,
        "invalid_runner_sha256": EXPECTED_INVALID_RUNNER_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_attention_term_reconstruction": pred_a,
            "pred_b_query_projection_closure": pred_b,
            "pred_c_error_is_off_query": pred_c,
            "pred_d_exact_partition": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "attention_term_reconstruction_max_abs": {str(k): v for k, v in term_error.items()},
            "query_projection_max_abs": {str(k): v for k, v in query_error.items()},
            "off_query_projection_max_abs": {str(k): v for k, v in off_query_error.items()},
            "full_sequence_projection_max_abs": {str(k): v for k, v in full_error.items()},
            "partition_rows": partition_rows, "record_count": len(records),
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
        },
        "diagnostic_records": records, "terminal": terminal, "reason": reason,
        "scope_boundary": "Post-outcome instrument diagnostic only; v1 remains immutable and invalid.",
        "next_action": (
            "run an immutable release audit over v1 rows plus this diagnostic"
            if terminal == "screen" else "retain the source-compression evidence as unreleased"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"],
        "query_projection_max_abs": result["score"]["query_projection_max_abs"],
        "off_query_projection_max_abs": result["score"]["off_query_projection_max_abs"],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
