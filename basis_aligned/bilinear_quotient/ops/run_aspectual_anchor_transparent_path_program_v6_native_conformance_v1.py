#!/usr/bin/env python3
"""Execute released program v6 on real confirmation tensors."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_api pred_b_real_tensor_program_conformance pred_c_recovery_conformance pred_d_direction_and_effect pred_e_exact_coverage
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v6 as program
import circuit_fast_screen_producer as producer
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_mlp12_14_bilinear_compression_split_v1 as empirical
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_v6_native_conformance_v1.json"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v6.py"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v6_result.json"
PARENT_RESULT = ROOT / "circuits/followups/aspectual_anchor_mlp12_14_bilinear_compression_split_v1_result.json"
CAPTURE_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp12_14_bilinear_compression_split_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v6_native_conformance_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v6_native_conformance_v1"
EXPECTED_PRIOR_SHA256 = "08458e2ddc00728c6305763010343bcc7b042298d91e9690c7079fd65b816463"
EXPECTED = {
    PROGRAM: "5997c8bae8b07a795dd36371636cb62d23db596e1ec53cf5978eab541082412f",
    RELEASE: "fe1c4364dc4c47ef640d1f9dc610f553890735a425a79f2baac9cad83ffff442",
    PARENT_RESULT: "21842ebd543ce8cb68fba680056224b0330da666004efb26f96ea72679ff558f",
    CAPTURE_RUNNER: "110c18124deec634ec7b34e06048ca3f4b69e3d5483471cbeafd8dca255d4262",
}
MODEL_FORWARDS_MAX = 10
EXAMPLE_EVALUATIONS_MAX = 80


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery missing or nonfinite")
    return {"count": len(values), "mean_recovery": statistics.fmean(values), "mean_absolute_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    release = json.loads(RELEASE.read_text())
    parent = json.loads(PARENT_RESULT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID or release.get("terminal") != "release"
        or parent.get("terminal") != "screen" or program.program_manifest()["program_id"] != program.PROGRAM_ID
    ):
        raise ExperimentError("authority terminal or API changed")
    _selection, confirmation, spec, _full_mlp_reference = empirical.validate_static()
    reference = {
        ("confirmation", record["family"], record["row_id"]): (record["answer_logit"], record["foil_logit"])
        for record in parent["intervention_logits"]
        if record.get("phase") == "confirmation" and record.get("arm_id") == "selected_two_each"
    }
    if len(reference) != 16:
        raise ExperimentError("selected bilinear reference changed")
    expected_family = parent["score"]["confirmation"]["arms"]["selected_two_each"]["families"]
    expected_mean = parent["score"]["confirmation"]["arms"]["selected_two_each"]["mean_target_recovery"]
    return confirmation, spec, reference, expected_family, expected_mean


def main() -> None:
    rows, spec, reference, expected_family, expected_mean = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_transparent_path_program_v6_native_conformance_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows": len(rows),
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = empirical.component_parent.utc_now(), time.perf_counter()
    backend = empirical.BilinearSuffixBackend.load("cuda")
    native, values, records = {}, {family: [] for family in ("A1", "A2")}, []
    manual_base_max_abs = writer_tensor_error_max_abs = reference_logit_max_abs = 0.0
    forward_calls = evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            role_banks = backend.role_positions(base_batch, donor_batch)
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            base_manual, base_capture = backend.capture_suffix_heads(base_batch)
            _writer, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(base_batch, donor_batch, base_bilinear, donor_bilinear)
            forward_calls += 4
            evaluations += 4 * len(chunk)
            writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
            for native_pair, manual_pair in zip(base_native.answer_foil, base_manual.answer_foil):
                manual_base_max_abs = max(manual_base_max_abs, abs(native_pair[0] - manual_pair[0]), abs(native_pair[1] - manual_pair[1]))
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), family, side, answer, foil)

            attention_delta = {}
            for boundary in program.SUFFIX_SOURCE_BOUNDARIES:
                bp, bv, _be = backend.attention_terms(base_batch, base_capture, boundary)
                hp, hv, _he = backend.attention_terms(base_batch, hybrid_capture, boundary)
                attention_delta[boundary] = backend.projected_source_delta(base_batch, role_banks, (bp, bv), (hp, hv), boundary, mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary])
            mlp_states = {boundary: (*backend.mlp_states(base_capture, boundary), *backend.mlp_states(hybrid_capture, boundary)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
            output_state = base_capture["resid18"].clone()
            for i, query in enumerate(base_batch.semantic_positions):
                states = {boundary: tuple(value[i, query].float() for value in mlp_states[boundary]) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
                delta = program.compiled_sparse_suffix_delta(
                    hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float(),
                    lambda0_by_boundary={boundary: backend.model.transformer.h[boundary].lambdas[0].float() for boundary in program.SUFFIX_BOUNDARIES},
                    source_attention_delta_by_boundary={boundary: attention_delta[boundary][i, query] for boundary in program.SUFFIX_SOURCE_BOUNDARIES},
                    mlp_states_by_boundary=states,
                    down_weight_by_boundary={boundary: backend.model.transformer.h[boundary].mlp.Down.weight.float() for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY},
                )
                output_state[i, query] = (output_state[i, query].float() + delta).to(output_state.dtype)
            output = backend.final_readout(base_batch, output_state)
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                expected = reference[("confirmation", family, str(row["row_id"]))]
                reference_logit_max_abs = max(reference_logit_max_abs, abs(answer - expected[0]), abs(foil - expected[1]))
                values[family].append(value)
                records.append({"family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})

    summaries = {family: summarize(values[family]) for family in ("A1", "A2")}
    mean_recovery = statistics.fmean(summaries[family]["mean_recovery"] for family in ("A1", "A2"))
    recovery_error = max(abs(mean_recovery - expected_mean), *(abs(summaries[family]["mean_recovery"] - expected_family[family]["mean_recovery"]) for family in ("A1", "A2")))
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in rows for side in ("base", "donor"))
    pred_a = capability and program.program_manifest()["compiled_suffix_boundaries"] == program.SUFFIX_BOUNDARIES and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3
    pred_b = reference_logit_max_abs <= 0.125
    pred_c = recovery_error <= 1.0e-5
    pred_d = mean_recovery > 0.0 and all(summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_e = len(records) == 16 and len({record["row_id"] for record in records}) == 16 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    predictions = {"pred_a_authority_capability_and_api": pred_a, "pred_b_real_tensor_program_conformance": pred_b, "pred_c_recovery_conformance": pred_c, "pred_d_direction_and_effect": pred_d, "pred_e_exact_coverage": pred_e}
    terminal = "screen" if all(predictions.values()) else "invalid"
    result = {
        "schema": "aspectual_anchor_transparent_path_program_v6_native_conformance_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "program_sha256": EXPECTED[PROGRAM], "predictions": predictions,
        "score": {"families": summaries, "mean_recovery": mean_recovery, "reference_recovery_max_abs": recovery_error, "reference_logit_max_abs": reference_logit_max_abs, "manual_base_scored_logit_max_abs": manual_base_max_abs, "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs, "forward_calls": forward_calls, "example_evaluations": evaluations, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0},
        "intervention_logits": records, "terminal": terminal,
        "reason": "v6_executes_on_checkpoint_tensors" if terminal == "screen" else "authority_api_control_recovery_or_coverage_invalid",
        "next_action": "compile exact final normalization and scored readout" if terminal == "screen" else "retain empirical runner implementation",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "predictions": predictions, "mean_recovery": mean_recovery, "reference_logit_max_abs": reference_logit_max_abs, "recovery_error": recovery_error, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
