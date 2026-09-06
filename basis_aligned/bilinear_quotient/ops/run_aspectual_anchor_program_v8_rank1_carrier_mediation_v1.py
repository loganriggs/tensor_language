#!/usr/bin/env python3
"""Test whether program-v8 displacement acts through the rank-one resid18 carrier."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_full_control pred_b_full_program_effect pred_c_rank1_mediation pred_d_orthogonal_exclusion pred_e_exact_coverage
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v8 as program
import circuit_fast_screen_producer as producer
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_mlp12_14_bilinear_compression_split_v1 as empirical
import run_aspectual_anchor_program_v7_fresh_construction_transfer_v1 as fresh_runner
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v8_rank1_carrier_mediation_v1.json"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v8.py"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v8_result.json"
RANK1 = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
LEXICAL_REFERENCE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v6_native_conformance_v1_result.json"
FRESH_REFERENCE = ROOT / "circuits/followups/aspectual_anchor_program_v7_fresh_construction_transfer_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v8_rank1_carrier_mediation_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v8_rank1_carrier_mediation_v1"
EXPECTED_PRIOR_SHA256 = "1634a39673236bbebf144465e0910531274703081c7fba50f0a2d2b721d15105"
EXPECTED = {
    PROGRAM: "87eb67f3a96904534c8d3ddca5e1df59fa14efd88d1174e1cc2805435346bb57",
    RELEASE: "06a72441279946bd9facf302e545b8381ffde6d46a6dd33a72128dddb2ce8994",
    RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    LEXICAL_REFERENCE: "3d6e70a3c7d5786bb915647dfb0958f9d7c9de051cb170d6d427cae98b4633df",
    FRESH_REFERENCE: "8e1ea19c94ef269d2e9c7c0577568a0e1fe2e8bc6640e016377270df2dc68129",
}
ARMS = ("full_program", "rank1_carrier", "orthogonal_remainder")
MODEL_FORWARDS_MAX = 28
EXAMPLE_EVALUATIONS_MAX = 336


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
    rank1 = json.loads(RANK1.read_text())
    lexical_reference = json.loads(LEXICAL_REFERENCE.read_text())
    fresh_reference = json.loads(FRESH_REFERENCE.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID or release.get("terminal") != "release"
        or rank1.get("terminal") != lexical_reference.get("terminal") or rank1.get("terminal") != fresh_reference.get("terminal") or rank1.get("terminal") != "screen"
        or rank1["rank"] != 1 or rank1["basis"]["shape"] != [1152, 1]
        or rank1["basis"]["sha256"] != "123c6e098fcccf68bd9b881bb81c6b95858a258baa688b79a947a3043bb61e39"
    ):
        raise ExperimentError("authority terminal, rank, or basis changed")
    _selection, lexical_rows, lexical_spec, _reference = empirical.validate_static()
    fresh_rows, fresh_spec = fresh_runner.validate_static()
    fresh_rows = tuple(row for row in fresh_rows if row["transform_id"] in ("A1", "A2"))
    lexical_ref = {(record["family"], record["row_id"]): (record["answer_logit"], record["foil_logit"]) for record in lexical_reference["intervention_logits"]}
    fresh_ref = {(record["family"], record["row_id"]): (record["answer_logit"], record["foil_logit"]) for record in fresh_reference["intervention_logits"] if record["arm_id"] == "program_v7"}
    if len(lexical_rows) != len(lexical_ref) or len(fresh_rows) != len(fresh_ref) or len(lexical_rows) + len(fresh_rows) != 48:
        raise ExperimentError("population or reference coverage changed")
    return lexical_rows, lexical_spec, lexical_ref, fresh_rows, fresh_spec, fresh_ref, rank1


def main() -> None:
    lexical_rows, lexical_spec, lexical_ref, fresh_rows, fresh_spec, fresh_ref, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_program_v8_rank1_carrier_mediation_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows": 48, "arms": list(ARMS), "rank": 1,
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
    q = backend.torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=backend.torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    values = {panel: {arm: [] for arm in ARMS} for panel in ("lexical_A1", "lexical_A2", "fresh_A1", "fresh_A2")}
    records, native = [], {}
    manual_base_max_abs = writer_tensor_error_max_abs = full_reference_max_abs = 0.0
    forward_calls = evaluations = 0

    panels = (
        ("lexical", lexical_rows, lexical_spec, lexical_ref),
        ("fresh", fresh_rows, fresh_spec, fresh_ref),
    )
    for panel_name, panel_rows, spec, reference in panels:
        for family in ("A1", "A2"):
            family_rows = [row for row in panel_rows if row["transform_id"] == family]
            panel = f"{panel_name}_{family}"
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
                for i, (row, query) in enumerate(zip(chunk, base_batch.semantic_positions)):
                    states = {boundary: tuple(value[i, query].float() for value in mlp_states[boundary]) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
                    full_delta = program.compiled_sparse_suffix_delta(
                        hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float(),
                        lambda0_by_boundary={boundary: backend.model.transformer.h[boundary].lambdas[0].float() for boundary in program.SUFFIX_BOUNDARIES},
                        source_attention_delta_by_boundary={boundary: attention_delta[boundary][i, query] for boundary in program.SUFFIX_SOURCE_BOUNDARIES},
                        mlp_states_by_boundary=states,
                        down_weight_by_boundary={boundary: backend.model.transformer.h[boundary].mlp.Down.weight.float() for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY},
                    )
                    carrier_delta = q * backend.torch.dot(full_delta, q)
                    for arm, delta in (("full_program", full_delta), ("rank1_carrier", carrier_delta), ("orthogonal_remainder", full_delta - carrier_delta)):
                        answer_tensor, foil_tensor = program.exact_scored_pair(base_capture["resid18"][i, query].float() + delta, backend.model.lm_head, answer_id=base_batch.answer_ids[i], foil_id=base_batch.foil_ids[i])
                        answer, foil, value = suffix.recovery(row, (float(answer_tensor), float(foil_tensor)), native)
                        values[panel][arm].append(value)
                        records.append({"panel": panel, "arm_id": arm, "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})
                        if arm == "full_program":
                            expected = reference[(family, str(row["row_id"]))]
                            full_reference_max_abs = max(full_reference_max_abs, abs(answer - expected[0]), abs(foil - expected[1]))
                forward_calls += len(ARMS)
                evaluations += len(ARMS) * len(chunk)

    summaries = {panel: {arm: summarize(values[panel][arm]) for arm in ARMS} for panel in values}
    carrier_fraction = {panel: summaries[panel]["rank1_carrier"]["mean_recovery"] / summaries[panel]["full_program"]["mean_recovery"] for panel in summaries}
    orthogonal_fraction = {panel: summaries[panel]["orthogonal_remainder"]["mean_absolute_recovery"] / summaries[panel]["full_program"]["mean_absolute_recovery"] for panel in summaries}
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in lexical_rows + fresh_rows for side in ("base", "donor"))
    pred_a = capability and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and full_reference_max_abs <= 0.125
    pred_b = all(summaries[panel]["full_program"]["mean_recovery"] > 0.0 and summaries[panel]["full_program"]["direction_fraction"] >= 0.75 for panel in summaries)
    pred_c = all(carrier_fraction[panel] >= 0.70 and summaries[panel]["rank1_carrier"]["direction_fraction"] >= 0.75 for panel in summaries)
    pred_d = all(orthogonal_fraction[panel] <= 0.40 for panel in summaries)
    pred_e = len(records) == 144 and len({(record["panel"], record["arm_id"], record["row_id"]) for record in records}) == 144 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "program_v8_displacement_is_rank1_carrier_mediated", "null": "program_v8_displacement_not_primarily_rank1_mediated", "invalid": "authority_capability_instrument_control_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_program_v8_rank1_carrier_mediation_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "program_sha256": EXPECTED[PROGRAM], "basis_sha256": rank1["basis"]["sha256"],
        "predictions": {"pred_a_authority_capability_and_exact_full_control": pred_a, "pred_b_full_program_effect": pred_b, "pred_c_rank1_mediation": pred_c, "pred_d_orthogonal_exclusion": pred_d, "pred_e_exact_coverage": pred_e},
        "score": {"panels": summaries, "rank1_to_full_recovery_fraction": carrier_fraction, "orthogonal_to_full_absolute_fraction": orthogonal_fraction, "full_reference_logit_max_abs": full_reference_max_abs, "manual_base_scored_logit_max_abs": manual_base_max_abs, "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs, "forward_calls": forward_calls, "example_evaluations": evaluations, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0},
        "intervention_logits": records, "terminal": terminal, "reason": reason,
        "next_action": "compile rank1 carrier projection into program v9 explanation" if terminal == "screen" else "retain v8 full resid18 displacement and do not force carrier mediation",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "carrier_fraction": carrier_fraction, "orthogonal_fraction": orthogonal_fraction, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
