#!/usr/bin/env python3
"""Split compression of exact recurrence terms after projection onto the rank-one carrier."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_scalar_closure pred_b_selection_is_well_formed pred_c_fresh_scalar_compression pred_d_fresh_scored_effect_compression pred_e_exact_coverage
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v9 as program
import circuit_fast_screen_producer as producer
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_mlp12_14_bilinear_compression_split_v1 as empirical
import run_aspectual_anchor_program_v8_rank1_carrier_mediation_v1 as mediation_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_rank1_scalar_term_compression_split_v1.json"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v9.py"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v9_result.json"
RANK1 = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
MEDIATION = ROOT / "circuits/followups/aspectual_anchor_program_v8_rank1_carrier_mediation_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.rank1_scalar_term_compression_split_v1"
EXPECTED_PRIOR_SHA256 = "2a7d57dafa5f87f8405318cad35f6c47bda93116e8cf70948ba27202a10fe982"
EXPECTED = {
    PROGRAM: "30003dad7dc08bd9bebb7e43e58e8c652975195ec31a2fccbe36aafbba352436",
    RELEASE: "dc2cd67daa2fbed6ae9ddde33e9c44fae11b694e4f351b596b541878df9a9106",
    RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    MEDIATION: "313ccce304a18f8b0d63547bb973964f0b6f93a506765e4bb87c31e40c128aa9",
}
TERMS = (
    "carried_resid10", "attention11", "mlp11_left_change", "mlp11_right_change",
    "mlp12_left_change", "mlp12_right_change", "mlp14_left_change", "mlp14_right_change",
    "attention15", "mlp15_left_change", "mlp15_bilinear_interaction",
)
SELECTED_WIDTH = 5
MODEL_FORWARDS_MAX = 24
EXAMPLE_EVALUATIONS_MAX = 288


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
    lexical_rows, lexical_spec, _lexical_ref, fresh_rows, fresh_spec, _fresh_ref, rank1 = mediation_parent.validate_static()
    mediation = json.loads(MEDIATION.read_text())
    reference = {(record["panel"], record["row_id"]): (record["answer_logit"], record["foil_logit"]) for record in mediation["intervention_logits"] if record["arm_id"] == "rank1_carrier"}
    if (
        prior.get("candidate_id") != CANDIDATE_ID or tuple(prior["frozen_terms"]) != TERMS
        or release.get("terminal") != "release" or mediation.get("terminal") != "screen"
        or len(reference) != 48 or rank1["basis"]["sha256"] != program.RANK1_BASIS_SHA256
    ):
        raise ExperimentError("authority, term inventory, reference, or basis changed")
    return lexical_rows, lexical_spec, fresh_rows, fresh_spec, rank1, reference


def main() -> None:
    lexical_rows, lexical_spec, fresh_rows, fresh_spec, rank1, reference = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_rank1_scalar_term_compression_split_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "selection_rows": 16, "confirmation_rows": 32,
        "term_count": len(TERMS), "selected_width": SELECTED_WIDTH, "scored_arms": 2,
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
    native, captures, contribution_records = {}, [], []
    manual_base_max_abs = writer_tensor_error_max_abs = mlp_tensor_error_max_abs = 0.0
    tensor_closure_max_abs = scalar_closure_max_abs = 0.0
    forward_calls = evaluations = 0

    for phase, rows, spec in (("selection", lexical_rows, lexical_spec), ("confirmation", fresh_rows, fresh_spec)):
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
                mlp_terms = {}
                for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY:
                    projected, error = backend.projected_mlp_terms(base_capture, hybrid_capture, boundary)
                    mlp_terms[boundary] = projected
                    mlp_tensor_error_max_abs = max(mlp_tensor_error_max_abs, error)
                for i, (row, query) in enumerate(zip(chunk, base_batch.semantic_positions)):
                    lambdas = {boundary: backend.model.transformer.h[boundary].lambdas[0].float() for boundary in program.SUFFIX_BOUNDARIES}
                    def later_product(boundary):
                        value = backend.torch.ones((), device=backend.device, dtype=backend.torch.float32)
                        for later in range(boundary + 1, 18):
                            value = value * lambdas[later]
                        return value
                    term_vectors = {
                        "carried_resid10": (hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float()) * later_product(9),
                        "attention11": attention_delta[11][i, query] * later_product(11),
                        "mlp11_left_change": mlp_terms[11]["left_change"][i, query] * later_product(11),
                        "mlp11_right_change": mlp_terms[11]["right_change"][i, query] * later_product(11),
                        "mlp12_left_change": mlp_terms[12]["left_change"][i, query] * later_product(12),
                        "mlp12_right_change": mlp_terms[12]["right_change"][i, query] * later_product(12),
                        "mlp14_left_change": mlp_terms[14]["left_change"][i, query] * later_product(14),
                        "mlp14_right_change": mlp_terms[14]["right_change"][i, query] * later_product(14),
                        "attention15": attention_delta[15][i, query] * later_product(15),
                        "mlp15_left_change": mlp_terms[15]["left_change"][i, query] * later_product(15),
                        "mlp15_bilinear_interaction": mlp_terms[15]["bilinear_interaction"][i, query] * later_product(15),
                    }
                    states = {boundary: tuple(value[i, query].float() for value in (*backend.mlp_states(base_capture, boundary), *backend.mlp_states(hybrid_capture, boundary))) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
                    full_delta = program.compiled_sparse_suffix_delta(
                        hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float(),
                        lambda0_by_boundary=lambdas,
                        source_attention_delta_by_boundary={boundary: attention_delta[boundary][i, query] for boundary in program.SUFFIX_SOURCE_BOUNDARIES},
                        mlp_states_by_boundary=states,
                        down_weight_by_boundary={boundary: backend.model.transformer.h[boundary].mlp.Down.weight.float() for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY},
                    )
                    summed = sum(term_vectors.values())
                    tensor_closure_max_abs = max(tensor_closure_max_abs, float((summed - full_delta).abs().max()))
                    total_amplitude = float(backend.torch.dot(full_delta, q))
                    amplitudes = {term: float(backend.torch.dot(term_vectors[term], q)) for term in TERMS}
                    scalar_closure_max_abs = max(scalar_closure_max_abs, abs(sum(amplitudes.values()) - total_amplitude))
                    contribution_records.append({"phase": phase, "panel": f"{'lexical' if phase == 'selection' else 'fresh'}_{family}", "family": family, "row_id": str(row["row_id"]), "total_amplitude": total_amplitude, "term_amplitudes": amplitudes, "base_resid18": base_capture["resid18"][i, query].float().detach(), "answer_id": base_batch.answer_ids[i], "foil_id": base_batch.foil_ids[i]})
                captures.append((phase, family, chunk))

    selection_records = [record for record in contribution_records if record["phase"] == "selection"]
    selection_scores = {term: statistics.fmean((1.0 if record["total_amplitude"] >= 0.0 else -1.0) * record["term_amplitudes"][term] for record in selection_records) for term in TERMS}
    ranking = sorted(TERMS, key=lambda term: (-selection_scores[term], TERMS.index(term)))
    selected_terms = tuple(term for term in TERMS if term in ranking[:SELECTED_WIDTH])
    scored_values = {panel: {arm: [] for arm in ("full_rank1", "selected_five")} for panel in ("lexical_A1", "lexical_A2", "fresh_A1", "fresh_A2")}
    scored_records, reference_logit_max_abs = [], 0.0
    for record in contribution_records:
        full_amplitude = record["total_amplitude"]
        selected_amplitude = sum(record["term_amplitudes"][term] for term in selected_terms)
        for arm, amplitude in (("full_rank1", full_amplitude), ("selected_five", selected_amplitude)):
            delta = q * amplitude
            answer_tensor, foil_tensor = program.exact_scored_pair(record["base_resid18"] + delta, backend.model.lm_head, answer_id=record["answer_id"], foil_id=record["foil_id"])
            row = next(row for row in (lexical_rows if record["phase"] == "selection" else fresh_rows) if str(row["row_id"]) == record["row_id"])
            answer, foil, value = suffix.recovery(row, (float(answer_tensor), float(foil_tensor)), native)
            scored_values[record["panel"]][arm].append(value)
            scored_records.append({"phase": record["phase"], "panel": record["panel"], "arm_id": arm, "family": record["family"], "row_id": record["row_id"], "answer_logit": answer, "foil_logit": foil, "recovery": value})
            if arm == "full_rank1":
                expected = reference[(record["panel"], record["row_id"])]
                reference_logit_max_abs = max(reference_logit_max_abs, abs(answer - expected[0]), abs(foil - expected[1]))
    forward_calls += 2 * len(captures)
    evaluations += 2 * len(contribution_records)
    summaries = {panel: {arm: summarize(scored_values[panel][arm]) for arm in scored_values[panel]} for panel in scored_values}
    confirmation_records = [record for record in contribution_records if record["phase"] == "confirmation"]
    scalar_fraction, scalar_alignment, scored_fraction = {}, {}, {}
    for family in ("A1", "A2"):
        rows_family = [record for record in confirmation_records if record["family"] == family]
        scalar_fraction[family] = statistics.fmean((1.0 if record["total_amplitude"] >= 0.0 else -1.0) * sum(record["term_amplitudes"][term] for term in selected_terms) for record in rows_family) / statistics.fmean(abs(record["total_amplitude"]) for record in rows_family)
        scalar_alignment[family] = sum(sum(record["term_amplitudes"][term] for term in selected_terms) * record["total_amplitude"] > 0.0 for record in rows_family) / len(rows_family)
        panel = f"fresh_{family}"
        scored_fraction[family] = summaries[panel]["selected_five"]["mean_recovery"] / summaries[panel]["full_rank1"]["mean_recovery"]
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in lexical_rows + fresh_rows for side in ("base", "donor"))
    pred_a = capability and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and mlp_tensor_error_max_abs <= 5.0e-3 and tensor_closure_max_abs <= 1.0e-4 and scalar_closure_max_abs <= 1.0e-4 and reference_logit_max_abs <= 0.125
    pred_b = len(selected_terms) == SELECTED_WIDTH and len(set(selected_terms)) == SELECTED_WIDTH
    pred_c = all(scalar_fraction[family] >= 0.80 and scalar_alignment[family] >= 0.75 for family in ("A1", "A2"))
    pred_d = all(scored_fraction[family] >= 0.80 and summaries[f"fresh_{family}"]["selected_five"]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_e = len(scored_records) == 96 and len({(record["phase"], record["arm_id"], record["row_id"]) for record in scored_records}) == 96 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "five_terms_compress_rank1_carrier_on_fresh_constructions", "null": "five_terms_do_not_retain_enough_fresh_carrier_effect", "invalid": "authority_capability_closure_control_or_coverage_invalid"}[terminal]
    serializable_contributions = [{key: value for key, value in record.items() if key not in {"base_resid18", "answer_id", "foil_id"}} for record in contribution_records]
    result = {
        "schema": "aspectual_anchor_rank1_scalar_term_compression_split_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "basis_sha256": rank1["basis"]["sha256"],
        "predictions": {"pred_a_authority_capability_and_exact_scalar_closure": pred_a, "pred_b_selection_is_well_formed": pred_b, "pred_c_fresh_scalar_compression": pred_c, "pred_d_fresh_scored_effect_compression": pred_d, "pred_e_exact_coverage": pred_e},
        "score": {"selection_scores": selection_scores, "ranking": ranking, "selected_terms": list(selected_terms), "fresh_scalar_fraction": scalar_fraction, "fresh_scalar_alignment": scalar_alignment, "fresh_scored_fraction": scored_fraction, "scored_panels": summaries, "tensor_closure_max_abs": tensor_closure_max_abs, "scalar_closure_max_abs": scalar_closure_max_abs, "full_rank1_reference_logit_max_abs": reference_logit_max_abs, "manual_base_scored_logit_max_abs": manual_base_max_abs, "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs, "mlp_bilinear_tensor_reconstruction_max_abs": mlp_tensor_error_max_abs, "forward_calls": forward_calls, "example_evaluations": evaluations, "scored_record_count": len(scored_records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0},
        "term_amplitudes": serializable_contributions, "intervention_logits": scored_records, "terminal": terminal, "reason": reason,
        "next_action": "compile selected scalar carrier terms into program v10" if terminal == "screen" else "retain all eleven explicit carrier terms",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "ranking": ranking, "selected_terms": list(selected_terms), "scalar_fraction": scalar_fraction, "scored_fraction": scored_fraction, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
