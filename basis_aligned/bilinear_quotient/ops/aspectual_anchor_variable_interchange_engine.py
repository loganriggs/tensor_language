#!/usr/bin/env python3
"""Reusable capture-and-score engine for frozen aspectual variable interchanges."""

# BQGATE: LIBRARY -- returns measurements to hash-bound experiment runners; writes no result.
from __future__ import annotations

import math
import statistics
import time

import run_aspectual_anchor_program_v8_cross_construction_variable_interchange_v1 as parent


ARMS = parent.ARMS


class InterchangeEngineError(RuntimeError):
    pass


def component_key(row):
    return (str(row["transform_id"]), int(row["group_number"]))


def summarize(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise InterchangeEngineError("missing/nonfinite recovery")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def run_interchange(source_selector):
    """Capture v8 variables once and score a caller-selected source for every target.

    ``source_selector(panel, target_key, panel_components)`` returns the source key.
    The engine never selects sources from outcomes and never writes an artifact.
    """
    lexical_rows, lexical_spec, fresh_rows, fresh_spec = parent.validate_static()
    started_utc, started = parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = parent.empirical.BilinearSuffixBackend.load("cuda")
    native, components = {}, {"lexical": {}, "fresh": {}}
    manual_base_max_abs = writer_tensor_error_max_abs = mlp_tensor_error_max_abs = 0.0
    forward_calls = evaluations = capture_chunks = 0
    lambdas = {boundary: backend.model.transformer.h[boundary].lambdas[0].float() for boundary in parent.program.SUFFIX_BOUNDARIES}
    weights = {boundary: backend.model.transformer.h[boundary].mlp.Down.weight.float() for boundary in parent.program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}

    for panel, rows, spec in (("lexical", lexical_rows, lexical_spec), ("fresh", fresh_rows, fresh_spec)):
        for family in ("A1", "A2"):
            family_rows = [row for row in rows if row["transform_id"] == family]
            for chunk in parent.producer._chunks(family_rows, spec.batch_size):
                capture_chunks += 1
                base_batch = parent.producer._batch(spec, chunk, "base")
                donor_batch = parent.producer._batch(spec, chunk, "donor")
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
                        answer, foil = parent.producer._finite_pair(pair)
                        native[(str(row["row_id"]), side)] = parent.producer.NativeLogitEvidence(str(row["row_id"]), family, side, answer, foil)
                attention = {}
                for boundary in parent.program.SUFFIX_SOURCE_BOUNDARIES:
                    bp, bv, _ = backend.attention_terms(base_batch, base_capture, boundary)
                    hp, hv, _ = backend.attention_terms(base_batch, hybrid_capture, boundary)
                    attention[boundary] = backend.projected_source_delta(base_batch, role_banks, (bp, bv), (hp, hv), boundary, parent.mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary])
                mlp_states = {boundary: (*backend.mlp_states(base_capture, boundary), *backend.mlp_states(hybrid_capture, boundary)) for boundary in parent.program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
                for boundary in parent.program.SUFFIX_MLP_FACTORS_BY_BOUNDARY:
                    _terms, error = backend.projected_mlp_terms(base_capture, hybrid_capture, boundary)
                    mlp_tensor_error_max_abs = max(mlp_tensor_error_max_abs, error)
                for i, (row, query) in enumerate(zip(chunk, base_batch.semantic_positions)):
                    key = component_key(row)
                    if key in components[panel]:
                        raise InterchangeEngineError(f"duplicate component key: {panel}/{key}")
                    components[panel][key] = {
                        "row": row,
                        "base_resid18": base_capture["resid18"][i, query].float().detach(),
                        "initial": (hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float()).detach(),
                        "attention": {boundary: attention[boundary][i, query].detach() for boundary in parent.program.SUFFIX_SOURCE_BOUNDARIES},
                        "mlp": {boundary: tuple(value[i, query].float().detach() for value in mlp_states[boundary]) for boundary in parent.program.SUFFIX_MLP_FACTORS_BY_BOUNDARY},
                        "answer_id": base_batch.answer_ids[i],
                        "foil_id": base_batch.foil_ids[i],
                    }

    values = {f"{panel}_{family}": {arm: [] for arm in ARMS} for panel in ("lexical", "fresh") for family in ("A1", "A2")}
    records, pair_records = [], []
    for panel, panel_components in components.items():
        for target_key, target in panel_components.items():
            source_key = source_selector(panel, target_key, panel_components)
            if source_key not in panel_components or source_key == target_key:
                raise InterchangeEngineError(f"invalid source key: {panel}/{target_key}->{source_key}")
            source = panel_components[source_key]
            target_row, source_row = target["row"], source["row"]
            pair_records.append({
                "panel": panel,
                "target_family": target_key[0],
                "source_family": source_key[0],
                "target_group": target_key[1],
                "source_group": source_key[1],
                "target_row_id": str(target_row["row_id"]),
                "source_row_id": str(source_row["row_id"]),
                "same_direction": target_row["direction_id"] == source_row["direction_id"],
                "different_reporter": target_row["reporter"] != source_row["reporter"],
                "different_period": target_row["object_name"] != source_row["object_name"],
            })
            panel_family = f"{panel}_{target_key[0]}"
            groups = {
                "target_full": (target, target, target),
                "source_full": (source, source, source),
                "swap_initial": (source, target, target),
                "swap_attention": (target, source, target),
                "swap_mlp": (target, target, source),
            }
            for arm, (initial_owner, attention_owner, mlp_owner) in groups.items():
                delta = parent.program.compiled_sparse_suffix_delta(
                    initial_owner["initial"],
                    lambda0_by_boundary=lambdas,
                    source_attention_delta_by_boundary=attention_owner["attention"],
                    mlp_states_by_boundary=mlp_owner["mlp"],
                    down_weight_by_boundary=weights,
                )
                answer_tensor, foil_tensor = parent.program.exact_scored_pair(target["base_resid18"] + delta, backend.model.lm_head, answer_id=target["answer_id"], foil_id=target["foil_id"])
                answer, foil, recovery = parent.suffix.recovery(target_row, (float(answer_tensor), float(foil_tensor)), native)
                values[panel_family][arm].append(recovery)
                records.append({"panel": panel, "target_family": target_key[0], "source_family": source_key[0], "arm_id": arm, "target_row_id": str(target_row["row_id"]), "source_row_id": str(source_row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": recovery})

    forward_calls += len(ARMS) * capture_chunks
    evaluations += len(ARMS) * 64
    summaries = {panel: {arm: summarize(arm_values) for arm, arm_values in arms.items()} for panel, arms in values.items()}
    ratios = {panel: {arm: summaries[panel][arm]["mean_recovery"] / summaries[panel]["target_full"]["mean_recovery"] for arm in ARMS[1:]} for panel in summaries}
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in lexical_rows + fresh_rows for side in ("base", "donor"))
    return {
        "started_utc": started_utc,
        "finished_utc": parent.empirical.component_parent.utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "capability": capability,
        "manual_base_scored_logit_max_abs": manual_base_max_abs,
        "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
        "mlp_bilinear_tensor_reconstruction_max_abs": mlp_tensor_error_max_abs,
        "panels": summaries,
        "recovery_fraction_vs_target_full": ratios,
        "intervention_records": records,
        "pair_records": pair_records,
        "forward_calls": forward_calls,
        "example_evaluations": evaluations,
    }
