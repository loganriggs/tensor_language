#!/usr/bin/env python3
"""Test literal Task14 projected-write prototypes on unrelated behaviors."""

# BQGATE: EXPERIMENT pred_a_authority_native_capability pred_b_projected_write_noop_replay pred_c_all_ten_hooks_live pred_d_numbered_list_preserved pred_e_bracket_preserved pred_f_literal_complete_program_test
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_candidate_task14_head11_3_cross_circuit_collateral as authority
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v1.json"
PROTOTYPES = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
PROGRAM_VALIDATION = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
PRIOR_COLLATERAL = ROOT / "circuits/fast_screens/task14_head11_3_cross_circuit_collateral_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v1_result.json"
PRIOR_ART_SHA256 = "34757ca5e35c1b98a43af024fd9e7435fb6fe1abb74e0d5619105a80857a1cc0"
PROTOTYPE_SHA256 = "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07"
PROGRAM_VALIDATION_SHA256 = "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0"
PRIOR_COLLATERAL_SHA256 = "57590fa85d1bec2f8235b8111c700721d3c5050140b9eccbd8196c703382806d"
AUTHORITY_FILE_SHA256 = "066799f544056db56b96b6d5d64bcb7a092b8cb98cab3e36e0d897ad4f0646a6"
LAYER = 11
WIDTH = 1152
MAX_NOOP_LOGIT_ERROR = 1e-4
MAX_INSTALL_ERROR = 1e-5
MAX_MEDIAN_EFFECT = 0.10
MAX_ROW_EFFECT = 0.25
MIN_ROWS_UNDER_EFFECT = 14
MAX_ANSWER_FLIPS = 1
PRED_KEYS = (
    "pred_a_authority_native_capability",
    "pred_b_projected_write_noop_replay",
    "pred_c_all_ten_hooks_live",
    "pred_d_numbered_list_preserved",
    "pred_e_bracket_preserved",
    "pred_f_literal_complete_program_test",
)


class CollateralError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_prototypes() -> dict[str, dict[str, object]]:
    if _sha256(PROTOTYPES) != PROTOTYPE_SHA256:
        raise CollateralError("prototype artifact changed")
    artifact = json.loads(PROTOTYPES.read_text())
    selected = {
        key: value for key, value in artifact.get("prototypes", {}).items()
        if ".cardinality_" in key
    }
    if artifact.get("terminal") != "prototype_artifact" or len(selected) != 10 or any(len(item.get("coordinates", [])) != WIDTH for item in selected.values()):
        raise CollateralError("ten-vector prototype program is invalid")
    return selected


def validate_preflight() -> None:
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (PROGRAM_VALIDATION, PROGRAM_VALIDATION_SHA256, "program validation"),
        (PRIOR_COLLATERAL, PRIOR_COLLATERAL_SHA256, "prior collateral"),
        (Path(authority.__file__), AUTHORITY_FILE_SHA256, "collateral authority"),
    ):
        if _sha256(path) != expected:
            raise CollateralError(f"{label} changed")
    if json.loads(PROGRAM_VALIDATION.read_text())["score"]["predictions"]["pred_f_target_free_program_and_price"] is not True:
        raise CollateralError("upstream program provenance gate failed")
    _load_prototypes()


def derive_price() -> dict[str, int]:
    return {
        "physical_model_forwards": 3, "example_evaluations": 384,
        "nonzero_program_installations": 320, "zero_add_replays": 32,
        "native_evaluations": 32, "backwards": 0, "parameter_updates": 0,
    }


def compile_plan() -> dict[str, object]:
    validate_preflight()
    rows = authority.build_rows()
    return {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_plan_v1",
        "candidate_id": "subject_verb.number_agreement.mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v1",
        "split": "FROZEN_SELECT_CROSS_CIRCUIT",
        "row_count": len(rows), "behaviors": list(authority.BEHAVIORS),
        "prototype_count": 10, "projected_write_layer": LAYER,
        "projected_write_width": WIDTH, "position": "registered final prediction position",
        "prior_art_sha256": PRIOR_ART_SHA256, "prototype_artifact_sha256": PROTOTYPE_SHA256,
        "authority_sha256": authority.validate_rows(rows),
        "bars": {
            "maximum_noop_absolute_logit_error": MAX_NOOP_LOGIT_ERROR,
            "maximum_install_absolute_error": MAX_INSTALL_ERROR,
            "maximum_median_absolute_scaled_change": MAX_MEDIAN_EFFECT,
            "maximum_row_absolute_scaled_change": MAX_ROW_EFFECT,
            "minimum_rows_at_or_below_row_bar": MIN_ROWS_UNDER_EFFECT,
            "maximum_answer_flips": MAX_ANSWER_FLIPS,
            "pool_behaviors": False, "pool_prototypes": False,
        },
        "price": derive_price(),
    }


def _batch(rows) -> producer.ModelBatch:
    return producer.ModelBatch(
        row_ids=tuple(str(row["run_id"]) for row in rows), side="base",
        token_rows=tuple(tuple(int(token) for token in row["ids"]) for row in rows),
        answer_ids=tuple(int(row["answer_id"]) for row in rows),
        foil_ids=tuple(int(row["foil_id"]) for row in rows),
        semantic_positions=tuple(int(row["semantic_position"]) for row in rows),
    )


def _projected_add(executor, batch: producer.ModelBatch, vectors):
    torch, F, model = executor.torch, executor.F, executor.model
    tokens, lengths = executor._tensor_batch(batch)
    if vectors is None:
        vectors = torch.zeros((len(batch.row_ids), WIDTH), dtype=torch.float32, device=executor.device)
    if tuple(vectors.shape) != (len(batch.row_ids), WIDTH):
        raise CollateralError("projected-write batch has wrong shape")
    maximum_install_error = 0.0
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            if layer == LAYER:
                before = attention
                attention = attention.clone()
                for index, position in enumerate(batch.semantic_positions):
                    addition = vectors[index].to(dtype=attention.dtype)
                    attention[index, position] += addition
                    error = float((attention[index, position].float() - before[index, position].float() - addition.float()).abs().max())
                    maximum_install_error = max(maximum_install_error, error)
            x = live + attention
            x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
        pairs = tuple((
            float(logits[index, lengths[index] - 1, batch.answer_ids[index]].float()),
            float(logits[index, lengths[index] - 1, batch.foil_ids[index]].float()),
        ) for index in range(len(lengths)))
    return pairs, maximum_install_error


def _score(rows, native_pairs, noop_pairs, program_pairs, prototype_norms):
    native = {row["row_id"]: pair for row, pair in zip(rows, native_pairs)}
    noop_error = max(abs(x - y) for native_pair, noop_pair in zip(native_pairs, noop_pairs) for x, y in zip(native_pair, noop_pair))
    scales = {}
    for behavior in authority.BEHAVIORS:
        margins = [pair[0] - pair[1] for row, pair in zip(rows, native_pairs) if row["behavior"] == behavior]
        scales[behavior] = statistics.median(margins)
    evidence = []
    for item, pair in zip(program_pairs["rows"], program_pairs["pairs"]):
        native_margin = native[item["source_row_id"]][0] - native[item["source_row_id"]][1]
        patched_margin = pair[0] - pair[1]
        effect = abs(patched_margin - native_margin) / scales[item["behavior"]]
        evidence.append({
            "row_id": item["source_row_id"], "behavior": item["behavior"],
            "prototype_key": item["prototype_key"], "native_margin": native_margin,
            "patched_margin": patched_margin, "normalized_absolute_effect": effect,
            "answer_flipped": patched_margin <= 0,
        })
    summaries = {}
    for behavior in authority.BEHAVIORS:
        for prototype_key in sorted(prototype_norms):
            subset = [item for item in evidence if item["behavior"] == behavior and item["prototype_key"] == prototype_key]
            effects = [item["normalized_absolute_effect"] for item in subset]
            passed = statistics.median(effects) <= MAX_MEDIAN_EFFECT and sum(effect <= MAX_ROW_EFFECT for effect in effects) >= MIN_ROWS_UNDER_EFFECT and sum(item["answer_flipped"] for item in subset) <= MAX_ANSWER_FLIPS
            summaries[f"{behavior}.{prototype_key}"] = {
                "behavior": behavior, "prototype_key": prototype_key, "row_count": len(subset),
                "native_scale": scales[behavior], "median_normalized_absolute_effect": statistics.median(effects),
                "rows_at_or_below_0_25": sum(effect <= MAX_ROW_EFFECT for effect in effects),
                "answer_flips": sum(item["answer_flipped"] for item in subset), "passed_preservation": passed,
            }
    native_capable = all((pair[0] - pair[1]) > 0 for pair in native_pairs) and all(scale > 0 for scale in scales.values())
    live = len(prototype_norms) == 10 and all(math.isfinite(norm) and norm > 0 for norm in prototype_norms.values()) and program_pairs["maximum_install_error"] <= MAX_INSTALL_ERROR
    numbered = all(item["passed_preservation"] for item in summaries.values() if item["behavior"] == "numbered_list")
    bracket = all(item["passed_preservation"] for item in summaries.values() if item["behavior"] == "bracket_pending_opener")
    complete = len(evidence) == 320 and len({(item["row_id"], item["prototype_key"]) for item in evidence}) == 320
    predictions = dict(zip(PRED_KEYS, (native_capable, noop_error <= MAX_NOOP_LOGIT_ERROR, live, numbered, bracket, complete)))
    return {
        "noop_max_absolute_logit_error": noop_error,
        "maximum_install_absolute_error": program_pairs["maximum_install_error"],
        "prototype_l2_norms": prototype_norms, "behavior_scales": scales,
        "behavior_prototype_results": summaries, "evidence": evidence,
        "predictions": predictions,
    }


def evaluate(executor):
    source_rows = authority.build_rows()
    rows = [{**row, "run_id": row["row_id"]} for row in source_rows]
    batch = _batch(rows)
    native_output = executor.native(batch, capture=False)
    noop_pairs, noop_install_error = _projected_add(executor, batch, None)
    prototypes = _load_prototypes()
    program_rows, vectors = [], []
    for row in source_rows:
        for prototype_key, prototype in sorted(prototypes.items()):
            program_rows.append({
                **row, "source_row_id": row["row_id"],
                "run_id": f"{row['row_id']}:{prototype_key}", "prototype_key": prototype_key,
            })
            vectors.append(prototype["coordinates"])
    torch = executor.torch
    vector_tensor = torch.tensor(vectors, dtype=torch.float32, device=executor.device)
    program_pairs, install_error = _projected_add(executor, _batch(program_rows), vector_tensor)
    scores = _score(
        source_rows, native_output.answer_foil, noop_pairs,
        {"rows": program_rows, "pairs": program_pairs, "maximum_install_error": max(noop_install_error, install_error)},
        {key: float(value["l2_norm"]) for key, value in prototypes.items()},
    )
    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise CollateralError(f"refusing overwrite {OUT}")
    authority_runner = __import__("run_circuit_fast_screen_task14_head11_3_cross_circuit_collateral")
    authority_runner._verify_checkpoint()
    executor = producer.Bilin18TorchBackend.load("cuda")
    scores = evaluate(executor)
    terminal = "screen" if all(scores["predictions"].values()) else "null" if scores["predictions"][PRED_KEYS[0]] and scores["predictions"][PRED_KEYS[1]] and scores["predictions"][PRED_KEYS[2]] and scores["predictions"][PRED_KEYS[5]] else "invalid"
    reason = "both_unrelated_behaviors_preserved_for_all_ten_writes" if terminal == "screen" else "literal_program_has_cross_circuit_collateral" if terminal == "null" else "instrument_or_completeness_failed"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal, "reason": reason,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "score": scores,
        "limits": "Two unrelated behaviors establish narrow collateral breadth, not universal selectivity.",
    })
    print(json.dumps({"terminal": terminal, "reason": reason, "predictions": scores["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
