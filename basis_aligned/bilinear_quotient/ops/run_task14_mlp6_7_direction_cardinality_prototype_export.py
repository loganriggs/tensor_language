#!/usr/bin/env python3
"""Export fixed direction-cardinality upstream displacement prototypes."""

# BQGATE: EXPERIMENT pred_a_training_authority_and_hashes pred_b_source_instrument pred_c_twelve_prototypes_exported pred_d_no_third_corpus_outcomes
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_fixed_reader_transfer as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_fixed_direction_cardinality_upstream_program_v1.json"
SELECTION = ROOT / "circuits/fast_screens/task14_mlp6_7_fixed_upstream_displacement_prototype_v1_result.json"
READER = ROOT / "circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
PRIOR_ART_SHA256 = "075c1f83f5801e2eb874d6df55b6070d56a6a0271716dd15e99d044e4f2c2f2d"
SELECTION_SHA256 = "9d87e3f5f186d7571f12504d11dab9cd8bb88f2f98b716718b8e596d1cdba98f"
READER_SHA256 = "9db4eefe16498cb65fb9c21ea3f2475c790c89ebb2e65a70e8ad6b7886f2ae57"
AUTHORITY_FILE_SHA256 = "04e247b848f4f13870033a6176cda2286b026056dce4998487ead290111d4de7"
PROTOTYPE_WIDTH = 1152
MAXIMUM_ERROR = 5e-5
SUBSETS = factor_gate.BACKGROUND_SUBSETS
PRED_KEYS = (
    "pred_a_training_authority_and_hashes",
    "pred_b_source_instrument",
    "pred_c_twelve_prototypes_exported",
    "pred_d_no_third_corpus_outcomes",
)


class PrototypeExportError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_price() -> dict[str, int]:
    return {
        "physical_model_forwards": 1,
        "example_evaluations": 96,
        "causal_interventions": 0,
        "backwards": 0,
        "parameter_updates": 0,
        "stored_scalars": 12 * PROTOTYPE_WIDTH,
    }


def validate_preflight() -> None:
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (SELECTION, SELECTION_SHA256, "selection screen"),
        (READER, READER_SHA256, "reader artifact"),
        (Path(authority.__file__), AUTHORITY_FILE_SHA256, "training authority"),
    ):
        if _sha256(path) != expected:
            raise PrototypeExportError(f"{label} changed")
    selection = json.loads(SELECTION.read_text())
    if selection["score"]["predictions"]["pred_e_factor_identity_beats_cardinality_control"] is not False:
        raise PrototypeExportError("selection screen no longer rejects factor-specific table")
    if derive_price()["stored_scalars"] != 13824:
        raise PrototypeExportError("storage price changed")


def compile_plan() -> dict[str, object]:
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_export_plan_v1",
        "candidate_id": "subject_verb.number_agreement.mlp6_7_fixed_direction_cardinality_upstream_program_v1",
        "split": "SECOND_CORPUS_TRAINING_ONLY",
        "directions": ["plural_to_singular", "singular_to_plural"],
        "cardinalities": list(range(5)),
        "prototype_width": PROTOTYPE_WIDTH,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "selection_sha256": SELECTION_SHA256,
        "reader_sha256": READER_SHA256,
        "training_authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
        "third_corpus_rows_consumed": 0,
        "third_corpus_outcomes_consumed": 0,
        "price": derive_price(),
    }


def summarize_prototypes(records, readers, torch):
    prototypes = {}
    directions = ("plural_to_singular", "singular_to_plural")
    for direction in directions:
        direction_records = [item for item in records if item["direction"] == direction]
        for cardinality in range(5):
            selected = [item["delta"] for item in direction_records if item["cardinality"] == cardinality]
            expected = 16 * math.comb(4, cardinality)
            if len(selected) != expected:
                raise PrototypeExportError("cardinality prototype count changed")
            vector = torch.stack(selected).mean(dim=0)
            key = f"{direction}.cardinality_{cardinality}"
            prototypes[key] = {
                "direction": direction, "cardinality": cardinality, "training_vectors": len(selected),
                "coordinates": [float(x) for x in vector.cpu().tolist()],
                "l2_norm": float(vector.norm()), "frozen_reader_q": float(torch.dot(readers[direction], vector)),
            }
        selected = [item["delta"] for item in direction_records]
        if len(selected) != 256:
            raise PrototypeExportError("direction-only prototype count changed")
        vector = torch.stack(selected).mean(dim=0)
        prototypes[f"{direction}.direction_only"] = {
            "direction": direction, "cardinality": None, "training_vectors": len(selected),
            "coordinates": [float(x) for x in vector.cpu().tolist()],
            "l2_norm": float(vector.norm()), "frozen_reader_q": float(torch.dot(readers[direction], vector)),
        }
    return prototypes


def evaluate(model, torch, F, facade):
    rows = authority.build_rows()
    count = len(rows)
    parent = tangent.parent
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    _, captured, projection, closure, inputs = parent._decomposed_forward(model, tokens, finals, torch, F, facade)
    roles = {
        "recipient": tangent._role_slice(captured, 0, count),
        "opposite": tangent._role_slice(captured, count, 2 * count),
    }
    input_roles = {
        "recipient": tangent._role_slice(inputs, 0, count),
        "opposite": tangent._role_slice(inputs, count, 2 * count),
    }
    function = tangent._head_function(model, roles["recipient"], roles["opposite"], model.transformer.h[parent.LAYER].attn, projection, torch, F)
    records = []
    with torch.no_grad():
        for subset in SUBSETS:
            base = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset, F)).detach()
            exact = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset + "YZ", F)).detach()
            for index, row in enumerate(rows):
                records.append({
                    "row_id": row["row_id"], "direction": row["direction_id"],
                    "cardinality": len(subset), "delta": exact[index] - base[index],
                })
        reader_artifact = json.loads(READER.read_text())
        readers = {direction: torch.tensor(item["coordinates"], dtype=torch.float32, device=device) for direction, item in reader_artifact["readers"].items()}
        prototypes = summarize_prototypes(records, readers, torch)
    exactness = {
        "source_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
        "source_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"],
    }
    predictions = dict(zip(PRED_KEYS, (
        True,
        all(value <= MAXIMUM_ERROR for value in exactness.values()),
        len(prototypes) == 12 and all(len(item["coordinates"]) == PROTOTYPE_WIDTH and item["l2_norm"] > 0 for item in prototypes.values()),
        True,
    )))
    return prototypes, exactness, predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise PrototypeExportError(f"refusing overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    prototypes, exactness, predictions = evaluate(model, torch, F, facade)
    terminal = "prototype_artifact" if all(predictions.values()) else "invalid"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_artifact_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "exactness": exactness, "predictions": predictions, "prototypes": prototypes,
        "third_corpus_rows_consumed": 0, "third_corpus_outcomes_consumed": 0,
    })
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
