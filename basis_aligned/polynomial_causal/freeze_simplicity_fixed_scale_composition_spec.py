#!/usr/bin/env python3
"""Rung455: freeze a fixed-scale composition metric from already-open bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
SOURCE = Path(__file__).resolve()
OUT = BQ / "simplicity_fixed_scale_composition_spec_v1.json"
FAMILIES = ("mlp0", "mlp_pca", "vocabulary")
FILES = {
    "mlp0": {
        "bundle": BQ / "simplicity_mlp0_complete_candidate_consequences_bundle.pt",
        "result": BQ / "simplicity_mlp0_complete_candidate_consequences_results.json",
    },
    "mlp_pca": {
        "bundle": BQ / "simplicity_mlp_pca_complete_candidate_consequences_bundle.pt",
        "result": BQ / "simplicity_mlp_pca_complete_candidate_consequences_results.json",
    },
    "vocabulary": {
        "bundle": BQ / "simplicity_vocabulary_complete_candidate_consequences_bundle.pt",
        "result": BQ / "simplicity_vocabulary_complete_candidate_consequences_results.json",
    },
}
HASHES = {
    FILES["mlp0"]["bundle"]: "db58eca4de2d057640a0b72c196587c74ebdebcee8572c8147f2858348554980",
    FILES["mlp0"]["result"]: "388f7d4a49ac037cbe62346f0b239ca03f7a0a6b6284ff49ad2a06921a0b2a70",
    FILES["mlp_pca"]["bundle"]: "9225f3e4562a6752d8b121b4f4f7f6f9a51d031f1032145c2e912f16eafc4faf",
    FILES["mlp_pca"]["result"]: "0f99365bdb9a21fb4674cc5695d89435127bd3225de7767e4d4d177a6191344e",
    FILES["vocabulary"]["bundle"]: "7a1163eb3fd333b5496b9ef7637c99b764b44435a05de733806656cd9dd435ab",
    FILES["vocabulary"]["result"]: "2275efc6f2f690e30f8e212c7b3b2d32e37e15c031c94d201372a74b0557e27d",
}
RECONSTRUCTION_TOLERANCE = 2e-4


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def committed_source() -> tuple[str, str]:
    relative = str(SOURCE.relative_to(REPO))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=REPO, check=True)
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO)
    digest = hashlib.sha256(blob).hexdigest()
    if sha256(SOURCE) != digest:
        raise RuntimeError("rung455 freezer is not the committed HEAD blob")
    return commit, digest


def edges(family: str, result: dict[str, object]) -> list[tuple[str, str]]:
    if family == "mlp0":
        return [("256", "384"), ("384", "448"), ("448", "512"), ("512", "640")]
    if family == "mlp_pca":
        return [
            ("mlp_pca_p8_17_r256", "mlp_pca_p8_17_r384"),
            ("mlp_pca_p8_17_r384", "mlp_pca_p8_17_r512"),
        ]
    return [tuple(pair) for pair in result["structured_rank_edges"]]


def direction_receipt(values: dict[str, float], pairs: list[tuple[str, str]]) -> dict[str, object]:
    records = []
    for lower, higher in pairs:
        difference = values[lower] - values[higher]
        records.append({"lower_rank": lower, "higher_rank": higher,
                        "lower_minus_higher": difference, "ordered": difference > 0})
    correct = sum(record["ordered"] for record in records)
    return {"correct": correct, "total": len(records),
            "accuracy": correct / len(records), "edges": records}


def write_create_only(payload: dict[str, object]) -> None:
    descriptor = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())


def main() -> None:
    if OUT.exists():
        raise RuntimeError("rung455 output namespace already exists")
    for path, expected in HASHES.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    commit, source_hash = committed_source()

    family_payload = {}
    max_reconstruction_error = 0.0
    for family in FAMILIES:
        bundle = torch.load(FILES[family]["bundle"], map_location="cpu", weights_only=False)
        result = json.loads(FILES[family]["result"].read_text())
        if set(bundle) < {"schema", "native_ce", "partner_ce", "arms"}:
            raise RuntimeError(f"bundle schema changed: {family}")
        if tuple(bundle["arms"]) != tuple(result["arms"]):
            raise RuntimeError(f"arm identity changed: {family}")
        native = bundle["native_ce"].float()
        partner = bundle["partner_ce"].float()
        if native.shape != partner.shape or native.ndim != 1:
            raise RuntimeError(f"token geometry changed: {family}")
        fixed_denominator = float((partner - native).double().norm())
        if not math.isfinite(fixed_denominator) or fixed_denominator <= 0:
            raise RuntimeError(f"invalid fixed denominator: {family}")

        arms = {}
        old_values = {}
        raw_values = {}
        fixed_values = {}
        for name, arm in bundle["arms"].items():
            candidate = arm["candidate_ce"].float()
            joint = arm["candidate_partner_ce"].float()
            physical = (joint - native).double()
            additive = ((candidate - native) + (partner - native)).double()
            interaction = physical - additive
            raw_norm = float(interaction.norm())
            moving_denominator = float(additive.norm())
            registered_ratio = raw_norm / moving_denominator
            old_value = result["arms"][name]["full"]["composition_normalized_error"]
            reconstruction_error = abs(registered_ratio - old_value)
            max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
            if reconstruction_error > RECONSTRUCTION_TOLERANCE:
                raise RuntimeError(f"registered ratio reconstruction mismatch: {family}/{name}")
            fixed_score = raw_norm / fixed_denominator
            values = (raw_norm, moving_denominator, registered_ratio, fixed_score,
                      reconstruction_error, float(interaction.square().mean().sqrt()))
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite metric: {family}/{name}")
            if abs(fixed_score - raw_norm / fixed_denominator) > 1e-15:
                raise RuntimeError(f"fixed-scale identity mismatch: {family}/{name}")
            arms[name] = {
                "raw_interaction_l2": raw_norm,
                "raw_interaction_rms": values[-1],
                "candidate_dependent_additive_l2": moving_denominator,
                "registered_ratio_reconstructed": registered_ratio,
                "registered_ratio_stored": old_value,
                "registered_ratio_reconstruction_abs_error": reconstruction_error,
                "fixed_partner_effect_l2": fixed_denominator,
                "fixed_scale_composition_error": fixed_score,
            }
            old_values[name] = registered_ratio
            raw_values[name] = raw_norm
            fixed_values[name] = fixed_score

        pairs = edges(family, result)
        family_payload[family] = {
            "schema": bundle["schema"],
            "token_count": native.numel(),
            "arm_ids": list(bundle["arms"]),
            "fixed_partner_effect_l2": fixed_denominator,
            "arms": arms,
            "descriptive_already_open_adjacent_ladders": {
                "registered_ratio": direction_receipt(old_values, pairs),
                "raw_interaction_l2": direction_receipt(raw_values, pairs),
                "fixed_scale_composition_error": direction_receipt(fixed_values, pairs),
            },
        }

    payload = {
        "schema": "simplicity_fixed_scale_composition_spec_v1",
        "status": "frozen_from_already_open_outcomes_before_independent_vocabulary_test",
        "rung": 455,
        "source_commit": commit,
        "source_sha256": source_hash,
        "inputs": {str(path): digest for path, digest in HASHES.items()},
        "definition": {
            "interaction": "L_PQ - L_P - L_Q + L_N",
            "registered_denominator": "||(L_P-L_N)+(L_Q-L_N)||_2",
            "fixed_denominator": "||L_Q-L_N||_2",
            "fixed_scale_score": "||L_PQ-L_P-L_Q+L_N||_2 / ||L_Q-L_N||_2",
            "compact_trace_reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        },
        "families": family_payload,
        "max_registered_ratio_reconstruction_abs_error": max_reconstruction_error,
        "outcome_access": {
            "already_open_complete_teaching_bundles_loaded": True,
            "row_cache_loaded": False,
            "model_loaded": False,
            "independent_vocabulary_outcome_loaded": False,
            "sealed_opened": False,
        },
        "pred_a_exact_old_sources": True,
        "pred_b_registered_metric_reconstruction": True,
        "pred_c_fixed_scale_definition_valid": True,
        "pred_d_new_outcomes_closed": True,
        "strong_null_specification_invalid": False,
        "scientific_status": {
            "rung454_repaired": False,
            "vocabulary_family_counted": False,
            "teaching_family_count": 2,
            "predictor_fit": False,
            "direction_receipts_are_posthoc_descriptive": True,
        },
        "next_step": "preregister_independent_192_document_vocabulary_fixed_scale_test",
    }
    write_create_only(payload)
    print(json.dumps({
        "status": "complete", "rung": 455, "output": str(OUT), "sha256": sha256(OUT),
        "max_reconstruction_error": max_reconstruction_error,
        "descriptive_direction_accuracy": {
            family: {metric: receipt["accuracy"] for metric, receipt in
                     item["descriptive_already_open_adjacent_ladders"].items()}
            for family, item in family_payload.items()
        },
        "new_outcomes_opened": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
