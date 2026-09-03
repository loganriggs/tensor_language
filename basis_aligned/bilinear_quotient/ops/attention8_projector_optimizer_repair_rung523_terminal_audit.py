#!/usr/bin/env python3
"""Independently audit rung 523's FIT/VALIDATION-only terminal decision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Mapping

import torch


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
REPO = ROOT.parent.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import attention8_projector_optimizer_repair_rung523_math as repair  # noqa: E402
import attention8_selective_shared_projector_rung522_archive as archive  # noqa: E402


DEFAULT_RESULT = ROOT / "attention8_projector_optimizer_repair_rung523_results.json"
DEFAULT_ARCHIVE = ROOT / "attention8_projector_optimizer_repair_rung523_work/candidate_frames.pt"
DEFAULT_OUTPUT = ROOT / "attention8_projector_optimizer_repair_rung523_terminal_audit.json"
EXPECTED_BUCKETS = {
    "fit_d0_full_attention8": 95,
    "fit_health": 120,
    "full_attention8_comparator": 36,
    "native_capture": 131,
    "native_replay": 131,
    "self_donor": 2,
}
EXPECTED_LEDGER = {
    "optimization_forwards": 9_000,
    "optimization_backwards": 9_000,
    "inference_forwards": 515,
    "removal_forwards": 0,
    "inference_by_bucket": EXPECTED_BUCKETS,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    """Normalize tuples and JSON-compatible scalars without changing precision."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _health(record: Mapping[str, object]) -> repair.FitHealth:
    return repair.FitHealth(
        losses=record["loss_history"],
        initial_common_validation=float(record["initial_common_validation"]),
        final_common_validation=float(record["final_common_validation"]),
        orthonormality_error=float(record["orthonormality_error"]),
        projector_distance=float(record["projector_distance"]),
        finite_gradients=bool(record["finite_gradients"]),
        model_gradients_absent=bool(record["model_gradients_absent"]),
    )


def _ledger(execution: Mapping[str, object]) -> dict[str, object]:
    observed = {key: execution.get(key) for key in EXPECTED_LEDGER}
    if observed != EXPECTED_LEDGER:
        raise ValueError(f"execution ledger changed: {observed} != {EXPECTED_LEDGER}")
    runtime = execution.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime <= 0:
        raise ValueError("runtime_seconds is not finite and positive")
    return observed


def audit_terminal_result(
    result: Mapping[str, object],
    artifact: Mapping[str, object],
    *,
    artifact_file_sha256: str,
) -> dict[str, object]:
    """Verify the archive, score, seal, and frozen closure decision."""
    if result.get("rung") != 523:
        raise ValueError("result is not rung 523")
    if result.get("status") != "raw_adam_through_qr_closed":
        raise ValueError("rung 523 did not reach the registered closure boundary")
    for field in ("test_accessible", "test_opened", "omitted_targets_evaluated"):
        if result.get(field) is not False:
            raise ValueError(f"{field} changed or is not literal false")
    if result.get("claim_level") != "FIT/VALIDATION-only optimizer calibration; no circuit evidence":
        raise ValueError("claim level changed")

    seal = result.get("seal")
    if not isinstance(seal, Mapping):
        raise ValueError("seal is absent")
    if seal.get("test_split_requested") is not False:
        raise ValueError("TEST was requested")
    if seal.get("scientific_a_through_d_scored") is not False:
        raise ValueError("scientific predictions were scored")
    requested = seal.get("requested_splits")
    if requested != ["FIT", "VALIDATION", "VALIDATION"]:
        raise ValueError(f"split access changed: {requested}")

    execution = result.get("execution_price")
    if not isinstance(execution, Mapping):
        raise ValueError("execution ledger is absent")
    ledger = _ledger(execution)

    receipt = result.get("frame_artifact")
    if not isinstance(receipt, Mapping):
        raise ValueError("frame artifact receipt is absent")
    if receipt.get("file_sha256") != artifact_file_sha256:
        raise ValueError("frame artifact file hash differs")
    if receipt.get("frame_count") != 45:
        raise ValueError("frame receipt count changed")
    if artifact.get("schema") != "attention8-projector-optimizer-repair-rung523-frames-v1":
        raise ValueError("frame artifact schema changed")

    frames = artifact.get("frames")
    records = artifact.get("records")
    diagnostics = artifact.get("diagnostics")
    result_records = result.get("candidate_records")
    result_diagnostics = result.get("candidate_diagnostics")
    result_cells = result.get("prospective_cells")
    if not all(isinstance(value, Mapping) for value in (
        frames, records, diagnostics, result_records, result_diagnostics, result_cells
    )):
        raise ValueError("candidate artifact or result maps are absent")
    expected_arms = set(repair.PROSPECTIVE_ARMS)
    if any(set(value) != expected_arms for value in (
        records, diagnostics, result_records, result_diagnostics, result_cells
    )):
        raise ValueError("prospective arm census changed")
    if _json_value(records) != _json_value(result_records):
        raise ValueError("artifact/result candidate records differ")
    if _json_value(diagnostics) != _json_value(result_diagnostics):
        raise ValueError("artifact/result diagnostics differ")

    recomputed_cells: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, object]] = {}
    expected_frame_keys = set()
    for arm in repair.PROSPECTIVE_ARMS:
        arm_records = records[arm]
        if not isinstance(arm_records, list) or len(arm_records) != 15:
            raise ValueError(f"{arm} record census changed")
        recomputed = repair.score_candidate_cell([_health(record) for record in arm_records])
        recomputed_cells[arm] = recomputed
        if _json_value(recomputed) != _json_value(result_cells[arm]):
            raise ValueError(f"stored score differs from records for {arm}")
        failure_counts: Counter[str] = Counter()
        validation_ratios = []
        validation_improved = 0
        train_improved = 0
        maximum_losses = []
        for record, fit_score in zip(arm_records, recomputed["fits"], strict=True):
            if record.get("arm") != arm:
                raise ValueError(f"record arm mismatch in {arm}")
            key = f"{arm}:{record['frame_id']}"
            expected_frame_keys.add(key)
            frame = frames.get(key)
            if not isinstance(frame, torch.Tensor) or frame.shape != (1152, 4):
                raise ValueError(f"invalid frame tensor {key}")
            if archive.tensor_sha256(frame) != record.get("frame_sha256"):
                raise ValueError(f"frame hash differs for {key}")
            failure_counts.update(fit_score["failures"])
            initial_validation = float(record["initial_common_validation"])
            final_validation = float(record["final_common_validation"])
            validation_ratios.append(final_validation / initial_validation)
            validation_improved += int(final_validation < initial_validation)
            losses = [float(value) for value in record["loss_history"]]
            train_improved += int(sum(losses[-20:]) < sum(losses[:20]))
            maximum_losses.append(max(losses))
        summaries[arm] = {
            "passing_fits": recomputed["passing_fit_count"],
            "fit_count": 15,
            "validation_improved_count": validation_improved,
            "training_window_improved_count": train_improved,
            "median_final_over_initial_validation": statistics.median(validation_ratios),
            "median_maximum_training_loss": statistics.median(maximum_losses),
            "maximum_training_loss": max(maximum_losses),
            "losses_above_100": recomputed["spike_count_strictly_above_100"],
            "losses_above_1000": recomputed["extreme_count_strictly_above_1000"],
            "per_fit_failure_counts": dict(sorted(failure_counts.items())),
        }
    if set(frames) != expected_frame_keys or len(frames) != 45:
        raise ValueError("frame tensor census differs from records")

    decision = repair.adoption_decision(recomputed_cells)
    if _json_value(decision) != _json_value(result.get("decision")):
        raise ValueError("terminal decision differs from recomputed scores")
    if decision["adopted_arm"] is not None or decision["licenses_sealed_rung522_repeat"]:
        raise ValueError("closed optimizer family unexpectedly licensed a rerun")
    if any(bool(cell["passes"]) for cell in recomputed_cells.values()):
        raise ValueError("expected all three prospective arms to fail")
    registered = result.get("registered_predictions")
    expected_predictions = {
        "pred_a_fixed_scale_high_lr_passes": False,
        "pred_b_row_specific_low_lr_passes": False,
        "pred_c_fixed_scale_low_lr_passes": False,
    }
    if registered != expected_predictions:
        raise ValueError("registered prediction scoring changed")

    return {
        "schema": "rung523-terminal-optimizer-audit-v1",
        "passes": True,
        "status": result["status"],
        "test_opened": False,
        "omitted_targets_evaluated": False,
        "frame_artifact_file_sha256": artifact_file_sha256,
        "frame_count": 45,
        "exact_call_ledger": ledger,
        "arm_summaries": summaries,
        "decision": decision,
        "interpretation": (
            "fixed normalization removes catastrophic spikes but no arm is healthy in all "
            "15 fits; raw Adam through differentiable QR is closed, with no circuit null"
        ),
    }


def _validate_dependencies(result: Mapping[str, object]) -> None:
    dependencies = result.get("dependency_sha256")
    if not isinstance(dependencies, Mapping):
        raise ValueError("dependency hash map is absent")
    for relative, expected in dependencies.items():
        path = REPO / str(relative)
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"dependency changed: {relative}")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.result.open(encoding="utf-8") as source:
        result = json.load(source)
    _validate_dependencies(result)
    artifact_sha256 = _file_sha256(args.archive)
    artifact = torch.load(args.archive, map_location="cpu", weights_only=False)
    audit = {
        "result_file_sha256": _file_sha256(args.result),
        **audit_terminal_result(
            result, artifact, artifact_file_sha256=artifact_sha256
        ),
    }
    _atomic_json(args.output, audit)
    print(json.dumps({
        "output": str(args.output),
        "passes": audit["passes"],
        "status": audit["status"],
        "test_opened": audit["test_opened"],
        "passing_fits": {
            arm: values["passing_fits"] for arm, values in audit["arm_summaries"].items()
        },
    }, indent=2, sort_keys=True), flush=True)
    return audit


if __name__ == "__main__":
    main()
