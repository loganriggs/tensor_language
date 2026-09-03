#!/usr/bin/env python3
"""Run the sealed FIT/VALIDATION-only rung-523 optimizer diagnosis.

Scientific non-claim: this calibrates the optimizer used by rung 522.  It never
evaluates an omitted target, never reads TEST, and cannot establish a circuit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
POLY = ROOT.parent / "polynomial_causal"
REPO = POLY.parent.parent
PREREG = POLY / "ATTENTION8_PROJECTOR_OPTIMIZER_REPAIR_RUNG523_PREREGISTRATION.md"
MATH_PATH = OPS / "attention8_projector_optimizer_repair_rung523_math.py"
R522_RUNNER = OPS / "attention8_selective_shared_projector_rung522_run.py"
R522_ARCHIVE_MODULE = OPS / "attention8_selective_shared_projector_rung522_archive.py"
R522_SCHEDULER = OPS / "attention8_selective_shared_projector_rung522_scheduler.py"
R522_STATE = OPS / "attention8_selective_shared_projector_rung522_state_guard.py"
BASELINE_ARCHIVE = ROOT / "attention8_selective_shared_projector_rung522_work/frames_pretest.pt"
DEFAULT_OUTPUT = ROOT / "attention8_projector_optimizer_repair_rung523_results.json"
DEFAULT_WORK = ROOT / "attention8_projector_optimizer_repair_rung523_work"

FROZEN_SHA256 = {
    PREREG: "930a751ff6b7f6c69ae6765b569aa31172b5b5aea334ed1f639d17111861e035",
    MATH_PATH: "0d16b27cdf107efcf40f425bdc1e81350b07d3367db83eeded61a49d676e39e1",
    R522_RUNNER: "b9ff888e808cca1459c469ea15c111a421ebbb0a2d56999c10378099c5e305d0",
    R522_ARCHIVE_MODULE: "02680d4912d48d4199b6aaa607d1c77120822217e8e56b40a61d80bddb33dec9",
    R522_SCHEDULER: "d840318d5b675ce762f6c9a0d451c11550c6520b97ffbb762672ec703af5540f",
    R522_STATE: "028a21352506236ae99c4181925494ed144993fd2186cb14d61fb8a16fe00d9c",
    BASELINE_ARCHIVE: "2b8d3709714903890c4ae935a07da7284ac3253b7b2242d055023b33adeca2bb",
}

EXPECTED_OPTIMIZATION_FORWARDS = 9_000
EXPECTED_OPTIMIZATION_BACKWARDS = 9_000
EXPECTED_INFERENCE_BY_BUCKET = {
    "native_capture": 131,
    "native_replay": 131,
    "self_donor": 2,
    "fit_d0_full_attention8": 95,
    "full_attention8_comparator": 36,
    "fit_health": 120,
}
EXPECTED_INFERENCE_FORWARDS = sum(EXPECTED_INFERENCE_BY_BUCKET.values())

# pred_a fixed target/map scaling alone repairs the optimizer at learning rate .03.
# pred_b lowering the learning rate alone repairs the row-specific optimizer.
# pred_c both changes together repair the optimizer when neither single change does.


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frozen_dependencies() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        value = _file_sha256(path)
        if value != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {value} != {expected}")
        observed[str(path.relative_to(REPO))] = value
    return observed


_DEPENDENCY_SHA256 = _validate_frozen_dependencies()
if os.environ.get("BQLIB_DRYRUN") == "1":
    print(
        "DRYRUN OK: rung523 FIT/VALIDATION only; 3 arms x 15 fits x 200 updates; "
        "9000 forwards + 9000 backwards; 515 inference; TEST unreachable",
        flush=True,
    )
    raise SystemExit(0)


import torch  # noqa: E402

for _path in (OPS, POLY, ROOT, REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import attention8_projector_optimizer_repair_rung523_math as repair  # noqa: E402
import attention8_selective_shared_projector_rung522_archive as archive  # noqa: E402
import attention8_selective_shared_projector_rung522_run as r522  # noqa: E402
import attention8_selective_shared_projector_rung522_state_guard as r522_state  # noqa: E402
import attention8_shared_private_das_rung521 as stage_a  # noqa: E402
import bilin18_observed_model_facade as facade  # noqa: E402


class FitValidationOnlyState:
    """Small executable seal: FIT and VALIDATION are the only legal split names."""

    def __init__(self) -> None:
        self.inference_events = 0
        self.optimization_forwards = 0
        self.optimization_backwards = 0
        self.requested_splits: list[str] = []

    def authorize_split_access(self, split: str) -> None:
        if split not in {"FIT", "VALIDATION"}:
            raise RuntimeError(f"rung523 forbids split access: {split}")
        self.requested_splits.append(split)

    def record_inference_events(self, count: int) -> None:
        if count < 0:
            raise ValueError("negative inference count")
        self.inference_events += count
        if self.inference_events > EXPECTED_INFERENCE_FORWARDS:
            raise RuntimeError("rung523 inference ceiling exceeded")

    def record_optimization_events(self, forwards: int, backwards: int) -> None:
        if forwards < 0 or backwards < 0:
            raise ValueError("negative optimization count")
        self.optimization_forwards += forwards
        self.optimization_backwards += backwards
        if self.optimization_forwards > EXPECTED_OPTIMIZATION_FORWARDS:
            raise RuntimeError("rung523 optimization-forward ceiling exceeded")
        if self.optimization_backwards > EXPECTED_OPTIMIZATION_BACKWARDS:
            raise RuntimeError("rung523 optimization-backward ceiling exceeded")

    def assert_terminal(self) -> None:
        if set(self.requested_splits) - {"FIT", "VALIDATION"}:
            raise RuntimeError("rung523 touched a forbidden split")
        if self.optimization_forwards != EXPECTED_OPTIMIZATION_FORWARDS:
            raise RuntimeError("rung523 optimization-forward count changed")
        if self.optimization_backwards != EXPECTED_OPTIMIZATION_BACKWARDS:
            raise RuntimeError("rung523 optimization-backward count changed")
        if self.inference_events != EXPECTED_INFERENCE_FORWARDS:
            raise RuntimeError("rung523 inference count changed")


def real_specs() -> tuple[r522_state.FrameSpec, ...]:
    specs = tuple(
        value for value in r522_state.EXPECTED_FRAME_SPECS.values()
        if value.family == "real_leave_one_out"
    )
    if len(specs) != repair.EXPECTED_FITS_PER_ARM:
        raise RuntimeError("real leave-one-out frame census changed")
    if any(spec.omitted_target is None or len(spec.training_targets) != 2 for spec in specs):
        raise RuntimeError("real leave-one-out specification changed")
    return specs


def _objective(
    responses: Mapping[str, r522.core.TargetResponse],
    fixed_scales: Mapping[str, torch.Tensor],
    *,
    map_index: int,
    scale_mode: str,
) -> tuple[torch.Tensor, str, dict[str, float]]:
    losses = {}
    for target in sorted(responses):
        response = responses[target]
        if scale_mode == "row_specific":
            denominator = response.full_member.square().mean() + 1e-12
        elif scale_mode == "fixed_target_map":
            denominator = fixed_scales[target][map_index]
        else:
            raise ValueError(f"unknown scale mode {scale_mode}")
        losses[target] = repair.normalized_target_loss(
            response.full_member,
            response.projected_member,
            response.projected_control,
            denominator=denominator,
            control_coefficient=24.0,
        )
    stacked = torch.stack([losses[target] for target in sorted(losses)])
    maximum, index = torch.max(stacked, dim=0)
    name = sorted(losses)[int(index.detach().cpu())]
    return maximum, name, {
        target: float(loss.detach().cpu()) for target, loss in losses.items()
    }


@dataclass(frozen=True)
class CandidateRecord:
    arm: str
    frame_id: str
    omitted_target: str
    seed: int
    scale_mode: str
    learning_rate: float
    frame_sha256: str
    fit_scheduler_sha256: str
    validation_scheduler_sha256: str
    loss_history: tuple[float, ...]
    maximizing_targets: tuple[str, ...]
    initial_common_validation: float
    final_common_validation: float
    orthonormality_error: float
    projector_distance: float
    finite_gradients: bool
    model_gradients_absent: bool

    def health(self) -> repair.FitHealth:
        return repair.FitHealth(
            losses=self.loss_history,
            initial_common_validation=self.initial_common_validation,
            final_common_validation=self.final_common_validation,
            orthonormality_error=self.orthonormality_error,
            projector_distance=self.projector_distance,
            finite_gradients=self.finite_gradients,
            model_gradients_absent=self.model_gradients_absent,
        )


def _common_validation(
    callback: r522.ProjectedResponseCallback,
    frame: torch.Tensor,
    fixed_scales: Mapping[str, torch.Tensor],
) -> tuple[float, dict[str, float]]:
    with torch.no_grad():
        responses = callback(frame, -1)
        maximum, _name, per_target = _objective(
            responses, fixed_scales, map_index=0, scale_mode="fixed_target_map"
        )
    return float(maximum.detach().cpu()), per_target


def _fit_candidate(
    instrument: r522.Rung522Instrument,
    spec: r522_state.FrameSpec,
    training_callback: r522.ProjectedResponseCallback,
    validation_callback: r522.ProjectedResponseCallback,
    fixed_scales: Mapping[str, torch.Tensor],
    *,
    arm: str,
    scale_mode: str,
    learning_rate: float,
) -> tuple[torch.Tensor, CandidateRecord, dict[str, object]]:
    initial = r522.core.deterministic_haar_frame(
        r522.D, r522.RANK, spec.seed, dtype=torch.float32, device=instrument.device
    )
    raw = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        (raw,), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
    initial_validation, initial_per_target = _common_validation(
        validation_callback, initial, fixed_scales
    )
    losses: list[float] = []
    maximizing: list[str] = []
    per_target_digest = hashlib.sha256()
    for update in range(repair.UPDATES_PER_FIT):
        optimizer.zero_grad(set_to_none=True)
        frame = r522.core.differentiable_qr_retraction(raw)
        before_forward = instrument.ledger.optimization_forwards
        before_backward = instrument.ledger.optimization_backwards
        responses = training_callback(frame, update)
        if instrument.ledger.optimization_forwards != before_forward + 1:
            raise RuntimeError("candidate update did not execute one model forward")
        objective, target, per_target = _objective(
            responses,
            fixed_scales,
            map_index=update % 4,
            scale_mode=scale_mode,
        )
        if not bool(torch.isfinite(objective).detach().cpu()):
            raise FloatingPointError(f"non-finite objective at {arm}/{spec.frame_id}/{update}")
        objective.backward()
        instrument.ledger.charge("optimization_backward")
        instrument.state.record_optimization_events(0, 1)
        if instrument.ledger.optimization_backwards != before_backward + 1:
            raise RuntimeError("candidate update did not execute one backward")
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all().detach().cpu()):
            raise FloatingPointError(f"non-finite gradient at {arm}/{spec.frame_id}/{update}")
        r522.core.assert_parameters_have_no_gradients(tuple(instrument.model.parameters()))
        optimizer.step()
        value = float(objective.detach().cpu())
        losses.append(value)
        maximizing.append(target)
        per_target_digest.update(json.dumps(
            {"update": update, "losses": per_target}, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode())

    final = r522.core.differentiable_qr_retraction(raw.detach()).cpu().contiguous()
    final_validation, final_per_target = _common_validation(
        validation_callback, final.to(instrument.device), fixed_scales
    )
    initial_cpu = initial.detach().cpu().float()
    orthonormality = float(
        (final.mT @ final - torch.eye(r522.RANK, dtype=torch.float32)).abs().amax()
    )
    overlap = (initial_cpu.mT @ final).square().sum()
    distance = float((2 * r522.RANK - 2 * overlap).clamp_min(0).sqrt())
    record = CandidateRecord(
        arm=arm,
        frame_id=spec.frame_id,
        omitted_target=str(spec.omitted_target),
        seed=spec.seed,
        scale_mode=scale_mode,
        learning_rate=learning_rate,
        frame_sha256=archive.tensor_sha256(final),
        fit_scheduler_sha256=training_callback.balanced.fingerprint,
        validation_scheduler_sha256=validation_callback.balanced.fingerprint,
        loss_history=tuple(losses),
        maximizing_targets=tuple(maximizing),
        initial_common_validation=initial_validation,
        final_common_validation=final_validation,
        orthonormality_error=orthonormality,
        projector_distance=distance,
        finite_gradients=True,
        model_gradients_absent=True,
    )
    diagnostics = {
        "initial_common_validation_by_target": initial_per_target,
        "final_common_validation_by_target": final_per_target,
        "per_update_per_target_loss_sha256": per_target_digest.hexdigest(),
    }
    del initial, raw, optimizer
    return final, record, diagnostics


def _callbacks(
    instrument: r522.Rung522Instrument,
    spec: r522_state.FrameSpec,
    fit_pairs: Mapping[str, r522.TargetPairs],
    validation_pairs: Mapping[str, r522.TargetPairs],
    validation_full_map0: torch.Tensor,
) -> tuple[r522.ProjectedResponseCallback, r522.ProjectedResponseCallback]:
    fit_scheduler = r522._make_balanced_scheduler(
        spec, fit_pairs, instrument.data["row_masks"]["fit"]
    )
    validation_scheduler = r522._make_balanced_scheduler(
        spec, validation_pairs, instrument.data["row_masks"]["validation"]
    )
    return (
        r522.ProjectedResponseCallback(
            instrument,
            spec,
            split="fit",
            pairs=fit_pairs,
            balanced=fit_scheduler,
            full_by_map=instrument.full_fit_d0,
            optimization=True,
        ),
        r522.ProjectedResponseCallback(
            instrument,
            spec,
            split="validation",
            pairs=validation_pairs,
            balanced=validation_scheduler,
            full_by_map=validation_full_map0,
            optimization=False,
            fixed_health_batch=True,
        ),
    )


def _atomic_torch(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as sink:
            torch.save(value, sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _baseline_score(
    instrument: r522.Rung522Instrument,
    loaded: archive.LoadedFrameArchive,
    fit_pairs: Mapping[str, r522.TargetPairs],
    validation_pairs: Mapping[str, r522.TargetPairs],
    validation_full_map0: torch.Tensor,
    fixed_scales: Mapping[str, torch.Tensor],
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    fits = []
    common = {}
    for spec in real_specs():
        _training, validation = _callbacks(
            instrument, spec, fit_pairs, validation_pairs, validation_full_map0
        )
        initial = r522.core.deterministic_haar_frame(
            r522.D, r522.RANK, spec.seed, dtype=torch.float32, device=instrument.device
        )
        final = loaded.frames[spec.frame_id].to(instrument.device)
        initial_value, initial_by_target = _common_validation(validation, initial, fixed_scales)
        final_value, final_by_target = _common_validation(validation, final, fixed_scales)
        archived = loaded.records[spec.frame_id]
        health = archived.health_record_payload
        fits.append(repair.FitHealth(
            losses=archived.fit_record_payload["loss_history"],
            initial_common_validation=initial_value,
            final_common_validation=final_value,
            orthonormality_error=float(health["orthonormality_error"]),
            projector_distance=float(health["projector_distance_from_initialization"]),
            finite_gradients=True,
            model_gradients_absent=True,
        ))
        common[spec.frame_id] = {
            "initial": initial_value,
            "final": final_value,
            "initial_by_target": initial_by_target,
            "final_by_target": final_by_target,
        }
    return repair.score_candidate_cell(fits), common


def _assert_exact_ledger(ledger: r522.CallLedger, state: FitValidationOnlyState) -> None:
    if ledger.optimization_forwards != EXPECTED_OPTIMIZATION_FORWARDS:
        raise RuntimeError("optimization-forward ledger changed")
    if ledger.optimization_backwards != EXPECTED_OPTIMIZATION_BACKWARDS:
        raise RuntimeError("optimization-backward ledger changed")
    if ledger.inference_forwards != EXPECTED_INFERENCE_FORWARDS:
        raise RuntimeError("inference-forward ledger changed")
    if ledger.inference_by_bucket != EXPECTED_INFERENCE_BY_BUCKET:
        raise RuntimeError(
            f"inference buckets changed: {ledger.inference_by_bucket} != "
            f"{EXPECTED_INFERENCE_BY_BUCKET}"
        )
    if ledger.removal_forwards != 0:
        raise RuntimeError("rung523 unexpectedly performed a removal call")
    state.assert_terminal()


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-directory", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args(argv)
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids rung523 execution")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    if args.work_directory.exists():
        raise FileExistsError(f"refusing to reuse work directory: {args.work_directory}")
    args.work_directory.mkdir(parents=True, exist_ok=False)
    started = time.time()

    loaded = archive.load_frame_archive(BASELINE_ARCHIVE)
    data, design, preflight = stage_a.preflight()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    ledger = r522.CallLedger()
    state = FitValidationOnlyState()
    instrument = r522.Rung522Instrument(model, data, design, ledger, state)
    capture = instrument.capture_pretest_splits()
    fit_full_receipt = instrument.precompute_full_fit_d0()
    validation_full = instrument.evaluate_swap(
        "validation", frame=None, inference_bucket="full_attention8_comparator"
    )
    if instrument.full_fit_d0 is None:
        raise RuntimeError("FIT full-attention response cache is absent")

    fit_pairs = {
        target: r522._combined_fit_pairs(design, target)
        for target in r522_state.FITTED_TARGETS
    }
    validation_pairs = r522._exclusive_pairs_for_cell(design, "validation")
    validation_full_map0 = validation_full.map_responses["D0:forward"][:1]
    fixed_scales = {
        target: repair.fixed_target_map_scales(
            instrument.full_fit_d0,
            fit_pairs[target].mask("member").view(1000, r522.TOKENS),
            instrument.split_rows["fit"],
        )
        for target in r522_state.FITTED_TARGETS
    }

    baseline_score, baseline_common = _baseline_score(
        instrument,
        loaded,
        fit_pairs,
        validation_pairs,
        validation_full_map0,
        fixed_scales,
    )

    arm_definitions = {
        repair.ROW_LOW_LR: ("row_specific", 0.003),
        repair.FIXED_HIGH_LR: ("fixed_target_map", 0.03),
        repair.FIXED_LOW_LR: ("fixed_target_map", 0.003),
    }
    frames: dict[str, torch.Tensor] = {}
    records: dict[str, list[CandidateRecord]] = {name: [] for name in repair.PROSPECTIVE_ARMS}
    diagnostics: dict[str, dict[str, object]] = {name: {} for name in repair.PROSPECTIVE_ARMS}
    total = len(repair.PROSPECTIVE_ARMS) * len(real_specs())
    completed = 0
    for arm in repair.PROSPECTIVE_ARMS:
        scale_mode, learning_rate = arm_definitions[arm]
        for spec in real_specs():
            training, validation = _callbacks(
                instrument, spec, fit_pairs, validation_pairs, validation_full_map0
            )
            frame, record, fit_diagnostics = _fit_candidate(
                instrument,
                spec,
                training,
                validation,
                fixed_scales,
                arm=arm,
                scale_mode=scale_mode,
                learning_rate=learning_rate,
            )
            key = f"{arm}:{spec.frame_id}"
            frames[key] = frame
            records[arm].append(record)
            diagnostics[arm][spec.frame_id] = fit_diagnostics
            completed += 1
            print(
                f"RUNG523 FIT {completed:02d}/{total} {key} "
                f"final_validation={record.final_common_validation:.6g}",
                flush=True,
            )
            if completed % 10 == 0:
                torch.cuda.empty_cache()

    cell_scores = {
        arm: repair.score_candidate_cell([record.health() for record in records[arm]])
        for arm in repair.PROSPECTIVE_ARMS
    }
    decision = repair.adoption_decision(cell_scores)
    _assert_exact_ledger(ledger, state)

    artifact_payload = {
        "schema": "attention8-projector-optimizer-repair-rung523-frames-v1",
        "frames": frames,
        "records": {
            arm: [asdict(record) for record in records[arm]]
            for arm in repair.PROSPECTIVE_ARMS
        },
        "diagnostics": diagnostics,
    }
    artifact_path = args.work_directory / "candidate_frames.pt"
    artifact_sha256 = _atomic_torch(artifact_path, artifact_payload)
    result = {
        "schema_version": 1,
        "rung": 523,
        "status": (
            "optimizer_repair_adopted"
            if decision["licenses_sealed_rung522_repeat"]
            else "raw_adam_through_qr_closed"
        ),
        "claim_level": "FIT/VALIDATION-only optimizer calibration; no circuit evidence",
        "test_accessible": False,
        "test_opened": False,
        "omitted_targets_evaluated": False,
        "dependency_sha256": _DEPENDENCY_SHA256,
        "checkpoint": checkpoint.__dict__,
        "stage_a_preflight": preflight,
        "capture": capture,
        "fit_full_attention8": fit_full_receipt,
        "fixed_fit_target_map_scales": {
            target: [float(value) for value in scales]
            for target, scales in fixed_scales.items()
        },
        "baseline": {
            "arm": repair.ROW_HIGH_LR,
            "archive_file_sha256": loaded.file_sha256,
            "score": baseline_score,
            "common_validation": baseline_common,
        },
        "prospective_cells": cell_scores,
        "registered_predictions": {
            "pred_a_fixed_scale_high_lr_passes": bool(
                cell_scores[repair.FIXED_HIGH_LR]["passes"]
            ),
            "pred_b_row_specific_low_lr_passes": bool(
                cell_scores[repair.ROW_LOW_LR]["passes"]
            ),
            "pred_c_fixed_scale_low_lr_passes": bool(
                cell_scores[repair.FIXED_LOW_LR]["passes"]
            ),
        },
        "candidate_records": {
            arm: [asdict(record) for record in records[arm]]
            for arm in repair.PROSPECTIVE_ARMS
        },
        "candidate_diagnostics": diagnostics,
        "decision": decision,
        "frame_artifact": {
            "path": str(artifact_path.resolve()),
            "file_sha256": artifact_sha256,
            "frame_count": len(frames),
        },
        "execution_price": {
            **ledger.snapshot(),
            "runtime_seconds": time.time() - started,
        },
        "seal": {
            "requested_splits": state.requested_splits,
            "test_split_requested": False,
            "scientific_a_through_d_scored": False,
        },
    }
    r522._atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "candidate_passes": decision["candidate_passes"],
        "adopted_arm": decision["adopted_arm"],
        "test_opened": False,
    }, indent=2, sort_keys=True), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    main()
