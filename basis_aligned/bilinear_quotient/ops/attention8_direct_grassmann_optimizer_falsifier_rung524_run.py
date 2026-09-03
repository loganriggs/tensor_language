#!/usr/bin/env python3
"""Run rung 524's CPU-only planted direct-Grassmann optimizer falsifier."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
POLY = ROOT.parent / "polynomial_causal"
REPO = ROOT.parent.parent
PREREG = POLY / "ATTENTION8_DIRECT_GRASSMANN_OPTIMIZER_FALSIFIER_RUNG524_PREREGISTRATION.md"
IMPLEMENTATION = POLY / "ATTENTION8_DIRECT_GRASSMANN_OPTIMIZER_FALSIFIER_RUNG524_IMPLEMENTATION_RECEIPT.md"
MATH_PATH = OPS / "attention8_direct_grassmann_optimizer_falsifier_rung524_math.py"
DEFAULT_OUTPUT = ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_results.json"
DEFAULT_ARCHIVE = ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_frames.pt"

for path in (OPS,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import attention8_direct_grassmann_optimizer_falsifier_rung524_math as gm  # noqa: E402


FROZEN_SHA256 = {
    PREREG: "70acb0efb86dfe6357bca258fd3c47af0a4ef4d0008641dfd8ef8c63dc507321",
    MATH_PATH: "b38c3551ac537940c8c8b72b95e37db4e22389185e6fae34cb5cc25c1d9b4072",
}
PLANT_SEED = 524000
FIT_SEED = 524100
VALIDATION_SEED = 524200
OOD_SEED = 524300
INITIAL_SEEDS = tuple(range(524400, 524405))
TARGETS = tuple(range(gm.TARGET_COUNT))
MAPS = tuple(range(gm.MAP_COUNT))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_dependencies() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        value = _file_sha256(path)
        if value != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {value} != {expected}")
        observed[str(path.relative_to(REPO))] = value
    return observed


@dataclass(frozen=True)
class SplitData:
    name: str
    member_inputs: torch.Tensor  # [target,map,example,D]
    target_outputs: torch.Tensor  # [target,map,example,output]
    control_inputs: torch.Tensor  # [target,map,example,D]


@dataclass(frozen=True)
class FitRecord:
    omitted_target: int
    seed: int
    accepted_updates: int
    failed_update: int | None
    finite_losses: bool
    finite_gradients: bool
    maximum_evaluated_loss: float
    initial_fit_loss: float
    final_fit_loss: float
    initial_validation_loss: float
    final_validation_loss: float
    orthonormality_error: float
    projector_error: float
    minimum_principal_cosine: float
    frame_sha256: str
    loss_history: tuple[float, ...]
    accepted_step_sizes: tuple[float, ...]
    backtracks_per_update: tuple[int, ...]

    def pretest(self) -> gm.PretestFit:
        return gm.PretestFit(
            accepted_updates=self.accepted_updates,
            finite_losses=self.finite_losses,
            finite_gradients=self.finite_gradients,
            maximum_evaluated_loss=self.maximum_evaluated_loss,
            orthonormality_error=self.orthonormality_error,
            initial_fit_loss=self.initial_fit_loss,
            final_fit_loss=self.final_fit_loss,
            initial_validation_loss=self.initial_validation_loss,
            final_validation_loss=self.final_validation_loss,
            projector_error=self.projector_error,
            minimum_principal_cosine=self.minimum_principal_cosine,
        )


class SplitSeal:
    """Construct OOD data only after every frozen FIT/VALIDATION gate passes."""

    def __init__(self) -> None:
        self.requested = ["FIT", "VALIDATION"]
        self.ood_opened = False

    def open_ood(self, pretest: Mapping[str, object]) -> None:
        if pretest.get("pretest_passes") is not True:
            raise RuntimeError("cannot open OOD after a failed pretest")
        self.requested.append("OOD")
        self.ood_opened = True


def _random_frame(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return gm.canonical_qr(torch.randn(
        gm.D, gm.RANK, generator=generator, dtype=torch.float64
    ))


def _planted_objects() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(PLANT_SEED)
    planted = gm.canonical_qr(torch.randn(
        gm.D, gm.RANK, generator=generator, dtype=torch.float64
    ))
    readouts = torch.randn(
        gm.TARGET_COUNT, gm.MAP_COUNT, gm.OUTPUT_DIMENSION, gm.D,
        generator=generator, dtype=torch.float64,
    ) / math.sqrt(gm.D)
    return planted, readouts


def _orthogonalize(values: torch.Tensor, planted: torch.Tensor) -> torch.Tensor:
    return values - (values @ planted) @ planted.mT


def _student_t3(shape: Sequence[int], generator: torch.Generator) -> torch.Tensor:
    numerator = torch.randn(*shape, generator=generator, dtype=torch.float64)
    denominator_normals = torch.randn(
        *shape, 3, generator=generator, dtype=torch.float64
    )
    chi_square = denominator_normals.square().sum(dim=-1)
    # Standard t_3 has variance 3; divide by sqrt(3) for unit variance.
    return numerator / torch.sqrt(chi_square)


def build_split(
    name: str,
    *,
    seed: int,
    examples: int,
    planted: torch.Tensor,
    readouts: torch.Tensor,
    ood: bool,
) -> SplitData:
    if name not in {"FIT", "VALIDATION", "OOD"}:
        raise ValueError("unknown split")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (gm.TARGET_COUNT, gm.MAP_COUNT, examples)
    coefficients = (
        _student_t3((*shape, gm.RANK), generator)
        if ood else
        torch.randn(*shape, gm.RANK, generator=generator, dtype=torch.float64)
    )
    nuisance = torch.randn(*shape, gm.D, generator=generator, dtype=torch.float64)
    controls = torch.randn(*shape, gm.D, generator=generator, dtype=torch.float64)
    if ood:
        coordinate_scale = torch.linspace(0.5, 1.5, gm.D, dtype=torch.float64)
        nuisance = nuisance * coordinate_scale
        controls = controls * coordinate_scale.flip(0)
    nuisance = 0.25 * _orthogonalize(nuisance, planted)
    controls = _orthogonalize(controls, planted)
    member = coefficients @ planted.mT + nuisance
    planted_member = (member @ planted) @ planted.mT
    outputs = torch.einsum("tmnd,tmod->tmno", planted_member, readouts)
    return SplitData(
        name=name,
        member_inputs=member.contiguous(),
        target_outputs=outputs.contiguous(),
        control_inputs=controls.contiguous(),
    )


def fixed_fit_scales(fit: SplitData) -> torch.Tensor:
    if fit.name != "FIT":
        raise ValueError("fixed scales must come from FIT")
    scales = fit.target_outputs.square().mean(dim=(2, 3)) + 1e-12
    if scales.shape != (gm.TARGET_COUNT, gm.MAP_COUNT):
        raise RuntimeError("scale shape changed")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise RuntimeError("invalid fixed scale")
    return scales


def normalized_objective(
    frame: torch.Tensor,
    data: SplitData,
    readouts: torch.Tensor,
    scales: torch.Tensor,
    targets: Sequence[int],
) -> torch.Tensor:
    losses = []
    for target in targets:
        for map_index in MAPS:
            members = data.member_inputs[target, map_index]
            controls = data.control_inputs[target, map_index]
            readout = readouts[target, map_index]
            projected_member = (members @ frame) @ frame.mT
            projected_control = (controls @ frame) @ frame.mT
            prediction = projected_member @ readout.mT
            control_prediction = projected_control @ readout.mT
            member_loss = (
                prediction - data.target_outputs[target, map_index]
            ).square().mean() / scales[target, map_index]
            control_loss = control_prediction.square().mean() / scales[target, map_index]
            losses.append(member_loss + gm.CONTROL_COEFFICIENT * control_loss)
    return torch.stack(losses).max()


def _evaluated_objective(
    frame: torch.Tensor,
    data: SplitData,
    readouts: torch.Tensor,
    scales: torch.Tensor,
    targets: Sequence[int],
    ledger: Counter[str],
    bucket: str,
) -> torch.Tensor:
    ledger[bucket] += 1
    return normalized_objective(frame, data, readouts, scales, targets)


def fit_one(
    omitted_target: int,
    seed: int,
    *,
    planted: torch.Tensor,
    readouts: torch.Tensor,
    fit: SplitData,
    validation: SplitData,
    scales: torch.Tensor,
    ledger: Counter[str],
) -> tuple[torch.Tensor, FitRecord]:
    training_targets = tuple(target for target in TARGETS if target != omitted_target)
    frame = _random_frame(seed)
    with torch.no_grad():
        initial_fit = float(_evaluated_objective(
            frame, fit, readouts, scales, training_targets, ledger, "fit_evaluation"
        ))
        initial_validation = float(_evaluated_objective(
            frame, validation, readouts, scales, TARGETS, ledger, "validation_evaluation"
        ))
    evaluated_losses = [initial_fit, initial_validation]
    losses = []
    step_sizes = []
    backtracks = []
    finite_gradients = True
    failed_update = None
    for update in range(gm.UPDATES):
        leaf = frame.detach().clone().requires_grad_(True)
        objective = _evaluated_objective(
            leaf, fit, readouts, scales, training_targets, ledger, "fit_gradient_evaluation"
        )
        ledger["gradient_evaluations"] += 1
        gradient, = torch.autograd.grad(objective, leaf)
        tangent = gm.grassmann_tangent(leaf, gradient)
        if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(tangent).all()):
            finite_gradients = False
            failed_update = update
            break
        current = float(objective.detach())
        evaluated_losses.append(current)
        squared_norm = float(tangent.detach().square().sum())
        accepted = False
        step = gm.INITIAL_STEP
        for backtrack in range(gm.MAX_BACKTRACKS + 1):
            with torch.no_grad():
                candidate = gm.retract(frame, -step * tangent.detach())
                candidate_loss = float(_evaluated_objective(
                    candidate, fit, readouts, scales, training_targets,
                    ledger, "line_search_evaluation",
                ))
            evaluated_losses.append(candidate_loss)
            if math.isfinite(candidate_loss) and candidate_loss <= current - gm.ARMIJO * step * squared_norm:
                frame = candidate
                losses.append(candidate_loss)
                step_sizes.append(step)
                backtracks.append(backtrack)
                accepted = True
                break
            step *= gm.BACKTRACK_FACTOR
        if not accepted:
            failed_update = update
            break

    with torch.no_grad():
        final_fit = float(_evaluated_objective(
            frame, fit, readouts, scales, training_targets, ledger, "fit_evaluation"
        ))
        final_validation = float(_evaluated_objective(
            frame, validation, readouts, scales, TARGETS, ledger, "validation_evaluation"
        ))
    evaluated_losses.extend((final_fit, final_validation))
    orthonormality = float(
        (frame.mT @ frame - torch.eye(gm.RANK, dtype=frame.dtype)).abs().max()
    )
    record = FitRecord(
        omitted_target=omitted_target,
        seed=seed,
        accepted_updates=len(losses),
        failed_update=failed_update,
        finite_losses=all(math.isfinite(value) for value in evaluated_losses),
        finite_gradients=finite_gradients,
        maximum_evaluated_loss=max(evaluated_losses),
        initial_fit_loss=initial_fit,
        final_fit_loss=final_fit,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
        orthonormality_error=orthonormality,
        projector_error=gm.projector_error(frame, planted),
        minimum_principal_cosine=gm.minimum_principal_cosine(frame, planted),
        frame_sha256=_tensor_sha256(frame),
        loss_history=tuple(losses),
        accepted_step_sizes=tuple(step_sizes),
        backtracks_per_update=tuple(backtracks),
    )
    return frame.detach().contiguous(), record


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


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
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


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dependencies = _validate_dependencies()
    if args.dry_run:
        print(
            "DRYRUN OK: CPU float64; 15 fits x 200 direct Grassmann updates; "
            "OOD constructed only after all pretest gates pass",
            flush=True,
        )
        return {"dry_run": True, "dependency_sha256": dependencies}
    if args.output.exists() or args.archive.exists():
        raise FileExistsError("rung524 output or archive already exists")
    if torch.cuda.is_initialized():
        raise RuntimeError("rung524 must not initialize CUDA")
    started = time.time()
    planted, readouts = _planted_objects()
    fit = build_split(
        "FIT", seed=FIT_SEED, examples=gm.FIT_EXAMPLES,
        planted=planted, readouts=readouts, ood=False,
    )
    validation = build_split(
        "VALIDATION", seed=VALIDATION_SEED, examples=gm.VALIDATION_EXAMPLES,
        planted=planted, readouts=readouts, ood=False,
    )
    scales = fixed_fit_scales(fit)
    seal = SplitSeal()
    ledger: Counter[str] = Counter()
    frames = {}
    records = []
    for omitted_target in TARGETS:
        for seed in INITIAL_SEEDS:
            frame, record = fit_one(
                omitted_target, seed, planted=planted, readouts=readouts,
                fit=fit, validation=validation, scales=scales, ledger=ledger,
            )
            key = f"omit{omitted_target}:seed{seed}"
            frames[key] = frame
            records.append(record)
            print(
                f"RUNG524 FIT {len(records):02d}/15 {key} accepted={record.accepted_updates} "
                f"val_ratio={record.final_validation_loss / record.initial_validation_loss:.6g} "
                f"projector_error={record.projector_error:.6g}",
                flush=True,
            )
    pretest = gm.score_pretest([record.pretest() for record in records])
    ood_score = None
    ood_losses = None
    if pretest["pretest_passes"]:
        seal.open_ood(pretest)
        ood = build_split(
            "OOD", seed=OOD_SEED, examples=gm.OOD_EXAMPLES,
            planted=planted, readouts=readouts, ood=True,
        )
        with torch.no_grad():
            ood_losses = [
                float(_evaluated_objective(
                    frame, ood, readouts, scales, TARGETS, ledger, "ood_evaluation"
                ))
                for frame in frames.values()
            ]
        ood_score = gm.score_ood(ood_losses)
    decision = gm.final_decision(pretest, ood_score)
    artifact = {
        "schema": "attention8-direct-grassmann-rung524-frames-v1",
        "frames": frames,
        "planted_frame": planted,
        "readouts": readouts,
        "fixed_fit_scales": scales,
        "records": [asdict(record) for record in records],
    }
    archive_sha256 = _atomic_torch(args.archive, artifact)
    result = {
        "schema_version": 1,
        "rung": 524,
        "status": (
            "direct_subspace_instrument_passes"
            if decision["instrument_passes"] else
            "direct_subspace_instrument_falsified"
        ),
        "claim_level": "planted optimizer-instrument test; no circuit evidence",
        "dependency_sha256": dependencies,
        "implementation_receipt_sha256": _file_sha256(IMPLEMENTATION),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "revision": os.environ.get("BILIN18_REVISION", "committed-worktree"),
        "design": {
            "dimensions": {
                "D": gm.D, "rank": gm.RANK, "targets": gm.TARGET_COUNT,
                "maps": gm.MAP_COUNT, "output": gm.OUTPUT_DIMENSION,
            },
            "examples": {
                "FIT": gm.FIT_EXAMPLES,
                "VALIDATION": gm.VALIDATION_EXAMPLES,
                "OOD": gm.OOD_EXAMPLES,
            },
            "seeds": {
                "plant": PLANT_SEED, "FIT": FIT_SEED,
                "VALIDATION": VALIDATION_SEED, "OOD": OOD_SEED,
                "initializations": INITIAL_SEEDS,
            },
            "optimizer": {
                "updates": gm.UPDATES,
                "initial_step": gm.INITIAL_STEP,
                "armijo": gm.ARMIJO,
                "backtrack_factor": gm.BACKTRACK_FACTOR,
                "maximum_backtracks": gm.MAX_BACKTRACKS,
                "control_coefficient": gm.CONTROL_COEFFICIENT,
            },
        },
        "frame_archive": {
            "path": str(args.archive.resolve()),
            "file_sha256": archive_sha256,
            "frame_count": len(frames),
        },
        "records": [asdict(record) for record in records],
        "pretest": pretest,
        "ood_losses": ood_losses,
        "ood": ood_score,
        "decision": decision,
        "seal": {
            "requested_splits": seal.requested,
            "ood_opened": seal.ood_opened,
        },
        "execution_price": {
            **dict(sorted(ledger.items())),
            "runtime_seconds": time.time() - started,
            "model_forwards": 0,
            "model_backwards": 0,
            "gpu_calls": 0,
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "pretest": pretest,
        "ood": ood_score,
        "decision": decision,
        "output": str(args.output),
    }, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
