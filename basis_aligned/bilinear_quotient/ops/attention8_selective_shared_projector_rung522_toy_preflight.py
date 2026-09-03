#!/usr/bin/env python3
"""CPU planted-subspace preflight for rung 522.

Prediction: with a rank-4 selective subspace and an equally member-predictive
rank-4 broad subspace planted in a 32-dimensional toy response, the frozen
max-over-target objective recovers the selective projector for all five seeds.
The held-out target must have normalized projector overlap >= .90 with the
planted selective subspace, overlap <= .10 with the broad subspace, response
cosine >= .99, optimally scaled residual <= .05, and concentration >= 4.

Null: a recovery-only objective cannot distinguish the two planted subspaces;
failure of the control-penalized fit means the proposed optimizer is not a live
instrument and blocks model science.  The broad and selective projectors are
constructed to have exactly equal member response, so differing member power
cannot make the test pass.

Price: five CPU-only fits x 200 updates, each on two full-batch toy targets.
There are no model calls, CUDA calls, or learned model parameters.  The largest
learned object is a 32 x 4 float64 frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


if os.environ.get("BQLIB_DRYRUN") == "1":
    print(
        "DRYRUN OK: CPU-only rung522 toy; 5 seeds x 200 updates; "
        "dimension=32 rank=4; no model or CUDA imports",
        flush=True,
    )
    raise SystemExit(0)


import torch  # noqa: E402

import attention8_selective_shared_projector_rung522_math as core  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).with_name(
    "attention8_selective_shared_projector_rung522_toy_preflight_results.json"
)
TOY_SEEDS = tuple(range(52200, 52205))
TARGETS = ("toy_a", "toy_b", "toy_heldout")


@dataclass(frozen=True)
class ToyTarget:
    member_displacement: torch.Tensor
    member_reader: torch.Tensor
    control_displacement: torch.Tensor
    control_reader: torch.Tensor


@dataclass(frozen=True)
class ToyProblem:
    selective: torch.Tensor
    broad: torch.Tensor
    train: dict[str, ToyTarget]
    validation: dict[str, ToyTarget]
    heldout: ToyTarget


def _target(
    selective: torch.Tensor,
    broad: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> ToyTarget:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rank = selective.shape[1]
    member_code = torch.randn(samples, rank, generator=generator, dtype=torch.float64)
    member_read = torch.randn(samples, rank, generator=generator, dtype=torch.float64)
    control_code = torch.randn(samples, rank, generator=generator, dtype=torch.float64)
    control_read = torch.randn(samples, rank, generator=generator, dtype=torch.float64)
    # The two orthogonal rank-4 copies carry identical member codes and readers.
    # Consequently P_selective and P_broad produce exactly the same member
    # response.  Only the broad copy exists for controls.
    return ToyTarget(
        member_displacement=member_code @ selective.mT + member_code @ broad.mT,
        member_reader=member_read @ selective.mT + member_read @ broad.mT,
        control_displacement=control_code @ broad.mT,
        control_reader=control_read @ selective.mT + control_read @ broad.mT,
    )


def make_problem(*, dimension: int = 32, rank: int = 4, samples: int = 192) -> ToyProblem:
    if dimension < 2 * rank:
        raise ValueError("toy needs two orthogonal rank-sized subspaces")
    joint = core.deterministic_haar_frame(dimension, 2 * rank, 52122)
    selective, broad = joint[:, :rank], joint[:, rank:]
    train = {
        name: _target(selective, broad, seed=53000 + index, samples=samples)
        for index, name in enumerate(TARGETS[:2])
    }
    validation = {
        name: _target(selective, broad, seed=54000 + index, samples=samples)
        for index, name in enumerate(TARGETS[:2])
    }
    heldout = _target(selective, broad, seed=55000, samples=samples)
    return ToyProblem(
        selective=selective,
        broad=broad,
        train=train,
        validation=validation,
        heldout=heldout,
    )


def _responses(
    datasets: dict[str, ToyTarget], frame: torch.Tensor
) -> dict[str, core.TargetResponse]:
    result = {}
    for name, target in datasets.items():
        result[name] = core.TargetResponse(
            full_member=core.full_bilinear_response(
                target.member_displacement, target.member_reader
            ),
            projected_member=core.projected_bilinear_response(
                target.member_displacement, target.member_reader, frame
            ),
            projected_control=core.projected_bilinear_response(
                target.control_displacement, target.control_reader, frame
            ),
        )
    return result


def _objective(problem: ToyProblem, frame: torch.Tensor, split: str) -> core.ObjectiveResult:
    datasets = problem.train if split == "train" else problem.validation
    return core.exact_max_target_objective(
        _responses(datasets, frame),
        control_coefficient=core.OptimizerConfig().control_coefficient,
        epsilon=core.OptimizerConfig().loss_epsilon,
    )


def _heldout_metrics(problem: ToyProblem, frame: torch.Tensor) -> dict[str, float]:
    target = problem.heldout
    full = core.full_bilinear_response(target.member_displacement, target.member_reader)
    projected = core.projected_bilinear_response(
        target.member_displacement, target.member_reader, frame
    )
    control = core.projected_bilinear_response(
        target.control_displacement, target.control_reader, frame
    )
    return {
        **core.signed_response_metrics(projected, full),
        **core.response_concentration(projected, control),
        "selective_projector_overlap": float(
            core.daslib.normalized_projector_overlap(frame, problem.selective)
        ),
        "broad_projector_overlap": float(
            core.daslib.normalized_projector_overlap(frame, problem.broad)
        ),
    }


def run_preflight() -> dict[str, object]:
    config = core.OptimizerConfig()
    problem = make_problem(rank=config.rank)
    selective_objective = float(_objective(problem, problem.selective, "validation").maximum)
    broad_objective = float(_objective(problem, problem.broad, "validation").maximum)
    # This is the exact anti-confound: member losses are equal before the
    # control term is added, despite a strictly ordered full objective.
    selective_member = max(
        float(loss.member)
        for loss in _objective(problem, problem.selective, "validation").per_target.values()
    )
    broad_member = max(
        float(loss.member)
        for loss in _objective(problem, problem.broad, "validation").per_target.values()
    )
    if abs(selective_member - broad_member) > 1e-12:
        raise RuntimeError("toy construction does not equalize selective and broad member power")
    if not selective_objective < broad_objective:
        raise RuntimeError("toy control term does not reject the planted broad projector")

    seed_results = []
    for seed in TOY_SEEDS:
        fit = core.fit_projector(
            problem.selective.shape[0],
            seed,
            lambda frame, _step: _responses(problem.train, frame),
            lambda frame, _step: _responses(problem.validation, frame),
            config=config,
            dtype=torch.float64,
            device="cpu",
        )
        metrics = _heldout_metrics(problem, fit.frame)
        passed = bool(
            fit.healthy
            and metrics["selective_projector_overlap"] >= 0.90
            and metrics["broad_projector_overlap"] <= 0.10
            and metrics["signed_cosine"] >= 0.99
            and metrics["relative_residual"] <= 0.05
            and metrics["member_to_control_concentration"] >= 4.0
        )
        seed_results.append(
            {
                "seed": seed,
                "healthy": fit.healthy,
                "health_failures": list(fit.health_failures),
                "initial_validation_objective": fit.initial_validation_objective,
                "final_validation_objective": fit.final_validation_objective,
                "first_20_loss_mean": sum(fit.loss_history[:20]) / 20,
                "final_20_loss_mean": sum(fit.loss_history[-20:]) / 20,
                "final_orthonormality_error": fit.final_orthonormality_error,
                "projector_distance_from_initialization": (
                    fit.projector_distance_from_initialization
                ),
                "maximizing_target_counts": {
                    name: fit.maximizing_target_history.count(name) for name in TARGETS[:2]
                },
                **metrics,
                "passed": passed,
            }
        )

    prediction = all(bool(result["passed"]) for result in seed_results)
    payload: dict[str, object] = {
        "experiment": "attention8_selective_shared_projector_rung522_toy_preflight",
        "cpu_only": True,
        "config": config.__dict__,
        "toy": {
            "dimension": problem.selective.shape[0],
            "rank": problem.selective.shape[1],
            "samples_per_target_split": len(problem.heldout.member_displacement),
            "selective_member_loss": selective_member,
            "broad_member_loss": broad_member,
            "selective_total_objective": selective_objective,
            "broad_total_objective": broad_objective,
        },
        "seeds": seed_results,
        "prediction": prediction,
        "model_science_opened": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["content_sha256_without_hash_field"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_preflight()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["prediction"]:
        raise SystemExit("RUNG522 TOY PREFLIGHT FAILED: model science remains closed")


if __name__ == "__main__":
    main()
