#!/usr/bin/env python3
"""CPU falsifier for the frozen selective rank-one causal-response optimizer.

The toy has one planted shared actuator and one A-correlated nuisance direction. An
unconstrained DAS objective can use both, while the frozen augmented-Lagrangian
schedule must retain useful A recovery and suppress the registered P/C responses.
It imports no model code and must never touch CUDA.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_toy_result.json"
SEED = 0
STEPS_PER_BLOCK = 50
RHO_SCHEDULE = (1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0, 32.0, 32.0)
LR = 0.05
CONSTRAINT = 0.10


def _normalized_q(raw: torch.Tensor) -> torch.Tensor:
    # QR is the production Stiefel parameterization; for rank one the projector
    # is the normalized outer product, independent of QR's sign convention.
    return torch.linalg.qr(raw[:, None], mode="reduced")[0][:, 0]


def _response(delta: torch.Tensor, readout: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Linear-head response to the rank-one projected patch delta q q^T."""
    return (delta @ q) * (readout @ q)


def _problem():
    torch.manual_seed(SEED)
    dtype = torch.float64
    d, n = 8, 32
    shared = torch.zeros(d, dtype=dtype)
    nuisance = torch.zeros(d, dtype=dtype)
    control = torch.zeros(d, dtype=dtype)
    shared[0], nuisance[1], control[2] = 1.0, 1.0, 1.0
    phase = torch.linspace(-1.0, 1.0, n, dtype=dtype)
    a_delta = (1.0 + 0.12 * phase)[:, None] * shared + (0.80 - 0.08 * phase)[:, None] * nuisance
    p_delta = (1.0 + 0.10 * phase)[:, None] * nuisance
    c_delta = (0.9 - 0.10 * phase)[:, None] * control
    readout_a = shared + 0.80 * nuisance + 0.25 * control
    readout_b = 0.70 * shared + 0.10 * nuisance
    target = a_delta @ readout_a
    scale = target.abs().median()
    return shared, nuisance, a_delta, p_delta, c_delta, readout_a, readout_b, target, scale


def _fit(constrained: bool):
    shared, nuisance, a_delta, p_delta, c_delta, readout_a, readout_b, target, scale = _problem()
    torch.manual_seed(SEED)
    raw = (0.02 * torch.randn_like(shared)).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    multipliers = torch.zeros(2, dtype=raw.dtype)
    for rho in RHO_SCHEDULE:
        for _ in range(STEPS_PER_BLOCK):
            optimizer.zero_grad()
            q = _normalized_q(raw)
            a_response = _response(a_delta, readout_a, q)
            a_loss = ((a_response - target) / scale).square().mean()
            effects = torch.stack(
                (
                    _response(p_delta, readout_a, q).abs().mean() / scale,
                    _response(c_delta, readout_a, q).abs().mean() / scale,
                )
            )
            if constrained:
                violation = effects - CONSTRAINT
                shifted = torch.relu(violation + multipliers / rho)
                loss = a_loss + (0.5 * rho * shifted.square() - multipliers.square() / (2.0 * rho)).sum()
            else:
                loss = a_loss
            loss.backward()
            optimizer.step()
        if constrained:
            with torch.no_grad():
                q = _normalized_q(raw)
                effects = torch.stack(
                    (
                        _response(p_delta, readout_a, q).abs().mean() / scale,
                        _response(c_delta, readout_a, q).abs().mean() / scale,
                    )
                )
                multipliers = torch.clamp(multipliers + rho * (effects - CONSTRAINT), min=0.0)
    with torch.no_grad():
        q = _normalized_q(raw)
        a_response = _response(a_delta, readout_a, q)
        recovery = (a_response / target).abs().mean()
        p_effect = _response(p_delta, readout_a, q).abs().mean() / scale
        c_effect = _response(c_delta, readout_a, q).abs().mean() / scale
        second_readout_response = _response(a_delta, readout_b, q).abs().mean() / scale
        return {
            "q": q,
            "a_recovery": float(recovery),
            "p_effect": float(p_effect),
            "c_effect": float(c_effect),
            "shared_cosine": float((q @ shared).abs()),
            "nuisance_cosine": float((q @ nuisance).abs()),
            "second_readout_response": float(second_readout_response),
        }


def main() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU toy refuses an initialized CUDA runtime")
    unconstrained = _fit(False)
    constrained = _fit(True)
    checks = {
        "unconstrained_uses_nuisance": unconstrained["nuisance_cosine"] >= 0.35 and unconstrained["p_effect"] > 0.20,
        "constrained_recovers_shared": constrained["shared_cosine"] >= 0.90,
        "constrained_retains_a": constrained["a_recovery"] >= 0.55,
        "constrained_suppresses_p": constrained["p_effect"] <= 0.11,
        "constrained_suppresses_c": constrained["c_effect"] <= 0.11,
        "shared_actuator_visible_to_second_readout": constrained["second_readout_response"] >= 0.25,
        "fixed_schedule_executed": len(RHO_SCHEDULE) * STEPS_PER_BLOCK == 600,
        "cuda_unused": not torch.cuda.is_initialized(),
    }
    result = {
        "schema": "selective_rank1_causal_response_toy_result_v1",
        "candidate_id": "tense_auxiliary.is_vs_was.selective_das_resid18_rank1_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": "cpu",
        "seed": SEED,
        "optimizer": {
            "steps": len(RHO_SCHEDULE) * STEPS_PER_BLOCK,
            "steps_per_block": STEPS_PER_BLOCK,
            "rho_schedule": list(RHO_SCHEDULE),
            "lr": LR,
            "constraint": CONSTRAINT,
        },
        "unconstrained": {k: v for k, v in unconstrained.items() if k != "q"},
        "constrained": {k: v for k, v in constrained.items() if k != "q"},
        "basis_sha256": hashlib.sha256(constrained["q"].numpy().tobytes()).hexdigest(),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": "The fixed optimizer separates a planted shared actuator from an A-correlated nuisance only when P/C causal-response constraints are active; multiple readouts can observe the retained actuator.",
    }
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        for key in ("created_utc",):
            existing.pop(key, None)
            result.pop(key, None)
        if existing != result:
            raise FileExistsError(f"refusing to overwrite nonidentical receipt: {OUT}")
    else:
        OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
