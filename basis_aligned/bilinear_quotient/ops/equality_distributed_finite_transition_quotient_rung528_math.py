"""CPU scoring primitives for rung 528's finite-transition quotient.

These functions contain no model or data access.  The GPU runner will use them
unchanged after the raw post-MLP12 boundary instrument passes its own tests.
"""

from __future__ import annotations

import math
from typing import Any

import torch


CONTINUATIONS = ("native", "without_A14", "without_M17", "without_both")


def fit_signed_scale(target: torch.Tensor, source: torch.Tensor) -> float:
    """Least-squares scalar beta for target ~= beta * source."""
    target = target.double().reshape(-1)
    source = source.double().reshape(-1)
    den = float(torch.dot(source, source))
    if not math.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(torch.dot(source, target) / den)


def relation_metrics(target: torch.Tensor, source: torch.Tensor, beta: float) -> dict[str, float]:
    """Cosine and target-relative error for one frozen signed scale."""
    target = target.double().reshape(-1)
    source = source.double().reshape(-1)
    target2 = float(torch.dot(target, target))
    source2 = float(torch.dot(source, source))
    cross = float(torch.dot(target, source))
    cosine = cross / math.sqrt(max(target2 * source2, 1e-300))
    error2 = float(torch.dot(target - beta * source, target - beta * source))
    residual = math.sqrt(max(error2, 0.0) / max(target2, 1e-300))
    return {"cosine": cosine, "relative_residual": residual}


def factorial_interaction(effects: torch.Tensor) -> torch.Tensor:
    """Two-continuation interaction in the frozen native/A/M/both order."""
    if effects.shape[-2] != 4:
        raise ValueError("continuation axis must contain exactly four arms")
    native, without_a14, without_m17, without_both = effects.unbind(-2)
    return without_both - without_a14 - without_m17 + native


def score_pair(
    target_circuit: torch.Tensor,
    source_circuit: torch.Tensor,
    target_task: torch.Tensor,
    source_task: torch.Tensor,
) -> dict[str, Any]:
    """Apply rung 528's pre-control discovery gates to two transition responses.

    Each tensor has shape ``[2 document halves, 4 continuations, coordinates]``.
    The scale is fitted once on half zero's concatenated circuit coordinates and
    is reused everywhere else.
    """
    expected_prefix = (2, 4)
    for name, value in {
        "target_circuit": target_circuit,
        "source_circuit": source_circuit,
        "target_task": target_task,
        "source_task": source_task,
    }.items():
        if value.ndim != 3 or tuple(value.shape[:2]) != expected_prefix:
            raise ValueError(f"{name} must have shape [2,4,coordinates]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")

    beta = fit_signed_scale(target_circuit[0], source_circuit[0])
    circuit = [relation_metrics(target_circuit[h], source_circuit[h], beta) for h in range(2)]
    task = [relation_metrics(target_task[h], source_task[h], beta) for h in range(2)]
    per_continuation = [
        [relation_metrics(target_circuit[h, c], source_circuit[h, c], beta) for c in range(4)]
        for h in range(2)
    ]
    circuit_rms = torch.stack((
        target_circuit.double().square().mean(-1).sqrt(),
        source_circuit.double().square().mean(-1).sqrt(),
    ))
    task_norm = torch.stack((
        target_task.double().reshape(2, -1).norm(dim=-1),
        source_task.double().reshape(2, -1).norm(dim=-1),
    ))
    material = bool(
        (circuit_rms >= 5e-4).all()
        and (task_norm >= 2.5e-4).all()
    )
    passes_without_controls = bool(
        material
        and math.isfinite(beta)
        and 0.25 <= beta <= 4.0
        and circuit[0]["cosine"] >= .90
        and circuit[0]["relative_residual"] <= .35
        and circuit[1]["cosine"] >= .80
        and circuit[1]["relative_residual"] <= .50
        and task[0]["cosine"] >= .70
        and task[0]["relative_residual"] <= .65
        and task[1]["cosine"] >= .70
        and task[1]["relative_residual"] <= .65
        and all(row[c]["cosine"] >= (.65 if h == 0 else .55)
                for h, row in enumerate(per_continuation) for c in range(4))
    )
    return {
        "beta": beta,
        "material": material,
        "circuit": circuit,
        "task": task,
        "per_continuation": per_continuation,
        "passes_without_controls": passes_without_controls,
    }


def planted_suite(seed: int = 20260903528) -> dict[str, Any]:
    """Deterministic positive, wrong-sign, and unrelated controls."""
    generator = torch.Generator().manual_seed(seed)
    circuit = torch.randn(2, 4, 32, generator=generator, dtype=torch.float64) * .01
    task = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .01
    target_circuit = 1.5 * circuit
    target_task = 1.5 * task
    positive = score_pair(target_circuit, circuit, target_task, task)
    wrong_sign = score_pair(target_circuit, -circuit, target_task, -task)
    unrelated_circuit = torch.randn(2, 4, 32, generator=generator, dtype=torch.float64) * .01
    unrelated_task = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .01
    unrelated = score_pair(target_circuit, unrelated_circuit, target_task, unrelated_task)
    interaction_arms = torch.stack((
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 5.0]),
        torch.tensor([7.0, 11.0]),
        torch.tensor([13.0, 19.0]),
    ))
    interaction = factorial_interaction(interaction_arms.unsqueeze(0)).squeeze(0)
    return {
        "positive": positive,
        "wrong_sign": wrong_sign,
        "unrelated": unrelated,
        "factorial_interaction": interaction.tolist(),
        "passes": bool(
            positive["passes_without_controls"]
            and abs(positive["beta"] - 1.5) < 1e-12
            and not wrong_sign["passes_without_controls"]
            and abs(wrong_sign["beta"] + 1.5) < 1e-12
            and not unrelated["passes_without_controls"]
            and torch.equal(interaction, torch.tensor([4.0, 5.0]))
        ),
    }
