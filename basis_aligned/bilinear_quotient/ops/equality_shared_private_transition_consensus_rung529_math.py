"""CPU-only algebra and frozen scoring gates for rung 529.

Rung 529 asks whether several physically distinct equality actions contain one
shared post-MLP12 state change plus action-specific remainder.  These helpers
deliberately do not fit a basis or reduce rank.  They construct the registered
leave-one-action-out state, enumerate every single-donor comparison, and apply
the discovery gates before any GPU result is visible.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

import equality_distributed_finite_transition_quotient_rung528_math as r528


ACTIONS = ("N", "P", "Z7", "Z8")
CONTINUATIONS = r528.CONTINUATIONS
BETAS = {
    "N": 1.0,
    "P": 0.595594568993135,
    "Z7": 0.8070768390655048,
    "Z8": 0.7212548186259912,
}
WRONG_FOR_SOURCE = {"Z7": "W7", "Z8": "W8"}


def _validate_actions(values: Mapping[str, torch.Tensor], name: str) -> None:
    if tuple(values) != ACTIONS:
        raise ValueError(f"{name} keys must be in frozen order {ACTIONS}")
    shape = values[ACTIONS[0]].shape
    for action in ACTIONS:
        value = values[action]
        if value.shape != shape:
            raise ValueError(f"{name} tensors must share one shape")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name}[{action}] contains nonfinite values")


def aligned_states(
    deltas: Mapping[str, torch.Tensor],
    betas: Mapping[str, float] = BETAS,
) -> dict[str, torch.Tensor]:
    """Put each physical transition into native-action units: s_a=beta_a*delta_a."""
    _validate_actions(deltas, "deltas")
    if tuple(betas) != ACTIONS:
        raise ValueError(f"betas keys must be in frozen order {ACTIONS}")
    output = {}
    for action in ACTIONS:
        beta = float(betas[action])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("all frozen betas must be finite and positive")
        output[action] = deltas[action] * beta
    return output


def leave_one_out_decomposition(
    deltas: Mapping[str, torch.Tensor],
    betas: Mapping[str, float] = BETAS,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return registered consensus and exact private remainder for every target."""
    states = aligned_states(deltas, betas)
    output: dict[str, dict[str, torch.Tensor]] = {}
    for target in ACTIONS:
        donors = tuple(action for action in ACTIONS if action != target)
        consensus = torch.stack([states[action] for action in donors]).mean(0) / float(betas[target])
        private = deltas[target] - consensus
        output[target] = {
            "consensus": consensus,
            "private": private,
            "reconstruction": consensus + private,
        }
    return output


def single_donor_states(
    deltas: Mapping[str, torch.Tensor],
    betas: Mapping[str, float] = BETAS,
) -> dict[str, dict[str, torch.Tensor]]:
    """Enumerate every registered target<-donor physical state substitution."""
    aligned_states(deltas, betas)  # fail closed before constructing controls
    return {
        target: {
            donor: deltas[donor] * (float(betas[donor]) / float(betas[target]))
            for donor in ACTIONS if donor != target
        }
        for target in ACTIONS
    }


def wrong_sign_consensus_states(
    deltas: Mapping[str, torch.Tensor],
    wrong_deltas: Mapping[str, torch.Tensor],
    betas: Mapping[str, float] = BETAS,
) -> dict[str, dict[str, torch.Tensor]]:
    """Replace one eligible Z source by its wrong-sign control in each consensus.

    A target's own state is absent from its leave-one-out consensus, hence target
    Z7 has only the W8 control and target Z8 has only W7.
    """
    states = aligned_states(deltas, betas)
    if tuple(wrong_deltas) != ("W7", "W8"):
        raise ValueError("wrong_deltas keys must be in frozen order ('W7', 'W8')")
    for key, value in wrong_deltas.items():
        if value.shape != deltas["N"].shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"wrong_deltas[{key}] is malformed")

    output: dict[str, dict[str, torch.Tensor]] = {}
    for target in ACTIONS:
        donors = tuple(action for action in ACTIONS if action != target)
        target_controls = {}
        for source, wrong_name in WRONG_FOR_SOURCE.items():
            if source not in donors:
                continue
            terms = [
                float(betas[action]) * (wrong_deltas[wrong_name] if action == source else deltas[action])
                for action in donors
            ]
            target_controls[wrong_name] = torch.stack(terms).mean(0) / float(betas[target])
        output[target] = target_controls
    return output


def _validate_effect(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or tuple(value.shape[:2]) != (2, 4):
        raise ValueError(f"{name} must have shape [2,4,coordinates]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


def compare_effects(target: torch.Tensor, observed: torch.Tensor) -> list[dict[str, float]]:
    """Compare two effect tensors separately on the two registered halves."""
    _validate_effect(target, "target")
    _validate_effect(observed, "observed")
    if target.shape != observed.shape:
        raise ValueError("target and observed shapes differ")
    return [r528.relation_metrics(target[half], observed[half], 1.0) for half in range(2)]


def score_discovery_target(
    target_circuit: torch.Tensor,
    target_task: torch.Tensor,
    consensus_circuit: torch.Tensor,
    consensus_task: torch.Tensor,
    single_circuits: Mapping[str, torch.Tensor],
    wrong_circuits: Mapping[str, torch.Tensor],
    permutation_cosines: Sequence[float],
) -> dict[str, Any]:
    """Apply every response-level discovery gate for one frozen target.

    State reconstruction, edit liveness, and call reconciliation are instrument
    properties and remain the responsibility of the GPU runner.
    """
    for name, value in {
        "target_circuit": target_circuit,
        "target_task": target_task,
        "consensus_circuit": consensus_circuit,
        "consensus_task": consensus_task,
    }.items():
        _validate_effect(value, name)
    if target_circuit.shape != consensus_circuit.shape:
        raise ValueError("circuit effect shapes differ")
    if target_task.shape != consensus_task.shape:
        raise ValueError("task effect shapes differ")
    if len(single_circuits) != 3:
        raise ValueError("all three single donors are required")
    if not wrong_circuits:
        raise ValueError("at least one applicable wrong-sign control is required")
    if len(permutation_cosines) != 16 or not all(math.isfinite(float(x)) for x in permutation_cosines):
        raise ValueError("exactly 16 finite permutation cosines are required")

    circuit = compare_effects(target_circuit, consensus_circuit)
    task = compare_effects(target_task, consensus_task)
    continuations = [
        [r528.relation_metrics(target_circuit[h, c], consensus_circuit[h, c], 1.0)
         for c in range(4)]
        for h in range(2)
    ]
    single = {}
    for donor, value in single_circuits.items():
        _validate_effect(value, f"single_circuits[{donor}]")
        if value.shape != target_circuit.shape:
            raise ValueError("single-donor circuit shape differs")
        single[donor] = compare_effects(target_circuit, value)
    wrong = {}
    for control, value in wrong_circuits.items():
        _validate_effect(value, f"wrong_circuits[{control}]")
        if value.shape != target_circuit.shape:
            raise ValueError("wrong-sign circuit shape differs")
        wrong[control] = compare_effects(target_circuit, value)

    target_rms = target_circuit.double().square().mean(-1).sqrt()
    consensus_rms = consensus_circuit.double().square().mean(-1).sqrt()
    target_task_norm = target_task.double().reshape(2, -1).norm(dim=-1)
    consensus_task_norm = consensus_task.double().reshape(2, -1).norm(dim=-1)
    live = bool(
        (target_rms >= 5e-4).all()
        and (consensus_rms >= 5e-4).all()
        and (target_task_norm >= 2.5e-4).all()
        and (consensus_task_norm >= 2.5e-4).all()
    )
    best_single_error = [min(rows[h]["relative_residual"] for rows in single.values()) for h in range(2)]
    max_wrong_cosine = max(rows[0]["cosine"] for rows in wrong.values())
    permutation_q95 = float(torch.quantile(
        torch.tensor(tuple(float(x) for x in permutation_cosines), dtype=torch.float64), .95))
    gate_values = {
        "live": live,
        "circuit_d0": circuit[0]["cosine"] >= .90 and circuit[0]["relative_residual"] <= .35,
        "circuit_d1": circuit[1]["cosine"] >= .80 and circuit[1]["relative_residual"] <= .50,
        "task_both": all(row["cosine"] >= .70 and row["relative_residual"] <= .65 for row in task),
        "continuations": all(
            continuations[h][c]["cosine"] >= (.65 if h == 0 else .55)
            for h in range(2) for c in range(4)),
        "d0_beats_every_single_by_005": all(
            circuit[0]["relative_residual"] <= rows[0]["relative_residual"] - .05
            for rows in single.values()),
        "d1_within_002_of_best_single": circuit[1]["relative_residual"] <= best_single_error[1] + .02,
        "d0_control_margin": circuit[0]["cosine"] >= max(max_wrong_cosine, permutation_q95) + .10,
    }
    return {
        "circuit": circuit,
        "task": task,
        "continuations": continuations,
        "single": single,
        "wrong": wrong,
        "permutation_q95": permutation_q95,
        "best_single_error": best_single_error,
        "gates": gate_values,
        "passes_response_gates": bool(all(gate_values.values())),
    }


def planted_suite(seed: int = 20260903529) -> dict[str, Any]:
    """Positive shared/private construction plus singleton and sign controls."""
    generator = torch.Generator().manual_seed(seed)
    common = torch.randn(2, 4, 48, generator=generator, dtype=torch.float64) * .012
    task_common = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .012
    # Independent action-specific responses make each singleton noisier than
    # the mean of the other three, while retaining exact state construction.
    private = {
        action: torch.randn(2, 4, 48, generator=generator, dtype=torch.float64) * .0036
        for action in ACTIONS
    }
    private_task = {
        action: torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .0036
        for action in ACTIONS
    }
    circuit = {action: common + private[action] for action in ACTIONS}
    task = {action: task_common + private_task[action] for action in ACTIONS}

    target = "Z7"
    donors = [action for action in ACTIONS if action != target]
    consensus_circuit = torch.stack([circuit[action] for action in donors]).mean(0)
    consensus_task = torch.stack([task[action] for action in donors]).mean(0)
    singles = {action: circuit[action] for action in donors}
    wrong = {"W8": -consensus_circuit}
    permuted = []
    for seed_offset in range(16):
        order = torch.randperm(common.shape[-1], generator=torch.Generator().manual_seed(seed + seed_offset + 1))
        permuted.append(r528.relation_metrics(circuit[target][0], consensus_circuit[0, :, order], 1.0)["cosine"])
    score = score_discovery_target(
        circuit[target], task[target], consensus_circuit, consensus_task,
        singles, wrong, permuted)

    state_deltas = {
        action: torch.randn(7, 1152, generator=generator, dtype=torch.float64)
        for action in ACTIONS
    }
    decomposition = leave_one_out_decomposition(state_deltas)
    exact_error = max(
        float((decomposition[action]["reconstruction"] - state_deltas[action]).abs().max())
        for action in ACTIONS)
    return {
        "score": score,
        "maximum_reconstruction_error": exact_error,
        "passes": bool(score["passes_response_gates"] and exact_error <= 1e-12),
    }
