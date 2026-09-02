#!/usr/bin/env python3
"""RUNG509 -- coupled Left/Right/source assignments with shared causal responses.

This implementation is intentionally incomplete until the frozen finite-response
collector is wired.  The CPU-exercised core below fixes the dictionary parameterization,
loss, seeds, matching, and eligibility calculations before any rung509 model outcome.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mlp10_exact_source_pair_causal_split_rung507 as parent


PREREG = POLY / "MLP10_COUPLED_CAUSAL_DICTIONARY_RUNG509_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "mlp10_exact_source_family_factorial_rung508_results.json"
PARENT_BUNDLE = ROOT / "mlp10_exact_source_family_factorial_rung508_bundle.pt"
OUT = ROOT / "mlp10_coupled_causal_dictionary_rung509_results.json"
BUNDLE = ROOT / "mlp10_coupled_causal_dictionary_rung509_bundle.pt"

ATOMS = 8
RESPONSES = 34
SEEDS = (5090, 5091, 5092)
STEPS = 2000
LEARNING_RATE = .02
WEIGHT_DECAY = 1e-4
ENTROPY_WEIGHT = .01
DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)

SOURCE_PAIRS = tuple(parent.SOURCE_PAIRS)
PAIR_LEFT = torch.tensor([left for left, _right in SOURCE_PAIRS], dtype=torch.long)
PAIR_RIGHT = torch.tensor([right for _left, right in SOURCE_PAIRS], dtype=torch.long)


def assignment(logit_left: torch.Tensor, logit_right: torch.Tensor) -> torch.Tensor:
    """Return [action, pair, atom] simplex weights, symmetric in the two inputs."""
    if logit_left.shape != logit_right.shape or logit_left.ndim != 3:
        raise ValueError("source logits must both have shape [action, atom, source]")
    if logit_left.shape[1:] != (ATOMS, len(parent.NAMED_SOURCES)):
        raise ValueError("source-logit dimensions changed")
    score = (
        logit_left[:, :, PAIR_LEFT]
        + logit_right[:, :, PAIR_RIGHT]
        + logit_left[:, :, PAIR_RIGHT]
        + logit_right[:, :, PAIR_LEFT]
    ).permute(0, 2, 1)
    return score.softmax(-1)


def coupled_prediction(gates: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    """Predict [action,pair,response] from gates and shared atom responses."""
    if gates.shape != (len(parent.SOURCES), len(SOURCE_PAIRS), ATOMS):
        raise ValueError("assignment dimensions changed")
    if responses.shape != (ATOMS, RESPONSES):
        raise ValueError("response dimensions changed")
    return torch.einsum("apk,kc->apc", gates, responses)


def standardized_loss(prediction: torch.Tensor, target: torch.Tensor,
                      gates: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape[-1] != RESPONSES:
        raise ValueError("finite-response tensor dimensions changed")
    if scale.shape != (RESPONSES,) or bool((scale <= 0).any()):
        raise ValueError("response scale must be positive")
    fit = ((prediction - target) / scale).square().mean()
    entropy = -(gates.clamp_min(1e-12) * gates.clamp_min(1e-12).log()).sum(-1).mean()
    return fit + ENTROPY_WEIGHT * entropy


def best_permutation(left: torch.Tensor, right: torch.Tensor) -> tuple[int, ...]:
    """Exact maximum-cosine atom matching for the fixed eight-atom budget."""
    if left.shape != right.shape or left.shape != (ATOMS, RESPONSES):
        raise ValueError("atom response dimensions changed")
    left = torch.nn.functional.normalize(left.double(), dim=1)
    right = torch.nn.functional.normalize(right.double(), dim=1)
    similarity = left @ right.T
    best_score = float("-inf")
    best = None
    for permutation in itertools.permutations(range(ATOMS)):
        score = sum(float(similarity[index, permutation[index]]) for index in range(ATOMS))
        if score > best_score:
            best_score, best = score, permutation
    assert best is not None
    return tuple(best)


def fit_dictionary(target: torch.Tensor, seed: int) -> dict[str, torch.Tensor | float | int]:
    """Fit one fixed-seed CPU dictionary; target is finite, never a gradient."""
    if target.shape != (len(parent.SOURCES), len(SOURCE_PAIRS), RESPONSES):
        raise ValueError("target must have shape [4,253,34]")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains nonfinite values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left = torch.nn.Parameter(.01 * torch.randn(
        len(parent.SOURCES), ATOMS, len(parent.NAMED_SOURCES), generator=generator))
    right = torch.nn.Parameter(.01 * torch.randn(
        len(parent.SOURCES), ATOMS, len(parent.NAMED_SOURCES), generator=generator))
    response = torch.nn.Parameter(.01 * torch.randn(ATOMS, RESPONSES, generator=generator))
    scale = target.square().mean((0, 1)).sqrt().clamp_min(1e-8)
    optimizer = torch.optim.Adam((left, right, response), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    for _step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        gates = assignment(left, right)
        prediction = coupled_prediction(gates, response)
        loss = standardized_loss(prediction, target, gates, scale)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        gates = assignment(left, right)
        prediction = coupled_prediction(gates, response)
        loss = standardized_loss(prediction, target, gates, scale)
    return {
        "seed": seed,
        "left_logits": left.detach(),
        "right_logits": right.detach(),
        "responses": response.detach(),
        "assignments": gates.detach(),
        "prediction": prediction.detach(),
        "scale": scale.detach(),
        "loss": float(loss),
    }


def dry_run() -> None:
    torch.manual_seed(509)
    left = torch.randn(len(parent.SOURCES), ATOMS, len(parent.NAMED_SOURCES))
    right = torch.randn_like(left)
    gates = assignment(left, right)
    assert gates.shape == (4, 253, 8)
    assert torch.allclose(gates.sum(-1), torch.ones(4, 253), atol=1e-6)
    response = torch.randn(ATOMS, RESPONSES)
    target = coupled_prediction(gates, response)
    assert target.shape == (4, 253, 34)
    assert best_permutation(response, response) == tuple(range(ATOMS))
    print("rung509 dry-run: coupled assignment, prediction, and exact matching pass")


def main() -> None:
    if "--dry-run" in sys.argv:
        dry_run()
        return
    raise RuntimeError(
        "rung509 finite collector not yet wired; fail closed before model/GPU execution")


if __name__ == "__main__":
    main()
