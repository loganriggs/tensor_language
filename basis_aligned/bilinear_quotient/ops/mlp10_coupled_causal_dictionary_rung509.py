#!/usr/bin/env python3
"""RUNG509 -- coupled Left/Right assignments with shared finite causal responses."""

# BQGATE: EXPERIMENT
# pred_a: exact live response/assignment/intervention instrument
# pred_b: two to eight restart- and half-stable coupled atoms
# pred_c: held-out response forecast plus at least two physical confirmations
# pred_d: at least one discovery-frozen pair rule predicts confirmation
# pred_e: a confirmed shared atom changes majority exact source pairs across actions

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp10_exact_source_pair_causal_split_rung507 as parent


PREREG = POLY / "MLP10_COUPLED_CAUSAL_DICTIONARY_RUNG509_PREREGISTRATION.md"
ADDENDUM = POLY / "MLP10_COUPLED_CAUSAL_DICTIONARY_RUNG509_PREFLIGHT_ADDENDUM.md"
IDENTIFIABILITY_REPAIR = (
    POLY / "MLP10_COUPLED_CAUSAL_DICTIONARY_RUNG509_IDENTIFIABILITY_REPAIR.md")
PARENT_SOURCE = ROOT / "ops/mlp10_exact_source_family_factorial_rung508.py"
PARENT_RESULT = ROOT / "mlp10_exact_source_family_factorial_rung508_results.json"
PARENT_BUNDLE = ROOT / "mlp10_exact_source_family_factorial_rung508_bundle.pt"
OUT = ROOT / "mlp10_coupled_causal_dictionary_rung509_results.json"
BUNDLE = ROOT / "mlp10_coupled_causal_dictionary_rung509_bundle.pt"
IDENTIFIABILITY_OUT = (
    ROOT / "mlp10_coupled_causal_dictionary_rung509_identifiability_results.json")
HASHES = {
    PREREG: "aa65e25c218951112d4599e5b5869ee6b8f84f89d93b044dc080b9dc3c9c5b58",
    ADDENDUM: "278ec89ca580e3f2681c71e7f13651c586838d0dd54d60157e130bcffdd2eaab",
    IDENTIFIABILITY_REPAIR: "381988395edd4d54c1d08ba99bef336ed0ca708fc48497dc479887e0d647f5bf",
    PARENT_SOURCE: "9715eec0d74dfba1b6931430dc9a31d2fa0b9dd017e1f063ba304ac83d24b7ce",
    PARENT_RESULT: "05060565f25a5b59a233f5b336ee9882e330ea3e39f8d7f6b27e715aab5825ba",
    PARENT_BUNDLE: "45b4a2245a8b3d740014ff0b8e3d575766b50f2f88a253d9235b8fd0e5cbd23d",
}

ATOMS = 8
RESPONSES = 34
SEEDS = (5090, 5091, 5092)
STEPS = 2000
LEARNING_RATE = .02
WEIGHT_DECAY = 1e-4
ENTROPY_WEIGHT = .01
ARCHETYPE_ENTROPY_WEIGHT = .01
DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
ATOM_NAMES = tuple(f"ATOM{i}" for i in range(ATOMS))
SYNTHETIC_SEEDS = (5090, 5091, 5092, 15090, 15091, 15092)

SOURCE_PAIRS = tuple(parent.SOURCE_PAIRS)
PAIR_LEFT = torch.tensor([left for left, _right in SOURCE_PAIRS], dtype=torch.long)
PAIR_RIGHT = torch.tensor([right for _left, right in SOURCE_PAIRS], dtype=torch.long)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Predict [action,pair,response] from assignments and shared responses."""
    if gates.shape != (len(parent.SOURCES), len(SOURCE_PAIRS), ATOMS):
        raise ValueError("assignment dimensions changed")
    if responses.shape != (ATOMS, RESPONSES):
        raise ValueError("response dimensions changed")
    return torch.einsum("apk,kc->apc", gates, responses)


def standardized_loss(prediction: torch.Tensor, target: torch.Tensor,
                      gates: torch.Tensor, scale: torch.Tensor,
                      archetype_weights: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape[-1] != RESPONSES:
        raise ValueError("finite-response tensor dimensions changed")
    if scale.shape != (RESPONSES,) or bool((scale <= 0).any()):
        raise ValueError("response scale must be positive")
    fit = ((prediction - target) / scale).square().mean()
    entropy = -(gates.clamp_min(1e-12) * gates.clamp_min(1e-12).log()).sum(-1).mean()
    archetype_entropy = torch.zeros((), dtype=fit.dtype, device=fit.device)
    if archetype_weights is not None:
        if archetype_weights.shape != (ATOMS, len(parent.SOURCES) * len(SOURCE_PAIRS)):
            raise ValueError("archetype weights have the wrong dimensions")
        archetype_entropy = -(
            archetype_weights.clamp_min(1e-12)
            * archetype_weights.clamp_min(1e-12).log()).sum(-1).mean()
    return fit + ENTROPY_WEIGHT * entropy + ARCHETYPE_ENTROPY_WEIGHT * archetype_entropy


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


def fit_dictionary(target: torch.Tensor, seed: int) -> dict[str, object]:
    """Fit one fixed-seed CPU dictionary; target contains finite effects only."""
    target = target.float().cpu()
    if target.shape != (len(parent.SOURCES), len(SOURCE_PAIRS), RESPONSES):
        raise ValueError("target must have shape [4,253,34]")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains nonfinite values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left = torch.nn.Parameter(.01 * torch.randn(
        len(parent.SOURCES), ATOMS, len(parent.NAMED_SOURCES), generator=generator))
    right = torch.nn.Parameter(.01 * torch.randn(
        len(parent.SOURCES), ATOMS, len(parent.NAMED_SOURCES), generator=generator))
    archetype_logits = torch.nn.Parameter(.01 * torch.randn(
        ATOMS, len(parent.SOURCES) * len(SOURCE_PAIRS), generator=generator))
    scale = target.square().mean((0, 1)).sqrt().clamp_min(1e-8)
    optimizer = torch.optim.Adam((left, right, archetype_logits), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    flat_target = target.view(-1, RESPONSES)
    for _step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        gates = assignment(left, right)
        archetype_weights = archetype_logits.softmax(-1)
        responses = archetype_weights @ flat_target
        prediction = coupled_prediction(gates, responses)
        loss = standardized_loss(
            prediction, target, gates, scale, archetype_weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        gates = assignment(left, right)
        archetype_weights = archetype_logits.softmax(-1)
        responses = archetype_weights @ flat_target
        prediction = coupled_prediction(gates, responses)
        loss = standardized_loss(
            prediction, target, gates, scale, archetype_weights)
    return {
        "seed": seed, "left_logits": left.detach(), "right_logits": right.detach(),
        "archetype_logits": archetype_logits.detach(),
        "archetype_weights": archetype_weights.detach(),
        "anchor_indices": archetype_weights.argmax(-1).detach(),
        "anchor_weights": archetype_weights.max(-1).values.detach(),
        "responses": responses.detach(), "assignments": gates.detach(),
        "prediction": prediction.detach(), "scale": scale.detach(), "loss": float(loss),
    }


def planted_separable_case() -> dict[str, object]:
    """Build the frozen factorized toy with one distinct near-pure row per atom."""
    true_left = torch.zeros(len(parent.SOURCES), ATOMS, len(parent.NAMED_SOURCES))
    true_right = torch.zeros_like(true_left)
    for atom in range(ATOMS):
        action = atom % len(parent.SOURCES)
        true_left[action, atom, atom] = 2.0
        true_right[action, atom, atom] = 2.0
    true_assignments = assignment(true_left, true_right)
    raw = torch.randn(
        RESPONSES, ATOMS, generator=torch.Generator(device="cpu").manual_seed(9))
    true_responses = torch.linalg.qr(raw).Q.T.contiguous()
    target = coupled_prediction(true_assignments, true_responses)
    flat_target = target.reshape(-1, RESPONSES)
    anchors, purities = [], []
    for atom in range(ATOMS):
        similarities = torch.nn.functional.cosine_similarity(
            flat_target, true_responses[atom].unsqueeze(0), dim=1)
        anchor = int(similarities.argmax())
        anchors.append(anchor)
        purities.append(float(true_assignments.reshape(-1, ATOMS)[anchor, atom]))
    if len(set(anchors)) != ATOMS or min(purities) < .99:
        raise RuntimeError("synthetic case lost its eight distinct near-pure anchors")
    return {
        "target": target, "true_assignments": true_assignments,
        "true_responses": true_responses, "anchor_indices": anchors,
        "anchor_assignment_weights": purities,
    }


def synthetic_identifiability_audit() -> dict[str, object]:
    """Run the frozen pre-model truth-recovery gate without choosing a restart."""
    case = planted_separable_case()
    fits = [fit_dictionary(case["target"], seed) for seed in SYNTHETIC_SEEDS]
    rows = []
    all_response_cosines, all_assignment_cosines = [], []
    all_anchor_matches, all_anchor_weights = [], []
    expected_anchors = case["anchor_indices"]
    for fit in fits:
        permutation = best_permutation(case["true_responses"], fit["responses"])
        order = torch.tensor(permutation, dtype=torch.long)
        responses = fit["responses"][order]
        assignments = fit["assignments"][:, :, order]
        anchors = [int(value) for value in fit["anchor_indices"][order]]
        anchor_weights = [float(value) for value in fit["anchor_weights"][order]]
        response_cosines = [
            _cosine(case["true_responses"][atom], responses[atom])
            for atom in range(ATOMS)]
        assignment_cosines = [
            _cosine(case["true_assignments"][:, :, atom], assignments[:, :, atom])
            for atom in range(ATOMS)]
        anchor_matches = [
            observed == expected for observed, expected in zip(anchors, expected_anchors)]
        all_response_cosines.extend(response_cosines)
        all_assignment_cosines.extend(assignment_cosines)
        all_anchor_matches.extend(anchor_matches)
        all_anchor_weights.extend(anchor_weights)
        rows.append({
            "seed": fit["seed"], "loss": fit["loss"],
            "permutation_to_truth": list(permutation),
            "response_cosines": response_cosines,
            "assignment_cosines": assignment_cosines,
            "anchor_indices": anchors, "anchor_weights": anchor_weights,
            "anchor_identity_matches": anchor_matches,
        })
    thresholds = {
        "minimum_response_cosine": .90,
        "minimum_assignment_cosine": .80,
        "minimum_anchor_weight": .90,
        "all_anchor_identities_must_match": True,
    }
    holds = bool(
        min(all_response_cosines) >= thresholds["minimum_response_cosine"]
        and min(all_assignment_cosines) >= thresholds["minimum_assignment_cosine"]
        and min(all_anchor_weights) >= thresholds["minimum_anchor_weight"]
        and all(all_anchor_matches))
    return {
        "status": "instrument_passed" if holds else "instrument_failed",
        "rung": 509, "model_loaded": False, "model_outcomes_opened": False,
        "cpu_fits": len(fits), "fit_steps_each": STEPS,
        "synthetic_seeds": list(SYNTHETIC_SEEDS),
        "expected_anchor_indices": expected_anchors,
        "expected_anchor_assignment_weights": case["anchor_assignment_weights"],
        "thresholds": thresholds, "fits": rows,
        "summary": {
            "minimum_response_cosine": min(all_response_cosines),
            "minimum_assignment_cosine": min(all_assignment_cosines),
            "minimum_anchor_weight": min(all_anchor_weights),
            "anchor_identity_matches": sum(all_anchor_matches),
            "anchor_identity_total": len(all_anchor_matches),
        },
        "holds": holds,
        "registered_failure_route": (
            "downstream_predictive_state_quotient_without_latent_dictionary"),
    }


def require_identifiable_instrument(
        audit: dict[str, object] | None = None) -> dict[str, object]:
    if audit is None:
        audit = synthetic_identifiability_audit()
    if not audit["holds"]:
        raise RuntimeError(
            "rung509 model execution blocked by the frozen synthetic "
            "identifiability gate; take the registered no-latent quotient route")
    return audit


def align_fits(fits: list[dict]) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align every fit to half0/seed5090, then average without choosing a seed."""
    if len(fits) != 6:
        raise ValueError("exactly six discovery fits are required")
    canonical = fits[0]["responses"]
    aligned = []
    for fit in fits:
        permutation = best_permutation(canonical, fit["responses"])
        order = torch.tensor(permutation, dtype=torch.long)
        row = dict(fit)
        row["responses"] = fit["responses"][order]
        row["assignments"] = fit["assignments"][:, :, order]
        row["archetype_weights"] = fit["archetype_weights"][order]
        row["anchor_indices"] = fit["anchor_indices"][order]
        row["anchor_weights"] = fit["anchor_weights"][order]
        row["permutation_to_canonical"] = permutation
        aligned.append(row)
    mean_gates = torch.stack([fit["assignments"] for fit in aligned]).mean(0)
    mean_responses = torch.stack([fit["responses"] for fit in aligned]).mean(0)
    mean_scale = torch.stack([fit["scale"] for fit in aligned]).mean(0)
    return aligned, mean_gates, mean_responses, mean_scale


def _cosine(left, right) -> float:
    return parent.state_parent.cosine(
        torch.as_tensor(left, dtype=torch.float64).flatten(),
        torch.as_tensor(right, dtype=torch.float64).flatten())


def stable_atoms(aligned: list[dict]) -> tuple[list[int], dict[str, dict]]:
    checks: dict[str, dict] = {}
    prelim: list[int] = []
    half_runs = (aligned[:3], aligned[3:])
    mean_gate = torch.stack([fit["assignments"] for fit in aligned]).mean(0)
    for atom in range(ATOMS):
        restart_response, restart_assignment = [], []
        for runs in half_runs:
            for left, right in itertools.combinations(runs, 2):
                restart_response.append(_cosine(
                    left["responses"][atom], right["responses"][atom]))
                restart_assignment.append(_cosine(
                    left["assignments"][:, :, atom], right["assignments"][:, :, atom]))
        half_response = [_cosine(aligned[index]["responses"][atom],
                                 aligned[index + 3]["responses"][atom]) for index in range(3)]
        half_assignment = [_cosine(aligned[index]["assignments"][:, :, atom],
                                   aligned[index + 3]["assignments"][:, :, atom])
                           for index in range(3)]
        support, support_ok = {}, True
        for action_index, source in enumerate(parent.SOURCES):
            weights = mean_gate[action_index, :, atom]
            majority = torch.nonzero(weights >= .50).flatten().tolist()
            max_mass_share = float(weights.max() / weights.sum().clamp_min(1e-30))
            row_ok = len(majority) >= 2 and max_mass_share <= .80
            support[source] = {
                "majority_terms": [parent.PAIR_NAMES[index] for index in majority],
                "majority_count": len(majority), "maximum_mass_share": max_mass_share,
                "holds": row_ok,
            }
            support_ok &= row_ok
        anchors = [int(fit["anchor_indices"][atom]) for fit in aligned]
        anchor_weights = [float(fit["anchor_weights"][atom]) for fit in aligned]
        anchor_ok = len(set(anchors)) == 1 and min(anchor_weights) >= .90
        holds = bool(
            min(restart_response) >= .80 and min(restart_assignment) >= .75
            and min(half_response) >= .70 and min(half_assignment) >= .65
            and support_ok and anchor_ok)
        checks[ATOM_NAMES[atom]] = {
            "restart_response_cosines": restart_response,
            "restart_assignment_cosines": restart_assignment,
            "half_response_cosines": half_response,
            "half_assignment_cosines": half_assignment,
            "anchor_indices": anchors, "anchor_weights": anchor_weights,
            "anchor_holds": anchor_ok, "support": support, "preliminary_holds": holds,
        }
        if holds:
            prelim.append(atom)
    repeated_anchors = {
        int(aligned[0]["anchor_indices"][atom]) for atom in prelim
    }
    distinct_anchor_ok = len(repeated_anchors) == len(prelim)
    eligible = []
    for atom in prelim:
        diversity = {
            ATOM_NAMES[other]: _cosine(aligned[0]["responses"][atom],
                                       aligned[0]["responses"][other])
            for other in prelim if other != atom
        }
        diverse = (not diversity or max(diversity.values()) <= .90) and distinct_anchor_ok
        checks[ATOM_NAMES[atom]]["eligible_response_cosines"] = diversity
        checks[ATOM_NAMES[atom]]["diversity_holds"] = diverse
        checks[ATOM_NAMES[atom]]["all_preliminary_anchors_distinct"] = distinct_anchor_ok
        checks[ATOM_NAMES[atom]]["holds"] = bool(diverse)
        if diverse:
            eligible.append(atom)
    for atom in set(range(ATOMS)) - set(prelim):
        checks[ATOM_NAMES[atom]]["diversity_holds"] = False
        checks[ATOM_NAMES[atom]]["holds"] = False
    return eligible, checks


def _empty_diagnostics() -> dict:
    row = parent._empty_diagnostics()
    row.update({
        "patches": 0, "patches_expected": 0, "patches_exact": False,
        "exact_term_partition_relative_squared": 0.0,
        "assignment_partition_relative_squared": 0.0,
    })
    return row


def _update_diagnostics(total: dict, row: dict) -> None:
    parent._update_diagnostics(total, row)


def _term_outputs(mlp, factors) -> list[torch.Tensor]:
    return [parent._pair_output(mlp, factors, index) for index in range(len(SOURCE_PAIRS))]


def _weighted_hidden(factors: dict, weights: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(len(parent.NAMED_SOURCES), len(parent.NAMED_SOURCES),
                         device=factors["left"].device, dtype=factors["left"].dtype)
    weights = weights.to(device=matrix.device, dtype=matrix.dtype)
    pair_left, pair_right = PAIR_LEFT.to(matrix.device), PAIR_RIGHT.to(matrix.device)
    matrix[pair_left, pair_right] = weights
    off = pair_left != pair_right
    matrix[pair_right[off], pair_left[off]] = weights[off]
    return torch.einsum("btsh,su,btuh->bth", factors["left"], matrix, factors["right"])


def _circuit_fingerprint(collection: dict, arm: str, source: str, window: str) -> torch.Tensor:
    source_index = parent.SOURCES.index(source)
    arm_index = collection["arms"].index(arm)
    intact_index = collection["arms"].index("intact")
    if window == "pooled":
        target = collection["circuit_sums"][source_index, arm_index].sum(0)
        intact = collection["circuit_sums"][source_index, intact_index].sum(0)
        counts = collection["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        target = collection["circuit_sums"][source_index, arm_index, half]
        intact = collection["circuit_sums"][source_index, intact_index, half]
        counts = collection["circuit_counts"][half]
    effects = (target - intact) / counts.clamp_min(1)
    return effects[0] - effects[1]


def response_tensor(collection: dict, window: str) -> torch.Tensor:
    """Return [source,253,34] finite task-plus-circuit responses."""
    if len(collection["circuit_tags"]) != 30:
        raise ValueError("rung509 requires exactly 30 frozen circuit coordinates")
    rows = []
    for source in parent.SOURCES:
        terms = []
        for term in parent.PAIR_NAMES:
            task = parent.finite_vector(collection, term, collection, source, window)
            circuit = _circuit_fingerprint(collection, term, source, window)
            terms.append(torch.cat((task.double(), circuit.double())))
        rows.append(torch.stack(terms))
    result = torch.stack(rows)
    if result.shape != (4, 253, 34) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("finite response tensor changed")
    return result


@torch.no_grad()
def collect_exact(model, rows, task_masks, circuit_masks, circuit_tags, scales, bounds):
    lo, hi, _split = bounds
    documents = hi - lo
    arms = ("intact",) + tuple(parent.PAIR_NAMES)
    task = torch.zeros(4, len(arms), documents, len(parent.TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(parent.TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(4, len(arms), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[parent.TARGET].mlp
    for start in range(lo, hi, parent.BATCH):
        stop = start + parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in parent.TASK_CELLS}
        direct_logits, _, direct_diag, _ = parent._forward(model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        _update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = parent._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        _update_diagnostics(diagnostics, absent_diag)
        base_task[local:local + parent.BATCH] = parent._task_sums(
            parent._nll(absent_logits, batch_rows).detach().cpu()[None], masks)[0]
        counts[local:local + parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in parent.TASK_CELLS], -1)
        nll_rows = []
        absent_terms = _term_outputs(mlp, absent["factors"])
        absent_hidden_sum = parent._sum_unordered_pair_hidden(absent["factors"])
        diagnostics["exact_term_partition_relative_squared"] = max(
            diagnostics["exact_term_partition_relative_squared"],
            parent._relative_squared(absent_hidden_sum,
                                     absent["factors"]["left"].sum(2)
                                     * absent["factors"]["right"].sum(2)))
        for source in parent.SOURCES:
            logits, current, current_diag, _ = parent._forward(
                model, tokens, scales, action=source, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            _update_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                difference = logits.detach().float() - direct_logits.float()
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"], float(difference.abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    float(difference.square().sum())
                    / max(float(direct_logits.float().square().sum()), 1e-30))
            current_terms = _term_outputs(mlp, current["factors"])
            current_hidden_sum = parent._sum_unordered_pair_hidden(current["factors"])
            diagnostics["exact_term_partition_relative_squared"] = max(
                diagnostics["exact_term_partition_relative_squared"],
                parent._relative_squared(current_hidden_sum,
                                         current["factors"]["left"].sum(2)
                                         * current["factors"]["right"].sum(2)))
            source_nll = [parent._nll(logits, batch_rows).detach().cpu()]
            for term_index in range(len(SOURCE_PAIRS)):
                delta = current_terms[term_index] - absent_terms[term_index]
                replacement = current["deployed_write"] - delta.to(current["deployed_write"].dtype)
                patched_logits, _captures, patch_diag, patch_audit = parent.score_parent.run_forward(
                    model, tokens, action=source, scales=scales,
                    patch_writes={"m10": replacement})
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                source_nll.append(parent._nll(patched_logits, batch_rows).detach().cpu())
            nll_rows.extend(source_nll)
        nll_stack = torch.stack(nll_rows).view(4, len(arms), parent.BATCH, parent.TOKENS)
        task[:, :, local:local + parent.BATCH] = parent._task_sums(
            nll_stack.view(-1, parent.BATCH, parent.TOKENS), masks).view(
                4, len(arms), parent.BATCH, len(parent.TASK_CELLS))
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(
            nll_stack.view(4 * len(arms), -1).double(), matrix.T,
        ).view(4, len(arms), 2, 2, len(circuit_tags))
    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches, "analytical": batches * (1 + 4 * len(arms))}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["patches_expected"] = batches * 4 * (len(arms) - 1)
    diagnostics["patches_exact"] = diagnostics["patches"] == diagnostics["patches_expected"]
    return {
        "bounds": bounds, "arms": arms, "task": task, "task_counts": counts,
        "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "diagnostics": diagnostics,
    }


@torch.no_grad()
def collect_weighted(model, rows, task_masks, scales, bounds, gates, pair_atoms=()):
    lo, hi, _split = bounds
    documents = hi - lo
    specs = tuple((atom,) for atom in range(ATOMS)) if not pair_atoms else tuple(pair_atoms)
    arms = ATOM_NAMES if not pair_atoms else tuple(
        f"{ATOM_NAMES[left]}+{ATOM_NAMES[right]}" for left, right in specs)
    if gates.shape != (4, 253, 8):
        raise ValueError("mean assignment tensor changed")
    stored_arms = arms if pair_atoms else ("intact",) + arms
    task = torch.zeros(4, len(stored_arms), documents,
                       len(parent.TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(parent.TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[parent.TARGET].mlp
    for start in range(lo, hi, parent.BATCH):
        stop = start + parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in parent.TASK_CELLS}
        absent_logits, absent, absent_diag, _ = parent._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        _update_diagnostics(diagnostics, absent_diag)
        base_task[local:local + parent.BATCH] = parent._task_sums(
            parent._nll(absent_logits, batch_rows).detach().cpu()[None], masks)[0]
        counts[local:local + parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in parent.TASK_CELLS], -1)
        nll_rows = []
        for source_index, source in enumerate(parent.SOURCES):
            logits, current, current_diag, _ = parent._forward(
                model, tokens, scales, action=source, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            _update_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            atom_outputs = []
            for atom in range(ATOMS):
                hidden = _weighted_hidden(current["factors"], gates[source_index, :, atom])
                hidden -= _weighted_hidden(absent["factors"], gates[source_index, :, atom])
                atom_outputs.append(parent._linear(hidden, mlp.Down.weight.float()))
            named_delta = current["factors"]["semantic_output"] \
                - absent["factors"]["semantic_output"]
            diagnostics["assignment_partition_relative_squared"] = max(
                diagnostics["assignment_partition_relative_squared"],
                parent._relative_squared(torch.stack(atom_outputs).sum(0), named_delta))
            source_nll = [] if pair_atoms else [parent._nll(logits, batch_rows).detach().cpu()]
            for spec in specs:
                delta = sum((atom_outputs[atom] for atom in spec),
                            torch.zeros_like(atom_outputs[0]))
                replacement = current["deployed_write"] - delta.to(current["deployed_write"].dtype)
                patched_logits, _captures, patch_diag, patch_audit = parent.score_parent.run_forward(
                    model, tokens, action=source, scales=scales,
                    patch_writes={"m10": replacement})
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                source_nll.append(parent._nll(patched_logits, batch_rows).detach().cpu())
            nll_rows.extend(source_nll)
        nll_stack = torch.stack(nll_rows).view(4, len(stored_arms), parent.BATCH, parent.TOKENS)
        task[:, :, local:local + parent.BATCH] = parent._task_sums(
            nll_stack.view(-1, parent.BATCH, parent.TOKENS), masks).view(
                4, len(stored_arms), parent.BATCH, len(parent.TASK_CELLS))
    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": 0, "analytical": batches * (1 + 4 * (1 + len(specs)))}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["patches_expected"] = batches * 4 * len(specs)
    diagnostics["patches_exact"] = diagnostics["patches"] == diagnostics["patches_expected"]
    return {"bounds": bounds, "arms": stored_arms, "task": task,
            "task_counts": counts, "base_task": base_task,
            "source_task": task[:, 0] if not pair_atoms else None,
            "diagnostics": diagnostics}


def _instrument(collection: dict, *, exact=False) -> bool:
    d = collection["diagnostics"]
    ok = bool(
        d["calls_exact"] and d["patches_exact"] and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= parent.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= parent.DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and math.isfinite(d["score_delta_predeployment_relative_squared"])
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0)
    if exact:
        return bool(ok and d["exact_term_partition_relative_squared"] <= 1e-10
                    and d["native_replay_logit_max_abs"] == 0.0
                    and d["native_replay_relative_squared"] <= 1e-12)
    return bool(ok and d["assignment_partition_relative_squared"] <= 1e-8)


def _atom_checks(collection: dict, atoms: list[int]) -> tuple[list[int], dict]:
    passing, checks = [], {}
    for atom in atoms:
        name = ATOM_NAMES[atom]
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in parent.SOURCES:
            vector = parent.finite_vector(collection, name, collection, source)
            repeat = parent._comparison(
                parent.finite_vector(collection, name, collection, source, "half0"),
                parent.finite_vector(collection, name, collection, source, "half1"))
            all_copy, off_target = parent.finite_all_off(collection, name, collection, source)
            source_holds = bool(
                float(torch.linalg.vector_norm(vector)) >= .00025
                and repeat["cosine"] >= .50 and repeat["norm_ratio"] <= 3
                and abs(all_copy) >= .00025 and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": vector.tolist(), "repeat": repeat,
                "all_copy_effect_nat": all_copy, "off_target_effect_nat": off_target,
                "holds": source_holds,
            }
            holds &= source_holds
        native = parent.finite_vector(collection, name, collection, "N")
        for source in parent.SOURCES[1:]:
            metric = parent._comparison(
                native, parent.finite_vector(collection, name, collection, source))
            metric["holds"] = bool(metric["cosine"] >= .70 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[name] = row
        if holds:
            passing.append(atom)
    return passing, checks


def _confirmation_checks(discovery: dict, confirmation: dict,
                         atoms: list[int]) -> tuple[list[int], dict]:
    confirmed, checks = [], {}
    for atom in atoms:
        name = ATOM_NAMES[atom]
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in parent.SOURCES:
            before = parent.finite_vector(discovery, name, discovery, source)
            vector = parent.finite_vector(confirmation, name, confirmation, source)
            transfer = parent._comparison(before, vector)
            repeat = parent._comparison(
                parent.finite_vector(confirmation, name, confirmation, source, "half0"),
                parent.finite_vector(confirmation, name, confirmation, source, "half1"))
            all_copy, off_target = parent.finite_all_off(
                confirmation, name, confirmation, source)
            source_holds = bool(
                float(torch.linalg.vector_norm(vector)) >= .00025
                and transfer["cosine"] >= .60 and transfer["norm_ratio"] <= 3
                and repeat["cosine"] >= .50 and repeat["norm_ratio"] <= 3
                and abs(all_copy) >= .00025 and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": vector.tolist(), "discovery_transfer": transfer,
                "repeat": repeat, "all_copy_effect_nat": all_copy,
                "off_target_effect_nat": off_target, "holds": source_holds,
            }
            holds &= source_holds
        native = parent.finite_vector(confirmation, name, confirmation, "N")
        for source in parent.SOURCES[1:]:
            metric = parent._comparison(
                native, parent.finite_vector(confirmation, name, confirmation, source))
            metric["holds"] = bool(metric["cosine"] >= .65 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[name] = row
        if holds:
            confirmed.append(atom)
    return confirmed, checks


def heldout_forecast(mean_gates, mean_responses, scale, discovery_target,
                     confirmation_target) -> dict:
    prediction = coupled_prediction(mean_gates, mean_responses)
    error = float((((prediction - confirmation_target) / scale).square().mean()))
    baseline_response = discovery_target.mean((0, 1))
    baseline = baseline_response.view(1, 1, -1).expand_as(confirmation_target)
    baseline_error = float((((baseline - confirmation_target) / scale).square().mean()))
    generator = torch.Generator().manual_seed(509)
    permuted = torch.empty_like(mean_gates)
    for action in range(4):
        order = torch.randperm(253, generator=generator)
        permuted[action] = mean_gates[action, order]
    control = coupled_prediction(permuted, mean_responses)
    control_error = float((((control - confirmation_target) / scale).square().mean()))
    holds = bool(error <= .75 * baseline_error and error <= .80 * control_error)
    return {
        "standardized_mse": error, "one_response_baseline_mse": baseline_error,
        "permuted_assignment_control_mse": control_error,
        "ratio_to_baseline": error / max(baseline_error, 1e-30),
        "ratio_to_permutation": error / max(control_error, 1e-30), "holds": holds,
    }


def _source_changing(atom: int, gates: torch.Tensor) -> tuple[bool, dict]:
    sets = {}
    for source_index, source in enumerate(parent.SOURCES):
        indices = torch.nonzero(gates[source_index, :, atom] >= .50).flatten().tolist()
        sets[source] = [parent.PAIR_NAMES[index] for index in indices]
    changes = any(set(sets[left]) != set(sets[right])
                  for left, right in itertools.combinations(parent.SOURCES, 2))
    return changes, sets


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_finite_source_family_instrument") is True
        and result.get("pred_b_sparse_finite_family_split") is False
        and result.get("strong_null") is True
        and result.get("next_step")
        == "coupled_left_right_output_dictionary_with_finite_prediction"
    ):
        raise RuntimeError("rung508 route changed")
    rows, task_masks, circuit_masks, scales, _discovery_tags, validation_tags, metadata = \
        parent.validate_inputs()
    if len(SOURCE_PAIRS) != 253 or len(validation_tags) != 30:
        raise RuntimeError("exact-term or circuit response vocabulary changed")
    return rows, task_masks, circuit_masks, scales, list(validation_tags), {
        **metadata, "rung508_result_sha256": sha256(PARENT_RESULT),
        "rung508_bundle_sha256": sha256(PARENT_BUNDLE),
        "documents": {"discovery": list(DISCOVERY), "unused": [748, 752],
                      "confirmation": list(CONFIRMATION)},
        "response_coordinates": {"task": list(parent.GRAD_CELLS[:4]),
                                 "circuits": list(validation_tags)},
    }


def _bundle_collection(collection):
    return {key: value for key, value in collection.items() if key != "diagnostics"}


def _gpu_smoke():
    require_identifiable_instrument()
    rows, task_masks, circuit_masks, scales, circuit_tags, _metadata = validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    bounds = (500, 504, 502)
    exact = collect_exact(
        model, rows, task_masks, circuit_masks, circuit_tags, scales, bounds)
    gates = torch.full((4, 253, 8), 1 / 8)
    atoms = collect_weighted(model, rows, task_masks, scales, bounds, gates)
    pairs = collect_weighted(
        model, rows, task_masks, scales, bounds, gates, pair_atoms=((0, 1),))
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "exact": _instrument(exact, exact=True), "atoms": _instrument(atoms),
        "pairs": _instrument(pairs), "all_253_term_patches": exact["diagnostics"]["patches"] == 1012,
        "all_8_atom_patches": atoms["diagnostics"]["patches"] == 32,
        "one_pair_patch_per_source": pairs["diagnostics"]["patches"] == 4,
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 509,
        "scientific_outcomes_retained": False, "checks": checks,
        "exact_diagnostics": exact["diagnostics"],
        "atom_diagnostics": atoms["diagnostics"], "pair_diagnostics": pairs["diagnostics"],
        "full_forwards": sum(sum(row["diagnostics"]["calls"].values())
                             for row in (exact, atoms, pairs)), "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"rung509 CUDA smoke failed: "
                           f"{sorted(name for name, value in checks.items() if not value)}")


def dry_run() -> None:
    torch.manual_seed(509)
    left = torch.randn(4, 8, 22)
    right = torch.randn_like(left)
    gates = assignment(left, right)
    assert gates.shape == (4, 253, 8)
    assert torch.allclose(gates.sum(-1), torch.ones(4, 253), atol=1e-6)
    response = torch.randn(8, 34)
    target = coupled_prediction(gates, response)
    assert target.shape == (4, 253, 34)
    assert best_permutation(response, response) == tuple(range(8))
    assert 2 * (63116 + 2294) + 2 * 62 * (1 + 4 * (1 + math.comb(8, 2))) == 145328
    print(json.dumps({
        "status": "dry_run_passed", "rung": 509, "model_loaded": False,
        "outcomes_opened": False, "atoms": ATOMS, "response_coordinates": RESPONSES,
        "maximum_conditional_forwards": 145328,
    }, indent=2, sort_keys=True))


def run_synthetic_audit() -> None:
    started = time.time()
    result = synthetic_identifiability_audit()
    result["runtime_s"] = time.time() - started
    result["source_hashes"] = {
        str(path): digest for path, digest in HASHES.items()}
    result["implementation_sha256"] = sha256(Path(__file__))
    dump(result, IDENTIFIABILITY_OUT)
    print(json.dumps(result, indent=2, sort_keys=True))


def _score_pairs(single_discovery, single_confirmation, pair_discovery,
                 pair_confirmation, confirmed):
    rules, checks, predictable = {}, {}, []
    for left, right in itertools.combinations(confirmed, 2):
        left_name, right_name = ATOM_NAMES[left], ATOM_NAMES[right]
        name = f"{left_name}+{right_name}"
        rule = parent.fit_composition(
            single_discovery, pair_discovery, left_name, right_name)
        rules[name] = rule
        if not rule["identified"]:
            continue
        row, holds = {"sources": {}}, True
        for source in parent.SOURCES:
            left_v = parent.finite_vector(single_confirmation, left_name,
                                          single_confirmation, source)
            right_v = parent.finite_vector(single_confirmation, right_name,
                                           single_confirmation, source)
            joint_v = parent.finite_vector(pair_confirmation, name,
                                           single_confirmation, source)
            predicted = parent.predict_composition(rule, left_v, right_v)
            cos = parent.state_parent.cosine(predicted, joint_v)
            residual = parent._relative_residual(joint_v, predicted)
            half_cos = []
            for window in ("half0", "half1"):
                lv = parent.finite_vector(single_confirmation, left_name,
                                          single_confirmation, source, window)
                rv = parent.finite_vector(single_confirmation, right_name,
                                          single_confirmation, source, window)
                jv = parent.finite_vector(pair_confirmation, name,
                                          single_confirmation, source, window)
                half_cos.append(parent.state_parent.cosine(
                    parent.predict_composition(rule, lv, rv), jv))
            all_copy, off_target = parent.finite_all_off(
                pair_confirmation, name, single_confirmation, source)
            source_ok = bool(cos >= .70 and residual <= .65 and min(half_cos) > 0
                             and abs(all_copy) >= .00025
                             and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "prediction_cosine": cos, "prediction_relative_residual": residual,
                "half_prediction_cosines": half_cos,
                "all_copy_effect_nat": all_copy, "off_target_effect_nat": off_target,
                "holds": source_ok,
            }
            holds &= source_ok
        row["holds"] = bool(holds)
        checks[name] = row
        if holds:
            predictable.append(name)
    return rules, checks, predictable


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        dry_run()
        return
    if "--synthetic-identifiability-audit" in sys.argv:
        run_synthetic_audit()
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        _gpu_smoke()
        return
    require_identifiable_instrument()
    started = time.time()
    rows, task_masks, circuit_masks, scales, circuit_tags, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung509 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["exact_discovery"] = collect_exact(
        model, rows, task_masks, circuit_masks, circuit_tags, scales, DISCOVERY)
    calibration = {"discovery": parent._calibration(
        collections["exact_discovery"]["base_task"],
        collections["exact_discovery"]["source_task"],
        collections["exact_discovery"]["task_counts"], DISCOVERY)}
    calibration_ok = {"discovery": parent.state_parent.calibration_holds(
        calibration["discovery"])}
    discovery_targets = {
        window: response_tensor(collections["exact_discovery"], window)
        for window in ("half0", "half1", "pooled")}
    fits = [fit_dictionary(discovery_targets[window], seed)
            for window in ("half0", "half1") for seed in SEEDS]
    aligned, mean_gates, mean_responses, mean_scale = align_fits(fits)
    eligible, stability_checks = stable_atoms(aligned)
    identifying = 2 <= len(eligible) <= 8

    discovery_checks, discovery_pass = {}, []
    confirmation_checks, confirmed, forecast = {}, [], {}
    composition_rules, composition_checks, predictable_pairs = {}, {}, []
    source_maps = {}
    if calibration_ok["discovery"] and identifying:
        collections["atom_discovery"] = collect_weighted(
            model, rows, task_masks, scales, DISCOVERY, mean_gates)
        discovery_pass, discovery_checks = _atom_checks(
            collections["atom_discovery"], eligible)
    if len(discovery_pass) >= 2:
        collections["exact_confirmation"] = collect_exact(
            model, rows, task_masks, circuit_masks, circuit_tags, scales, CONFIRMATION)
        exact_confirmation = collections["exact_confirmation"]
        calibration["confirmation"] = parent._calibration(
            exact_confirmation["base_task"], exact_confirmation["source_task"],
            exact_confirmation["task_counts"], CONFIRMATION)
        calibration_ok["confirmation"] = parent.state_parent.calibration_holds(
            calibration["confirmation"])
        forecast = heldout_forecast(
            mean_gates, mean_responses, mean_scale,
            discovery_targets["pooled"], response_tensor(exact_confirmation, "pooled"))
        collections["atom_confirmation"] = collect_weighted(
            model, rows, task_masks, scales, CONFIRMATION, mean_gates)
        confirmed, confirmation_checks = _confirmation_checks(
            collections["atom_discovery"], collections["atom_confirmation"], discovery_pass)
    pred_c_pre = bool(calibration_ok.get("confirmation", False)
                      and forecast.get("holds", False) and 2 <= len(confirmed) <= 8)
    if pred_c_pre:
        pairs = tuple(itertools.combinations(confirmed, 2))
        collections["pair_discovery"] = collect_weighted(
            model, rows, task_masks, scales, DISCOVERY, mean_gates, pair_atoms=pairs)
        collections["pair_confirmation"] = collect_weighted(
            model, rows, task_masks, scales, CONFIRMATION, mean_gates, pair_atoms=pairs)
        composition_rules, composition_checks, predictable_pairs = _score_pairs(
            collections["atom_discovery"], collections["atom_confirmation"],
            collections["pair_discovery"], collections["pair_confirmation"], confirmed)
    for atom in confirmed:
        changes, maps = _source_changing(atom, mean_gates)
        source_maps[ATOM_NAMES[atom]] = {"changes": changes, "majority_terms": maps}

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and all(_instrument(collection, exact=name.startswith("exact_"))
                for name, collection in collections.items()))
    pred_b = bool(pred_a and calibration_ok["discovery"] and identifying)
    pred_c = bool(pred_b and pred_c_pre)
    pred_d = bool(pred_c and predictable_pairs)
    pred_e = bool(pred_c and any(row["changes"] for row in source_maps.values()))
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)
    if not pred_a:
        next_step = "repair_coupled_response_instrument_only"
    elif not pred_b:
        next_step = "downstream_predictive_state_quotient_without_latent_dictionary"
    elif not pred_c:
        next_step = "close_coupled_dictionary_grouping_form"
    elif not pred_d:
        next_step = "model_higher_order_suffix_state_for_identified_atoms"
    elif not pred_e:
        next_step = "audit_score_specific_paths_without_shared_realization_claim"
    else:
        next_step = "validate_confirmed_atoms_on_ood_code_and_build_subprogram"

    bundle_payload = {
        "schema": "rung509_coupled_finite_causal_dictionary_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "fits": aligned, "mean_assignments": mean_gates,
        "mean_responses": mean_responses, "mean_scale": mean_scale,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 509,
        "claim_level": "coupled_dictionary_screen_until_all_causal_gates_pass",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "score_sources": list(parent.SOURCES),
        "named_sources": list(parent.NAMED_SOURCES), "exact_terms": list(parent.PAIR_NAMES),
        "calibration": calibration, "calibration_holds": calibration_ok,
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "analysis": {
            "fit_losses": [{"seed": fit["seed"], "loss": fit["loss"],
                            "permutation_to_canonical": fit["permutation_to_canonical"]}
                           for fit in aligned],
            "stable_atoms": [ATOM_NAMES[index] for index in eligible],
            "stability_checks": stability_checks,
            "discovery_physical_atoms": [ATOM_NAMES[index] for index in discovery_pass],
            "discovery_physical_checks": discovery_checks,
            "heldout_exact_term_forecast": forecast,
            "confirmed_atoms": [ATOM_NAMES[index] for index in confirmed],
            "confirmation_checks": confirmation_checks,
            "composition_rules": composition_rules,
            "composition_checks": composition_checks,
            "predictable_pairs": predictable_pairs,
            "source_realization_maps": source_maps,
        },
        'pred_a_exact_live_response_assignment_intervention_instrument': pred_a,
        'pred_b_two_to_eight_stable_coupled_atoms': pred_b,
        'pred_c_heldout_forecast_and_two_physical_confirmations': pred_c,
        'pred_d_pair_composition_predicts_confirmation': pred_d,
        'pred_e_shared_atom_changes_source_realization': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values()),
            "backwards": 0, "cpu_fits": 6, "eligible_atoms": len(eligible),
            "discovery_physical_atoms": len(discovery_pass),
            "confirmed_atoms": len(confirmed),
            "maximum_conditional_forwards": 145328,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 509, "pred_a": pred_a, "pred_b": pred_b,
        "pred_c": pred_c, "pred_d": pred_d, "pred_e": pred_e,
        "strong_null": strong_null,
        "stable_atoms": result["analysis"]["stable_atoms"],
        "discovery_physical_atoms": result["analysis"]["discovery_physical_atoms"],
        "confirmed_atoms": result["analysis"]["confirmed_atoms"],
        "predictable_pairs": predictable_pairs,
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
