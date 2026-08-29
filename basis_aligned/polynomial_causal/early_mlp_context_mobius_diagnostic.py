#!/usr/bin/env python3
"""Descriptive product-poset/Mobius analysis of the closed context-cross grid.

This is deliberately post-outcome and cannot turn the failed prospective rank-3/4
test into a pass.  It asks which sparse hierarchical grammar should be frozen on a
new mask family or adjacent physical cut.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import early_mlp_context_cross_v1 as registry
import early_mlp_context_cross_v1_lifecycle as lifecycle
import score_early_mlp_context_cross_v1 as sealed_score


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "early_mlp_context_mobius_diagnostic_results.json"
REPORT = HERE / "EARLY_MLP_CONTEXT_MOBIUS_DIAGNOSTIC.md"

PREFIX_SETS = tuple(frozenset(mask) for mask in registry.PREFIX_MASKS)
SUFFIX_SETS = tuple(frozenset(mask) for mask in registry.SUFFIX_MASKS)
INNER = tuple(range(1, 8))
STAGE_CELLS = {
    "discovery": registry.RANK3_DISCOVERY_CELLS,
    "validation": registry.RANK4_VALIDATION_CELLS,
    "heldout": registry.HELDOUT_CELLS,
}
PREFIX_NAMES = (
    "empty", "MLP0", "MLP0+MLP1", "MLP1", "MLP0+MLP1+MLP2",
    "MLP2", "MLP1+MLP2", "MLP0+MLP2",
)
SUFFIX_ATOM_NAMES = (
    "empty",
    "attention3",
    "MLP3",
    "local attention3-MLP3 synergy",
    "additional attention4-8 beyond attention3",
    "additional MLP3-8 / broad-block synergy",
    "additional attention9-17",
    "additional MLP9-17 / deep-block synergy",
)
K_VALUES = (1, 2, 4, 8, 12, 16, 24, 32)
BOOTSTRAP_DRAWS = 1_000


def zeta_matrix(sets: tuple[frozenset, ...]) -> np.ndarray:
    """Return Z[x,a]=1 when atom/subset a is contained in intervention x."""

    return np.asarray(
        [[float(atom <= intervention) for atom in sets] for intervention in sets],
        dtype=np.float64,
    )


def mobius_coefficients(
    values: np.ndarray,
    row_sets: tuple[frozenset, ...] = PREFIX_SETS,
    column_sets: tuple[frozenset, ...] = SUFFIX_SETS,
) -> np.ndarray:
    """Invert values = Z_row @ coefficients @ Z_column.T exactly."""

    values = np.asarray(values, dtype=np.float64)
    zr, zc = zeta_matrix(row_sets), zeta_matrix(column_sets)
    if values.shape != (len(row_sets), len(column_sets)):
        raise ValueError("value grid shape differs from the two posets")
    return np.linalg.solve(zr, np.linalg.solve(zc, values.T).T)


def reconstruct(
    coefficients: np.ndarray,
    row_sets: tuple[frozenset, ...] = PREFIX_SETS,
    column_sets: tuple[frozenset, ...] = SUFFIX_SETS,
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return zeta_matrix(row_sets) @ coefficients @ zeta_matrix(column_sets).T


def interaction(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    return cost - cost[:, [0]] - cost[[0], :] + cost[0, 0]


def design_matrix() -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    zr = zeta_matrix(PREFIX_SETS)[1:, 1:]
    zc = zeta_matrix(SUFFIX_SETS)[1:, 1:]
    coordinates = tuple((i, j) for i in INNER for j in INNER)
    return np.kron(zr, zc), coordinates


def omp(
    x: np.ndarray, y: np.ndarray, count: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Deterministic normalized-correlation OMP followed by exact least squares."""

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not 1 <= count <= x.shape[1]:
        raise ValueError("invalid OMP inputs")
    norms = np.linalg.norm(x, axis=0)
    available = norms > 0
    if int(np.sum(available)) < count:
        raise ValueError("OMP requests more terms than the training data identify")
    selected: list[int] = []
    residual = y.copy()
    coefficients = np.empty(0, dtype=np.float64)
    for _ in range(count):
        correlations = np.full(x.shape[1], -np.inf, dtype=np.float64)
        correlations[available] = np.abs(x[:, available].T @ residual) / norms[available]
        correlations[selected] = -np.inf
        chosen = int(np.argmax(correlations))
        selected.append(chosen)
        coefficients = np.linalg.lstsq(x[:, selected], y, rcond=None)[0]
        residual = y - x[:, selected] @ coefficients
    full = np.zeros(x.shape[1], dtype=np.float64)
    full[selected] = coefficients
    return tuple(selected), full


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = truth - prediction
    rmse = float(np.sqrt(np.mean(error**2)))
    rms = float(np.sqrt(np.mean(truth**2)))
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "rmse": rmse,
        "nre_to_zero_interaction": rmse / rms if rms else float("nan"),
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator else float("nan"),
    }


def leave_one_cell_out(
    x: np.ndarray, y: np.ndarray, *, omp_count: int | None = None,
    fixed_columns: Iterable[int] | None = None,
) -> dict[str, float]:
    if (omp_count is None) == (fixed_columns is None):
        raise ValueError("choose exactly one support rule")
    prediction = np.empty_like(y)
    fixed = None if fixed_columns is None else tuple(fixed_columns)
    for heldout in range(len(y)):
        train = np.arange(len(y)) != heldout
        if omp_count is not None:
            _, coefficients = omp(x[train], y[train], omp_count)
            prediction[heldout] = x[heldout] @ coefficients
        else:
            coefficients = np.linalg.lstsq(x[train][:, fixed], y[train], rcond=None)[0]
            prediction[heldout] = x[heldout, fixed] @ coefficients
    return metrics(y, prediction)


def _cost_grid(bundle) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    grid = np.empty((8, 8), dtype=np.float64)
    documents: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stage_name, cells in STAGE_CELLS.items():
        stage = getattr(bundle, stage_name)
        ce = stage.ce_sum.detach().cpu().numpy().astype(np.float64)
        tokens = stage.document_token_count.detach().cpu().numpy().astype(np.float64)
        documents[stage_name] = (ce, tokens)
        values = ce.sum(axis=0) / tokens.sum()
        for ordinal, cell in enumerate(cells):
            grid[cell] = values[ordinal]
    return grid, documents


def _bootstrap_interaction(
    documents: dict[str, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator,
) -> np.ndarray:
    first = next(iter(documents.values()))[0]
    indices = rng.integers(0, first.shape[0], size=first.shape[0])
    grid = np.empty((8, 8), dtype=np.float64)
    for stage_name, cells in STAGE_CELLS.items():
        ce, tokens = documents[stage_name]
        values = ce[indices].sum(axis=0) / tokens[indices].sum()
        for ordinal, cell in enumerate(cells):
            grid[cell] = values[ordinal]
    return interaction(grid)


def _coordinate_name(index: int, coordinates: tuple[tuple[int, int], ...]) -> str:
    i, j = coordinates[index]
    return f"{PREFIX_NAMES[i]} x {SUFFIX_ATOM_NAMES[j]}"


def analyze() -> dict:
    bundles, receipt = sealed_score.load_terminal_bundles(
        lifecycle.output_paths(), require_authoritative=True,
    )
    x, coordinates = design_matrix()
    roles: dict[str, dict] = {}
    role_vectors: dict[str, np.ndarray] = {}
    role_documents = {}
    for role, bundle in bundles.items():
        cost, documents = _cost_grid(bundle)
        delta = interaction(cost)
        coefficients = mobius_coefficients(delta)
        if not np.allclose(reconstruct(coefficients), delta, atol=1e-10, rtol=1e-10):
            raise RuntimeError("Mobius transform did not reconstruct the grid")
        y = delta[1:, 1:].reshape(-1)
        role_vectors[role] = y
        role_documents[role] = documents
        singular = np.linalg.svd(delta[1:, 1:], compute_uv=False)
        exact_mobius = coefficients[1:, 1:].reshape(-1)
        largest = np.argsort(-np.abs(exact_mobius))[:12]
        early_order_columns = {
            order: tuple(
                ordinal for ordinal, (i, _j) in enumerate(coordinates)
                if len(PREFIX_SETS[i]) <= order
            )
            for order in (1, 2)
        }
        roles[role] = {
            "interaction_rms": float(np.sqrt(np.mean(y**2))),
            "singular_values": singular.tolist(),
            "singular_energy_fraction": (
                np.cumsum(singular**2) / np.sum(singular**2)
            ).tolist(),
            "largest_exact_mobius_terms": [
                {
                    "name": _coordinate_name(int(index), coordinates),
                    "coefficient": float(exact_mobius[index]),
                }
                for index in largest
            ],
            "loo_omp": {
                str(k): leave_one_cell_out(x, y, omp_count=k) for k in K_VALUES
            },
            "loo_fixed_early_order": {
                str(order): leave_one_cell_out(
                    x, y, fixed_columns=columns,
                )
                for order, columns in early_order_columns.items()
            },
        }

    transfer: dict[str, dict] = {}
    role_names = tuple(roles)
    for source, target in ((role_names[0], role_names[1]), (role_names[1], role_names[0])):
        y_source, y_target = role_vectors[source], role_vectors[target]
        entries = {}
        for k in K_VALUES:
            support, source_coefficients = omp(x, y_source, k)
            target_refit = np.zeros(x.shape[1], dtype=np.float64)
            target_refit[list(support)] = np.linalg.lstsq(
                x[:, support], y_target, rcond=None,
            )[0]
            entries[str(k)] = {
                "support": [_coordinate_name(index, coordinates) for index in support],
                "direct_value_transfer": metrics(y_target, x @ source_coefficients),
                "target_refit_on_source_support": metrics(y_target, x @ target_refit),
            }
        transfer[f"{source}_to_{target}"] = entries

    bootstrap: dict[str, dict] = {}
    for role_index, role in enumerate(role_names):
        rng = np.random.default_rng(2026082901 + role_index)
        counts = {8: np.zeros(49, dtype=np.int64), 16: np.zeros(49, dtype=np.int64)}
        for _ in range(BOOTSTRAP_DRAWS):
            y = _bootstrap_interaction(role_documents[role], rng)[1:, 1:].reshape(-1)
            for k in counts:
                support, _ = omp(x, y, k)
                counts[k][list(support)] += 1
        bootstrap[role] = {
            str(k): [
                {
                    "name": _coordinate_name(int(index), coordinates),
                    "selection_frequency": float(counts[k][index] / BOOTSTRAP_DRAWS),
                }
                for index in np.argsort(-counts[k])[:k]
            ]
            for k in counts
        }

    correlation = float(np.corrcoef(role_vectors[role_names[0]], role_vectors[role_names[1]])[0, 1])
    direct_grid_transfer = {
        f"{source}_to_{target}": metrics(role_vectors[target], role_vectors[source])
        for source, target in ((role_names[0], role_names[1]), (role_names[1], role_names[0]))
    }
    return {
        "schema_version": 1,
        "status": "descriptive_post_outcome_not_prospective",
        "measurement_receipt_sha256": lifecycle.file_sha256(lifecycle.output_paths().receipt),
        "measurement_source_closure_sha256": receipt["source_closure_sha256"],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "roles": roles,
        "cross_role_interaction_correlation": correlation,
        "direct_full_grid_transfer": direct_grid_transfer,
        "sparse_support_transfer": transfer,
        "bootstrap_support_stability": bootstrap,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(result: dict) -> str:
    lines = [
        "# Early-MLP/context hierarchical Möbius diagnostic",
        "",
        "**Status: descriptive post-outcome analysis. This is not a new prospective pass.**",
        "",
        "The exact product-poset Möbius transform writes each measured interaction as",
        "a sum of contributions that appear for the first time at a particular early-MLP",
        "subset and suffix replacement set. The transform is exact; simplicity comes only",
        "from predicting cells after retaining a small, stable subset of its terms.",
        "",
        "The early side is a genuine three-factor Boolean lattice. The suffix side is",
        "only a registry of nested macro-replacements, not a physical factorial: for",
        "example, MLP-only layers 3--8 were never measured. Suffix Möbius coefficients",
        "therefore mix broad MLP main effects with attention-by-MLP synergy. They are",
        "useful macro-contrasts, not identified per-site mechanisms or tensor-program terms.",
        "The zeta basis is also non-orthogonal, so squared coefficient size is not",
        "Parseval energy. Simplicity below is judged by held-cell prediction instead.",
        "",
        "## Main findings",
        "",
        f"The two complete interaction grids have Pearson correlation "
        f"**{result['cross_role_interaction_correlation']:.4f}** across their 49 non-anchor cells.",
        "This measures transport across disjoint document populations before fitting any",
        "new values on the target role.",
        "",
        "### Singular spectrum (descriptive only)",
        "",
        "| role | rank-1 energy | rank-2 | rank-3 | rank-4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, entry in result["roles"].items():
        energy = entry["singular_energy_fraction"]
        lines.append(f"| {role} | {_fmt(energy[0])} | {_fmt(energy[1])} | {_fmt(energy[2])} | {_fmt(energy[3])} |")
    lines += [
        "",
        "High in-sample singular energy is not enough: the prospective fixed-pivot rank",
        "test already failed. The remaining question is whether a *sparse hierarchical*",
        "basis predicts an omitted intervention more reliably.",
        "",
        "### Leave-one-cell-out sparse Möbius prediction",
        "",
        "Each of the 49 non-anchor intervention cells is omitted in turn. Orthogonal",
        "matching pursuit selects a fixed number of Möbius terms using only the other 48",
        "cells. NRE is RMSE divided by the zero-interaction baseline; below 1 is useful.",
        "",
        "| role | terms | LOO NRE | LOO R2 |",
        "|---|---:|---:|---:|",
    ]
    for role, entry in result["roles"].items():
        for k in K_VALUES:
            metric = entry["loo_omp"][str(k)]
            lines.append(f"| {role} | {k} | {_fmt(metric['nre_to_zero_interaction'])} | {_fmt(metric['r2'])} |")
    lines += [
        "",
        "### Fixed early interaction order is not enough",
        "",
        "Keeping all suffix macro-contrasts but only singleton early-MLP terms uses 21",
        "coefficients; allowing early pairs uses 42. These hereditary models perform",
        "poorly despite their larger sizes:",
        "",
        "| role | maximum early order | terms | LOO NRE | LOO R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, entry in result["roles"].items():
        for order, count in ((1, 21), (2, 42)):
            metric = entry["loo_fixed_early_order"][str(order)]
            lines.append(
                f"| {role} | {order} | {count} | "
                f"{_fmt(metric['nre_to_zero_interaction'])} | {_fmt(metric['r2'])} |"
            )
    lines += [
        "",
        "The useful sparsity is structured across both early subsets and suffix",
        "macro-contrasts; it is not merely a low-degree polynomial in MLP0/1/2.",
        "",
        "### Cross-document support transport",
        "",
        "Support is selected on one role. `Direct` copies both that sparse grammar and its",
        "coefficients to the other role. `Refit` preserves only the grammar but allows the",
        "target role to re-estimate its coefficient values. This separates stable structure",
        "from stable numerical calibration.",
        "",
        "| direction | terms | direct NRE | refit NRE |",
        "|---|---:|---:|---:|",
    ]
    for direction, entries in result["sparse_support_transfer"].items():
        for k in K_VALUES:
            entry = entries[str(k)]
            lines.append(
                f"| {direction} | {k} | "
                f"{_fmt(entry['direct_value_transfer']['nre_to_zero_interaction'])} | "
                f"{_fmt(entry['target_refit_on_source_support']['nre_to_zero_interaction'])} |"
            )
    lines += [
        "",
        "The full source-role grid predicts the other role at NRE about 0.10, while the",
        "16-term direct transfers are about 0.19--0.20. These are strong same-corpus,",
        "disjoint-document transfers, but they are not a new corpus or semantic OOD test.",
        "",
        "### Bootstrap-stable eight-term candidates",
        "",
        "These terms are selected in at least 80% of 1,000 independent document",
        "bootstraps on both roles when OMP is limited to eight terms:",
        "",
        "| macro-contrast | skip7000 | skip11000 |",
        "|---|---:|---:|",
    ]
    frequency = {
        role: {
            item["name"]: item["selection_frequency"]
            for item in result["bootstrap_support_stability"][role]["8"]
        }
        for role in result["roles"]
    }
    common = sorted(set.intersection(*(set(values) for values in frequency.values())))
    for name in common:
        if min(frequency[role][name] for role in frequency) >= 0.8:
            lines.append(
                f"| {name} | {_fmt(frequency['skip7000'][name])} | "
                f"{_fmt(frequency['skip11000'][name])} |"
            )
    lines += [
        "",
        "These names inherit the suffix-aliasing warning above. An `additional MLP3-8",
        "/ broad-block synergy` coefficient cannot distinguish a broad MLP main effect",
        "from its interaction with the attention sites bundled in that mask.",
        "",
        "## Claim boundary and next test",
        "",
        "This analysis may nominate a sparse grammar, but every cell was already visible",
        "when the diagnostic was designed. A genuine result requires freezing the support",
        "rule and testing new suffix masks or an adjacent layer boundary. Token/logit-vector",
        "outcomes should accompany CE so scalar averaging cannot hide incompatible behavior.",
        "The cheapest de-aliasing test adds the missing MLP-only layers-3--8 suffix and",
        "crosses it with all eight early prefixes: eight new masks on each role. Together",
        "with the existing empty, attention-only, and all-sites columns, this completes",
        "the broad attention-by-MLP square. Its support rule and gates must be frozen",
        "before those new outcomes are opened.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    result = analyze()
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "cross_role_interaction_correlation": result["cross_role_interaction_correlation"],
        "results": str(RESULTS),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
