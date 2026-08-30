"""Test whether circuit-direction geometry predicts cross-circuit causal effects.

This is a read-only analysis of existing a8/a16/m16 artifacts.  It asks a narrow
question: does absolute cosine similarity between two fitted circuit directions
predict how strongly ablating one direction affects the other circuit?  It does not
reconstruct activation vectors or claim a tensor decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_matrix(nested: dict[str, dict[str, float]], names: list[str]) -> np.ndarray:
    return np.asarray([[nested[a][b] for b in names] for a in names], dtype=np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        raise ValueError("Spearman inputs must have equal size >= 2")
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_geometry(
    cosine: np.ndarray,
    causal: np.ndarray,
    *,
    seed: int = 0,
    permutation_draws: int = 20_000,
) -> dict[str, float | int]:
    """Compute off-diagonal and within-target geometry/causality agreement.

    The permutation test relabels the geometry matrix by the same row/column
    permutation, preserving its symmetry and spectrum while breaking its alignment to
    named causal effects.  It is a discovery p-value, not a protected confirmatory
    test.
    """
    cosine = np.asarray(cosine, dtype=np.float64)
    causal = np.asarray(causal, dtype=np.float64)
    if cosine.shape != causal.shape or cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise ValueError("cosine and causal matrices must be same-size square arrays")
    n = cosine.shape[0]
    mask = ~np.eye(n, dtype=bool)
    absolute_cosine = np.abs(cosine)
    observed = spearman(absolute_cosine[mask], causal[mask])

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutation_draws):
        permutation = rng.permutation(n)
        permuted = absolute_cosine[np.ix_(permutation, permutation)]
        statistic = spearman(permuted[mask], causal[mask])
        exceed += abs(statistic) >= abs(observed)

    within_target = []
    top1 = 0
    reciprocal_rank = []
    for target in range(n):
        sources = np.asarray([source for source in range(n) if source != target])
        geometry_scores = absolute_cosine[sources, target]
        causal_scores = causal[sources, target]
        within_target.append(spearman(geometry_scores, causal_scores))
        geometry_order = sources[np.argsort(-geometry_scores)]
        causal_best = int(sources[np.argmax(causal_scores)])
        top1 += int(int(geometry_order[0]) == causal_best)
        reciprocal_rank.append(1.0 / (1 + int(np.flatnonzero(geometry_order == causal_best)[0])))

    return {
        "n_circuits": n,
        "off_diagonal_pairs": int(mask.sum()),
        "off_diagonal_spearman": observed,
        "matrix_label_permutation_draws": permutation_draws,
        "matrix_label_two_sided_p": (exceed + 1) / (permutation_draws + 1),
        "mean_within_target_spearman": float(np.mean(within_target)),
        "top1_source_matches": top1,
        "top1_source_fraction": top1 / n,
        "mean_reciprocal_rank": float(np.mean(reciprocal_rank)),
    }


def _load_inputs(circuit_dir: Path) -> tuple[dict[str, object], dict[str, str]]:
    paths = {
        "a8_full": circuit_dir / "SUBSPACE.json",
        "a8_residual": circuit_dir / "RESIDUAL.json",
        "a16": circuit_dir / "A16.json",
        "m16": circuit_dir / "M16.json",
    }
    artifacts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    a8_full = artifacts["a8_full"]["groups"]["a8"]
    a8_residual = artifacts["a8_residual"]
    a16 = artifacts["a16"]
    m16 = artifacts["m16"]
    inputs = {
        "a8": {
            "names": a8_full["circuits"],
            "full_cosine": a8_full["cos"],
            "full_causal": a8_full["rank1_projection_concentration"],
            "residual_cosine": a8_residual["residual_cos"],
            "residual_causal": a8_residual["residual_rank1_concentration"],
            "shared_variance": a8_residual["shared_direction_variance_fraction"],
        },
        "a16": {
            "names": a16["circuits"],
            "full_cosine": a16["cos_full"],
            "full_causal": a16["concentration_full"],
            "residual_cosine": a16["cos_residual"],
            "residual_causal": a16["concentration_residual"],
            "shared_variance": a16["shared_variance_explained"],
        },
        "m16": {
            "names": m16["circuits"],
            "full_cosine": m16["cos_full"],
            "full_causal": m16["concentration_full"],
            "residual_cosine": m16["cos_residual"],
            "residual_causal": m16["concentration_residual"],
            "shared_variance": m16["shared_variance_explained"],
        },
    }
    return inputs, {name: _sha256(path) for name, path in paths.items()}


def build_receipt(circuit_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    inputs, parent_hashes = _load_inputs(circuit_dir)
    components: dict[str, object] = {}
    for component, entry in inputs.items():
        names = entry["names"]
        component_result = {"shared_variance": entry["shared_variance"]}
        for phase in ("full", "residual"):
            cosine = nested_matrix(entry[f"{phase}_cosine"], names)
            causal = nested_matrix(entry[f"{phase}_causal"], names)
            component_result[phase] = analyze_geometry(
                cosine,
                causal,
                seed={"a8": 11, "a16": 17, "m16": 23}[component]
                + (0 if phase == "full" else 100),
            )
        components[component] = component_result

    full_signs = [np.sign(components[name]["full"]["off_diagonal_spearman"]) for name in components]
    residual_max_abs = max(
        abs(components[name]["residual"]["off_diagonal_spearman"])
        for name in components
    )
    return {
        "schema": "component_geometry_causal_transport_v1",
        "claim_boundary": (
            "Retrospective CPU discovery analysis of already-opened a8/a16/m16 "
            "direction and rank-1 concentration summaries. No model, rows, protected "
            "outcomes, decomposition fit, circuit promotion, or confirmatory p-value."
        ),
        "question": (
            "Does absolute cosine geometry between fitted circuit directions transport "
            "as a predictor of cross-circuit causal damage?"
        ),
        "parent_sha256": parent_hashes,
        "components": components,
        "transport_summary": {
            "full_geometry_spearman_sign_agrees_all_components": bool(
                len(set(map(float, full_signs))) == 1
            ),
            "residual_max_abs_off_diagonal_spearman": float(residual_max_abs),
            "interpretation": (
                "Geometry is a proposal mechanism, not a transported causal or "
                "hierarchical simplicity metric. Any shared/private tensor fit must "
                "be selected on held-out causal response cells."
            ),
        },
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1] / "bilinear_quotient" / "circuits"
    parser.add_argument("--circuit-dir", type=Path, default=base)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "component_geometry_causal_transport_receipt.json",
    )
    args = parser.parse_args()
    receipt = build_receipt(args.circuit_dir.resolve())
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
