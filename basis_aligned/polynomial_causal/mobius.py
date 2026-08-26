"""Small, dependency-light utilities for causal intervention polynomials."""

from itertools import combinations

import torch


def subsets(n):
    """Return bit-mask subsets in increasing cardinality then numeric order."""
    return sorted(range(1 << n), key=lambda mask: (mask.bit_count(), mask))


def mobius_coefficients(values, n):
    """Boolean Mobius coefficients from values indexed by integer bit mask."""
    missing = set(range(1 << n)) - set(values)
    if missing:
        raise ValueError(f"missing intervention masks: {sorted(missing)}")
    coeffs = {}
    for mask in subsets(n):
        total = 0.0
        sub = mask
        while True:
            parity = (mask.bit_count() - sub.bit_count()) & 1
            total += (-1.0 if parity else 1.0) * values[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        coeffs[mask] = total
    return coeffs


def evaluate_multilinear(coeffs, alpha):
    """Evaluate a multilinear extension at an iterable of scalar alpha values."""
    result = 0.0
    for mask, coeff in coeffs.items():
        term = coeff
        for i, value in enumerate(alpha):
            if mask & (1 << i):
                term = term * value
        result = result + term
    return result


def design_row(alpha, max_degree):
    """Feature row [1, a_i, a_i*a_j, ...] up to max_degree."""
    alpha = list(alpha)
    row = []
    terms = []
    for degree in range(max_degree + 1):
        for inds in combinations(range(len(alpha)), degree):
            value = 1.0
            for i in inds:
                value *= alpha[i]
            row.append(value)
            terms.append(inds)
    return row, terms


def fit_effect_model(observations, max_degree):
    """Least-squares multilinear model.

    observations is an iterable of (alpha_tuple, scalar_value). Returns a dict with
    coefficient terms and a predict callable's data representation.
    """
    observations = list(observations)
    if not observations:
        raise ValueError("at least one observation is required")
    rows = []
    targets = []
    terms = None
    for alpha, target in observations:
        row, current_terms = design_row(alpha, max_degree)
        if terms is None:
            terms = current_terms
        elif terms != current_terms:
            raise ValueError("inconsistent intervention dimension")
        rows.append(row)
        targets.append(target)
    x = torch.tensor(rows, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.float64)
    coeff = torch.linalg.lstsq(x, y).solution
    return {"terms": terms, "coefficients": coeff}


def predict_effect(model, alpha):
    row, terms = design_row(alpha, max(len(term) for term in model["terms"]))
    if terms != model["terms"]:
        raise ValueError("intervention dimension does not match fitted model")
    x = torch.tensor(row, dtype=model["coefficients"].dtype)
    return float(x @ model["coefficients"])


def normalized_error(predicted, actual, floor=0.05):
    return abs(predicted - actual) / max(abs(actual), floor)

