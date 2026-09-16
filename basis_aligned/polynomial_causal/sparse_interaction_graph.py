"""Reusable exact sparse interaction graphs on Boolean intervention lattices.

The numerical values may be Python scalars, NumPy arrays, or Torch tensors.
Only discovery-time selection and reporting require NumPy; exact decomposition
and composition preserve the caller's array/tensor type and precision.

Sign convention: ``corners[S]`` is the measured outcome with ports in ``S``
edited.  ``dividends`` are ordinary Möbius coefficients of that outcome.
``damage_atoms`` negates nonempty dividends, so their full sum predicts
``corners[0] - corners[all_ports]``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations


def _validate_mask(mask: int, n_ports: int) -> None:
    if not isinstance(mask, int) or mask < 0 or mask >= (1 << n_ports):
        raise ValueError(f"invalid mask {mask!r} for {n_ports} ports")


def _shape(value) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    return tuple(shape)


def submasks(mask: int) -> tuple[int, ...]:
    """Return every submask in ascending order, including zero and ``mask``."""
    if not isinstance(mask, int) or mask < 0:
        raise ValueError("mask must be a nonnegative integer")
    values = []
    current = mask
    while True:
        values.append(current)
        if current == 0:
            break
        current = (current - 1) & mask
    return tuple(sorted(values))


def candidate_masks(n_ports: int, maximum_order: int,
                    minimum_order: int = 1) -> tuple[int, ...]:
    """Enumerate masks by order and then integer mask, excluding the baseline."""
    if n_ports < 1 or not 1 <= minimum_order <= maximum_order <= n_ports:
        raise ValueError("invalid port count or interaction-order interval")
    result = []
    for order in range(minimum_order, maximum_order + 1):
        for indices in combinations(range(n_ports), order):
            result.append(sum(1 << index for index in indices))
    return tuple(sorted(result, key=lambda mask: (mask.bit_count(), mask)))


def required_corners(masks: Sequence[int], n_ports: int) -> tuple[int, ...]:
    """Return the minimal corner union needed to identify ``masks`` exactly."""
    needed = {0}
    for mask in masks:
        _validate_mask(mask, n_ports)
        needed.update(submasks(mask))
    return tuple(sorted(needed))


def selected_dividends(corners: Mapping[int, object], masks: Sequence[int],
                       n_ports: int) -> dict[int, object]:
    """Compute selected Möbius coefficients by direct inclusion-exclusion."""
    masks = tuple(masks)
    needed = required_corners(masks, n_ports)
    missing = tuple(mask for mask in needed if mask not in corners)
    if missing:
        raise ValueError(f"missing corner masks: {list(missing)}")
    shapes = {_shape(corners[mask]) for mask in needed}
    if len(shapes) != 1:
        raise ValueError("all required corners must have matching shapes")
    result = {}
    for mask in masks:
        terms = submasks(mask)
        value = None
        order = mask.bit_count()
        for subset in terms:
            signed = corners[subset] if (order - subset.bit_count()) % 2 == 0 else -corners[subset]
            value = signed if value is None else value + signed
        result[mask] = value
    return result


def full_dividends(corners: Mapping[int, object], n_ports: int) -> dict[int, object]:
    """Compute the complete exact Möbius transform, including the baseline."""
    if n_ports < 1:
        raise ValueError("n_ports must be positive")
    return selected_dividends(corners, tuple(range(1 << n_ports)), n_ports)


def damage_atoms(corners: Mapping[int, object], masks: Sequence[int],
                 n_ports: int) -> dict[int, object]:
    """Return signed damage atoms (negative nonempty outcome dividends)."""
    if any(mask == 0 for mask in masks):
        raise ValueError("damage atoms cannot include the baseline mask")
    return {mask: -value for mask, value in selected_dividends(corners, masks, n_ports).items()}


def reconstruct(dividends: Mapping[int, object], mask: int):
    """Reconstruct one corner from a complete-enough dividend mapping."""
    missing = [subset for subset in submasks(mask) if subset not in dividends]
    if missing:
        raise ValueError(f"missing dividend masks: {missing}")
    values = [dividends[subset] for subset in submasks(mask)]
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


def compose(atoms: Mapping[int, object], masks: Sequence[int] | None = None):
    """Sum frozen graph atoms without fitting coefficients."""
    masks = tuple(atoms) if masks is None else tuple(masks)
    if not masks:
        raise ValueError("at least one graph atom is required")
    missing = [mask for mask in masks if mask not in atoms]
    if missing:
        raise ValueError(f"missing graph atoms: {missing}")
    shapes = {_shape(atoms[mask]) for mask in masks}
    if len(shapes) != 1:
        raise ValueError("all graph atoms must have matching shapes")
    result = atoms[masks[0]]
    for mask in masks[1:]:
        result = result + atoms[mask]
    return result


def validate_panels(panels: Mapping[str, Sequence[int]], discovery: str = "discovery") -> None:
    """Reject empty, duplicated, or discovery-overlapping evaluation panels."""
    if discovery not in panels:
        raise ValueError(f"missing discovery panel {discovery!r}")
    normalized = {}
    for name, indices in panels.items():
        values = tuple(int(index) for index in indices)
        if not values:
            raise ValueError(f"empty panel {name!r}")
        if len(values) != len(set(values)) or min(values) < 0:
            raise ValueError(f"invalid indices in panel {name!r}")
        normalized[name] = set(values)
    discovery_indices = normalized[discovery]
    for name, indices in normalized.items():
        if name != discovery and discovery_indices & indices:
            raise ValueError(f"panel {name!r} overlaps discovery")


def greedy_select(target, atoms: Mapping[int, object], discovery_indices: Sequence[int],
                  steps: int, candidates: Sequence[int] | None = None):
    """Select unscaled atoms on discovery rows only, with deterministic ties."""
    import numpy as np

    candidates = tuple(sorted(atoms)) if candidates is None else tuple(candidates)
    if not 0 < steps <= len(candidates) or len(candidates) != len(set(candidates)):
        raise ValueError("invalid greedy budget or duplicate candidates")
    missing = [mask for mask in candidates if mask not in atoms]
    if missing:
        raise ValueError(f"missing candidate atoms: {missing}")
    indices = np.asarray(tuple(discovery_indices), dtype=np.int64)
    if indices.size == 0 or np.any(indices < 0):
        raise ValueError("discovery indices must be nonempty and nonnegative")
    target_array = np.asarray(target)
    residual = target_array[indices].copy()
    denominator = max(float(np.linalg.norm(residual)), 1e-30)
    available = list(candidates)
    selected = []
    curve = []
    for _ in range(steps):
        def score(mask):
            candidate = np.asarray(atoms[mask])[indices]
            return float(np.linalg.norm(residual - candidate)), mask
        choice = min(available, key=score)
        residual -= np.asarray(atoms[choice])[indices]
        available.remove(choice)
        selected.append(choice)
        curve.append(float(np.linalg.norm(residual) / denominator))
    return tuple(selected), tuple(curve)


def metrics(target, prediction, indices: Sequence[int]) -> dict[str, float | int]:
    """Component-relative metrics over rows selected along the first axis."""
    import numpy as np

    indices = np.asarray(tuple(indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("metric indices must be nonempty")
    y = np.asarray(target)[indices].reshape(-1).astype(np.float64, copy=False)
    p = np.asarray(prediction)[indices].reshape(-1).astype(np.float64, copy=False)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("target and prediction must be finite with matching shapes")
    yn = float(np.linalg.norm(y)); pn = float(np.linalg.norm(p))
    return {
        "rows": int(indices.size),
        "target_rms": float(np.sqrt(np.mean(y * y))),
        "prediction_rms": float(np.sqrt(np.mean(p * p))),
        "relative_l2": float(np.linalg.norm(y - p) / max(yn, 1e-30)),
        "cosine": float(p @ y / max(pn * yn, 1e-30)),
        "aligned_recovery": float(p @ y / max(y @ y, 1e-30)),
    }


def evaluate_frozen(target, atoms: Mapping[int, object], masks: Sequence[int],
                    panels: Mapping[str, Sequence[int]]) -> dict[str, dict[str, float | int]]:
    """Evaluate one already-frozen graph on discovery and disjoint OOD panels."""
    validate_panels(panels)
    prediction = compose(atoms, masks)
    return {name: metrics(target, prediction, indices) for name, indices in panels.items()}
