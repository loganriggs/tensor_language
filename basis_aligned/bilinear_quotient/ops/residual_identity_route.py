"""Exact affine residual-skip transport used by controlled circuit interventions."""

# BQGATE: LIBRARY
from __future__ import annotations

import math


class ResidualIdentityRouteError(ValueError):
    pass


def skip_gain(lambda0_values):
    """Return the exact scalar coefficient multiplying a residual delta."""
    values = tuple(float(value) for value in lambda0_values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ResidualIdentityRouteError("skip coefficients must be nonempty and finite")
    gain = math.prod(values)
    if not math.isfinite(gain):
        raise ResidualIdentityRouteError("skip gain is not finite")
    return gain


def _transport_delta(native_entry, writer_entry, gain):
    if native_entry.shape != writer_entry.shape:
        raise ResidualIdentityRouteError("native and writer entry shapes differ")
    return (writer_entry - native_entry) * gain


def direct_add(native_final, native_entry, writer_entry, gain):
    """Install controlled identity carriage in a native final residual."""
    delta = _transport_delta(native_entry, writer_entry, gain)
    if native_final.shape != delta.shape:
        raise ResidualIdentityRouteError("native final and transported delta shapes differ")
    return native_final + delta


def direct_remove(writer_final, native_entry, writer_entry, gain):
    """Remove controlled identity carriage while retaining writer module writes."""
    delta = _transport_delta(native_entry, writer_entry, gain)
    if writer_final.shape != delta.shape:
        raise ResidualIdentityRouteError("writer final and transported delta shapes differ")
    return writer_final - delta
