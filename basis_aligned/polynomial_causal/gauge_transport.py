"""Pure helpers for the priced gauge-transport preregistration.

The functions here do not fit a transport and do not load the model.  They make the
coordinate convention and causal-response score executable before GPU outcomes are
observed.
"""

from __future__ import annotations

import math

import torch


def _matrix(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def physical_transport(
    destination_decoder: torch.Tensor,
    coordinate_transport: torch.Tensor,
    source_encoder: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``D_t @ A @ E_s`` with explicit shape checks."""
    _matrix("destination_decoder", destination_decoder)
    _matrix("coordinate_transport", coordinate_transport)
    _matrix("source_encoder", source_encoder)
    if destination_decoder.shape[1] != coordinate_transport.shape[0]:
        raise ValueError("destination coordinate dimensions disagree")
    if coordinate_transport.shape[1] != source_encoder.shape[0]:
        raise ValueError("source coordinate dimensions disagree")
    return destination_decoder @ coordinate_transport @ source_encoder


def rewrite_coordinate_gauge(
    destination_decoder: torch.Tensor,
    coordinate_transport: torch.Tensor,
    source_encoder: torch.Tensor,
    source_gauge: torch.Tensor,
    destination_gauge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the complete coordinate rewrite from the preregistration.

    Coordinates obey ``c_s' = G_s c_s`` and ``c_t' = G_t c_t``.  Solves are
    used instead of explicitly forming inverses.
    """
    original = physical_transport(
        destination_decoder, coordinate_transport, source_encoder
    )
    del original  # The call above performs all interface shape validation.
    _matrix("source_gauge", source_gauge)
    _matrix("destination_gauge", destination_gauge)
    ks = source_encoder.shape[0]
    kt = destination_decoder.shape[1]
    if source_gauge.shape != (ks, ks):
        raise ValueError("source_gauge has the wrong shape")
    if destination_gauge.shape != (kt, kt):
        raise ValueError("destination_gauge has the wrong shape")
    try:
        # D_t G_t^-1 and A G_s^-1, expressed as right-side solves.
        decoder_rewritten = torch.linalg.solve(
            destination_gauge.T, destination_decoder.T
        ).T
        transport_right = torch.linalg.solve(
            source_gauge.T, coordinate_transport.T
        ).T
    except torch.linalg.LinAlgError as error:
        raise ValueError("coordinate gauges must be nonsingular") from error
    transport_rewritten = destination_gauge @ transport_right
    encoder_rewritten = source_gauge @ source_encoder
    return decoder_rewritten, transport_rewritten, encoder_rewritten


def response_metrics(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    denominator_floor_fraction: float = 0.1,
) -> dict[str, float]:
    """Score intervention-minus-baseline response vectors.

    The aggregate normalized response error (NRE) is the preregistered primary
    metric.  The per-example q90 uses a scale floor to keep nearly-null examples
    from dominating; the floor is returned and must be frozen from discovery in a
    real evaluation.
    """
    if reference.shape != prediction.shape or reference.ndim < 2:
        raise ValueError("reference and prediction need equal [examples, ...] shapes")
    if not 0 <= denominator_floor_fraction <= 1:
        raise ValueError("denominator_floor_fraction must lie in [0, 1]")
    if not torch.isfinite(reference).all() or not torch.isfinite(prediction).all():
        raise ValueError("responses contain non-finite values")
    ref = reference.detach().double().reshape(reference.shape[0], -1)
    pred = prediction.detach().double().reshape(prediction.shape[0], -1)
    error = pred - ref
    ref_energy = float(ref.square().sum())
    if ref_energy == 0:
        raise ValueError("NRE is undefined for an all-zero reference family")
    nre = math.sqrt(float(error.square().sum()) / ref_energy)
    ref_norm = torch.linalg.vector_norm(ref, dim=1)
    nonzero = ref_norm[ref_norm > 0]
    median = float(nonzero.median()) if nonzero.numel() else 0.0
    floor = denominator_floor_fraction * median
    denom = ref_norm.clamp_min(floor if floor > 0 else torch.finfo(ref.dtype).tiny)
    relative = torch.linalg.vector_norm(error, dim=1) / denom
    dot = float((ref * pred).sum())
    cosine_denom = math.sqrt(float(ref.square().sum()) * float(pred.square().sum()))
    cosine = dot / cosine_denom if cosine_denom > 0 else float("nan")
    return {
        "nre": nre,
        "per_example_relative_q90": float(torch.quantile(relative, 0.9)),
        "response_cosine": cosine,
        "denominator_floor": floor,
    }


def powered_sign_agreement(
    reference_effect: torch.Tensor,
    predicted_effect: torch.Tensor,
    powered: torch.Tensor,
) -> float:
    """Fraction of preregistered powered scalar effects with matching sign."""
    if reference_effect.shape != predicted_effect.shape:
        raise ValueError("effect tensors must have equal shapes")
    if powered.shape != reference_effect.shape or powered.dtype != torch.bool:
        raise ValueError("powered must be a boolean tensor matching the effects")
    if int(powered.sum()) == 0:
        raise ValueError("at least one powered effect is required")
    ref = reference_effect[powered]
    pred = predicted_effect[powered]
    return float((torch.sign(ref) == torch.sign(pred)).double().mean())
