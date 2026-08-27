"""Pure helpers for the priced gauge-transport preregistration.

The functions here do not fit a transport and do not load the model.  They make the
coordinate convention and causal-response score executable before GPU outcomes are
observed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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


def fit_delta_ridge(
    source_delta: torch.Tensor,
    destination_delta: torch.Tensor,
    *,
    relative_ridge: float = 1e-3,
) -> torch.Tensor:
    """Fit ``destination_delta ~= source_delta @ weight`` with zero origin.

    Intervention responses are differences, so an affine intercept would introduce
    an arbitrary origin and is intentionally excluded.  The ridge is relative to
    the mean Gram diagonal, making the declared value invariant to sample count.
    """
    if source_delta.ndim != 2 or destination_delta.ndim != 2:
        raise ValueError("delta regression expects rank-2 [examples, coordinates] tensors")
    if source_delta.shape[0] != destination_delta.shape[0]:
        raise ValueError("source and destination deltas need the same examples")
    if source_delta.shape[0] == 0:
        raise ValueError("delta regression needs at least one example")
    if relative_ridge < 0 or not math.isfinite(relative_ridge):
        raise ValueError("relative_ridge must be finite and nonnegative")
    source = source_delta.detach().double()
    destination = destination_delta.detach().double()
    if not torch.isfinite(source).all() or not torch.isfinite(destination).all():
        raise ValueError("delta regression contains non-finite values")
    gram = source.T @ source
    scale = float(gram.diag().mean()) if gram.numel() else 0.0
    regularization = relative_ridge * max(scale, torch.finfo(gram.dtype).tiny)
    gram = gram.clone()
    gram.diagonal().add_(regularization)
    try:
        return torch.linalg.solve(gram, source.T @ destination)
    except torch.linalg.LinAlgError as error:
        raise ValueError("delta regression Gram matrix is singular") from error


def response_r2(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """Zero-origin response R2: ``1 - ||prediction-reference||^2/||reference||^2``."""
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction response shapes disagree")
    reference = reference.detach().double()
    prediction = prediction.detach().double()
    denominator = float(reference.square().sum())
    if denominator == 0:
        raise ValueError("response R2 is undefined for an all-zero reference")
    return 1.0 - float((prediction - reference).square().sum()) / denominator


def commuting_output_metrics(
    baseline_logits: torch.Tensor,
    early_intervention_logits: torch.Tensor,
    transported_logits: torch.Tensor,
) -> dict[str, float]:
    """Score whether a transported late patch commutes with an early intervention.

    ``E_out`` is the aggregate ``KL(early || transported) / KL(early || baseline)``.
    Centered-logit error removes the softmax-irrelevant per-token scalar gauge.
    Inputs may have any leading dimensions but share the final vocabulary axis.
    """
    if not (
        baseline_logits.shape
        == early_intervention_logits.shape
        == transported_logits.shape
    ):
        raise ValueError("all logit tensors must have identical shapes")
    if baseline_logits.ndim < 2:
        raise ValueError("logits need a final vocabulary dimension")
    tensors = (baseline_logits, early_intervention_logits, transported_logits)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("logits contain non-finite values")
    baseline = baseline_logits.detach().double()
    early = early_intervention_logits.detach().double()
    transported = transported_logits.detach().double()
    logp_early = F.log_softmax(early, dim=-1)
    p_early = logp_early.exp()
    numerator = float(
        (p_early * (logp_early - F.log_softmax(transported, dim=-1))).sum()
    )
    denominator = float(
        (p_early * (logp_early - F.log_softmax(baseline, dim=-1))).sum()
    )
    if denominator <= torch.finfo(torch.float64).eps:
        raise ValueError("early intervention has zero output KL relative to baseline")
    early_centered = early - early.mean(dim=-1, keepdim=True)
    transported_centered = transported - transported.mean(dim=-1, keepdim=True)
    baseline_centered = baseline - baseline.mean(dim=-1, keepdim=True)
    target_delta = early_centered - baseline_centered
    prediction_error = transported_centered - early_centered
    target_energy = float(target_delta.square().sum())
    if target_energy == 0:
        raise ValueError("early intervention has zero centered-logit response")
    return {
        "e_out": numerator / denominator,
        "centered_logit_relative_rmse": math.sqrt(
            float(prediction_error.square().sum()) / target_energy
        ),
        "early_vs_baseline_kl_sum": denominator,
        "early_vs_transported_kl_sum": numerator,
    }


def centered_logit_response_sums(
    baseline_raw_logits: torch.Tensor,
    early_raw_logits: torch.Tensor,
    transported_raw_logits: torch.Tensor,
) -> dict[str, float]:
    """Streaming-friendly centered pre-softcap response error components."""
    if not (
        baseline_raw_logits.shape
        == early_raw_logits.shape
        == transported_raw_logits.shape
    ):
        raise ValueError("all raw-logit tensors must have identical shapes")
    if baseline_raw_logits.ndim < 2:
        raise ValueError("raw logits need a final vocabulary dimension")
    values = tuple(
        tensor.detach().double() - tensor.detach().double().mean(dim=-1, keepdim=True)
        for tensor in (baseline_raw_logits, early_raw_logits, transported_raw_logits)
    )
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("raw logits contain non-finite values")
    baseline, early, transported = values
    target = early - baseline
    error = transported - early
    return {
        "centered_logit_error_sum_squares": float(error.square().sum()),
        "centered_logit_target_sum_squares": float(target.square().sum()),
    }


def haar_basis_in_support(
    support: torch.Tensor,
    rank: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a Haar rank-``rank`` basis inside an orthonormal support."""
    _matrix("support", support)
    if rank <= 0 or rank > support.shape[1]:
        raise ValueError("rank must lie in [1, support dimension]")
    identity = torch.eye(support.shape[1], dtype=support.dtype, device=support.device)
    if not torch.allclose(support.T @ support, identity, rtol=1e-5, atol=1e-6):
        raise ValueError("support columns must be orthonormal")
    gaussian = torch.randn(
        support.shape[1], rank,
        dtype=support.dtype, device=support.device, generator=generator,
    )
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -torch.ones_like(torch.diag(r)),
                        torch.ones_like(torch.diag(r)))
    return support @ (q * signs)


def beats_all_nulls(
    candidate: float,
    nulls: list[float],
    *,
    lower_is_better: bool,
) -> dict[str, float | bool | int]:
    """Exact finite-null gate; with 20 nulls the minimum p-value is 1/21."""
    if not nulls or not math.isfinite(candidate) or not all(map(math.isfinite, nulls)):
        raise ValueError("candidate and at least one null must be finite")
    if lower_is_better:
        null_at_least_as_good = sum(value <= candidate for value in nulls)
        passed = candidate < min(nulls)
    else:
        null_at_least_as_good = sum(value >= candidate for value in nulls)
        passed = candidate > max(nulls)
    return {
        "passed": passed,
        "null_at_least_as_good": null_at_least_as_good,
        "finite_null_p": (1 + null_at_least_as_good) / (1 + len(nulls)),
        "n_nulls": len(nulls),
    }
