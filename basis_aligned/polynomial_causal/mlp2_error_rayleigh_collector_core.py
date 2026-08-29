"""Pure control construction and reductions for the MLP2 error-Rayleigh collector."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


EPS = 1e-20
CONTROL_NAMES = ("ACTUAL", "DERANGED", "COV_RANDOM")
AMPLITUDES = (1.0 / 16.0, 1.0 / 8.0)
FEATURE_NAMES = (
    "local_mse",
    "ce_jvp_h16", "qlogit_h16", "q5_h16", "q6_h16",
    "kl_minus_h16", "kl_plus_h16", "dce_minus_h16", "dce_plus_h16",
    "ce_jvp_h8", "qlogit_h8", "q5_h8", "q6_h8",
    "kl_minus_h8", "kl_plus_h8", "dce_minus_h8", "dce_plus_h8",
)
FINITE_NAMES = (
    "direct_dce", "injected_dce", "logits_max_abs", "attention5_max_abs",
    "attention6_max_abs", "logits_exact", "attention5_exact", "attention6_exact",
)


def _field(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 3 \
            or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("error bank must be one finite [document, position, width] tensor")
    if value.shape[0] < 3:
        raise ValueError("error controls require at least three documents")
    return value.float().contiguous()


def _document_norm(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).square().sum(1).sqrt()


def rescale_document_norm(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value, reference = _field(value), _field(reference)
    if value.shape != reference.shape:
        raise ValueError("control and reference fields differ in shape")
    source, target = _document_norm(value), _document_norm(reference)
    if bool((source <= EPS).any()) or bool((target <= EPS).any()):
        raise ValueError("error control has a degenerate document norm")
    shape = (len(value),) + (1,) * (value.ndim - 1)
    return (value * (target / source).reshape(shape)).contiguous()


def control_error_bank(actual: torch.Tensor, seed: int) -> Mapping[str, torch.Tensor]:
    """Construct matched, whole-document deranged, and covariance-span controls."""
    actual = _field(actual)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("control seed must be a nonnegative integer")
    deranged = rescale_document_norm(actual.roll(1, dims=0), actual)
    centered = actual - actual.mean(dim=0, keepdim=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(len(actual), len(actual), generator=generator)
    weights.fill_diagonal_(0.0)
    row_norm = weights.square().sum(1, keepdim=True).sqrt()
    if bool((row_norm <= EPS).any()):
        raise RuntimeError("covariance-control mixing matrix is degenerate")
    weights = weights / row_norm
    random = torch.einsum("ij,jtw->itw", weights, centered)
    random = rescale_document_norm(random, actual)
    return {"ACTUAL": actual, "DERANGED": deranged, "COV_RANDOM": random}


def actual_write(native: torch.Tensor, candidate: torch.Tensor, alpha: float) -> torch.Tensor:
    """Endpoint-preserving implementation of native + alpha*(candidate-native)."""
    if native.shape != candidate.shape or native.dtype != candidate.dtype \
            or native.device != candidate.device or not isinstance(alpha, (int, float)):
        raise ValueError("actual interpolation inputs are malformed")
    if float(alpha) == 0.0:
        return native
    if float(alpha) == 1.0:
        return candidate
    return torch.lerp(native.float(), candidate.float(), float(alpha)).to(native.dtype)


def control_write(native: torch.Tensor, error: torch.Tensor, alpha: float) -> torch.Tensor:
    if native.shape != error.shape or native.device != error.device \
            or not isinstance(alpha, (int, float)):
        raise ValueError("control interpolation inputs are malformed")
    return (native.float() + float(alpha) * error.float()).to(native.dtype)


def document_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2] \
            or targets.dtype != torch.long or not bool(torch.isfinite(logits).all()):
        raise ValueError("CE inputs are malformed")
    return F.cross_entropy(
        logits.float().flatten(0, 1), targets.flatten(), reduction="none",
    ).reshape(targets.shape).mean(1).double().cpu()


def response_statistics(
    baseline_logits: torch.Tensor,
    plus_logits: torch.Tensor,
    minus_logits: torch.Tensor,
    baseline_attention5: torch.Tensor,
    plus_attention5: torch.Tensor,
    minus_attention5: torch.Tensor,
    baseline_attention6: torch.Tensor,
    plus_attention6: torch.Tensor,
    minus_attention6: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> Mapping[str, torch.Tensor]:
    """Reduce one antithetic response pair to document-level sufficient statistics."""
    if alpha <= 0 or baseline_logits.shape != plus_logits.shape \
            or baseline_logits.shape != minus_logits.shape \
            or targets.shape != baseline_logits.shape[:2]:
        raise ValueError("response logit inputs are malformed")
    for native, plus, minus in (
        (baseline_attention5, plus_attention5, minus_attention5),
        (baseline_attention6, plus_attention6, minus_attention6),
    ):
        if native.shape != plus.shape or native.shape != minus.shape \
                or native.shape[0] != baseline_logits.shape[0]:
            raise ValueError("response attention inputs are malformed")

    native = baseline_logits.float()
    plus = plus_logits.float()
    minus = minus_logits.float()
    delta = (plus - minus) / (2.0 * float(alpha))
    logp = F.log_softmax(native, dim=-1)
    probability = logp.exp()
    expected_delta = (probability * delta).sum(-1)
    selected_delta = delta.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ce_jvp = (expected_delta - selected_delta).mean(1)
    qlogit = ((probability * delta.square()).sum(-1) - expected_delta.square()).mean(1)

    def attention_energy(base: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor):
        derivative = (hi.float() - lo.float()) / (2.0 * float(alpha))
        axes = tuple(range(1, derivative.ndim))
        denominator = base.float().square().sum(axes)
        if bool((denominator <= EPS).any()):
            raise ValueError("native attention energy is degenerate")
        return derivative.square().sum(axes) / denominator

    q5 = attention_energy(baseline_attention5, plus_attention5, minus_attention5)
    q6 = attention_energy(baseline_attention6, plus_attention6, minus_attention6)

    def kl(changed: torch.Tensor):
        changed_logp = F.log_softmax(changed.float(), dim=-1)
        return (probability * (logp - changed_logp)).sum(-1).mean(1)

    baseline_ce = document_ce(native, targets)
    output = {
        "ce_jvp": ce_jvp.double().cpu(),
        "qlogit": qlogit.clamp_min(0).double().cpu(),
        "q5": q5.double().cpu(), "q6": q6.double().cpu(),
        "kl_minus": kl(minus).double().cpu(),
        "kl_plus": kl(plus).double().cpu(),
        "dce_minus": document_ce(minus, targets) - baseline_ce,
        "dce_plus": document_ce(plus, targets) - baseline_ce,
    }
    if any(value.shape != (len(targets),) or not torch.isfinite(value).all()
           for value in output.values()):
        raise RuntimeError("response reduction produced malformed statistics")
    return output


def pack_features(
    local_mse: torch.Tensor,
    by_amplitude: Mapping[float, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    if set(by_amplitude) != set(AMPLITUDES) or local_mse.ndim != 1:
        raise ValueError("feature inputs are incomplete")
    columns = [local_mse.double().cpu()]
    for amplitude in AMPLITUDES:
        value = by_amplitude[amplitude]
        columns.extend(value[name] for name in (
            "ce_jvp", "qlogit", "q5", "q6", "kl_minus", "kl_plus",
            "dce_minus", "dce_plus",
        ))
    packed = torch.stack(columns, dim=-1)
    if packed.shape != (len(local_mse), len(FEATURE_NAMES)) \
            or not torch.isfinite(packed).all():
        raise RuntimeError("packed response features are malformed")
    return packed


def replay_statistics(
    baseline_logits: torch.Tensor, direct_logits: torch.Tensor,
    injected_logits: torch.Tensor, baseline_attention5: torch.Tensor,
    direct_attention5: torch.Tensor, injected_attention5: torch.Tensor,
    baseline_attention6: torch.Tensor, direct_attention6: torch.Tensor,
    injected_attention6: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    baseline_ce = document_ce(baseline_logits, targets)
    direct_ce = document_ce(direct_logits, targets) - baseline_ce
    injected_ce = document_ce(injected_logits, targets) - baseline_ce

    def max_abs(left: torch.Tensor, right: torch.Tensor):
        return (left.float() - right.float()).abs().flatten(1).amax(1).double().cpu()

    def exact(left: torch.Tensor, right: torch.Tensor):
        return (left == right).flatten(1).all(1).double().cpu()

    columns = [
        direct_ce, injected_ce, max_abs(direct_logits, injected_logits),
        max_abs(direct_attention5, injected_attention5),
        max_abs(direct_attention6, injected_attention6),
        exact(direct_logits, injected_logits), exact(direct_attention5, injected_attention5),
        exact(direct_attention6, injected_attention6),
    ]
    packed = torch.stack(columns, dim=-1)
    if packed.shape != (len(targets), len(FINITE_NAMES)) or not torch.isfinite(packed).all():
        raise RuntimeError("finite replay statistics are malformed")
    return packed
