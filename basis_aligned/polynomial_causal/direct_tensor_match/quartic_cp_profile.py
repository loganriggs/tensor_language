"""Profile the output coefficients of a symmetric quartic CP polynomial."""
import torch
from quartic_cp import cp_gram, directional


def normalize_factors(factors):
    return [f / f.norm(dim=1, keepdim=True).clamp_min(1e-12) for f in factors]


def profile(gram, cross, ridge=1e-8, envelope=True):
    coefficients = torch.linalg.solve(
        gram + ridge * torch.eye(len(gram), dtype=gram.dtype, device=gram.device),
        cross.T,
    ).T
    if envelope:
        coefficients = coefficients.detach()
    loss = (coefficients * (coefficients @ gram)).sum() - 2 * (coefficients * cross).sum()
    loss = loss + ridge * coefficients.square().sum()
    return loss, coefficients


def cp_objective(teacher_coefficients, teacher_factors, factors, ridge=1e-8, envelope=True):
    gram = cp_gram(factors, factors)
    cross = teacher_coefficients @ cp_gram(teacher_factors, factors)
    return profile(gram, cross, ridge, envelope)


def native_objective(teacher, factors, ridge=1e-8, envelope=True):
    gram = cp_gram(factors, factors)
    cross = directional(*teacher, factors).T
    return profile(gram, cross, ridge, envelope)
