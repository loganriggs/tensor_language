"""CPU certificates for the product count of a vector-valued quadratic map.

For

    q(x) = sum_g d_g (l_g^T x)(r_g^T x),

only the symmetrized products ``sym(l_g tensor r_g)`` are observable.  If their
Gram matrix and the decoder-column Gram matrix are positive definite, the output-mode
unfolding has rank K.  Matrix-unfolding rank lower-bounds partially symmetric CP rank,
while the displayed K products give the matching upper bound.  Hence the exact
quadratic product rank is K.

The numerical certificate uses the standard floating-point model for float64 matrix
products, Weyl's inequality, and a deliberately conservative eigensolver backward-
error allowance.  Inputs are converted exactly from serialized float32 to float64.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


FLOAT64_U = 2.0 ** -53
EIGENSOLVER_SAFETY_FACTOR = 100.0


@dataclass(frozen=True)
class PositiveDefiniteCertificate:
    dimension: int
    lambda_min_computed: float
    lambda_max_computed: float
    construction_error_bound_frobenius: float
    eigensolver_backward_error_allowance: float
    total_spectral_error_bound: float
    certified_lambda_min_lower_bound: float
    margin_over_error_bound: float
    positive_definite_certified: bool


@dataclass(frozen=True)
class QuadraticRankCertificate:
    products: int
    width: int
    decoder: PositiveDefiniteCertificate
    symmetrized_products: PositiveDefiniteCertificate
    unfolding_rank_lower_bound: int
    explicit_product_upper_bound: int
    exact_quadratic_product_rank_certified: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(left: torch.Tensor, right: torch.Tensor, decoder: torch.Tensor) -> None:
    if any(not torch.is_tensor(value) or value.device.type != "cpu" or value.ndim != 2
           or not value.is_floating_point() or not bool(torch.isfinite(value).all())
           for value in (left, right, decoder)):
        raise ValueError("factors must be finite floating CPU matrices")
    if left.shape != right.shape or decoder.shape != (left.shape[1], left.shape[0]):
        raise ValueError("expected Left/Right [K,d] and decoder [d,K]")
    if left.shape[0] < 1 or left.shape[1] < 1:
        raise ValueError("factor dimensions must be positive")


def _gamma(length: int) -> float:
    value = length * FLOAT64_U
    if value >= 1:
        raise ValueError("dot product is outside the standard error model")
    return value / (1.0 - value)


def _product_with_error(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fl(a@b) and a componentwise absolute forward-error bound."""
    if a.dtype != torch.float64 or b.dtype != torch.float64 or a.shape[1] != b.shape[0]:
        raise ValueError("internal products require aligned float64 matrices")
    product = a @ b
    bound = _gamma(a.shape[1]) * (a.abs() @ b.abs())
    return product, bound


def _hadamard_product_error(
    a: torch.Tensor, a_error: torch.Tensor, b: torch.Tensor, b_error: torch.Tensor,
) -> torch.Tensor:
    rounding = FLOAT64_U / (1.0 - FLOAT64_U) * (a * b).abs()
    return (a_error * b.abs() + a.abs() * b_error + a_error * b_error + rounding)


def symmetrized_product_gram(
    left: torch.Tensor, right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gram of sym(l_g tensor r_g), plus a construction error bound.

    With sym(a tensor b)=(a tensor b+b tensor a)/2,

        <sym(l_g tensor r_g), sym(l_h tensor r_h)>
        = 1/2 [(l_g.l_h)(r_g.r_h) + (l_g.r_h)(r_g.l_h)].
    """
    if left.dtype != torch.float64 or right.dtype != torch.float64 or left.shape != right.shape:
        raise ValueError("symmetrized Gram requires aligned float64 factors")
    ll, ll_error = _product_with_error(left, left.T)
    rr, rr_error = _product_with_error(right, right.T)
    lr, lr_error = _product_with_error(left, right.T)
    direct = ll * rr
    crossed = lr * lr.T
    direct_error = _hadamard_product_error(ll, ll_error, rr, rr_error)
    crossed_error = _hadamard_product_error(lr, lr_error, lr.T, lr_error.T)
    gram = 0.5 * (direct + crossed)
    # Include the final addition and multiplication by 1/2.
    final_rounding = 3.0 * FLOAT64_U * (0.5 * (direct.abs() + crossed.abs()))
    error = 0.5 * (direct_error + crossed_error) + final_rounding
    return 0.5 * (gram + gram.T), error


def _pd_certificate(gram: torch.Tensor, construction_error: torch.Tensor) -> PositiveDefiniteCertificate:
    if gram.dtype != torch.float64 or construction_error.shape != gram.shape or gram.ndim != 2 or (
        gram.shape[0] != gram.shape[1]
    ):
        raise ValueError("positive-definite certificate inputs are malformed")
    values = torch.linalg.eigvalsh(0.5 * (gram + gram.T))
    construction = float(torch.linalg.vector_norm(construction_error))
    backward = float(
        EIGENSOLVER_SAFETY_FACTOR * gram.shape[0] * FLOAT64_U
        * torch.linalg.vector_norm(gram)
    )
    total = construction + backward
    lower = float(values[0]) - total
    margin = float(values[0]) / total if total > 0 else math.inf
    return PositiveDefiniteCertificate(
        dimension=gram.shape[0],
        lambda_min_computed=float(values[0]),
        lambda_max_computed=float(values[-1]),
        construction_error_bound_frobenius=construction,
        eigensolver_backward_error_allowance=backward,
        total_spectral_error_bound=total,
        certified_lambda_min_lower_bound=lower,
        margin_over_error_bound=margin,
        positive_definite_certified=lower > 0,
    )


def certify_quadratic_product_rank(
    left: torch.Tensor, right: torch.Tensor, decoder: torch.Tensor,
) -> QuadraticRankCertificate:
    """Certify exact product rank K when one symmetrized unfolding has rank K."""
    _validate(left, right, decoder)
    left64, right64, decoder64 = left.double(), right.double(), decoder.double()
    decoder_gram, decoder_error = _product_with_error(decoder64.T, decoder64)
    decoder_pd = _pd_certificate(decoder_gram, decoder_error)
    product_gram, product_error = symmetrized_product_gram(left64, right64)
    product_pd = _pd_certificate(product_gram, product_error)
    products = left.shape[0]
    certified = decoder_pd.positive_definite_certified and product_pd.positive_definite_certified
    return QuadraticRankCertificate(
        products=products,
        width=left.shape[1],
        decoder=decoder_pd,
        symmetrized_products=product_pd,
        unfolding_rank_lower_bound=products if certified else 0,
        explicit_product_upper_bound=products,
        exact_quadratic_product_rank_certified=certified,
    )

