"""Pure CPU algebra for the preregistered MLP1 folded-tensor diagnostic.

This module has no checkpoint loader, filesystem writes, model imports, or command-line
entry point.  Production code must bind its exact source before supplying real weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


ENERGY_THRESHOLDS = (0.90, 0.95, 0.99, 0.999)


@dataclass(frozen=True)
class BalancedFactors:
    down: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    bias: torch.Tensor
    dead_units: tuple[int, ...]
    term_norms: torch.Tensor
    max_log_defect_before: float
    weighted_log_defect_before: float
    max_log_defect_after: float
    weighted_log_defect_after: float


def _cpu_f64(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be float32 or float64")
    result = value.detach().to(dtype=torch.float64).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def validate_and_copy_factors(
    down: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return owned CPU-float64 factors in the registered ``D,L,R,b`` layout."""
    dmat = _cpu_f64(down, "down", 2)
    lmat = _cpu_f64(left, "left", 2)
    rmat = _cpu_f64(right, "right", 2)
    bvec = _cpu_f64(bias, "bias", 1)
    if lmat.shape != rmat.shape:
        raise ValueError("left and right shapes differ")
    output, hidden = dmat.shape
    if lmat.shape[0] != hidden:
        raise ValueError("hidden dimension differs across factors")
    if bvec.shape != (output,):
        raise ValueError("bias shape differs from output dimension")
    if min(output, hidden, lmat.shape[1]) <= 0:
        raise ValueError("factor dimensions must be positive")
    return dmat, lmat, rmat, bvec


def _log_defect(norms: torch.Tensor, log_mass: torch.Tensor) -> tuple[float, float]:
    live = (norms > 0).all(dim=1)
    if not bool(live.any()):
        return 0.0, 0.0
    defects = torch.log(norms[live]).std(dim=1, correction=0)
    live_log_mass = log_mass[live]
    weights = torch.exp(live_log_mass - live_log_mass.max())
    weighted = 0.0 if float(weights.sum()) == 0.0 else float((defects * weights).sum() / weights.sum())
    return float(defects.max()), weighted


def balance_factors(
    down: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
) -> BalancedFactors:
    """Put every nonzero CP term in its scale-balanced positive gauge.

    The bias is owned and returned unchanged.  Only exactly zero terms are deleted;
    this keeps the dead-unit rule invariant under every finite nonzero scalar gauge.
    """
    dmat, lmat, rmat, bvec = validate_and_copy_factors(down, left, right, bias)
    norms = torch.stack(
        (torch.linalg.vector_norm(dmat, dim=0),
         torch.linalg.vector_norm(lmat, dim=1),
         torch.linalg.vector_norm(rmat, dim=1)),
        dim=1,
    )
    dead = (norms == 0).any(dim=1)
    log_m = torch.zeros(norms.shape[0], dtype=torch.float64)
    live = ~dead
    log_m[live] = torch.log(norms[live]).mean(dim=1)
    term_norms = torch.exp(log_m)
    term_norms[dead] = 0.0
    log_mass = 3.0 * log_m
    before_max, before_weighted = _log_defect(norms, log_mass)

    for column, matrix, axis in ((0, dmat, 0), (1, lmat, 1), (2, rmat, 1)):
        scale = torch.ones_like(term_norms)
        scale[live] = torch.exp(log_m[live] - torch.log(norms[live, column]))
        if axis == 0:
            matrix.mul_(scale.unsqueeze(0))
            matrix[:, dead] = 0.0
        else:
            matrix.mul_(scale.unsqueeze(1))
            matrix[dead, :] = 0.0

    after_norms = torch.stack(
        (torch.linalg.vector_norm(dmat, dim=0),
         torch.linalg.vector_norm(lmat, dim=1),
         torch.linalg.vector_norm(rmat, dim=1)),
        dim=1,
    )
    after_max, after_weighted = _log_defect(after_norms, log_mass)
    return BalancedFactors(
        down=dmat,
        left=lmat,
        right=rmat,
        bias=bvec,
        dead_units=tuple(torch.nonzero(dead, as_tuple=False).flatten().tolist()),
        term_norms=term_norms,
        max_log_defect_before=before_max,
        weighted_log_defect_before=before_weighted,
        max_log_defect_after=after_max,
        weighted_log_defect_after=after_weighted,
    )


def bilinear_output(factors: BalancedFactors, inputs: torch.Tensor) -> torch.Tensor:
    """Execute the factored polynomial, including the separately stored bias."""
    x = _cpu_f64(inputs, "inputs", 2)
    if x.shape[1] != factors.left.shape[1]:
        raise ValueError("input width differs from factor width")
    products = (x @ factors.left.T) * (x @ factors.right.T)
    return products @ factors.down.T + factors.bias


def _rank_summary(squared_singular_values: torch.Tensor) -> dict[str, object]:
    values = squared_singular_values.detach().to(dtype=torch.float64, device="cpu")
    if values.ndim != 1 or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("squared singular values must be a finite nonnegative vector")
    total = float(values.sum())
    cumulative = torch.zeros_like(values) if total == 0.0 else torch.cumsum(values, dim=0) / total
    ranks: dict[str, int] = {}
    for threshold in ENERGY_THRESHOLDS:
        if total == 0.0:
            rank = 0
        else:
            rank = int(torch.searchsorted(cumulative, threshold, right=False)) + 1
        ranks[f"{threshold:.3f}"] = rank
    return {
        "squared_singular_values": values.tolist(),
        "cumulative_frobenius_fraction": cumulative.tolist(),
        "frobenius_squared": total,
        "energy_ranks": ranks,
    }


def balanced_down_svd(factors: BalancedFactors) -> dict[str, object]:
    singular = torch.linalg.svdvals(factors.down)
    return _rank_summary(singular.square())


def exact_folded_mode_grams(
    factors: BalancedFactors,
    *,
    hidden_block: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact output/input mode Grams without materializing ``T``.

    Peak algebraic workspace is a constant number of ``hidden_block x hidden``
    matrices plus the two returned Grams.  Accumulation is float64 on CPU.
    """
    if not isinstance(hidden_block, int) or isinstance(hidden_block, bool) or hidden_block <= 0:
        raise ValueError("hidden_block must be a positive integer")
    dmat, lmat, rmat = factors.down, factors.left, factors.right
    output, hidden = dmat.shape
    width = lmat.shape[1]
    gout = torch.zeros((output, output), dtype=torch.float64)
    gin = torch.zeros((width, width), dtype=torch.float64)
    dt = dmat.T
    lt = lmat.T
    rt = rmat.T

    for start in range(0, hidden, hidden_block):
        stop = min(start + hidden_block, hidden)
        db = dmat[:, start:stop]
        lb = lmat[start:stop]
        rb = rmat[start:stop]

        ll = lb @ lt
        rr = rb @ rt
        lr = lb @ rt
        rl = rb @ lt
        kernel = 0.5 * (ll * rr + lr * rl)
        gout.add_(db @ (kernel @ dt))

        output_kernel = db.T @ dmat
        gin.add_(0.25 * (lb.T @ ((output_kernel * rr) @ lmat)))
        # Across the complete hidden-index sum, the omitted companion cross
        # term is this matrix's transpose.  The final symmetrization supplies
        # both halves; doubling here preserves their registered 1/4 weights.
        gin.add_(0.50 * (lb.T @ ((output_kernel * rl) @ rmat)))
        gin.add_(0.25 * (rb.T @ ((output_kernel * ll) @ rmat)))

    return 0.5 * (gout + gout.T), 0.5 * (gin + gin.T)


def _spectrum_from_gram(gram: torch.Tensor, *, negative_relative_tolerance: float) -> dict[str, object]:
    if negative_relative_tolerance < 0 or not torch.isfinite(torch.tensor(negative_relative_tolerance)):
        raise ValueError("negative_relative_tolerance must be finite and nonnegative")
    matrix = _cpu_f64(gram, "gram", 2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square")
    asymmetry = float(torch.linalg.vector_norm(matrix - matrix.T))
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(matrix).flip(0)
    scale = max(1.0, float(eigenvalues.abs().max()))
    minimum = float(eigenvalues.min())
    if minimum < -negative_relative_tolerance * scale:
        raise ValueError("gram has a materially negative eigenvalue")
    clipped = eigenvalues.clamp_min(0.0)
    result = _rank_summary(clipped)
    result.update({"minimum_raw_eigenvalue": minimum, "gram_asymmetry_frobenius": asymmetry})
    return result


def folded_hosvd_spectra(
    factors: BalancedFactors,
    *,
    hidden_block: int = 64,
    negative_relative_tolerance: float = 1e-10,
    trace_relative_tolerance: float = 1e-10,
) -> dict[str, object]:
    gout, gin = exact_folded_mode_grams(factors, hidden_block=hidden_block)
    output = _spectrum_from_gram(gout, negative_relative_tolerance=negative_relative_tolerance)
    input_mode_1 = _spectrum_from_gram(gin, negative_relative_tolerance=negative_relative_tolerance)
    trace_out = float(torch.trace(gout))
    trace_in = float(torch.trace(gin))
    trace_scale = max(1.0, abs(trace_out), abs(trace_in))
    trace_residual = abs(trace_out - trace_in) / trace_scale
    if trace_residual > trace_relative_tolerance:
        raise ValueError("mode Gram traces disagree beyond the registered tolerance")
    return {
        "output_mode": output,
        "input_mode_1": input_mode_1,
        "input_mode_2": dict(input_mode_1),
        "input_mode_equality": "exact_by_registered_partial_symmetry",
        "output_gram_trace": trace_out,
        "input_gram_trace": trace_in,
        "relative_trace_residual": trace_residual,
        "hidden_block": hidden_block,
        # Four factor Grams, one output-factor Gram, one Hadamard temporary,
        # and conservative allowance for the block-by-width matmul temporary.
        "peak_block_workspace_elements_conservative_bound": (
            6 * hidden_block * factors.down.shape[1]
            + 2 * hidden_block * factors.left.shape[1]
        ),
    }


def hosvd_bases(
    output_gram: torch.Tensor,
    input_gram: torch.Tensor,
    *,
    output_rank: int,
    input_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gout = _cpu_f64(output_gram, "output_gram", 2)
    gin = _cpu_f64(input_gram, "input_gram", 2)
    if gout.shape[0] != gout.shape[1] or gin.shape[0] != gin.shape[1]:
        raise ValueError("mode Grams must be square")
    if not 1 <= output_rank <= gout.shape[0] or not 1 <= input_rank <= gin.shape[0]:
        raise ValueError("requested HOSVD rank is outside its mode")
    _, uo = torch.linalg.eigh(0.5 * (gout + gout.T))
    _, ui = torch.linalg.eigh(0.5 * (gin + gin.T))
    return uo[:, -output_rank:].flip(1).contiguous(), ui[:, -input_rank:].flip(1).contiguous()


def project_symmetric_hosvd_core(
    factors: BalancedFactors,
    output_basis: torch.Tensor,
    input_basis: torch.Tensor,
) -> torch.Tensor:
    """Project the folded tensor to a requested, explicitly bounded HOSVD core."""
    uo = _cpu_f64(output_basis, "output_basis", 2)
    ui = _cpu_f64(input_basis, "input_basis", 2)
    if uo.shape[0] != factors.down.shape[0] or ui.shape[0] != factors.left.shape[1]:
        raise ValueError("basis ambient dimension differs from factor dimensions")
    eye_o = torch.eye(uo.shape[1], dtype=torch.float64)
    eye_i = torch.eye(ui.shape[1], dtype=torch.float64)
    if not torch.allclose(uo.T @ uo, eye_o, atol=1e-10, rtol=1e-10):
        raise ValueError("output basis is not orthonormal")
    if not torch.allclose(ui.T @ ui, eye_i, atol=1e-10, rtol=1e-10):
        raise ValueError("input basis is not orthonormal")
    dp = uo.T @ factors.down
    lp = factors.left @ ui
    rp = factors.right @ ui
    unsym = torch.einsum("an,nb,nc->abc", dp, lp, rp)
    return 0.5 * (unsym + unsym.transpose(1, 2))


def _price_record(float_storage: int, integer_storage: int, multiply_adds: int,
                  products: int, bias_additions: int) -> dict[str, int]:
    if min(float_storage, integer_storage, multiply_adds, products, bias_additions) < 0:
        raise ValueError("prices must be nonnegative")
    return {
        "float_storage": int(float_storage),
        "integer_storage": int(integer_storage),
        "multiply_adds_per_token": int(multiply_adds),
        "bilinear_products_per_token": int(products),
        "bias_additions_per_token": int(bias_additions),
        "scalar_multiplications_per_token": int(multiply_adds + products),
    }


def native_price(output: int, hidden: int, width: int) -> dict[str, int]:
    return _price_record(2 * hidden * width + output * hidden + output, 0,
                         2 * hidden * width + output * hidden, hidden, output)


def down_rank_price(output: int, hidden: int, width: int, rank: int) -> dict[str, int]:
    if not 1 <= rank <= min(output, hidden):
        raise ValueError("Down rank is outside matrix dimensions")
    return _price_record(2 * hidden * width + rank * (hidden + output) + output, 0,
                         2 * hidden * width + rank * (hidden + output), hidden, output)


def dense_tucker_price(output: int, width: int, output_rank: int, input_rank: int) -> dict[str, int]:
    if not 1 <= output_rank <= output or not 1 <= input_rank <= width:
        raise ValueError("Tucker rank is outside tensor dimensions")
    pairs = input_rank * (input_rank + 1) // 2
    coefficients = output_rank * pairs
    return _price_record(width * input_rank + output * output_rank + coefficients + output, 0,
                         width * input_rank + coefficients + output * output_rank, pairs, output)


def sparse_tucker_price(output: int, width: int, output_rank: int, input_rank: int,
                        coefficients: int, active_pairs: int) -> dict[str, int]:
    max_coefficients = output_rank * input_rank * (input_rank + 1) // 2
    max_pairs = input_rank * (input_rank + 1) // 2
    if not 0 <= coefficients <= max_coefficients:
        raise ValueError("sparse coefficient count is outside core dimensions")
    if coefficients == 0 and active_pairs != 0:
        raise ValueError("an empty sparse core cannot have an active pair")
    if coefficients > 0 and not 1 <= active_pairs <= min(coefficients, max_pairs):
        raise ValueError("active pair count is inconsistent")
    return _price_record(width * input_rank + output * output_rank + coefficients + output,
                         3 * coefficients,
                         width * input_rank + coefficients + output * output_rank,
                         active_pairs, output)


def cp_price(output: int, width: int, rank: int) -> dict[str, int]:
    if rank <= 0:
        raise ValueError("CP rank must be positive")
    return _price_record(rank * (2 * width + output) + output, 0,
                         rank * (2 * width + output), rank, output)


def sparse_core_curve(
    core: torch.Tensor,
    keep_counts: Iterable[int],
    *,
    ambient_output: int,
    ambient_width: int,
) -> list[dict[str, object]]:
    """Return deterministic symmetric-COO curves in folded Frobenius currency."""
    tensor = _cpu_f64(core, "core", 3)
    output_rank, input_rank, other_input = tensor.shape
    if input_rank != other_input:
        raise ValueError("core input modes differ")
    if not torch.allclose(tensor, tensor.transpose(1, 2), atol=1e-10, rtol=1e-10):
        raise ValueError("core is not symmetric in its input modes")
    requested = tuple(keep_counts)
    if any(not isinstance(k, int) or isinstance(k, bool) for k in requested):
        raise TypeError("keep counts must be integers")
    records: list[tuple[float, int, int, int]] = []
    for a in range(output_rank):
        for b in range(input_rank):
            for c in range(b, input_rank):
                value = float(tensor[a, b, c])
                energy = value * value * (1.0 if b == c else 2.0)
                records.append((energy, a, b, c))
    records.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    total = sum(row[0] for row in records)
    result: list[dict[str, object]] = []
    for keep in requested:
        if not 0 <= keep <= len(records):
            raise ValueError("keep count is outside symmetric core size")
        chosen = records[:keep]
        active_pairs = len({(b, c) for _, _, b, c in chosen})
        retained = 1.0 if total == 0.0 else sum(row[0] for row in chosen) / total
        result.append({
            "scalar_coefficients": keep,
            "active_input_pairs": active_pairs,
            "retained_core_frobenius_fraction": retained,
            "indices": [[a, b, c] for _, a, b, c in chosen],
            "price": sparse_tucker_price(
                ambient_output, ambient_width, output_rank, input_rank, keep, active_pairs
            ),
        })
    return result


def price_ladder(output: int, hidden: int, width: int,
                 down_ranks: Sequence[int], cp_ranks: Sequence[int]) -> Mapping[str, object]:
    return {
        "native": native_price(output, hidden, width),
        "down_rank": {str(rank): down_rank_price(output, hidden, width, rank) for rank in down_ranks},
        "cp_rank": {str(rank): cp_price(output, width, rank) for rank in cp_ranks},
    }
