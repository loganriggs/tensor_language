"""MDL accounting + distortion conventions for qk_mdl — FROZEN as of Tier 0.4
(2026-07-15). Per spec §4/§6: any change to these invalidates prior numbers and
must be announced in LOG.md with affected tables rerun. Ratios between codebooks
at matched epsilon are the headline quantities (robust to the bit conventions).

DESCRIPTION LENGTH (bits):
  - continuous parameter (block mean, factor entry, Fourier coeff): 32 bits each
  - discrete choice among n options (partition label, support index): log2(n) bits
  - exceptions (entry stored exactly): 32 + log2(#entries) bits each
  DL(codebook fit) = 32 * n_floats + sum(discrete bits) + exception bits.

DISTORTION (weight/matrix level, used by the synthetic battery and any folded-
matrix fit): relative squared Frobenius error
  D(Mhat) = ||Mhat - M||_F^2 / ||M||_F^2
No centering is applied (tick 0: the no-softmax models have NO per-query gauge).

DISTORTION (model level, for real heads; RESOLVED by Logan 2026-07-15: "MSE and
CE delta seem good for now. Would highlight the CE delta one"):
  - HEADLINE / binding audit: downstream delta-CE with the compressed head
    patched in. Tables are gated and ranked by delta-CE.
  - search-loop / secondary: relative pattern MSE over a fixed eval token
    batch, E||Phat - P||^2 / E||P||^2 (cheap enough for inner k/r searches).
  - epsilon_pattern for the search loop is CALIBRATED against delta-CE: chosen
    so the SVD baseline at that epsilon has comfortably small delta-CE, then
    frozen; every reported row carries both numbers.

EPSILON: battery runs at eps = 1.5 * (plant noise floor). For real heads, eps is
calibrated in Tier 1 so that full-rank-minus-one SVD is comfortably inside
(spec §4), with DL-vs-eps curves reported for headline tables.
"""


def dl_bits(n_floats: int = 0, discrete_bits: float = 0.0,
            n_exceptions: int = 0, exception_pool: int = 1) -> float:
    import math
    exc = n_exceptions * (32 + (math.log2(exception_pool) if exception_pool > 1 else 0))
    return 32 * n_floats + discrete_bits + exc


def dl_svd(r: int, n_rows: int, n_cols: int) -> float:
    """Truncated SVD: r left factors + r right factors + r singular values."""
    return dl_bits(n_floats=r * (n_rows + n_cols + 1))


def dl_sparse_dict(n_atoms: int, dim: int, nnz: int, include_bias: bool = True) -> float:
    """Overcomplete sparse dictionary: atoms (+ bias) at 32 bits per float, then per
    nonzero: a 32-bit coefficient + log2(n_atoms) support-index bits. Matches the
    qk_sae_control.bits_dict convention (Phase 0)."""
    import math
    return dl_bits(n_floats=n_atoms * dim + (dim if include_bias else 0) + nnz,
                   discrete_bits=nnz * math.log2(max(n_atoms, 2)))


def dl_bicluster(k_r: int, k_c: int, n_rows: int, n_cols: int) -> float:
    import math
    return dl_bits(n_floats=k_r * k_c,
                   discrete_bits=n_rows * math.log2(max(k_r, 2))
                   + n_cols * math.log2(max(k_c, 2)))


def dl_toeplitz_fourier(n_modes: int) -> float:
    """c(delta) as mean + n_modes complex Fourier coefficients."""
    return dl_bits(n_floats=1 + 2 * n_modes)


def dl_toeplitz_full(n_rows: int, n_cols: int) -> float:
    return dl_bits(n_floats=n_rows + n_cols - 1)
