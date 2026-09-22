# Exact graph edit: completing products of pair features

22 September 2026, 02:08 UTC. The lean conditional program contains atoms

$$
pq+a p+b q,
$$

where p and q are quadratic features (each a product of two affine input projections), and a,b are fixed scalar correction coefficients. The exact identity

$$
pq+a p+b q=(p+b)(q+a)-ab
$$

allows us to use one root product of shifted quadratic features. The term minus ab is constant and can be absorbed into the output bias after multiplication by each atom's output vector. This is an exact arithmetic rewrite; it introduces no new approximation, fit or change of target.

For 512 atoms, this removes 1,024 scalar coefficient multiplications per state. The variable-product count remains 1,536, additions remain unchanged, and stored floating coefficients remain 848,912 including the common writer. Coefficient multiplications fall from 828,416 to 827,392, excluding the common writer. The structural saving is only about 0.12% of that count; it should not be presented as a major compression result. The scalar coefficients become pair biases, so they are still stored.

Fifteen toy checks cover five structures and all three pairings: generic random factors, zero corrections, repeated factors, large corrections and constant-only input features. All pass. On all 16,384 opened native inputs, compiled programs agree with the original lean programs to about 1e-15 relative error in float64 and 1.1e-7 in float32. Constants are compiled in float64 before casting, reducing rounding loss. Source artifacts and helpers used by queued native tests are unchanged.

Seven rotating-order warm CPU rounds on two threads yield speedups of 1.06/1.07 times for batch one, 1.018/1.020 for batch64 and 1.010/1.014 for batch2048 across the two seeds. These small timing differences are descriptive on a shared machine, not a demonstrated GPU or end-to-end model improvement. Fewer pointwise operations may matter more than their share of scalar multiply counts, but the benchmark does not isolate that mechanism.

This is a concrete instance of the proposed second stage: rewrite the arithmetic graph after decomposition, preserving its function. It neither fixes inherited small-output errors nor identifies meanings for the quadratic intermediates. The compiled form is available through a separate helper; existing queued intervention tests retain their original evaluator, avoiding any mid-run mutation.

[Compiler and evaluator](shifted_pair_cp.py) · [Controls and benchmark](audit_shifted_pair_cp.py) · [Results](SHIFTED_PAIR_CP_V1.json).
