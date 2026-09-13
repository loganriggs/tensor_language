# One private correction vector and a shared writer product

13 September 2026. This reuses the existing fixed-writer cache to simplify the
[complementary tail executor](POLYNOMIAL_TAIL_MLP10_V1_MATH.md).

For the three response vectors $v_0,v_1,v_2$, define
$P_{00}=K(v_0,v_0)$ and $P_{02}=K(v_0,v_2)$. The omitted degree-three and
degree-four coefficients in physical amplitude coordinates are

$$
N_3=-4\gamma\beta P_{00}-\gamma P_{02},
\qquad N_4=\gamma^2P_{00}.
$$

Therefore the omitted self numerator is

$$
E(a)=\frac{a^5}{2\rho(a)^2}
\left[-\gamma P_{02}+(a\gamma^2-4\gamma\beta)P_{00}\right].
$$

The linear response vector $v_1$ cancels from this correction. Only $v_2$
needs new Left/Right projections, followed by one Down write. The fixed writer
$v_0$, its two hidden projections and $P_{00}$ are shared across contexts using
`fixed_writer_products_v1.prepare_global`. This is the same function as the
previous two-tail executor, expressed without dimensionless scaling or a
duplicated fourth-degree vector. It is not a newly fitted approximation.

Actual checkpoint weights, Gaussian pristine contexts and signed amplitudes
$-2,-0.5,0,0.5,1,2$ give full-output and own-tail errors below
$4.00\times10^{-15}$. Checking the tail separately prevents its small size
relative to native output from hiding an inaccurate correction. A mismatched
writer cache raises an error.

Reused preparation takes 3.36/6.80/18.74 ms for context batches 1/16/64, versus
5.18/15.45/51.74 ms for the previous two-tail preparation: speedups
1.54/2.27/2.76×. All pass the declared 1.1× bar. One-time global preparation
took 4.05 ms in this run. It is excluded from reused timings and must be paid;
the single-context preparation saving covers it after roughly three contexts.
This setup timing is descriptive, not a stable latency guarantee.

The shared cache holds 11520 scalars and each context holds one 1152-vector.
The previous representation holds 2305 scalars per context. Thus total extra
state becomes smaller starting at ten contexts. Common upstream response
state and dense native maps remain charged on both sides. The global cache
could already be present for other consumers; no such reuse is assumed in
this standalone comparison. Execution still evaluates the native bilinear
layer, so this is a preparation/state improvement, not whole-model compression.

CPU receipt: `SHARED_POLYNOMIAL_TAIL_V1_RESULT.json`. Native implementation
replay is registered in `ops/run_shared_tail_native_v1.py`, with frozen binding
`SHARED_TAIL_NATIVE_V1_BINDING.json`: use the existing 48-template panel and
compare all prior scores within $10^{-5}$ while retaining the previous native
fidelity bars. This is not an independent fresh-text panel. Read
`SHARED_TAIL_NATIVE_V1_RESULT.json` for its authoritative terminal outcome.

The managed native replay completed in 5.09 seconds with all five predicates
true. Maximum score disagreement with the previous three-term program is
$3.81\times10^{-6}$. A separate relative-effect audit finds 0.0616% target and
0.2068% control disagreement in family 0, and exact effect agreement in the
other three families, with no material sign reversals. This checks the effect
scale rather than relying only on absolute score agreement. The native
prefix/background/suffix remain in use. See
`SHARED_TAIL_NATIVE_V1_RELATIVE_AUDIT.json`.
