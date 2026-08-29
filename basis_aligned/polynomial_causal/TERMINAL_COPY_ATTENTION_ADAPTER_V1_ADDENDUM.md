# Terminal copy attention adapter v1 — execution addendum

**Status:** source implementation and known-answer tests only; the terminal-copy model
run remains **NO-GO**. This addendum closes the per-head tensor-formula implementation
gap, not the fresh-row, scorer, checkpoint, or result-lifecycle gates.

## Exact object

For one attention site, the adapter copies the six bias-free projection matrices
`q`, `k`, `q2`, `k2`, `v`, and `c_proj`, plus the learned value-bus scalar and rotary
frequencies. It retains no reference to the native module. Given the live site-entry
state, it evaluates the checkpoint's exact two-QK formula:

$$
P_{h q k}=
\left(\frac{\langle \operatorname{RoPE}(\widehat q_{qh}),
\operatorname{RoPE}(\widehat k_{kh})\rangle}{128}\right)
\left(\frac{\langle \operatorname{RoPE}(\widehat q'_{qh}),
\operatorname{RoPE}(\widehat k'_{kh})\rangle}{128}\right),
$$

with causal entries retained and future entries set to zero. Hats denote the native
per-head RMS normalization. There is deliberately no softmax or row normalization.
The value at later layers is the native learned mixture of the current site's value
and the shared block-0 value bus.

The implementation also preserves the native contraction layout: it first forms a
`[batch, head, query, d_head]` value result, then transposes and materializes the
contiguous `[batch, query, head, d_head]` tensor consumed by `c_proj`. The spent v1
checkpoint check showed that requesting the transposed layout directly from einsum is
real-number equivalent but not bfloat16-kernel equivalent.

Before the output projection, each head occupies a disjoint 128-column slice. If
$z_h$ is head $h$'s mixed value result and $W_O^{(h)}$ is the matching column block of
`c_proj`, the physical residual-stream write attributed to that head is

$$
w_h=z_h\left(W_O^{(h)}\right)^\top.
$$

Therefore the native attention write has the exact additive certificate

$$
w_{\mathrm{attention}}=\sum_{h=0}^{8}w_h.
$$

This identity is the reason per-head extraction/removal is principled here: the
intervention changes a physical additive residual write after all nonlinear QK and
value computations, rather than pretending the heads are independent upstream.

## Ownership and sealing

`terminal_copy_attention_adapter.py` clones every source tensor and charges all of
them in its price receipt. After construction it makes zero native projection calls.
A transaction returns cloned sums for requested head sets, records the requested sets
and all-head recomposition error, and revokes its internal per-head tensor on exit.
This prevents a collector from retaining a mutable alias and silently changing a
later arm.

## Known-answer boundary

The synthetic native implementation covers both the block-0 value-bus creation and a
later attention call reusing that bus. Tests require bit equality of the adapter's
unpartitioned full write with the native formula. The separately projected head sum
must recompose within relative error $10^{-6}$ because splitting `c_proj` changes
floating-point accumulation order even though the real-number identity is exact.
Tests also require additivity of disjoint head subsets within that tolerance, zero
native calls after cloning, price accounting, clone non-aliasing, transaction
revocation, and fail-closed handling of projection biases or invalid head sets.

Passing these tests does **not** establish checkpoint identity on CUDA. The future
source-closed collector must perform a pre-outcome native-vs-adapter replay at every
named site, bind the observed tolerance, prove hook cleanup, and preserve any failure.
The remaining launch blockers are fresh four-role rows, a frozen scorer/bootstrap
authority, checkpoint/source authority, an explicit omission authority for the
optional late-MLP screen (or its adapter), and an empty create-only result namespace.
