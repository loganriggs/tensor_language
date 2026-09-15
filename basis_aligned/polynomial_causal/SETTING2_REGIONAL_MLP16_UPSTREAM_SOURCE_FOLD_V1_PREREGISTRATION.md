# Setting2 regional MLP16 upstream-source fold V1

## Question

The task-matched regional census identified earlier-residual × MLP16 (`ep`) as
the largest MLP17/unembedding interaction on all four construction families.
Which propagated embedding, attention, or MLP write inside that earlier residual
supplies the interaction?

For blocks $0,\ldots,17$, let $\lambda_{l,0}$ be the learned carried-residual
coefficient and $\lambda_{l,1}$ the embedding-reinjection coefficient. Capture the
actual attention and MLP writes through block 16 at the regional prediction token.
Expand the raw block-17 input as

$$
r_{17}=c_E x_0+
\sum_{l=0}^{16}c_l\left(a_l+m_l\right),
$$

where

$$
c_l=\prod_{j=l+1}^{17}\lambda_{j,0}
$$

and $c_E$ follows the exact recurrence
$c_E^{(l)}=\lambda_{l,0}c_E^{(l-1)}+\lambda_{l,1}$. The propagated MLP16 write is
$p=c_{16}m_{16}$; every other term sums to the earlier residual $e$.

Fold each individual source $s\subset e$ through the symmetrized MLP17 cross term
with $p$, the native shared input denominator, and each row's UK-minus-US
unembedding reader. Rank the 34 source terms by their 48-pair normalized cue-change
norm.

## Frozen predictions

1. **Instrument:** propagated sources reconstruct the raw block-17 input to
   relative error at most $10^{-5}$, and their folded source × MLP16 terms sum to
   the direct `ep` term to relative error at most $10^{-8}$.
2. **Concentration:** retaining the five highest change-norm sources reconstructs
   the `ep` cue-change vector with relative error at most `.50`.
3. **Earlier attention involvement:** at least one attention-layer source has
   change-norm ratio at least `.10` relative to the complete `ep` change.
4. **Stable leading source:** the top source has global ratio at least `.15` and
   ratio at least `.05` in every construction family.

The ranking and top-five set are descriptive selection on already-open regional
activations. They license a separately frozen causal test but are not circuit
identification. Report coefficients, all source ratios/alignment, family ratios,
cancellation, and the top-five replay. No fit, behavior/logit outcome, new rows,
rank search, gradients, parameter updates, or quantization. Full source generation,
normalizers, native weights, bias, residual output, final RMS, and softcap remain
charged.
