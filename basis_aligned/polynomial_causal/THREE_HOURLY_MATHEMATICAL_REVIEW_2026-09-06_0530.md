# Three-hour mathematical circuit review — 2026-09-06 05:30 UTC

## Decision

The aspectual-anchor circuit has reached an affine-recurrence boundary. The correct next object is not a generic low-rank fit to the
resid10-to-resid18 map. It is an explicit decomposition of the two empirically dominant missing new-write blocks, 12 and 14, while
retaining exact checkpoint lambda carry and the already released block11/15 source and bilinear terms.

The block12/14 ranking is presently post-outcome diagnostic, not prospective evidence: v1 failed an incommensurate control, and v2 is
an explicitly labeled sole-control repair. If the repaired run passes, it licenses conditional mechanism resolution and engineering;
it does not turn the exposed confirmation split back into an unopened holdout.

## Exact suffix recurrence

At the final-subject query, write the paired hybrid-minus-base residual difference entering suffix block \(b\) as \(\delta_b\). For
the checkpoint's scalar residual carry \(\lambda_b\) and the block's paired new-write difference \(w_b\),

\[
\delta_{b+1}=\lambda_b\delta_b+w_b,\qquad b=10,\ldots,17.
\]

Consequently,

\[
\delta_{18}=\left(\prod_{k=10}^{17}\lambda_k\right)\delta_{10}
+\sum_{b=10}^{17}\left(\prod_{k=b+1}^{17}\lambda_k\right)w_b.
\]

This identity matters operationally. Once the captured new writes are fixed, the recurrence is affine and additive across boundaries;
there is no cross-block interaction term to discover inside this operator. Nonlinearity enters when a write itself is recomputed from a
changed state, and at the exact final normalization/readout. Therefore search effort should be spent resolving the material \(w_b\)
terms, not fitting interactions that the frozen recurrence cannot contain.

## Established terms and measured residual

The valid executable program v5 resolves the upstream writer, blocks6-9 transport, block11 H3 plus MLP11, and block15 H5 plus MLP15.
The block11 source bank is determiner+period+self; block15 is period+determiner+self. In their exact released source contexts, MLP11 is
compressed by Left-change+Right-change and MLP15 by Left-change+bilinear-interaction.

Starting from the exact writer hybrid's \(\delta_{10}\), lambda carry plus only these compressed block11/15 writes recovers
0.232978 mean signed recovery. The fully dense captured recurrence recovers 0.294940, so the sparse fraction is 0.789916 and the omitted
new-write bank contributes a material 0.061962 under this operator. Removing either compressed block11 or block15 damages both A1 and
A2, so neither established term should be dropped merely to simplify notation.

## Missing-block compression and the invalid v1 control

The frozen singleton/leave-one-out score over \(\{10,12,13,14,16,17\}\) ranked the blocks

\[
12 > 14 > 10 > 13 > 16 > 17.
\]

On the exposed confirmation rows, blocks12+14 recovered 0.779652 of the all-omitted-minus-none increment and reached 0.964257 of the
all-omitted total, with positive increments and perfect direction in A1 and A2. These figures are not yet released scientific evidence
because v1 terminated invalid.

The failed predicate compared the all-omitted arm to the full `writer_two_term` output. Those are different programs: all-omitted uses
the compressed attention/MLP terms at blocks11 and 15, whereas the writer output contains their full captured native writes. The parent
dense recurrence does match the full writer to 3.8e-6 scored logit, which confirms the recurrence identity but does not imply that the
compressed recurrence must do so. A zero-forward audit localized exactly this mismatch. The only defensible repair is to remove that
incommensurate comparison prospectively, keep its 0.253535 value as a descriptive diagnostic, rerun unchanged, and preserve the
post-outcome label.

## The next decomposition

Conditional on repaired v2 passing, define at each selected boundary \(b\in\{12,14\}\)

\[
w_b=a_b+m_b,
\]

where \(a_b\) and \(m_b\) are the full captured attention and MLP paired new-write differences at the query. Execute the complete
two-factor lattice \(G_b(\varnothing),G_b(\{a\}),G_b(\{m\}),G_b(\{a,m\})\) inside one shared-capture run. The exact two-player Shapley
allocations are

\[
\phi_a=\tfrac12[(G(a)-G(\varnothing))+(G(a,m)-G(m))],
\]

\[
\phi_m=\tfrac12[(G(m)-G(\varnothing))+(G(a,m)-G(a))].
\]

Although endpoint recovery can be nonlinear because of final normalization, this lattice exactly attributes the endpoint effect of the
two intervention terms and detects cancellation. It should be run jointly for both boundaries and both families; separate jobs would
waste model loads and make cross-boundary controls harder to compare.

If one component at a boundary retains at least 80% of its full-block increment with positive A1/A2 movement, resolve only that
component next. If both are material, retain both and resolve attention by head/source terms and MLP by the same Left/Right/bilinear
response basis already used at blocks11/15. Do not introduce a learned low-rank surrogate until this exact finite lattice fails to yield
a smaller explicit term set.

## Predictive and explanatory boundary

All present suffix programs remain paired-causal and checkpoint-dependent. Their inputs include captured base/donor states; they do not
yet generate a native margin from raw text, transfer to a sealed new construction, or replace the final normalization/unembedding with a
standalone predictor. A transparent recurrence over captured terms is a real mechanistic advance, but it is not a whole-model simulator.

The next mathematical review is due around **2026-09-06 08:30 UTC**. The next hourly strategic review remains due around **06:17 UTC**.
