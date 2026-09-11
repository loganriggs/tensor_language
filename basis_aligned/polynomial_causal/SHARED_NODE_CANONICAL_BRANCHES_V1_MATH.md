# Distinct branches under a shared reader

11 September 2026. This is a weight-only analysis of the completed spectral
shared graph. It asks whether multiple consumers are algebraically different,
before assigning them behavioral labels.

For one unit input reader $u$, the node-removal function can be written

$$
D_u(x)=(u^\top x)Mx,\qquad M=\sum_g c_g p_g^\top.
$$

Writers $c_g$ are in the saved full-unembedding metric coordinates. Symmetrizing
the two input tensor indices gives the exact coefficient norm

$$
\|D_u\|_F^2=\frac12\left(\|M\|_F^2+\|Mu\|_2^2\right)
=\frac12\|M(I+uu^\top)^{1/2}\|_F^2.
$$

Therefore a truncated SVD of $M(I+uu^\top)^{1/2}$ gives the best output/partner
rank approximation while holding $u$ fixed. Transform its right factors back
with $(I+uu^\top)^{-1/2}$. This reuses the existing fixed-reader projection
algebra. Thin QR on the two or three consumer writers and partners reduces the
SVD to a two- or three-dimensional core; no dense 1152-dimensional SVD is needed.

The [receipt](SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.json) passes all three
registered bars: identities agree within 7.53e-15; 10 of 12 parents need at least
two branches for 95% of their own energy; three have second/first singular-value
ratio at least 0.5. Parent 0 needs three branches; parents 3 and 9 need only one
at that tolerance. Parent 1 divides its energy 53.67% / 46.33% between two branches.
These are ranks of individual removal functions, not 10 discovered circuits.

The [token readouts](SHARED_NODE_TOKEN_READOUTS_V1_SPECTRAL.json) annotate all
25 branches without fitting to text. Undo the saved output whitener, apply the
entire unembedding, and list positive and negative real-token loadings while
accounting separately for padding rows. Metric replay error is 1.33e-13.
Parent 1's two positive lists include, respectively, ` runs`, ` moves`, ` calls`
and ` has`, ` is`, ` was`. The scalar input product can reverse these signs;
the lists do not establish what a branch does in context.

The MLP17 and module dossiers were checked before pursuing this node. Prior
shared-reader input overlap and the earlier gerund quadratic do not establish
an alias. Related grammar findings and their control failures remain relevant.
The [frozen native screen](SHARED_NODE_PARENT1_FINEWEB_V1_PREREGISTRATION.md)
tests separate and joint removal on 24 paired historical FineWeb documents,
with native capability and collateral-effect bars. It does not discover or
refit factors using those documents.

There are two limits to this decomposition. First, canonicalization is conditional
on this reader and fitted graph; it is not stability across fitting starts.
Second, summing all standalone node-removal banks double-counts interactions
between shared parents. Whole-graph execution requires the existing pair
correction; these banks cannot simply replace the full graph by concatenation.

The native-original matched fit also completed at 17:22:16: coefficient capture
11.85428%, numeric and improvement bars held, convergence missed (fresh gradient
1.986e-4; recent relative progress 4.680e-5). The native-graph comparison remains
in progress at this note's publication. This adds no convergence or circuit claim.
