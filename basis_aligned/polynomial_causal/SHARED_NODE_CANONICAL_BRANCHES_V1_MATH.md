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

## Native behavioral screen completed at 17:59 UTC

The [first native screen](SHARED_NODE_PARENT1_FINEWEB_V1_RESULT.json) passes
execution and native-capability checks, but **fails the predicted support
direction**. Removing the branches improves target likelihood rather than
damaging it. The original prediction remains failed.

| Prefix family | Remove branch0: mean CE added | Remove branch1 | Remove both |
|---|---:|---:|---:|
| Branch0 weight-token family | −0.0420 | −0.0316 | −0.0714 |
| Branch1 weight-token family | −0.0322 | −0.0543 | −0.0849 |
| Nearby control targets | +0.0103 | +0.0066 | +0.0171 |

Positive CE added is damage; negative is improvement. Native top20 fractions
are66.7% and91.7% in the two target families. Mean absolute control CE changes
are0.0150 and0.0081, below the registered0.05 limit. Native and branch0 physical
endpoint logits replay within8.31e-7 relative error. Execution used the registered
11 body forwards and88 sequences, taking1.31 seconds inside the run.

These observations suggest suppression, but do not establish two distinct
behaviors. Paired own-minus-other95% intervals are[-0.0356,0.0141] and
[-0.0584,0.0248], both including zero. The resampling units are cached prefixes;
raw source-document identity has not been independently verified.

## The sign comes mainly from the branch write

The [executed sign audit](SHARED_NODE_PARENT1_SIGN_AUDIT_V1.json) uses frozen
endpoint states and native target-token unembedding rows. Both own families
have a negative forward raw target contribution on87.5% of examples. Thus
positive token loadings in the weight-only writer list did not imply positive
contributions in context: the product of the input reads supplied the sign.

For the post-MLP state $h$ and removed contribution $\delta$, first evaluate the
target-score change at the old RMS denominator, then change to the new denominator.
Including the native tanh soft cap at both steps defines a direct score change
$d$ and an RMS correction $r$. Let $z$ denote the change in the log sum of
exponentiated logits. Then

$$
\Delta\mathrm{CE}=-d-r+z.
$$

| Own-family removal | Direct target-score gain $d$ | RMS correction $r$ | Implied log-normalizer change $z$ | CE added |
|---|---:|---:|---:|---:|
| Branch0 | +0.1203 | −0.0047 | +0.0736 | −0.0420 |
| Branch1 | +0.1255 | −0.0103 | +0.0608 | −0.0543 |

The RMS correction is smaller than the direct score gain; changing competing
token scores offsets part of that gain. The log-normalizer change is inferred
from measured CE and recomputed target scores, rather than independently replayed
over the vocabulary. FP64 accounting is exact and FP32/FP64 target scores differ
by at most3.20e-6. All registered sign-audit bars hold; they explain the failed
direction on these rows and do not validate an inhibitory circuit on new inputs.

The [separate-prefix follow-up](SHARED_NODE_PARENT1_SUPPRESSION_V1_PREREGISTRATION.md)
freezes the same factors on128 other cached FineWeb prefixes,384 endpoints. It
tests suppression as a new hypothesis and requires paired evidence of own-versus-other
branch specificity. No weights are fitted to the panel, and no OOD or isolated
sufficiency claim is made. Related MLP17 calibration and earlier quote-suppression
findings remain relevant controls, not grounds to declare this a novel circuit.

## Separating input gating from selected output weights

Different token loadings can yield apparent family specificity even if the input
read products do not prefer those families. A [frozen gating test](branch_input_gating_v1.py)
therefore compares the actual polynomial amplitudes before applying token-specific
output weights. Orient each unit full-unembedding writer toward positive mean
loading on its original token family. Its signed amplitude is then the input
product times the writer norm, with the corresponding orientation sign.

Within a prefix, compare branch0 suppression at the family0 versus family1
position, and branch1 suppression in the opposite order. The registered bars
require a paired standardized difference of at least0.25 and a positive95% lower
bound for each branch. These bars were fixed before outcomes on the separate
128-prefix panel; no reader or output coefficient is fitted to data.

The [24-prefix diagnostic](BRANCH_INPUT_GATING_V1_INITIAL.json) passes coefficient
replay to1.0e-16 but misses both gating bars. Standardized differences are−0.233
and+0.146, with both intervals crossing zero. Their overall amplitude correlation
is0.122. Thus the initial data neither establish own-family input gating nor
support calling the branches duplicate signals. The same test remains prospective
for the separate panel and will use its already-captured endpoint states.

### Separate suppression screen and prospective input-gating check, 18:26 UTC

The registered 128-prefix screen completed at 18:24:12. All three screen bars held. Removing branch 0 improved its own token-family CE by 0.04840; removing branch 1 improved its own family by 0.05736. Own-minus-other removal effects were -0.01659 and -0.03367, with paired prefix-bootstrap 95% intervals [-0.03069, -0.00271] and [-0.04901, -0.01779]. Mean absolute nearby-control CE changes were 0.00517 and 0.00507. Physical replay relative errors were below 8.97e-7. Execution used 50 body forwards, 400 sequences, 384 endpoints and 2.39 seconds inside the experiment. [Receipt](SHARED_NODE_PARENT1_SUPPRESSION_V1_RESULT.json).

This is selective suppression on separate historical FineWeb prefixes. The original support-direction prediction remains failed. It is neither OOD evidence nor isolated circuit sufficiency; native background remains installed. The two token families were selected from frozen output loadings, so stronger own-token effects can arise from the writers without different contextual input gates.

The prospectively registered input-gating test addresses that alternative. Its exact coefficient reconstruction holds at 9.70e-17, but standardized paired own-family suppression differences are only 0.0748 and 0.1536 (bar 0.25), and both mean-difference intervals include zero. Both specificity bars miss. All-context amplitude correlation is 0.0128; low correlation does not establish task-selective activation. The narrower supported statement is distinct token-facing effects of two branches sharing one reader. We have not established that their input gates recognize distinct behaviors. No fitting occurred. [Prospective result](BRANCH_INPUT_GATING_V1_SEPARATE.json).


### Exact branch-basis search, 18:33 UTC

The two branches have almost equal singular values, so the selected output basis deserves scrutiny. Write this shared-reader node as

$$
D_u(x)=(u^\top x)M x,\qquad M=OP^\top,
$$

where the columns of $O$ are orthonormal in the full-unembedding metric coordinates. Absorb the original writer norms into $P$. For any orthogonal two-by-two rotation $R$,

$$
(OR)(PR)^\top=OP^\top.
$$

Thus the entire node is unchanged, while each branch and its deletion can change. Physical token loadings are $F=UW^{-1}O$, with $W^\top W=U^\top U$, using all 50,304 rows and no centering. Then $F^\top F=I$. The normalization is essential: arbitrary rescaling must not manufacture a sparsity gain.

**Concentrating large loadings.** Existing varimax maximizes the sum of fourth powers of rotated loadings here; its column second-moment correction is constant. For token row $(a_t,b_t)$, define

$$
A=\frac14\sum_t(a_t^4-6a_t^2b_t^2+b_t^4),\qquad
B=\sum_ta_tb_t(a_t^2-b_t^2).
$$

The objective is a constant plus $A\cos(4\theta)+B\sin(4\theta)$. Therefore $\theta=\operatorname{atan2}(B,A)/4$ is a global maximizing angle. This is a closed-form restriction of the existing varimax method, not a new method family or an iterative convergence claim.

The optimum is -23.228 degrees. Fourth moment increases **20.12%**, passing the registered 10% bar. But squared token-loading overlap increases **12.86%**, failing the desired 25% reduction. The strongest verb loadings concentrate while broad overlap increases. More extreme top loadings need not mean more disjoint token support. Whole-function replay is 1.62e-15; the full-U metric identity is within 3.14e-14. [Code](branch_output_rotation_v1.py) · [Receipt](BRANCH_OUTPUT_ROTATION_V1.json).

**Minimizing overlap directly.** To red-team that miss, minimize

$$
L(\theta)=\sum_t\min\{(FR)_{t0}^2,(FR)_{t1}^2\}.
$$

Using $\min(s,t)=(s+t-|s-t|)/2$, this is equivalent to maximizing

$$
\sum_t\left|(a_t^2-b_t^2)\cos(2\theta)+2a_tb_t\sin(2\theta)\right|.
$$

Each term changes sign at a known angle. Sort those events, maintain the summed sine/cosine coefficients, and inspect each interval's endpoints and any interior maximum. This gives a global solution in $O(V\log V)$ time and $O(V)$ storage, without a model forward pass or iterative optimizer. Independent direct-angle evaluations do not beat the computed optimum; a rotated planted disjoint pair recovers overlap below 7e-33. Numeric identities hold within 6.16e-15.

The best overlap falls from **0.290313 to 0.278891**, only **3.93%**, so the registered 25% reduction bar misses even under its directly optimized objective. Fourth moment retains 109.10% of the original, passing the separate 95% retention bar. The optimum is 82.995 degrees, equivalent under column swap/sign to a small change from the original basis. The previous varimax choice had overlap 0.327647. [Code](branch_overlap_rotation_v1.py) · [Receipt](BRANCH_OVERLAP_ROTATION_V1.json).

This establishes a narrow optimization limit: a large reduction in this overlap measure is unavailable through orthogonal rotation of this frozen unit-writer subspace. It does not constrain different subspaces, shared readers, nonorthogonal representations, or graph topology. Neither rotation changes product count or storage once absorbed into the weights. Neither is adopted as a better circuit basis. Their individual deletions have not received new behavioral validation; the completed original-basis suppression and gating verdicts remain unchanged. These results favor searching beyond cosmetic coordinate rotation when seeking substantially different reusable branches.


### Frozen corpus-shift panel, 19:03 UTC

The original two branches now have a separate queued validation on four Pile domains:32documents each from Wikipedia, StackExchange, biomedical text, and legal/patent text. The Pile-CC/OpenWebText2 pools are not used. Exact text hashes exclude all2048documents from the earlier million-token panel; selected duplicates and exact previous endpoint windows are excluded too. Candidate factor discovery used weights, not this data. These source labels and exclusions do not verify pretraining disjointness or remove near-duplicates.

The fixed coverage held after8090source rows were visited. The selected biomedical documents comprise24PubMed Abstracts and8PubMed Central; legal/patent comprises24USPTO and8FreeLaw. Prefixes have190–513tokens; every test window has128inputs, and padding is never evaluated. There are384frozen endpoints and no factor fitting. This is a small validation panel, not a new million-token discovery campaign.

The pooled suppression/specificity bars match the previous test. An additional registered domain check requires negative own-family mean effects and small collateral changes for both branches in each domain. Native capability and complete effect matrices are reported separately by domain, preventing a pooled result from hiding a reversal. A prospective input-gating check uses the existing unmodified statistics and thresholds after endpoint states arrive. [Protocol](SHARED_NODE_PARENT1_CORPUS_SHIFT_V1_PREREGISTRATION.md) · [Rows/provenance](SHARED_NODE_PARENT1_CORPUS_SHIFT_V1_ROWS.json).

The runner is audited and queued after the two independent local graph refits. Its factors remain the original frozen branches regardless of those fits' outcomes. Standard enqueue reports the same two advisory lint warnings as the previous screen: actual replay checks are explicitly relative, and mean absolute collateral CE is the intended preservation metric. No bypass or changed bars. Once the receipt exists, run `branch_input_gating_v2.py corpus_shift`; this version only adds the new panel name to the existing gating computation.
