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


### Corpus-shift completion and domain heterogeneity, 19:12 UTC

All four registered corpus-screen bars held. Pooled own-family removal CE effects are-0.04184/-0.04688; own-minus-other contrasts are-0.01845/-0.02243, with paired95%intervals[-0.03140,-0.00556]/[-0.03758,-0.00573]. Mean absolute control changes are0.00987/0.01219. Both branches have negative own-family means and controls below0.05in every domain. Execution took2.45seconds inside the run, with50bodyforwards/400sequences. [Result](SHARED_NODE_PARENT1_CORPUS_SHIFT_V1_RESULT.json).

The domain table limits the interpretation. In Wikipedia, branch1's own-minus-other point estimate is **+0.03023**, reversing the pooled preference; its post-result paired interval[-0.01058,0.07555]includeszero. Wikipedia minus the other three domains is+0.07021, with stratified-bootstrap95%interval[0.02704,0.11765]. This supports heterogeneity of the relative effect, not a confirmed positive reversal within Wikipedia alone. Branch0native top20capability is only0.406/0.438in discussion/biomedical domains, although the pooled capability bar passes. The registered per-domain D checked suppression signs and collateral size, not relative branch preference or per-domain capability. Preserve its pass without silently broadening it. [Post-result analysis](PARENT1_CORPUS_DOMAIN_HETEROGENEITY_V1.json).

The prospectively specified input-gating check again misses: standardized own-family amplitude differences are0.0159/0.0031, and both intervals includezero. Exact coefficient replay holds9.69e-17. [Gating receipt](BRANCH_INPUT_GATING_V2_CORPUS_SHIFT.json). We therefore have replicated, pooled token-facing suppression under a corpus shift, while context-selective input gates and domain-invariant branch assignment remain unestablished. This is no OOD/extraction/circuit promotion. No factor was refitted to these texts.

## Do the branch amplitudes carry useful context? — 19:33 follow-up

The failed family-gating checks ask whether branch strength distinguishes the two broad token families. They do not determine whether its variation *within* a family helps prediction. Two controls separate those questions.

First, the exact spherical-constant part of the scalar product is

$$
a(x)=(u^\top x)(p^\top x)
=\alpha\|x\|^2+x^\top S_0x,
\qquad \alpha=\frac{u^\top p}{1152},\quad
S_0=\operatorname{sym}(up^\top)-\alpha I.
$$

This uses only the weights, with no activation mean fitted. Its coefficient energy is 0.0564%/0.0194% of the two scalar quadratics. On the separate FineWeb and corpus-shift caches, replacing the amplitude by this radial term leaves 87.9–92.8% relative RMS error; sign agreement is only 71.6–76.6%. Exact decomposition replay holds within $1.79\times10^{-15}$. A holds, B/C miss. The products therefore have substantial nonconstant variation, but this alone says nothing about whether that variation is useful. This quadratic trace split differs from older radial-versus-tangential *residual write* studies. [Code](branch_radial_control_v1.py) · [Receipt](BRANCH_RADIAL_CONTROL_V1.json).

Second, replace an amplitude by another document's amplitude while retaining its physical writer and the entire native background:

$$
h'_i=h_i+\big[a_j(x_{\pi(i)})-a_j(x_i)\big]w_j.
$$

Four fixed donor permutations stay within the same endpoint family and, for the corpus-shift panel, the same domain. The joint arm uses the same donor for both branches. These swaps preserve each stratum's amplitude distribution while changing its alignment to individual contexts. The native tail includes final RMS normalization, the complete unembedding and the tanh cap. No factor is refitted.

V1's 32-row tail batches missed the absolute cached CE replay allowance: $1.23171\times10^{-5}$ versus $10^{-5}$. It remains an invalid instrument. V2 changed only the batch size back to the original eight; cached CE replay then became exactly zero. Its analytic FP64 tail derivative agrees with autograd within $5.63\times10^{-16}$, and identity swaps are exact. [V1](BRANCH_CONTEXT_INTERCHANGE_V1_RESULT.json) · [V2](BRANCH_CONTEXT_INTERCHANGE_V2_RESULT.json).

| Mean replacement CE cost, nats | FineWeb | Four-domain corpus shift |
|---|---:|---:|
| Branch 0 | 0.00878 | 0.01142 |
| Branch 1 | 0.01943 | 0.01943 |
| Both | 0.02863 | 0.03071 |

Higher CE is worse. All four single-branch exact costs exceed the 0.005 bar, and their descriptive paired-document intervals exclude zero: V2 A/B hold. First-order costs are 0.00689/0.01795 on FineWeb and 0.00955/0.01769 on the corpus-shift panel. FineWeb branch 0's interval crosses zero, so the stricter C prediction misses. Joint-minus-singles mean effects are 0.000432 and -0.000137 nats, respectively; this is a limited composition observation on these interventions, not a general guarantee.

These results suggest useful context alignment, especially for branch 1, even though the broad family-gating test failed. They do not establish what contextual information is represented. Swaps also change the exact target-token pairing and may add nonlinear noise cost. The panels were previously inspected, and the fixed-donor intervals do not fully capture donor dependence. The next all-donor covariance test directly addresses the four-donor sampling limitation while preserving V2's C miss. [Protocol](BRANCH_ALL_DONOR_ALIGNMENT_V1_PREREGISTRATION.md).

### Averaging every donor and controlling the exact target token

The all-donor follow-up completed at 19:33:35. For amplitude $a_i$ and local loss sensitivity $g_i=\nabla_h\ell_i^\top w$ in a stratum of $n$ documents,

$$
\mathbb E_{i,j\ne i}\!\left[g_i(a_j-a_i)\right]
=-\frac{n}{n-1}\left(\overline{ag}-\bar a\,\bar g\right).
$$

This is minus the sample covariance, so we can average all nonself donors exactly. The bootstrap now resamples paired documents within domains and recomputes the covariance, rather than conditioning on four fixed donor assignments. All numerical and B/C bars hold; explicit pairwise sums agree within $1.39\times10^{-17}$.

| Expected first-order interchange cost | FineWeb, mean [95% interval] | Corpus shift, mean [95% interval] |
|---|---|---|
| Branch 0 | 0.00801 [0.00496, 0.01107] | 0.00986 [0.00670, 0.01269] |
| Branch 1 | 0.01629 [0.01211, 0.02079] | 0.01661 [0.01240, 0.01990] |

These are the exact donor-averaged **linearized** effects, not exact full-dose losses over all donors. This follow-up addresses a plausible methodological reason for V2's FineWeb branch 0 uncertainty, while the original four-donor C miss remains. [Extraction](BRANCH_ALL_DONOR_ALIGNMENT_V1_EXTRACTION.json) · [CPU analysis](branch_all_donor_alignment_v1.py) · [Result](BRANCH_ALL_DONOR_ALIGNMENT_V1_RESULT.json).

A subsequent CPU control restricts donors further to the **same actual target token**, family and domain. This removes variation in target identity as the sole explanation of the mean alignment. Cells need at least two documents; 253/256 target-family endpoints qualify on FineWeb and 244/256 on the corpus-shift panel. Control-family endpoints are excluded from this particular check. Conditional mean linear costs remain positive: 0.01266/0.02244 nats on FineWeb and 0.01025/0.02142 on the corpus-shift panel. A/B/C hold, with pairwise identity error $6.94\times10^{-17}$. These covered-cell means have no registered uncertainty claim and should not be compared directly with the three-family averages above as an improvement. [Code](branch_exact_token_alignment_v1.py) · [Result](BRANCH_EXACT_TOKEN_ALIGNMENT_V1.json).

The stronger supported interpretation is **context-dependent amplitude alignment within output-token families**, including a positive mean under exact-token conditioning. We still do not know the represented contextual variable. A branch can suppress a token on average yet vary that suppression usefully between contexts; deletion improving selected-token CE and swapping amplitudes worsening CE are therefore compatible. These are post-result diagnostics on existing panels, not fresh OOD confirmation, standalone extraction, semantic task identification, or a full four-property circuit. The next circuit-level question is how much of this alignment comes from the shared reader versus each private partner.

## Shared-reader versus private-partner interchange — 19:42

Write the common scalar as $s=u^\top x$ and each private partner as $p_j=v_j^\top x$, so its branch amplitude is $sp_j$. Using one donor for both ports gives the exact identity

$$
\Delta(sp_j)=p_j\Delta s+s\Delta p_j+\Delta s\,\Delta p_j.
$$

The first term changes the shared reader while retaining the current partner; the second changes the private partner; the third accounts for changing both together. This mixed term comes from the **bilinear computation**, even when the downstream loss is linearized. It is not the curvature of CE. Changing the shared graph port affects both consumers; displaying each consumer's contribution does not imply that a common reader can be physically changed for only one of its uses without splitting the node.

For loss sensitivity $g_j$, all-nonself donor expectations within each family/domain stratum are $-\operatorname{Cov}(s,p_jg_j)$, $-\operatorname{Cov}(p_j,sg_j)$, and a mixed remainder that sums with these to $-\operatorname{Cov}(sp_j,g_j)$. Covariances use the sample denominator $n-1$. Explicit all-pair identities hold within $3.04\times10^{-17}$ and reproduce the earlier total exactly. [Code](branch_port_interchange_v1.py) · [Result](BRANCH_PORT_INTERCHANGE_V1.json).

| Mean first-order CE cost | FineWeb branch 0 / 1 | Corpus shift branch 0 / 1 |
|---|---|---|
| Shared reader only | -0.02110 / -0.00322 | -0.01490 / 0.00302 |
| Private partner only | -0.01726 / -0.00834 | -0.01087 / 0.00117 |
| Mixed term | 0.04637 / 0.02785 | 0.03563 / 0.01242 |
| Complete product | 0.00801 / 0.01629 | 0.00986 / 0.01661 |

The shared-reader positive-cost prediction **fails**: B/C miss, despite A passing. Complete-product context alignment therefore does not immediately establish that each isolated port is a useful independently manipulable variable. The signs persist qualitatively under the exact-target covered-cell check.

### Red-team: single-port swaps change the mean product

Whole-product permutations preserve the amplitude distribution within a stratum. Swapping just $s$ breaks its correlation with $p_j$ and changes the mean amplitude by

$$
\overline{\Delta a_j}=-\operatorname{Cov}(s,p_j).
$$

The same mean shift occurs for private-only swaps; the mixed term contributes twice the opposite shift, so a complete-product swap has zero mean change. This creates a concrete difference between the interventions. Split their linear effects exactly as

$$
\mathbb E[g_j\Delta a_j]
=\bar g_j\,\mathbb E[\Delta a_j]
+\mathbb E[(g_j-\bar g_j)\Delta a_j].
$$

No activation mean is fitted into a model or substituted for a factor; this is statistical accounting of the measured intervention. The mean-shift contribution to the shared-reader cost is -0.02251/-0.01203 on FineWeb and -0.01987/-0.00586 on the corpus-shift panel. It explains 106.7% and 133.3% of branch 0's negative total, respectively. The centered-sensitivity remainders are **0.00141/0.00881** and **0.00496/0.00888** nats. The accounting A/B/C bars pass; identities hold within $1.34\times10^{-15}$. These positive remainder means have no registered confidence claim. [Code](branch_port_mean_accounting_v1.py) · [Receipt](BRANCH_PORT_MEAN_ACCOUNTING_V1.json).

The executed red-team explains why the raw shared-port miss is not evidence that the shared reader carries no useful context. It does **not** repair that miss or establish a standalone shared-reader circuit: the remainder is an accounting term, not a separately validated executable intervention. The supported computation remains the paired product with useful context alignment. Further port-level interpretation must control the changed product distribution and test finite interventions before claiming independent extraction or selective reuse.

### Closing the independent-reader screen with private-matched donors

The next control selected one nonself donor permutation per family/domain by minimizing private-partner distance. If $P$ contains the two partner readers, the distance is $\Delta p^\top(P^\top P)^{-1}\Delta p$: the squared minimum residual-input displacement needed to change those coordinates. This metric comes from the weights; it is not data-fitted whitening. Donor selection uses existing validation states and leaves every factor fixed. It approximates holding partners constant; it is not sampling a verified conditional distribution.

Quality bars held: private-distance energy fell to 3.52%/9.73% of independent-donor expectations, while common-reader perturbation energy retained 11.88%/14.17% on FineWeb/corpus shift. Stratum-averaged absolute product mean shifts fell to 2.25–10.01% of the independent-donor values. [Donor construction and receipt](BRANCH_PRIVATE_MATCHED_DONORS_V1.json).

The finite native-tail screen replayed baseline CE exactly and passed algebra/source checks. It nevertheless **failed C**, the registered positive common-reader cost of at least 0.001 nats on both panels:

| Mean finite CE cost | FineWeb | Corpus shift |
|---|---:|---:|
| Common reader only, both consumers | -0.000224 | -0.002166 |
| Both private partners | -0.000962 | -0.002144 |
| Coherent replacement of all three ports | 0.002368 | 0.006984 |

No mean compensation or parameter fitting was applied. These are descriptive means, not significance claims. The control is adequate under its frozen distance, perturbation-size and mean-shift bars, but examines only a limited range of conditional variation. This closes the current independent-reader screen; it does not prove the reader is globally irrelevant. The useful complete-product result remains. Further tuning of these donor controls would not establish a new circuit. [Finite result](BRANCH_PRIVATE_MATCHED_PORTS_V1_RESULT.json).

The 19:51 mathematical review instead produced an exact arithmetic rewrite of the frozen graph, with parent interventions preserved before its changed internal computations. It saves 25 variable multiplications without altering the represented tensor or increasing stored coefficients. This is a compiler result, not a rescue of the behavioral screen. [Derivation, literature and executed controls](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1951.md).

### Temporal cue response does not establish tense transfer

The descriptive [context atlas](BRANCH_PRODUCT_CONTEXT_ATLAS_V1.json) shows six positive, negative and near-zero examples per branch and panel, with the last 48 context tokens, prediction target, shared/private reader values, and original deletion effect. Branch 1's positive examples suggested present-tense contexts and its negative examples included past-tense biographies. This was a hypothesis selected from inspected examples. Distinct document identifiers do not guarantee independent or nonduplicate text. No factors were fitted to these data.

We froze that hypothesis and reused the historical V6 is/was authority: 64 rows, 16 lexical groups, two tense-changing prompt families A1/A2, answer-preserving paraphrases P and unrelated controls C. Its historical FIT label belongs to the older experiment; these frozen weight-derived branches were only evaluated. This is neither a new behavior discovery nor fresh OOD evidence. [Sealed rows](FROZEN_BRANCH_TENSE_V6_ROWS.json) · [Managed result](FROZEN_BRANCH_TENSE_V6_RESULT.json).

The model/physical intervention replay passed. Branch 1's amplitude was higher for present than past on all 32 A1/A2 pairs, with paired standardized differences 3.62 and 4.10. Nevertheless, the registered screen did not pass. Per-side native capability reached only 6/8 in some A1/P present cells, below 0.85; the older pooled 14/16 capability result remains consistent with this. The new criterion is stricter, not evidence of model regression. Activation and collateral criteria remain formally gated false.

More decisively, replacing both branch amplitudes with their donor values transferred under 1% of the native donor-versus-base answer-margin change, with inconsistent signs:

| Prompt family / direction | Native donor margin gap | Two-branch swap effect | Native MLP-output-only swap effect |
|---|---:|---:|---:|
| A1 present → past | 1.89246 | 0.000878 | -0.11113 |
| A1 past → present | 1.92645 | -0.016725 | -0.33540 |
| A2 present → past | 3.46660 | 0.016457 | -0.27025 |
| A2 past → present | 3.55660 | -0.000831 | -0.71061 |

The margin is the donor's answer logit minus its foil logit; a positive effect supports transfer toward the donor. The last column is the executed red-team check: swap the **entire native last-MLP output** while retaining the base residual background. It moves the margin in the opposite direction in all four cell means. Thus the weak branch transfer does not demonstrate that factorization lost a strong positive local MLP tense-transfer signal. That signal is absent under this particular native-output swap too. It also does not establish that the last MLP is irrelevant to tense, or rescue the branch screen.

The CPU computation preserves the full state norm and native capped two-token scores. With base pre-MLP background $b$, base/donor MLP outputs $m_b,m_d$, branch writers $W$ and amplitudes $a_b,a_d$, it compares

$$
h_b=b+m_b,\qquad
h_{\mathrm{bank}}=h_b+W(a_d-a_b),\qquad
h_{\mathrm{MLP}}=b+m_d.
$$

Each margin uses $30\tanh((Uh)_v/[30\sqrt{\operatorname{mean}(h^2)+\epsilon}])$ for both tokens. A separate explanatory calculation holds the base normalization denominator fixed; its MLP swap effect is still negative in every cell mean, so normalization alone does not explain the sign. This fixed-denominator arm is not native model execution. CPU replay differs from the earlier FP32 GPU margins by at most $7.24\times10^{-6}$ nats, within the registered $10^{-4}$ tolerance. Local accounting A passes; positive-local-transfer B and branch-coverage C fail. No denominator or earlier verdict is changed. [Code](frozen_branch_tense_local_anatomy_v1.py) · [Receipt](FROZEN_BRANCH_TENSE_LOCAL_ANATOMY_V1.json).

The useful distinction is now concrete: a factor can respond reliably to a temporal cue without its output carrying the tested answer change. The current evidence supports context-sensitive products, not an identified is/was circuit. Further rescue of this particular task screen is lower priority than finding what these frozen read/write products actually compute.
