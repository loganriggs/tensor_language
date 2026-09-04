# Three-hour mathematical review — 2026-09-04 03:30 UTC

## Goal and strategy correction

The target remains a smaller transparent tensor program that:

1. predicts the relevant computation on fresh and out-of-distribution text;
2. groups pieces of different heads or MLPs when downstream computation treats them as the same variable, and splits
   a native module when its pieces have different uses;
3. composes when several recovered computations are installed together;
4. permits selective removals, swaps, and edits without damaging unrelated behaviors; and
5. is simpler under literal storage, compute, edges, states, and executable-program price.

The late-MLP 768/384 split has a smooth spectrum and no discrete boundary. It is now closed as a research direction.
Rank, retained variance, and CE preservation are controls or prices, not circuit identification.

The immediate mathematical question is instead: **how can downstream computation define a basis that groups or
splits upstream attention writes, without assuming that a native attention head is a semantic unit?**

## Exact model object

bilin18 has residual width $D=1152$, 18 blocks, 9 attention heads per block, head width $d_h=128$, and bilinear-MLP
product width $H=4608$. At query position $q$, an attention head $h$ writes

$$
o_h(q)=W_h^O\sum_{k\le q}
\left(\frac{\langle W_h^Q\bar z_q,W_h^K\bar z_k\rangle}{128}\right)
\left(\frac{\langle W_h^{Q'}\bar z_q,W_h^{K'}\bar z_k\rangle}{128}\right)
W_h^V\bar z_k .
$$

There is no softmax. Holding normalization fixed, the attention score is degree four in its query/key inputs and the
write is degree five after including the value. A bilinear MLP at a later layer computes

$$
M(\bar z)=W_D\left[(W_L\bar z)\odot(W_R\bar z)\right]+b.
$$

It is quadratic in the normalized input $\bar z$. RMS normalization makes the complete network rational rather than
a single global polynomial, but the finite bilinear identity below remains exact because it uses the two actually
observed normalized states.

For a recipient input $x$, donor input $y$, an upstream attention-5 group $G$, and a downstream reader candidate $R$,
let $S_G(x\leftarrow y)$ mean that the exact projected residual write of every member of $G$ is transplanted from $y$
into $x$. The native head boundary is only the initial intervention coordinate; it is not the proposed final basis.

Let $Y_c(a,b)$ be the donor-directed target-logit margin for circuit family $c$, where $a\in\{0,1\}$ says whether $G$
is swapped and $b\in\{0,1\}$ says whether the downstream reader is present under a specified intervention. The
two-factor mixed difference is

$$
I_{c,G,R}=Y_c(1,1)-Y_c(0,1)-Y_c(1,0)+Y_c(0,0).
$$

This scalar is an exact contraction of the four observed model executions. It contains no Taylor or rank
approximation. The primary outcome should be the pairwise target-logit contrast: probability can saturate, CE adds
full-vocabulary log-normalization curvature, and either would create interactions in the outcome transform that are
not needed to answer whether the target computation changed. CE and full-vocabulary RMS remain collateral metrics.

## Symmetries and what is operationally identifiable

Inside one value/output channel,

$$
v\mapsto Gv,\qquad W^O\mapsto W^OG^{-1}
$$

leaves the projected residual write invariant. Paired Q/K changes of basis that preserve the dot product, exchange of
the two multiplied QK branches, bilinear-product permutations, and reciprocal product-coordinate scalings give the
other familiar gauges. Therefore raw Q, K, V, or MLP-product coordinates are not meanings by themselves.

Arbitrary cross-head rotation is not generally an architectural gauge because different heads compute different
position-dependent scalar patterns before their writes are added. Two pieces from different heads may nevertheless be
**operationally equivalent** on a registered task family if every registered downstream reader and intervention gives
them the same held-out response profile. This motivates the response tensor

$$
\mathcal I_{c,g,r,d},
$$

whose indices are circuit family $c$, upstream group or sub-head piece $g$, downstream reader $r$, and transplant
direction $d$. A grouping is a claim about equality or a frozen simple transformation of slices of this tensor on
held-out data. It is not a claim about cosine similarity of raw head outputs.

## What the multiple-mediator theorem gives us—and what it does not

Vaidyanathan et al. prove that multi-component activation-patching effects contain single-component interactions and
cross-component interactions. Their cross-interaction for a component set is the Boolean-lattice Möbius coefficient of
the intervention outcomes ([primary preprint](https://arxiv.org/abs/2606.27510)). This maps exactly to $I_{c,G,R}$
for our fixed two-component intervention lattice. It explains why single-head effects cannot determine whether two
components are redundant, complementary, or conditional on one another.

The theorem solves the accounting problem for the specified interventions: the four-cell coefficient is uniquely
defined, and a complete subset lattice reconstructs all measured coalition outcomes. It does **not** prove that $R$
mediates $G$. If $b=0$ means deleting the complete reader, $I_{c,G,R}$ only says that the causal effect of swapping $G$
depends on whether $R$ is present. Reader deletion can create an internal performance floor, disturb parallel paths,
or change later nonlinear operating points. Calling that mediation would overstate the result.

The rejected attention-5 design made exactly this mistake. Existing induction work did not show that L8H4 reads the
projected output of attention 5. R459 transplanted an L5H5 scalar score into L8H4 while retaining L8H4's payload; it did
not patch L5H5's residual write into L8H4. Existing pending-opener evidence shows that complete final-position L13H8
is a causally live site, but not that attention 5 supplies its relevant state. The proposed reader choices therefore
lacked the required upstream-to-reader link.

Causal-abstraction theory provides the stronger target: an interpretation is supported when interchange interventions
commute with a declared high-level causal model, rather than merely correlating at the observed state
([Geiger et al.](https://arxiv.org/abs/2301.04709)). Recent work on diagnosing causal abstractions further suggests
partitioning input pairs by their interchange success and using the failures to discover missing high-level variables,
rather than averaging them away ([Li et al.](https://arxiv.org/abs/2605.02234)). That is directly useful for separating
near/far induction, one/multiple predecessor, bracket type, and surface-form regimes.

## Exact translation of a reader-state change into bilinear weights

The cleanest route is to make a later bilinear MLP the first reader candidate. Run the recipient normally and with the
upstream group swapped. At MLP $R$, save the two *normalized* inputs

$$
\bar z_0,\qquad \bar z_1,
$$

and define $\delta=\bar z_1-\bar z_0$. Then the exact change in the MLP write is

$$
\begin{aligned}
M(\bar z_1)-M(\bar z_0)
=W_D\big[& (W_L\delta)\odot(W_R\bar z_0)\\
          &+(W_L\bar z_0)\odot(W_R\delta)\\
          &+(W_L\delta)\odot(W_R\delta)\big].
\end{aligned}
$$

The first term changes the left input while holding the right input at the recipient value; the second does the
opposite; the third is the finite interaction requiring both changes. This is not a low-rank approximation. The three
terms must exactly reconstruct the observed finite MLP-write change before any causal interpretation.

The output vectors are invariant to joint permutation and reciprocal rescaling of MLP product coordinates. Exchanging
the two bilinear sides swaps the first two labels but leaves their unordered pair and the joint term unchanged. Thus a
claim such as “the upstream variable reaches this MLP mainly through one side” is meaningful only up to the global
left/right exchange; the complete three-part response is the safer gauge-aware object.

Rung 485 already implements this finite left/right/joint construction for an MLP0-to-MLP1 path. The campaign should
reuse and generalize that tested algebra rather than write a new rank-reduction system.

## Executable consequence: reader liveness, necessity, and sufficiency must be separate

For each candidate upstream group and behavior, the next instrument should have three stages.

### 1. Reader-liveness screen

On FIT pairs only, perform the upstream swap once while caching all candidate later MLP normalized inputs. For each
candidate reader $R$, measure $\delta_R$ and the first-order target-margin contraction

$$
s_R=\nabla_{\bar z_R}Y\cdot\delta_R.
$$

The gradient is only a cheap screen. A candidate is eligible only if the physical upstream swap changes its state on
both document halves and the sign of $s_R$ agrees with the actual target-margin movement. The reader list and selection
rule are frozen before SELECT. One forward/backward collection can screen many readers, so semantic dataset design and
review—not GPU arithmetic—remain the main cost.

### 2. Reader-state clamp for necessity

Under the upstream swap, clamp the selected reader's normalized input back to the recipient state $\bar z_0$, while
leaving the rest of the swapped computation live. Compare the upstream-swap effect with and without this clamp.
This tests whether the swap's behavioral effect requires the change reaching that reader. It is more specific than
deleting the reader, because the reader retains its native recipient computation instead of being removed entirely.

The claim is still “necessary at this registered interface,” not “all causal influence is mediated here”: parallel
paths and interactions remain possible.

### 3. Reader-state injection for sufficiency

On the native recipient run, inject only the observed reader-state difference $\delta_R$ while leaving the upstream
group native. If this reproduces the signed donor-directed movement on held-out pairs, the reader-state change is
sufficient at that interface. Apply the exact bilinear identity above to split the resulting MLP write into left,
right, and joint weight contractions; then physically inject each term and their factorial combinations.

A reader-defined upstream group is identified only when the same frozen clamp/injection relation:

- predicts held-out and OOD pairs;
- works in both transplant directions;
- distinguishes answer-changing from active answer-preserving families;
- survives document halves and semantic subfamilies; and
- yields a reusable response profile across at least two behaviors or a stable within-module split for one behavior.

## Cost, norms, and falsifiers

The initial reader atlas stores only per-row norms, target-gradient contractions, chosen-reader states needed for the
next stage, hashes, and call audits. It does not store every hidden state. The approximation norm for the exact
bilinear reconstruction is maximum absolute vector error at the existing numerical threshold plus relative squared
error; behavioral decisions use donor-directed margin with document-cluster bootstrap intervals. CE, KL, top-1
changes, and the 62-circuit panel are collateral diagnostics, not selection labels.

This route is killed if:

1. the upstream swap does not reliably change any later reader state;
2. a gradient screen does not predict even the sign of the observed movement on held-out FIT halves;
3. clamping the selected reader state leaves the upstream behavioral effect unchanged;
4. injecting the reader-state change does not recover the signed effect;
5. the finite bilinear terms fail exact reconstruction; or
6. the same candidate moves answer-preserving or unrelated circuit controls comparably to the target.

The previous attention-5 head-group × L8H4/L13H8 deletion design is therefore blocked. Its advertised 5,847-forward
price was also incorrect under the existing separate-direction convention; the old factorial would cost 15,738
forwards, while the corrected clamp/injection design needs a new exact manifest before pricing.

## Decision and immediate action

The highest-information live action remains R593, because it repairs the invalid R592 instrument without changing its
science and directly tests the induction selector/content factorization. A different-agent exact review is in progress.

After that boundary, the mathematically preferred cross-head route is not another rank or output-similarity sweep. It
is a reusable **upstream-swap → reader-state atlas → reader clamp/injection → exact bilinear weight expansion**.
This directly addresses computational specification, cross-head grouping, within-module splitting, held-out
prediction, selective manipulation, and weight translation. The interaction-response tensor can later group upstream
pieces by downstream operational equivalence, but only after the clamp and injection experiments establish that its
reader axis is physically live.

