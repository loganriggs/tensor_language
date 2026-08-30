# Plain-language project update — 04:28 UTC, 2026-08-30

## UPDATE STARTS HERE

This update closes the MLP0 token/context section and starts the ten-circuit campaign.
The main scientific result since the previous explanation is that we now have one
very strong, genuinely tensor-native induction circuit on discovery data. The main
engineering result is a reusable way to reconstruct an attention layer from stored
tensors and globally keep or remove fixed heads without calling the native attention
module.

We do **not** yet have ten finished high-quality circuits. We have ten separately
specified behavioral targets, one newly strong circuit, one useful but nonspecific
primitive, an exact successor experiment ready for its production executor, and three
more campaigns now being built in priority order. This document distinguishes an
implemented experiment from a numerical result.

## 1. What finished about MLP0

Let the input to MLP0 before its RMS normalization be

$$
x=e_t+a,
$$

where $e_t$ is the contribution fixed by the current token and $a$ is the context
written by attention. RMSNorm multiplies the whole sum by one scalar
$\rho(t,a)$. The bilinear MLP gate is

$$
(L\rho x)\odot(R\rho x).
$$

Expanding it gives three exact branches:

$$
\begin{aligned}
TT &= \rho^2(Le_t)\odot(Re_t),\\
X  &= \rho^2\left[(Le_t)\odot(Ra)+(La)\odot(Re_t)\right],\\
CC &= \rho^2(La)\odot(Ra).
\end{aligned}
$$

`TT` means token-by-token, `X` means the two token-by-context cross terms, and `CC`
means context-by-context. They sum back to native MLP0 up to numerical precision.

On 96 held-out SELECT documents, their average causal CE benefits were:

| branch | CE benefit |
|---|---:|
| context-context (`CC`) | 1.1778 nat |
| token-token (`TT`) | 0.9281 nat |
| token-context (`X`) | 0.4008 nat |

Here CE is cross-entropy: lower is better, and a positive “benefit” means the model
becomes worse when the branch is absent. These three numbers do not add independently.
The extra pair interaction was `+1.7216` nat for `TT` with `X`, but `-1.1537` for
`TT` with `CC` and `-1.0328` for `X` with `CC`. The remaining three-way interaction
was only `+0.0244`.

The best current picture is therefore not three isolated modules. It is a lexical
token structure coupled strongly to a token-context interaction, plus a large
continuous context tensor that overlaps both. This is why the next MLP0 factorization
should share factors across branches or use a sparse hierarchy/DAG; fitting three
unrelated compressors would discard the largest interaction we measured.

This section is closed. Its registered two-hour cutoff is `05:46:40Z`, and a repaired
detached alarm touches the deadline sentinel then. Circuit work began well before the
cutoff rather than using the whole allowance.

## 2. What counts as a high-quality circuit

A circuit is not high quality merely because deleting a head changes a convenient set
of tokens. We are tracking six increasingly strong levels:

1. a candidate behavior;
2. causal localization to components;
3. a repeatable behavioral description;
4. upstream readers/writers and a mathematical mechanism;
5. a composable executable algebra; and
6. recursive reduction to token, position, and earlier-program primitives.

Separately, it must pass four practical tests:

- **extraction:** the proposed circuit alone restores the behavior from a background
  where its full owner components were deleted;
- **selective removal:** deleting the proposed operation harms the target behavior but
  has a small, bounded effect on matched controls and global text;
- **OOD transport:** the same frozen program and effect predictions work on held-out
  token identities, document types, or domains; and
- **execution closure:** the candidate does not secretly call the native component or
  use labels, parsers, argmax, TopK, or other unpriced routers while executing.

Evaluation masks may use a parser to decide which positions to score. They may not tell
the candidate which edge or component to change. That distinction is especially
important for bracket matching.

## 3. The previous-token result

The fixed previous-position tensor for layer-0 head 3 is highly extractable. After
deleting the native head, restoring only the offset-minus-one contraction recovered
`0.9421` of the head's CE effect. Recovery was nearly unchanged on unseen bigrams:
`0.9417` versus `0.9442` on seen bigrams. Shifting the same tensor to the wrong offset
recovered only `0.1529` or zero.

However, its removal damage was broad: about `+0.0625` nat on the nominal target and
about `+0.0632` on the matched self-attention control. Its specificity interval
included zero. The correct conclusion is that previous-token lookup is a real,
OOD-stable shared primitive, but not an independently removable semantic behavior.
This is useful upstream infrastructure for other circuits, not a claim that one
behavior owns the head.

## 4. The induction equality tensor

The strongest new circuit uses four heads: `L5H5`, `L7H3`, `L8H3`, and `L8H4`.
For query position $q$ and key position $k$, it uses the fixed relation

$$
M_{qk}=\langle e_{t_q},e_{t_{k-1}}\rangle\mathbf 1[1\leq k\leq q].
$$

$e_{t_i}$ is a one-hot vector for token identity. The inner product is one exactly
when the current query token equals the token immediately before key $k$. Thus key
$k$ contains the token that followed an earlier occurrence of the query token. For
head $h$, the extracted write is

$$
z_q^{(h)}=\sum_k M_{qk}A_{qk}^{(h)}v_k^{(h)},
$$

where $A_{qk}^{(h)}$ is the head's native continuous attention score and $v_k^{(h)}$
is its value. Every matching source is summed. There is no nearest-source rule,
argmax, TopK, or `has_match` branch.

On 192 SELECT documents:

- removing only equality-fetch edges damaged induction targets by `+0.51225` nat;
- target-minus-matched-control specificity was `+0.55251` nat;
- off-target damage was only `+0.006264` nat, with a 95% upper bound of `0.009824`;
- deleting all four heads and restoring only this tensor recovered `0.97397` of their
  target benefit, with interval `[0.94789, 0.99479]`; and
- a same-shaped cyclic permutation of vocabulary identity recovered approximately
  zero.

The replacement ran sequentially on the live residual stream, so later extracted
heads consumed the effects of earlier ones. Its analytical replay was bit-exact and
candidate arms made zero native Q/K/Q2/K2/V/O calls at the affected layers.

This explains an earlier confusing result. Replacing the four entire heads by means
had too much collateral because the heads also perform unrelated services. Removing
only the equality interaction isolates the induction service.

### Why held-out FINAL/OOD did not run yet

The original cached FINAL and code-OOD rows were frozen under a receipt that grants
access only if the older mean-replacement candidate passed. It did not. The new tensor
candidate cannot silently inherit that license. An independent audit caught this
before either outcome container was loaded. It also found that the first terminal
runner needed stronger atomic publication, persisted sufficient statistics, complete
call counts, and observed point estimates rather than averages of bootstrap draws.

That is a lifecycle NO-GO, not a negative scientific outcome. The SELECT result above
stands. The repair path is to freeze fresh disjoint natural and code roles for the new
candidate and reuse the mature transactional terminal infrastructure. No protected
FINAL/OOD outcome was opened.

## 5. Reusable attention intervention now added

The stored attention program already evaluated Q, K, Q2, K2, value mixing, causal
masking, RoPE, and output projection without retaining the native attention module.
It now also accepts a constant vector $w\in\mathbb R^H$ on the head index:

$$
o_{qhd}\leftarrow w_h o_{qhd}.
$$

Choosing $w_h=0$ for one head and one elsewhere deletes that head globally. This is a
diagonal tensor contraction. It does not inspect the token, context, target label, or
evaluation mask. The stored cost explicitly includes the $H$ scalar weights. The
identity, deletion, malformed-input, runtime, and facade suites pass 22 tests.

This small primitive is important because bracket and newline campaigns can now reuse
one exact intervention implementation instead of writing new hooks for every head.

## 6. Ordered successor status

The successor program targets layer-8 head 7. Its exact value/output map has two
sources: the 1,152-dimensional current residual stream and the saved **already
projected 128-dimensional** layer-0 H7 value bus. For fixed attention scores $A$, it
computes

$$
y_q=O\sum_k A_{qk}\left[(1-\lambda)Vx_k+\lambda b_k^{(0)}\right],
$$

where $b_k^{(0)}=V_0x_k^{(0)}$ is the shared bus. A pre-outcome audit of the interface
caught that the first successor scaffold unnecessarily treated this bus as another
1,152-dimensional state. The correct folded map is

$$
O\left[(1-\lambda)V\mid\lambda I_{128}\right],
$$

not a map that re-applies $V_0$. This matters both mathematically and for price. With
the layer-0 value producer charged once as shared model infrastructure, a rank-$s$
successor head costs

$$
4(128)(1152)+s(1152+128+1152)=589{,}824+2{,}432s
$$

stored floats. That is 609,280 at rank 8 and 901,120 at rank 128. The structured native
head itself costs 884,736 floats on the same shared bus, so rank 128 is an exact control,
not a compression; ranks below roughly 122 are cheaper. A standalone end-to-end claim
must additionally charge the shared layer-0 value producer once, not once per consumer.

The source-closed discovery has 17 frozen arms: ranks
`8,16,32,64,96,128`, same-price spectral nulls, and rank-128 source omissions. Prices
in the original scaffold are being corrected to the shared-bus currency before any
outcome is opened. A value-only object conditional on teacher attention scores is
explicitly not counted as an executable extraction.

Nineteen toy/source tests pass. No SELECT or OOD outcome has been opened. The remaining
implementation work is a source-closed layer-8 executor, transactional result owner,
and bootstrap finalizer. This is an implemented experiment scaffold, not a numerical
success.

## 7. The ten behaviorally distinct targets

Each has its own file under
`basis_aligned/bilinear_quotient/circuits/campaign_2026_08_30/`:

1. previous-token/bigram lookup;
2. induction copy;
3. ordered successor;
4. matched bracket closure;
5. article choice;
6. newline/structural boundary;
7. copied-entity continuation;
8. novel capitalization/register;
9. quote parity/closure; and
10. numeric/unit/date formatting.

These were chosen as distinct endpoints. The old circuit census has now produced a
particularly relevant hierarchy/DAG clue, but it is not yet being counted as five
finished mechanisms. Five layer-8 census directions are `0.894`-parallel and share one
direction that explains `91.6%` of their variance. Projecting that shared direction
out leaves near-orthogonal residuals (mean absolute cosine `0.359`), and four of five
residual directions are causally more selective for their own behavior than for the
other four. For `r.23.2.3`, residual-direction concentration rises from `3.649` to
`4.742` after removing the shared part.

This is the structure we hoped a sparse DAG could have: one common attention-8
substrate plus several smaller behavior-specific children. It also explains why the
first rank-1 probe was confusing—the large common direction caused damage but little
selectivity, masking the residual mechanisms. The exception `r.11.1.1` has no selective
residual yet and may really be a shadow of another leaf. A separate output-space
clustering did not reproduce the component-space grouping and was correctly rejected
by its preregistered validation gate. We therefore treat the shared-plus-residual
factorization as a promising entry point, not as five certified circuits.

## 8. Next work, in execution order

1. Finish the fresh-role transactional induction terminal and run natural/code OOD.
2. Complete the exact successor executor and run its frozen rank/source factorial.
3. Bracket closure: reconstruct all of layer-13 attention from stored tensors, globally
   remove H8 using the constant head projector, and score bracket stacks only as
   evaluation cells. Existing evidence is unusually strong: full H8 deletion costs
   `+0.8254` target/`+0.00376` global and removing the true-match edge costs `+0.6890`.
4. Newline: use the same executor for the fixed five-head crew. Existing damage is
   `+0.6166` on 7,397 newline targets and `+0.0049` elsewhere. Known TopK writer-pair
   compression is nonspecific and will not be retried as though it were positive.
5. Article choice: first falsify or confirm the compact rank-16 MLP0 output subspace on
   disjoint data against three equal-rank random projectors, then compose the exact
   front attention chain only if it passes.

Copied-entity, novel capitalization, quote closure, and numeric formatting remain in
the ten-circuit ledger but follow these higher-return campaigns. Quote machinery is
shared with bracket H8, capitalization is contaminated by copy service, and the old
numeric screen is underpowered with a failed null.

## 9. Current blockers

There is no missing checkpoint, FineWeb cache, code data, or GPU. The blockers are
specific and internal:

- the induction candidate needs fresh licensed terminal rows and the mature atomic
  lifecycle;
- successor needs its production tensor executor;
- bracket/newline need the fixed head projector integrated into that lifecycle; and
- ten high-quality circuits require ten independent removal/OOD claims, not ten names.

Work is parallelized across the induction lifecycle repair, successor, bracket, and
newline paths. Nothing above requires user input, and no task is waiting for it.

## UPDATE ENDS HERE
