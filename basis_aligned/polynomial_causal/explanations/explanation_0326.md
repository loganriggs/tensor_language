# Current bilin18 explanation — 03:26 UTC

**Date:** 2026-08-30
**What is new here:** sections 3 through 7 explain the completed sparse-MLP1 and
quadratic-router experiments. Section 8 gives the resulting plan.

## 1. The honest short answer

We still do not have a full reverse engineering of bilin18. On the strict whole-model
ledger:

| Quantity | Current value | Meaning |
|---|---:|---|
| Native values with certified simpler replacements | 29,196,288 / 545,904,054 = **5.348%** | These values can be removed inside an executable whole-model replacement under the registered test. |
| Measured causal CE assigned to named mechanisms | 0.57968 / 5.30682 = **10.923%** | This is the fraction of a deletion-based cross-entropy gap for which we have a named causal account. |
| Still unnamed in that ledger | **4.72714 nat = 89.077%** | This is the largest quantitative gap. |
| Fully validated terminal actions | **0 / 68** | No desired circuit yet passes extraction, selective removal, and OOD transport together. |

Here **CE**, or cross-entropy, is the model's average negative log-probability for the
actual next token. Lower is better. A difference of one nat multiplies the geometric
mean probability assigned to the correct token by $e$.

The strict percentages did not rise in this update because the new MLP1 programs are
still discovery results. They did, however, reveal a materially cheaper way to express
part of MLP1 and identify the next mathematical optimization more sharply.

## 2. What MLP1 computes

At one token, MLP1 receives a residual-stream vector

$$
x\in\mathbb{R}^{1152}.
$$

Its bilinear part computes

$$
g(x)=(Lx)\odot(Rx)\in\mathbb{R}^{4608},
$$

where $L$ and $R$ are learned linear maps and $\odot$ multiplies corresponding
coordinates. Its output is

$$
y(x)=Dg(x)+b\in\mathbb{R}^{1152}.
$$

$D$ is called the **Down matrix** because it maps the 4,608 product coordinates back
to the 1,152-dimensional residual stream. The output is then added to that stream.

Deleting MLP1 in the experiments below does **not** replace the whole block by the
empirical mean. The `ZERO` control removes the input-dependent term $Dg(x)$ while
retaining the native constant Down bias $b$. It therefore asks how much next-token
performance comes from the varying bilinear write rather than the free constant.

## 3. The sparse Down replacement that worked reasonably well

The first replacement retained the native $L$ and $R$ computation but replaced $Dg$
by

$$
\widehat y(x)=c+A\operatorname{TopK}_{32}(Eg(x))+b.
$$

Definitions:

- $E$ scores 512 candidate components from the 4,608 native products;
- `TopK` keeps only the 32 largest positive component scores for each token;
- $A$ turns those sparse scores into a 1,152-dimensional write;
- $c$ is a learned constant intercept;
- the native bias $b$ is added exactly.

This is like dictionary learning or an SAE in that each token uses only 32 of 512
components. Unlike a standard activation SAE, the target is the folded MLP1 output,
and success is measured by running the entire suffix of the language model.

On 96 held-out SELECT documents:

| Arm | CE |
|---|---:|
| Native MLP1 | 2.959766 |
| Sparse $P=512$ replacement | 3.101662 |
| Zero input-dependent MLP1 write | 4.011502 |

The causal recovery is

$$
R_{CE}=\frac{CE_{zero}-CE_{replacement}}
{CE_{zero}-CE_{native}}.
$$

For $P=512$, $R_{CE}=0.865084$: the program recovers **86.51%** of the CE damage
caused by deleting the varying MLP1 write. This is substantial, but below the frozen
90% gate, so FINAL remained sealed.

Its local output $R^2$ is 0.621211. Here

$$
R^2=1-\frac{\sum\|y-\widehat y\|^2}
{\sum\|y-\bar y\|^2}.
$$

An $R^2$ of 0.62 means the replacement explains 62% of held-out write variance around
the mean. It is not the same as 62% of behavior: the whole-model CE recovery is the
more causal number.

## 4. Why making that dictionary wider was poor return

We also tried 768 rather than 512 candidate components. It improved CE recovery from
86.51% to 88.48%, but required 1,474,560 additional learned constants. Once the native
$L$ and $R$ maps that this program still needs are included, it removes only 5.55% of
complete MLP1 storage. It still misses the 90% gate.

| Components | SELECT $R^2$ | CE recovery | Replacement constants | Complete-MLP storage saved |
|---:|---:|---:|---:|---:|
| 512 | 0.621211 | 86.51% | 2,950,272 | 14.81% |
| 768 | 0.639748 | 88.48% | 4,424,832 | 5.55% |

The $P=768$ run took **97.33 seconds**. This is why the current plan does not simply
increase the dictionary width until a threshold happens to pass.

## 5. The new tensor-factorized but hybrid pre-gate replacement

The expensive part left in the previous program is hidden by the notation $Eg(x)$:
it still computes all 4,608 native bilinear products and then a dense encoder. Because
the model is bilinear, every encoder score can instead be folded exactly into a
quadratic form on the original 1,152-dimensional input.

For encoder row $e_a$, define

$$
Q_a=\frac12\left(
L^T\operatorname{diag}(e_a)R+
R^T\operatorname{diag}(e_a)L
\right).
$$

Then the component's score is exactly

$$
e_a^Tg(x)=x^TQ_ax.
$$

This equality is the important tensor-network advantage: it composes the native
Left, Right, product, and encoder operations algebraically before approximation.

We approximated each symmetric matrix $Q_a$ by its largest signed eigenmodes,

$$
Q_a\approx\sum_{j=1}^{r}\lambda_{aj}v_{aj}v_{aj}^T,
$$

so its score becomes

$$
\widehat s_a(x)=\sum_{j=1}^{r}\lambda_{aj}(v_{aj}^Tx)^2.
$$

The quadratic score bank is a small tensor program: linear projections, scalar
squares, and weighted sums. **The complete replacement is not a pure tensor network,**
because TopK32 is a discrete comparison-and-selection operation. It is a hybrid
piecewise-quadratic program. It never calls native MLP1 Left, Right, or Down; exact
call counters verified zero calls to all three in every candidate arm.

This distinction matters. The hybrid can be composed as ordinary executable code and
is differentiable away from support boundaries. On any region where the selected 32
atoms are fixed, it is an ordinary quadratic tensor contraction. But there is no one
global polynomial coefficient tensor to contract with the downstream model unless we
also carry the support-indicator logic, whose size can grow combinatorially. It
therefore does not retain the clean global algebraic compositionality sought by the
project.

`rank 8` means eight one-dimensional quadratic modes **per one of the 512 routed
components**. It does not mean that all of MLP1 is an eight-dimensional function.

## 6. New numerical result

The strengthened search completed in **17.69 seconds**. It used a randomized
eigensolver with a 64-vector search space, five power iterations on $Q_a^2$, and a QR
orthogonalization at every iteration. Its worst relative eigenpair residual was
0.0819. That is adequate for a discovery screen, though not an exact optimality
certificate.

| Modes per component | CE | Recovery versus deletion | Fraction of $P=512$ recovery retained | Complete-MLP storage saved |
|---:|---:|---:|---:|---:|
| 1 | 3.490460 | 49.54% | 57.27% | 92.58% |
| 2 | 3.358208 | 62.12% | 71.80% | 88.88% |
| 4 | 3.285760 | 69.00% | 79.77% | 81.46% |
| 8 | 3.242361 | 73.13% | 84.54% | 66.64% |

Rank 8 agrees with the exact $P=512$ router's largest component 80.92% of the time
and recovers 64.34% of its positive TopK32 support. Its routed-code relative MSE is
0.2791, while its final decoded-write relative MSE is 0.2045.

The result is mixed but useful:

- **Positive:** a hybrid program that removes all native bilinear gates and Down computation
  preserves 73.13% of MLP1's measured CE effect while removing 66.64% of complete
  MLP1 storage. This is a much more genuinely executable simplification than sparse
  activations sitting on top of the full native gate vector.
- **Negative:** ordinary unweighted coefficient similarity leaves too much routing
  error. Rank 8 retains only 84.54% of the already-imperfect $P=512$ program's causal
  recovery and is not ready for composition or FINAL.

The primary receipt is
`mlp1_pregate_quadratic_router_v3_discovery.json` one directory above this file. Its
SHA-256 is
`400bd9a78c3bd460f3bd9ba35ea524a2d628f1bca26159b7002287995c133802`.

## 7. What the mathematics changed

The algebraic folding was directly useful: it turned every pre-selection SAE score
into an explicit quadratic tensor contraction on the residual stream and made it
possible to charge the native gates honestly. It did **not** turn TopK itself into a
tensor contraction.

The experiment also distinguishes two notions of tensor similarity.

The screen minimized unweighted coefficient error,

$$
\|Q_a-\widehat Q_a\|_F^2,
$$

where $\|\cdot\|_F$ is the Frobenius norm, the square root of the sum of squared matrix
entries. This treats every possible input-space direction equally.

But the router only needs to be correct on residual states that the model actually
visits. The directly relevant score objective is

$$
\mathbb E_{x\sim\text{real MLP1 inputs}}
\left[(x^T(Q_a-\widehat Q_a)x)^2\right].
$$

This is a **fourth-moment-weighted tensor norm**: $x$ occurs four times after expanding
the square. It can prefer a different low-rank basis from ordinary HOSVD or Frobenius
approximation. We then also care about discrete TopK support and whole-model CE, so the
full training objective should combine:

1. real-state score error;
2. routed sparse-code or decoded-write error;
3. CE through the exact downstream suffix;
4. an explicit price for vectors, eigenmodes, active components, and multiplies.

This is the highest-value mathematical move now. The present result is the necessary
coefficient-space control; it is not evidence that all low-rank quadratic routers fail.

## 8. Current plan, ranked by expected return

### 1. Split MLP0 exactly into fixed token, context, and interaction tensors

Before further optimizing a discrete router, exploit bilinearity directly. If the
pre-normalization MLP0 input is token-derived state $e_t$ plus attention context $a$,
then the common RMS scale $\rho(t,a)$ gives

$$
L[\rho(e_t+a)]\odot R[\rho(e_t+a)]
=\rho^2\big[(Le_t)\odot(Re_t)+(Le_t)\odot(Ra)
+(La)\odot(Re_t)+(La)\odot(Ra)\big].
$$

These are fixed token-token, token-context, context-token, and context-context tensor
branches. No TopK is required, and the four branches sum exactly to native MLP0. The
next experiment measures their independent and joint causal effects, then factors each
branch with the structure appropriate to it: lexical hierarchy for token-token,
low-rank continuous tensors for context-context, and class-conditioned block terms for
the cross branches.

### 2. Keep empirical-fourth-moment routing as a hybrid diagnostic

Fit a fixed rank-8 score bank on FIT residual states using the fourth-moment score
loss, then measure SELECT score error and physical CE. This remains useful for asking
whether Frobenius was the wrong norm, but even a pass produces a hybrid TopK program,
not the final tensor-network compiler.

### 3. Identify finite downstream response states

Inject controlled native-minus-compressed MLP1 writes under both native and compressed
MLP0 backgrounds. Seek a small state that predicts effects on held-out documents,
amplitudes, consumers, and paired edits. This is system identification rather than
local reconstruction and targets composition failures directly.

### 4. Establish one clean terminal behavior circuit

Use interaction-resolved interventions on the known copy/induction family or another
late behavior such as capitalization. A verified terminal reader supplies a concrete
observable for asking which early MLP0/1 directions matter. More late circuits help
early-layer interpretation only if they are specific enough to serve as measured
readers, not merely correlated probes.

### 5. Jointly factor early writers and downstream readers

Learn a shared sparse dictionary or DAG in which MLP0 writes a small set of components
and multiple later modules read sparse subsets. Charge graph edges and component
parameters, and require prediction on an unseen reader or a composed edit. This is the
best route to semantic components, but it needs the reader endpoints from step 3 to
avoid an arbitrary gauge rotation.

### 6. Search consumer-common invariant blocks

Given several verified downstream quadratic forms, find subspaces they approximately
preserve together. These would be gauge-robust candidate state variables. This remains
behind steps 2--4 because the real consumer panel is not yet rich enough.

## 9. Blockers, confusing results, and what would count as progress

There is **no external blocker** at present: the checkpoint, FineWeb rows, cached data,
and GPU are available. The immediate blocker is scientific: coefficient-low-rank and
local-write objectives do not yet preserve enough causal function.

The most informative oddity is that local and causal measures disagree in both
directions. The $P=512$ program has only 0.621 local $R^2$ but 0.865 CE recovery. The
rank-8 pre-gate program has imperfect router agreement yet 0.731 CE recovery. The
downstream network ignores or compensates for many local errors, but not predictably
enough yet. This is precisely why “simple” must be validated by an ability it buys:
cheaper execution, OOD prediction, circuit extraction, selective removal, or a
composable intervention—not reconstruction alone.

The next meaningful success is therefore not another nice-looking factorization. It
is one of:

- a lower-price pre-gate program that exceeds the $P=512$ causal recovery;
- a small response model that predicts unseen composed interventions;
- a terminal circuit whose extraction and selective removal transport OOD;
- or a joint writer/reader basis that predicts an unseen downstream reader.
