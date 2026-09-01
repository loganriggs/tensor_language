# Plain-English update — 2026-09-01 07:30 UTC

**The headline:** we now have both halves of a mathematical design rule for compression: singular-value tails tell
us how much prediction damage a rank choice causes, and a nearly universal 62-dimensional damage pattern tells us
which circuit certificates will fail. The remaining task is to predict the *size* of that pattern for a new mixed
allocation, then test one theory-chosen program instead of sweeping ranks.

## Our goal

We are trying to compile the 545,902,902-scalar bilin18 model into a substantially smaller explicit tensor program
that satisfies four constraints at once:

1. it predicts well on the census, fresh windows, and shifted text;
2. independently useful replacements remain useful when composed;
3. named interventions retain their signed effects and relative magnitudes; and
4. every tensor, factor, state, route table, exception, and fallback is included in the literal scalar/byte bill.

This is not ordinary low-rank approximation. A smaller model that loses its known circuits is a useful mapped tier,
not the executable, manipulable program we ultimately want.

## Where the physical frontier ended up

The context-weighted Q/K program forms a clean six-point staircase. Its smallest fully gated member uses rank56
for all 440 replaced query/key maps:

- **512,561,462 scalars**;
- census damage **+.01250780**;
- **43/62** circuit certificates;
- shifted-text mean/p95/max **+.005698/.037966/.056886**; and
- signed intervention cosine/error **.981641/.236669**, under bars tightened before seeing the result.

The next rank48 point is smaller at 508,055,862 scalars and remains predictively smooth (`+.01877572` census), but
certificates abruptly fall to **29/62**. We did not relax the bar, run a signed gate, or continue to rank40. Q/K56
is the end of the fully gated pure ladder.

Combining Q/K56 with the already successful MLP0 rank512 map reaches **507,253,046 scalars** and has nearly additive
loss: measured `+.01951292` versus `+.01854359` predicted, a ratio of `1.052`. Yet it also has only **29/62**
certificates. Moving the same approximation budget between Q/K and MLP0 therefore does not automatically evade the
certificate ledge.

## First new law: induced-metric tails predict damage

For a map `W` whose real inputs have covariance `C`, the relevant rank approximation is not ordinary SVD. It is
Eckart–Young in the input geometry:

`min_rank(W_r)<=r E||Wx-W_r x||^2 = min ||W C^(1/2)-W_r C^(1/2)||_F^2`.

So the exact local approximation error at rank `r` is the omitted squared singular-value tail of
`W C^(1/2)`. Summing that tail across the relevant maps gives `T_f(r)` for component family `f`. Empirically, the
physical census damage follows

`D_f(r) = a_f T_f(r)^(b_f)`.

For seven Q/K ranks, this law has log-space R2 **.9951** and leave-one-rank median error **7.9%**. For five MLP0
ranks it has R2 **.9947** and median error **6.9%**. The exponents differ—about **1.69** for Q/K and **1.08** for
MLP0—and forcing one common family-independent gain collapses R2 to `.394`. The median consequence per unit tail
energy differs by **3.30x**.

That gives a weighted water-filling rule. Add rank where the next stored scalar buys the largest decrease in
family-weighted singular tail. Equal ranks across different modules are not mathematically justified.

## Second new law: certificate failures lie on one ray

For each of the 62 known circuit members, divide its absolute compiled-model damage by half of its native ablation
effect. Call the result `v_i`; certificate `i` holds exactly while `v_i<1`.

Across Q/K ranks96 through48, the seven 62-dimensional vectors are almost scalar multiples of one fixed pattern:

`v_i(r) approximately equals s(r) k_i`.

The one-ray fit has R2 **.999452**. Leaving out each Q/K rank in turn and projecting it onto the ray reproduces all
seven certificate counts exactly, including the 43-to-29 cliff. More importantly, the shape transfers to
constructions that were not used to fit it:

- Q/K56 + MLP0-p512: cosine `.998523`, vector R2 `.989801`, projected/actual certificates `27/29`;
- context-value96: cosine `.997734`, vector R2 `.983549`, projected/actual `45/46`.

Conditional on the scalar intensity `s`, certificate count becomes the deterministic first-passage quantity

`number holding = #{i : s k_i < 1}`.

This explains why a smooth increase in aggregate loss can produce a sudden certificate cliff: many fixed margins
are crossed within one small interval of `s`.

There is one important limitation. For those held-out tests, projection used the observed member-damage vector to
estimate `s`. We have shown that the *shape* transfers; we have not yet shown that weights and rank alone predict a
new program's intensity. Building and cross-validating that prospective scale model is the next mathematical rung.

## Your finite-MoE distinction, tested literally

Generic top-k can select one of an enormous number of supports. It is a compact execution algorithm, but it is not
a small fixed tensor network unless its state/support space is explicitly bounded and priced. A small MoE router
is different: it chooses from a fixed finite expert bank, so its state bond, experts, and route table can all be
counted.

We tested exactly that at MLP0. All folded token inputs were clustered into four fixed PCA-space states. Each state
received its own context-weighted rank128 bilinear input subspace; the 50,304-entry token-to-state table was stored
literally. The complete MLP0 replacement costs **10,668,288 scalars**, 1,536 fewer than an equal-price global
rank517 subspace.

The states were balanced and real (`.161/.343/.255/.242` of the vocabulary), and clustering slightly beat a
balanced random router. But the decisive shared control won by about an order of magnitude:

| Program | FineWeb mean | WikiText mean | FineWeb max | WikiText max |
|---|---:|---:|---:|---:|
| Four clustered experts | .03405 | .04047 | .11286 | .10473 |
| Four random experts | .03896 | .04871 | — | — |
| One global p517 subspace | .00368 | .00210 | .02451 | .02696 |

All positive predictions failed and the strong null fired. We will not tune the number of states, expert rank, or
clustering after this result. The correct conclusion is narrow: this four-state token router spends its parameters
duplicating a subspace that is mostly shared across tokens. It does **not** prove that all finite MoE routers fail;
a router conditioned on a named contextual or behavioral state remains a distinct hypothesis.

## The folded full-embedding MLP0 direction

We also tested the stronger idea that all 50,304 folded token inputs plus the exact MLP0 weights might reveal a
block, hierarchy, or DAG directly in the bilinear function. Because raw hidden factors are gauge-nonidentifiable,
the probe used gauge-invariant quadratic contractions. Planted block controls were recovered exactly, including
after common gauge changes. Real MLP0, however, was no more reducible than its spectrum-matched null: the full1152D
graph-connectivity ratio was `1.294` and split overlap only `.172`.

So generic neuron partitions or a task-free tree/DAG are not visible in this tested invariant. That is a valuable
negative, not a rejection of the full-embedding idea. The route can reopen if we name an external variable—such as
a behavior, intervention, or contextual router state—and ask whether its conditioned contraction family has a
stable reducing subspace. The external variable is what can make the structure identifiable.

## What happens next

The immediate plan is:

1. fit a prospective model mapping Q/K and MLP singular tails to the certificate-ray intensity `s`;
2. cross-validate that model on saved configurations and reject it if its uncertainty exceeds roughly three
   certificates;
3. enumerate exact-price rank allocations on CPU using both the damage law and certificate constraint;
4. preregister one nontrivial candidate whose lower confidence bound can improve the current frontier; and
5. run that single candidate through census, certificates, non-overlapping shifted text, identity/price, and—only
   after those pass—the signed intervention gate.

### First result from that plan

The scale bridge has now been tested without training on any Q/K+MLP composition. The fitted law is

`log s = 4.21237 + 0.464992 log D`.

On the four held-out Q/K+MLP programs it achieves log-scale R2 `.952`, median relative scale error `4.4%`, and
certificate-count mean error `.5` when supplied measured damage. Going end to end—singular-tail component damage,
the preregistered cross-family interaction interval, then the intensity law—predicts all four held-out certificate
counts exactly on average.

The conservative exact-price enumeration has an equally useful negative answer. Among 42 calibrated Q/K-rank and
MLP0-rank allocations, no program smaller than the current 512,561,462-scalar rank56 point has a conservative
lower bound of 43 certificates. The closest cheaper configurations are predicted to retain substantially fewer:

- Q/K64 + MLP0-p512: 511,758,646 scalars, central/conservative 38/35 certificates;
- Q/K56 + MLP0-p640: 508,580,150 scalars, central/conservative 36/29 certificates.

Therefore no physical composition run is licensed under the existing 43-certificate frontier goal. This is a
discrete calibrated-grid result, not a proof about new metrics, untested intermediate ranks, or other component
families. It redirects effort toward changing the representation—especially the vocabulary sparse-residual and
vector-valued downstream routes—rather than rearranging the same Q/K and MLP0 rank budget.

### The staircase is deployment-dependent, not redundant

We also put storage and future prediction error into one minimum-description-length calculation. If a program has
`S` scalars and adds `D` nats per future token, its incremental description length relative to native is

`L_b(N) = b(S-S_native) + N D/ln(2)` bits,

under a hypothetical uniform `b` bits per scalar. Every fully gated Q/K point owns a nonempty interval on the
exact lower envelope. At 16 bits/scalar, the preferred program changes from Q/K56 to 64,72,80,88,96 and finally
native at approximately **11.6B, 16.9B, 26.3B, 43.8B, 64.0B, and 84.7B deployed tokens**. Eight-bit thresholds are
half as large and 32-bit thresholds double. The literal current tensor-byte ledger exactly matches the 32-bit
schedule for these Q/K factors.

This explains why there is no single best staircase point: small deployments favor storage; enormous deployments
amortize the higher-fidelity model. The 8/16-bit calculations assume quantization leaves loss unchanged and are
not physical claims. Rank48 is also MDL-optimal at sufficiently small token counts if circuit utility is ignored,
but its 29 certificates keep it outside the adopted set.

### The contextual finite-MoE alternative also fails

The first four-state router used token identity, so we tested the strongest obvious qualifier: route on the live
contextual MLP0 input. The literal program stored a PCA32 projection and four centroids, then selected one of four
fixed rank128 experts. Its 10,656,128-scalar price was bracketed by shared global ranks515 and516.

The states were populated and executable, but the shared control won overwhelmingly. Context routing added
`.0424/.0606` mean damage on FineWeb/WikiText, versus only `.00159/.00310` for the slightly cheaper global p515
subspace. It was about 27x/20x worse, failed the tail bars, did not beat random centroids on WikiText, and its
independent-fit WikiText mean was unstable. We therefore close simple four-state token-geometry and input-geometry
routers without tuning their state count or rank. A behavior-named state remains logically possible, but it needs
an independent mechanistic reason for its state variable and expert grammar.

### A different kind of win: physically storing Q/K factors in fp16

The MDL calculation suggested changing precision, so we tested it rather than assuming it. The exact Q/K56 rank
program was kept fixed, but its 31,539,200 factor scalars were stored as IEEE fp16 and explicitly converted to
fp32 for the existing contractions. There are no learned quantization scales or extra tensors.

The result is effectively identical to fp32 at every measured decision scale:

- census damage `.0125043` versus `.0125078`;
- census-vector mean absolute difference `.0001366`;
- the same 43/62 certificates;
- new shifted-text mean/p95/max `.00885/.03615/.07033`; and
- every dtype, map-count, fit, active-set, scalar, and byte assertion passed.

The scalar count remains 512,561,462, but literal storage falls to **1,871,225,452 bytes**, a **196,444,160-byte
(9.50%)** saving from native and 63.1 MB more saving than fp32-Q/K56. This is now undergoing the signed intervention
gate whose bars were frozen before the physical receipt landed. It improves storage, not execution latency: the
current implementation dequantizes to fp32 for computation.

The independent alternatives remain alive but ranked behind this bridge: an explicitly priced sparse-row repair
for the promising shared vocabulary code; a vector-valued suffix-Jacobian objective for MLP0 tails; and a
task-conditioned folded-MLP contraction/router whose state is named before fitting. The important methodological
change is that GPU time is now for falsifying one theory-selected program, not searching a rank grid.
