# Six independent routes for the next eight hours — 2026-09-01 02:39 UTC

(Damage means next-token cross-entropy added above the native model; lower is better. A stored-value price
means the complete standalone dependency graph, not the size of a replacement hook.)

## Our goal

We are trying to compile the 545,902,902-scalar bilin18 language model into a substantially smaller explicit
tensor program that remains predictive on fresh and shifted text, composes when several replacements are
installed together, and transports named causal interventions with the correct sign and relative magnitude.
The program must also be literally simpler: every required tensor, router state, lookup table, and executed
operation is counted.

The first honest point now exists. The mixed104 online-`c_v0` program uses exactly 539,595,062 scalars,
1.16% fewer than native, with census damage `+0.00469195`, `54/62` circuit certificates, signed-a16 effect
cosine `0.995879`, and WikiText-2 damage `+0.00485625`. That is a useful calibration point, not the desired
compression. The largest remaining targets are the eighteen bilinear MLPs (286.67M scalars) and the untied
input/output vocabulary maps (115.90M scalars).

## Correction: top-k is a compute policy, not the structural decomposition

A generic per-token top-k gate over `H` units has up to `binom(H,k)` possible supports. For the existing
`H=4608, k=1152` tier, the support index alone has roughly

$$
\log_2 {4608 \choose 1152} \approx 4608 H_2(1/4) \approx 3738\ \text{bits}
$$

of combinatorial capacity. It is a compact *algorithm* for deciding which dense units to execute, but it is
not a fixed small tensor network and it stores the full Left/Right/Down maps. We will therefore keep it as an
honest compute-sparse tier, not call it a discovered tensor decomposition.

A small-state MoE router is different. If a router chooses one of `S` named states and every state owns a
fixed expert subset, it can be written as

$$
y(x)=\sum_{s=1}^{S} r_s(x)\,T_s(x),
$$

with a router bond of dimension `S`. When `S` and the expert bank are small, the states, fixed subsets,
router, and executed experts can all be priced explicitly. This finite-state object is one of the structures
we will test; unconstrained top-k is its combinatorial null.

## The new sixth direction: fold the vocabulary into MLP0 and recover structure from the function

For a bilinear MLP,

$$
y(x)=D\big[(Lx)\odot(Rx)\big]+b
     =\sum_{u=1}^{H} d_u\,(l_u^\top x)(r_u^\top x)+b.
$$

At sequence length one, position zero, the block-0 input is a deterministic function of token identity: the
embedding, RMS normalization, block-0 remix, and self-only attention can all be folded exactly. Thus the
entire vocabulary supplies an exhaustive finite population `X={x_t}` rather than a sample of natural-text
activations. We can evaluate

$$
H_{tu}=(l_u^\top x_t)(r_u^\top x_t), \qquad Y=HD^\top+b
$$

for every token, and study the bilinear tensor

$$
\mathcal T=\sum_u d_u\otimes l_u\otimes r_u
$$

under the exact vocabulary-induced metric. This is stronger than applying an SAE only to observed output
activations: the inputs and weights are known, so the test can use the complete finite function and can ask
whether block, tree, DAG-like, or finite-router descriptions compress it.

There is already encouraging evidence in the repository. A previous weight-only fold recovered MLP0's
token-class geometry with per-token cosine `0.83`, pairwise-distance congruence `0.90`, and the same `3.30x`
class separation as the data-derived map. A flat `P=512, k=32` weight-action SAE recovered about 98.3% of
MLP0's live CE value, but only `R2≈0.739`; its function was seed-stable while individual atoms were not
(mean best-atom cosine about `0.52`). That is exactly the signature for trying structure above a flat SAE,
not another optimizer tweak.

Two qualifications matter:

1. General MLP0 inputs at later positions include context through block-0 attention. Only the one-token or
   position-zero map is exhaustively foldable from token identity. We will first identify that exact static
   component, then measure the contextual residual rather than claiming the finite vocabulary is all of MLP0.
2. Bilinear CP factors have permutation, scale, and Left/Right-swap gauges, and overcomplete factorizations
   can be non-unique. We will score recovered projectors, functional blocks, fixed router supports, and partial
   orders—not equality of raw hidden-unit labels. A real DAG claim also needs interventions or asymmetry;
   covariance clustering alone cannot orient an edge.

## The planted-to-real test

Before interpreting real weights, train bilinear toys whose embedding-to-output function contains a known
structure, then hide that structure behind unit permutation/rescaling and within-subspace rotations.

- **Block toy:** disjoint token/input groups write disjoint output subspaces.
- **Hierarchy toy:** child subspaces share parent features, with a known nesting tree.
- **DAG toy:** blocks have a known triangular parent-to-child innovation rule; score reachability and the
  transitive reduction, not an arbitrary dense edge representation.
- **Finite-router toy:** one of a small number of router states selects a fixed expert subset; compare its
  description length and recovery with unconstrained top-k.
- **Nulls:** a randomly rotated flat bilinear map, overlapping non-identifiable blocks, shuffled token labels,
  and a state count large enough to memorize tokens.

The recovery metrics are held-out function fidelity, adjusted Rand index for block/router assignments,
projector distance for nested subspaces, reachability and transitive-reduction F1 for the DAG, stability across
seeds and gauges, and literal description length. A planted route advances only if it recovers the known object
above its nulls. The same instrument then becomes a *screen* on real MLP0; it becomes an identification only
after held-out contextual positions and interventions agree.

## The other five independent directions

1. **Direct shared MLP tensor compression.** Fit layer-shared CP/Tucker/tensor-train factors to the eighteen
   Left/Right/Down tensors under a ridged real-input and causal-response metric. Test a literal replacement,
   not merely tensor similarity. A 25% saving would remove about 72M scalars.
2. **Joint vocabulary factorization.** Learn a shared vocabulary code for the untied input embedding and
   output head, plus separately priced sparse residuals for exceptional tokens. Rare-token and shifted-corpus
   loss are the decisive nulls.
3. **Causal-response coordinates.** Allocate tensor rank by signed intervention effect preserved per scalar,
   using held-out circuits and text. This tests whether the smallest weight singular directions matter because
   they align with a much smaller observable response basis.
4. **Predictive causal-state quotient.** Estimate a prefix/continuation/intervention Hankel rank and compile
   equivalence classes of residual states that have the same controlled future behavior. This can remove
   redundancy spanning modules rather than compressing each matrix separately.
5. **Executable error contracts and lower bounds.** Propagate local tensor approximation bounds through the
   residual stream and compare the empirical storage frontier with predictive-state/information lower bounds.
   The goal is to kill impossible compression targets cheaply and identify where precision is actually spent.

## Eight-hour allocation

Each direction gets one protected hour. The final two hours compare and exploit rather than starting a seventh
idea.

| UTC window | Work product |
|---|---|
| 02:40–03:40 | Direction 6: planted-to-real embedding-folded MLP0 structural screen |
| 03:40–04:40 | Direction 1: direct shared bilinear-MLP compression screen |
| 04:40–05:40 | Direction 2: joint vocabulary-map factorization screen |
| 05:40–06:40 | Direction 3: causal-response-coordinate screen |
| 06:40–07:40 | Direction 4: predictive causal-state/Hankel screen |
| 07:40–08:40 | Direction 5: executable error contract/lower-bound screen |
| 08:40–09:40 | Common scorecard, confound audit, and ranking |
| 09:40–10:40 | Execute the decisive follow-up for the best one or two routes; write the morning synthesis |

An hour is a decision budget, not a promise to burn sixty minutes after a result is decisive. Every checkpoint
must leave a durable record of the object tested, opposing prediction, receipt, null/limitation, literal price,
and next decision, then immediately begin the next lane. A long GPU job may finish later, but its lane still
owes a preregistration and a cheap discriminating screen within its hour.

## How the plan survives context resets

The active Codex durable goal contains the full six-route contract and the morning stopping condition. The
`bilin18-research-driver` skill restores the current explanation, rotation ledger, board, runner, and repository
state at every new research turn. At each elapsed hour it performs the strategic step-back and launches the
next concrete probe. The managed GPU runner serializes registered jobs while CPU mathematics and analysis
continue. The schedule is therefore stored in three independent places—the durable goal, the skill, and the
checked-in rotation ledger—rather than relying on conversational memory.

## Morning decision rule

Rank every route on measured signal, possible literal scalar/byte and compute saving, predictive and causal
fidelity, identifiability, robustness to corpus shift, and cost of the decisive next experiment. Continue the
highest expected-value route or two after 10:40 UTC. A beautiful latent structure that cannot be identified
under gauge/null controls remains a screen; a high-fidelity hook that stores the native graph is not a compiler
win; and a small finite-state router must beat the fully priced top-k and dense baselines.

## First live checkpoint: embedding-folded MLP0 (03:02 UTC)

The first direction produced a useful negative and a stricter standard for every later direction. The planted
teacher's support poset was recoverable directly from its own factors (Jaccard and reachability F1 both `1.0`).
But an independently trained bilinear student represented the same function with held-out R2 `0.999940` while
recovering only Jaccard `0.281` and reachability F1 `0.0`. A dense-random negative teacher produced the same
low structural scores. Therefore a structured function need not give gradient descent a canonical structured
factorization, even in the toy setting.

Adding the correct support-size spectrum as a hard prior improved recovery to Jaccard `0.820` and F1 `0.775`
at R2 `0.999927`, a real but sub-threshold signal. The decisive confound is that an incorrect pair-only prior
fit even better (R2 `0.999999`) while recovering the wrong graph (F1 `0.245`). Function fit alone therefore
cannot choose a structural prior. A real hierarchy or DAG must win on an external criterion: a smaller literal
program, held-out interventions, or shifted-distribution behavior.

On the exact 50,257-token position-zero population, small router-state labels were predictable above chance,
but fixed `K=512` expert subsets failed even with oracle state labels. The best legal fixed-subset program had
R2 `-0.052`, versus `0.754` for unconstrained per-token top-k. This kills the tested small-state/fixed-subset
model and strengthens the distinction between a small tensor network and a combinatorial execution policy.
Raw support recovery and this router are stopped; structured minimum-description fitting is parked until an
external discriminator exists. The rotation now tests direct cross-layer bilinear-tensor sharing.

## Second live checkpoint: direct shared MLP tensors (03:06 UTC)

The invariant atom comparison used the actual symmetric bilinear tensors, so permutation, reciprocal scaling,
Left/Right swaps, and coefficient signs could not hide reuse. The planted shared-factor control recovered every
atom and its rank-3 layer structure; an independent-bank negative recovered none. In the real model, however,
all 35 frozen layer pairs had zero matches even at invariant atom cosine `0.80`. The best median nearest cosine
was only `0.000516`, nearly the signed-coordinate null's `0.000393`. A separate polarization sketch put the
top-13 normalized layer-mode energy at `0.7724/0.7683`, indistinguishable from null `0.7716/0.7728`, far below
the frozen `0.95` bar.

This closes two attractive but unsupported raw-coefficient stories: sharing the native CP atoms between layers,
and representing the stack with a small number of whole-layer tensor templates. A 25% joint-bank saving would
require reusing at least `25.390625%` of 82,944 pooled native atoms; the observed adjacent-pair reuse proxy was
zero. This does not close newly fitted joint CP atoms under an activation or causal-response metric. It says
that such a method must genuinely refactor the functions; it cannot discover a large pre-existing common bank
by aligning the native factors. The rotation now moves to the two untied vocabulary matrices.

## Third live checkpoint: joint vocabulary code (03:15 UTC)

The useful executable family keeps the input embedding `E` exactly and reconstructs the output head as
`U_hat = E M + P_s V_s^T`. At residual rank 512 it costs `85,622,784` scalars, 73.88% of the two native
vocabulary maps, and its independent-SVD control is slightly cheaper. A data-free fit established real shared
geometry—`E M` alone explains 33.24% of output-weight energy, and the shared rank-512 program beats independent
rank 537 by more than 2x—but remained predictively bad at `+.743` FineWeb and `+.647` WikiText.

The frozen frequency diagnostic localized that failure enough to justify a separately preregistered metric
change. Refitting the same map and same rank under natural target-frequency weights reduced damage to `+.193`
FineWeb and `+.225` WikiText, while the identically weighted independent control remained at `+.552/+.778`.
All three prospective follow-up bars held. Thus the input/output maps share a *functionally useful* token code,
and consequence weighting matters far more than ordinary weight error.

This is the rotation's first route worth advancing, not an adoption. The gain comes with a clear rare-tail
failure: count>=10 targets are slightly better than native, but unseen targets cost `+1.57` FineWeb and `+.895`
WikiText. The 25%-saving ceiling leaves room for about 1,129 explicitly indexed residual rows beyond rank 512.
A fit-selected rare-row correction is therefore the natural exploit-phase test. Before that, the rotation moves
to causal-response coordinates, asking whether the same consequence-weighting lesson improves MLP rank allocation.

## Fourth live checkpoint: causal-response rank coordinates (03:20 UTC)

The tested coordinate was deliberately error-relative, not the already-rejected global observability quotient.
For native MLP0 outputs `y` and exact suffix-loss gradients `g`, the symmetric operator
`sym(E[(y-mean(y))^T g])` ranks directions by the signed first-order consequence of deleting the output component.
A rank-128 basis has a literal factorized-Down realization: `11,355,264` scalars for MLP0, 71.3% of native.

The response program cleared its absolute two-corpus bar (`+.0875` FineWeb, `+.0616` WikiText), but that result
was not causal-coordinate evidence. At the same rank and price, activation PCA scored `+.0500/+.0302` and Down
weight SVD `+.0621/+.0495`; response ranking lost at every tested rank. Its split-projector overlap was only
`.283`, versus PCA `.683` and random expectation `.111`. Thus ordinary low output rank explains the predictive
result, while the signed response basis is comparatively noisy and unstable. This direction is stopped. The
next probe uses a genuinely sequential, behavior-anchored finite state instead of another generic token splice.

## Fifth live checkpoint: predictive causal state (03:26 UTC)

The screen used natural 64-token prefixes, not unrelated prefix/suffix splices. Quote parity and open-parenthesis
state were asked to predict a nested bank of one- to three-action continuation log probabilities on both FineWeb
and WikiText. Quote parity had a real transferable signal: heldout and cross-corpus accuracy were `.875`. But it
explained only `.191/.239` of response variance, and parenthesis transfer was `.625`. Interaction ranks were 4–5,
not uniformly within the four-dimensional bar. Deleting delimiter head 13.8 reduced state separation by just
6–7%, while the matched head13.1 control was inert. All three positive predictions failed.

There is an additional scope warning: the short fixed suffix bank averaged 10–12 nats/token, so the behavior-
anchored object is still not a broad natural-continuation interface. The result supports quote parity as a weak,
distributed circuit classifier, not a small causal state capable of replacing native computation. It is parked;
generic Hankel expansion remains closed. The last independent direction now tests whether empirical error bounds
can cheaply rule compression candidates in or out.

## Sixth live checkpoint: executable error contract (03:31 UTC)

For the nine actual MLP0 projection programs from the prior direction, I measured the fraction of centered output
energy discarded locally and the finite end-to-end CE damage. A power law fitted only on eight skip7000 rows was
then frozen and tested on eight different skip11000 rows. It covered all nine heldout programs. The fitted exponent
was `1.746`, reasonably close to the old isotropic-random-error exponent `1.534`, so the relationship is not merely
a rank label in disguise.

The important limitation is width. The transferred interval spans a factor of `3.414`; calibration and validation
rank correlations were `.867/.917`. It correctly lower-bounded the rank-64 weight-SVD and response programs above
the `+.05` budget, but could not reject PCA rank64: its lower bound was `+.033` even though actual damage was
`+.096`. Thus omitted energy is a good cheap screening coordinate and a bad certificate for the strongest basis.
It is retained as an instrument, not claimed as a theorem or compression result. All six directions are now ready
for a common comparison; the leading executable route is the shared vocabulary code, with activation-PCA MLP0 as
the secondary seed and this error law as a possible low-cost screening tool.

## Common comparison and exploit choice (03:42 UTC)

The audit immediately paid for itself. Rung 302's label “shuffle” was actually the complementary binary partition,
so it could never change centroid R2. A corrected 64-permutation control preserved the classifier-level result but
dissolved the representation-level claim: every heldout classifier beat mean shuffled accuracy by at least `.10`,
while parenthesis R2 did not clear the shuffled 95th percentile. Predictive state is therefore retained only as a
circuit observation and killed as a compiler route.

Across all six directions, the vocabulary code is the only route with a large literal saving, an executable map,
and a shifted-corpus advantage over a slightly cheaper matched control. Activation-PCA MLP0 is second, because its
rank-256 program is small and predictive but untested under composition. The error power law ranks third as a cheap
screen, not a bound. The remaining forms either failed their controls or lack an executable candidate.

The exploit phase uses the small amount of room still available under a strict 25% vocabulary saving: at most
1,129 indexed exact residual rows. A diagonal Fisher score estimates, using fit contexts only, which rare rows'
logit errors contribute most to CE. That selection is compared prospectively against equal-price residual-norm and
random rare-row controls on new FineWeb and WikiText windows. A pass must repair the unseen-token tail as well as
aggregate CE; simply protecting common tokens again will not count.

## Exploit checkpoint: sparse rows fail (03:39 UTC)

The maximal exact-row hybrid under a strict 25% vocabulary saving stores 1,129 corrected rows. It did not repair
the tail. Fisher selection reduced aggregate damage by only `1.8%` on FineWeb and `4.4%` on WikiText; selecting
the largest residual-row norms was slightly better on FineWeb and essentially tied on WikiText. Unseen-target
damage moved by at most about 6% for any arm, far below the 40% bar. All registered predictions failed and the
null won.

This is informative because the three selectors chose almost disjoint sets (only 3–5% overlap) but all failed.
The rare failure is distributed across far more than a thousand token rows, not concentrated in a recoverable
exception table. Sparse corrections are stopped. The last vocabulary exploit therefore buys distributed residual
rank instead: ranks 640 and 768 relax vocabulary savings toward 20% and 15%, respectively. If that frontier cannot
reach predictive and unseen-tail bars against matched independent controls, the route is stopped and the secondary
activation-PCA MLP direction takes over.

## Exploit checkpoint: the shared-code frontier remains too costly (03:43 UTC)

Buying distributed residual rank works smoothly, but not fast enough. At rank 640 the shared program still adds
about `+.180/.197` FineWeb/WikiText while saving 20.4% of vocabulary. At rank 768 it adds `+.127/.145` while
saving 14.8%. The shared arms beat their price-matched independent SVDs by a large margin everywhere, confirming
that the input and output maps genuinely share a useful code. However, the unseen tail remains `+.724/.566` even
for the better sqrt-weighted rank-768 arm.

That distinction matters: this is a positive representation result and a negative compression decision. The code
is real, but the amount of distributed residual rank needed for low loss consumes most of the saving. I therefore
stop the vocabulary route at the preregistered boundary rather than chasing rank 896/1024. The exploit moves to
activation-PCA MLP outputs, where MLP0 rank256 already showed `+.0209/.0114` damage while saving 3.83M scalars.
The next test asks whether four calibration-selected layers compose on fresh FineWeb and WikiText for a literal
15.34M-scalar saving.

## Exploit checkpoint: PCA generalizes, naive allocation fails (03:49 UTC)

Rank-256 output PCA is not an MLP0 curiosity: 17 of 18 layers individually added at most `.04` CE on all three
populations. The problem is composition. Calibration damage barely ranked validation sensitivity (`rho=.298`),
and choosing the four smallest calibration layers produced `+.130/.122` damage. That is 1.78x/1.61x the sum of
their separate validation damages. A fixed, evenly spaced control quartet was substantially better at
`+.098/.084` for the same 15.34M-scalar saving.

This recovers a recurring law in a new object: local compression errors are broadly cheap but interact, so targeted
single-site ranking overfits while spread allocations can win. The positive part is that the fixed control sits
close enough to the desired range to justify one mathematical follow-up. I will measure every layer pair on two
calibration halves, estimate its excess-over-additive interaction, penalize interactions that do not replicate,
and enumerate all 3,060 four-layer subsets. Only the frozen risk-adjusted winner is evaluated on the disjoint
FineWeb/WikiText windows.

## Exploit checkpoint: pair interactions transfer, four layers narrowly miss (03:53 UTC)

The excess-over-additive pair term is exceptionally reproducible across calibration halves (`rho=.917`), even
though total pair damage ranks only reach `.476`. Penalizing non-replicating attractive interactions and enumerating
all quartets selects layers `{0,4,14,15}`. On heldout text that quartet lands `+.0876` FineWeb and `+.0586`
WikiText, a large improvement over the scalar-selected `+.130/.122` and a modest improvement over fixed-spaced
`+.098/.084`.

The registered result is still a miss: FineWeb is `.0076` above the `.08` bar, and its 10.9% improvement over the
fixed control misses the 15% requirement. But the stable interaction law and shifted-corpus improvement justify a
final smaller candidate. I now enumerate triples using only the already frozen calibration pair model, then test
the winner on untouched cache rows and a later WikiText window. Three layers save 11.50M scalars and remove three
of a quartet's six pair interactions; this is the last MLP exploit before the morning synthesis.

## Exploit checkpoint: targeting stops; the spread control gets a stability gate (03:56 UTC)

The pair-selected triple `{4,14,15}` adds `+.0668/.0672` on untouched FineWeb/WikiText. Its composition ratio is
only 1.19–1.21, confirming that the interaction model avoids much of the tax. But it misses the FineWeb `.06` bar
and loses badly to the fixed `{0,8,17}` control on WikiText (`+.0672` versus `+.0334`). Interaction-aware targeting
is therefore useful diagnostically but not robust enough to choose deployment layers.

The fixed control is scientifically interesting because it was chosen before seeing this population and lands
`+.0725/.0334` while saving 11.50M scalars. Rather than tune another subset, I promote this exact spaced triple to
a broad confirmation: 176 untouched rows from each FineWeb cache and 120 later WikiText rows, including row-p95
and worst-case gates. A fail ends MLP compression; a pass only earns integration and certificate work.

## Exploit checkpoint: the spread triple is stable (03:59 UTC)

The large gate confirms `{0,8,17}@r256`. Across 176 + 176 untouched FineWeb rows and 120 later WikiText rows,
mean damage is `+.0589/+.0600/+.0477`; every row-p95 is about `.10`, and the worst observed row is `+.154`.
Population means span only `.0123`. All three predictions hold.

This is the first MLP-side result tonight that earns adoption work. It remains much more expensive in CE per saved
scalar than the adopted QK program, and composition taxes are the central risk. The next exact object combines the
three factorized MLP Down maps with mixed104 online-c_v0. If the dependencies are disjoint, the proposed standalone
bill is `539,595,062 - 11,501,568 = 528,093,494` scalars. That number is not adopted until live active-set checks,
census, all 62 certificates, fresh windows, and a standalone dependency audit agree.

## Physical composition checkpoint: prediction composes, certificates collapse (04:05 UTC)

The combined program is physically coherent. Census damage is `+.06745`; subtracting adopted mixed104's
`+.00469` gives an MLP surcharge `+.06276`, consistent with the standalone triple. Every fresh window remains
below `+.069`, and all exact projector/QK/active-object tripwires pass. The standalone arithmetic is therefore
`528,093,494` scalars and `1,996,431,980` raw bytes.

The blocker is manipulability: only 8 of 62 certificates survive, missing the preregistered 10 and collapsing far
below the parent's 54. The triple is not adopted. One final capacity frontier tests its three two-layer subsets in
the same frozen mixed104 rebuild. If a pair restores certificate thresholds at a useful rate, it may earn fresh/OOD
confirmation; otherwise MLP compression closes as prediction-cheap but certificate-expensive.

## Physical pair frontier: capacity helps smoothly, not disproportionately (04:12 UTC)

All three rank-256 two-layer subsets remain predictively usable, but none clears the certificate gate. `{0,8}`
lands at `+.04021` with `16/62` certificates, `{0,17}` at `+.04970` with `17/62`, and `{8,17}` at `+.04726`
with `19/62`. Each saves exactly 7,667,712 scalars and proposes a 531,927,350-scalar standalone program.

Removing one projection therefore recovers 8–11 certificates from the triple, but the best arm misses the frozen
20-certificate bar by one and its surcharge per saved scalar is slightly worse than the triple's. The result rejects
a special toxic-layer explanation: the certificate count moves smoothly with total distortion, and every pair
lies in the expected aggregate-loss band. The next test buys rank rather than searching more subsets: compare
rank 384 and 512 on the frozen best pair at equal layer identities. This measures the certificate/scalar frontier
directly. If higher rank cannot restore certificates faster than its storage cost, the honest outcome is a
certificate-grade identification tier plus a lower-fidelity compression tier, not another adoption claim.

## Rank frontier: certificates are a thresholded one-dimensional damage law (04:19 UTC)

On the fixed `{8,17}` pair, rank 384 lands at `+.03379` with `24/62` certificates and 6,193,152 scalars saved;
rank 512 lands at `+.02490` with `32/62` and 4,718,592 saved. The frontier is monotone, but neither the registered
rank-specific certificate bar nor the required certificate recovery per returned scalar holds.

The stronger result comes from a prospective CPU calculation made before those two counts were visible. For every
certificate, I regressed its absolute member damage on whole-census damage using only the adopted parent, the three
rank-256 pairs, and the rank-256 triple. Median per-certificate R2 was `.947`. At census damage `.025`, this frozen
62-threshold model predicts exactly **32** valid certificates—the subsequently observed rank-512 count. Thus the
battery is close to a single common damage coordinate followed by 62 different thresholds. Buying ordinary PCA
rank moves down that coordinate; it does not change direction.

One qualitatively different falsifier remains. At the same rank256 and same `{8,17}` price, reserve 32 or 64 output
dimensions for certificate-loss gradient directions orthogonal to the leading activation-PCA space. Fit gradients
using one frozen half of tags and only 16 explicitly removed census rows; score the other tag half after excluding
those rows. The mathematical bound is

$$
|g^T(I-QQ^T)(y-\mu)| \leq \|(I-QQ^T)g\|\,\|(I-QQ^T)(y-\mu)\|.
$$

Ordinary PCA minimizes only the second factor. The hybrid sacrifices some activation variance to reduce the first.
It advances only if it beats both plain PCA and the already-frozen one-dimensional threshold prediction on held-out
tags. Otherwise the MLP campaign closes with a two-tier result.

## Constraint-orthogonal result: a real rotation, zero certificate movement (04:25 UTC)

The hybrid instrument worked. Grad32 overlaps plain PCA by only `.879` and captures 65–71% of the fit-gradient
energy; grad64 overlaps by `.766–.767` and captures 76–83%. Nevertheless, plain/grad32/grad64 score exactly
`19/19/19` full certificates and `10/10/10` held-out certificates after removing every gradient-fit row. Their
census damages are `+.047265/+.046353/+.047879`. Grad32 slightly improves aggregate loss but slightly worsens the
held-out normalized certificate margin; grad64 worsens both. Every positive prediction fails and the strong null
wins.

This closes activation-PCA MLP adoption tonight. Layer choice, number of layers, rank, interaction-aware allocation,
and an explicitly different certificate-gradient basis have all been tested. The prediction/compression tier remains
scientifically real, but it cannot be merged into the certificate-grade artifact under the frozen gates.

The next experiment returns to the user's original MLP0 idea without repeating this output-rank campaign or the old
coefficient-space HOSVD null. On the exhaustive position-zero token population, concatenate MLP0's Left and Right
linear maps and fit a **shared input encoder** under the exact token-induced metric. Deployment computes one latent
`z=B x`, then separate `A_L z` and `A_R z`, preserving the bilinear product and native Down. At latent rank `p`,

$$
N(p)=1152p+2(4608)p+4608(1152)+1152.
$$

Thus `p=512` saves 5,308,416 scalars (one third of MLP0), and `p=768` saves 2,654,208 (one sixth). Compare the
token-metric reduced-rank factorization against weight-SVD and input-PCA controls on held-out token identities, then
test the same literal maps on contextual FineWeb and WikiText. This asks whether the folded finite function exposes
input-side bilinear structure in the weights that ordinary coefficient metrics miss.
