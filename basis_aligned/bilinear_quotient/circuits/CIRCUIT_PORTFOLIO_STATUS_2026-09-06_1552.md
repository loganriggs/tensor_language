# Circuit portfolio status — 2026-09-06 15:52 UTC

## Count the evidence objects, not all of them as circuits

The repository currently exposes three different counts:

- `circuits/registry.json` contains **79 canonical legacy circuit records**. Most are
  census/localization records created before the current tier rubric; none carries an
  explicit current mechanistic or counterfactual tier, so 79 is not a count of completed
  explanations.
- The latest result for each `circuit_fast_screen_result_v1` candidate gives **64 unique
  behavioral candidates**: **49 screens** and **15 nulls**. A screen identifies a useful
  causal site under its registered controls; it is not a complete circuit.
- There are presently **three strongest computation-level linguistic program lines**:
  `aspectual_anchor.has_vs_had`, `tense_auxiliary.is_vs_was`, and
  `temporal_auxiliary.will_vs_had`. The compact Task14+bracket release is an additional
  useful 22-scalar executable behavioral program, but its own boundary certificate does
  not claim recursive causal/weight realization.

No line has earned Tier 5 / CF5. The portfolio is broad at localization and narrow at
full explanation.

## Current quality frontier

The temporal auxiliary line is the clearest current Tier-4 candidate:

- exact upstream writer: block 8 head 1 cue terms write the subject-onset state;
- localized readers: block 9 heads 1/4, block 11 head 3, and block 15 heads 5/1;
- operation identity: changed value vectors transported by near-native patterns account
  for about 100% of the reader response on both original and fresh cue families;
- actual-joint sparse response program: the three head groups plus MLP11/MLP14 reproduce
  **96.65%** of held-out A1 and **96.00%** of untouched A2 writer effect, with every row
  donorward and about tenfold lower rowwise error than the best singleton;
- remaining boundary: the replay is writer-conditioned/cached. The subspaces have not
  yet been converted into a compact weight-derived program and recursively connected to
  token/position primitives.

The aspectual and `is/was` lines have executable transparent programs and prospective
lexical transfer. Their causal recovery and circuit-boundary claims differ, so a single
ordinal comparison would hide important weaknesses. They should be re-audited under the
same Tier/CF rubric before being counted beside the temporal program.

## Incorporating subspace-to-weight translation

For an orthonormal residual subspace `U`, translate every attention head into exact
weight factors:

`Q_h U`, `K_h U`, `Q2_h U`, `K2_h U`, `V_h U`, `U' O_h`, and
`U' O_h V_h U`.

These distinguish routing reads, value reads, residual writes, and direct source-to-target
value transport. For a bilinear MLP, restrict its weights to the subspace tensor

`T[a,i,j] = sum_n (U' Down)[a,n] (Left U)[n,i] (Right U)[n,j]`.

The tensor exactly replays the quadratic MLP on inputs inside `U`; its Frobenius norm and
singular spectrum are invariant to orthogonal rotations of the chosen basis. Therefore
two tasks can share a weight-level circuit even if their fitted activation coordinates
are rotated.

The correct workflow is:

1. identify a causal subspace at one valid live boundary with full-rank closure;
2. map head-local axes through the exact output projection into residual space;
3. rank upstream writers and downstream readers using the contracted weight objects,
   against dimension-matched random-subspace controls;
4. intervene on the predicted weight-mediated edges and require held-out causal replay;
5. compare the gauge-invariant contracted tensors across tasks, then test the shared
   intersection and each task-specific complement separately.

This turns DAS from the final answer into a hypothesis generator for a literal tensor
program. It also diagnoses the scalar-objective cheat: a constrained-DAS direction that
mainly aligns with the final answer readout but lacks corresponding writer/reader weight
incidence should not be promoted as the circuit subspace.

The reusable implementation is `ops/subspace_weight_atlas.py`; its exact OV contraction,
bilinear replay, and basis-rotation invariance tests pass. The next experiment should
apply it to both the constrained-DAS and difference-in-means block11H3 axes, then test
whether the weight-predicted edges explain centered full-vocabulary causal effects.

## 16:01 prospective validation update

The first sealed application passed all gates. A rank-3 block8H1 writer subspace explains
95.73% of fit activation energy. Exact value-weight contractions, computed before opening
the evaluation head effects, rank `L15H5`, `L9H1`, `L11H3`, and `L9H4` first through
fourth; the fifth known reader `L15H1` is at percentile 0.875. Across all 81 downstream
heads, the value-weight score predicts absolute causal response with Spearman 0.526 on
A1-heldout and 0.535 on untouched A2, exceeding the corresponding routing-weight
correlations. Three of the top six weight-predicted readers independently clear the
registered causal-effect bar on both splits.

This changes the recommendation: first compare the aspectual, tense, and temporal writer
subspaces using their gauge-invariant weight contractions and causally test the shared
intersection/task-specific complements. The DAS-vs-DIM weight audit remains necessary for
identification, but weight-guided reader discovery is already licensed as an efficiency
tool rather than merely a proposed method.

## 18:51 subspace, weight, and live-edge update

The constrained-DAS red team does not support regularization as a sufficient repair.
On a truly fresh cue bank, frozen-axis difference-in-means beat unregularized scalar cDAS;
noise improved one panel slightly but worsened the other, while KL and aligned penalties
collapsed toward a still poorer difference-in-means-like basin. The stronger intervention
was multi-task fitting: a pooled rank-one direction beat both task-specific rank-one
baselines, and the frozen rank-two union beat every rank-one arm on a sealed third cue
bank. Its complement disturbance remained 0.328/0.344, however. The present diagnosis is
therefore an underspecified scalar/complement objective plus cue-conditioned rotation, not
simply too little regularization. Future optimization must use multiple environments and
an explicit invariance/selectivity constraint evaluated on held-out environments.

The rank-two H3 union has now been translated into exact checkpoint-weight interfaces.
The thin-QR gauge replay error is 6.36e-9; its two residual-write singular values are
10.77 and 10.15, so the second mode is not a numerical null. The weight atlas prospectively
ranked L9H1 first and L9H4 third among upstream heads. A live causal test then found that
L9H1/H4 plus the causally exposed L9H7 account for 99.17%/96.21% of the all-head projected
H3 response, while the other six heads account for only 2.62%/4.97%. Exact removal gives
the matching necessity result: the triple accounts for 99.42%/97.14% of all projected H3
removal and the complement only 2.40%/4.36%. Behavioral sufficiency and necessity close to
the all-head arms as well, with zero reconstruction, identity, and self-clamp error.

This licenses a selectively manipulable discovery-population edge:

`L8H1 cue write -> subject state -> L9H1/H4/H7 distributed refinement -> L11H3 rank-two read`.

The block-9 edge is about 8% of the total L8H1 writer effect, so it is a real serial branch
rather than the whole computation. Evidence is currently a preregistered discovery screen,
not stable identification: H7 was chosen on the v4 bank. The immediate promotion test is a
zero-fit replay of the frozen three-head sufficiency/necessity intervention on the earlier
Later/Previously cue bank, followed by a wholly new capability-first bank if it passes.

That promotion test is a strict near-miss, not a confirmation. On the 59 jointly capable
Later/Previously rows, triple necessity retains 99.47% of all-head H3 removal in A1 but
89.72% in A2, narrowly below the frozen 90% floor. The six-head complement remains small
(5.38%/12.42%), behavioral necessity closes in both panels, and every instrument control is
exact. Thus H1/H4/H7 are a recurrent dominant core, but the claim that they are the stable
minimal edge is falsified as registered. The efficient next diagnostic is the already
frozen nine-singleton inventory on this population, using the weight atlas's prediction
that H3 is the largest member of the six-head remainder; any revised set still requires a
new cue bank rather than threshold repair on these outcomes.

The singleton diagnostic preserves the dominant-core claim but falsifies that exact weight
prediction. H1/H4/H7 recover 100.63%/90.15% of the all-head sufficiency response. H3 is a
material A2 remainder at 5.65%, but H0 is larger at 11.07%. This is evidence that static
weight incidence ranks the large readers well but does not fully determine activation-
conditioned routing among small residual heads. H0 may extend the cross-cue core, but it is
post-outcome on this bank and cannot be added here. A sealed Tomorrow/Yesterday authority
has therefore been authored for capability-only gating before any four-head intervention.

Three capability-only attempts were rejected before causal testing: v5 had 32/32 A1 but
10/32 A2 jointly capable; v6 had 32/32 and 22/32; v7 Next-week/Last-week had 29/32 and
25/32. This is useful negative design evidence: changing both cue semantics and embedded
frame at once is an inefficient route to a high-quality confirmation bank. The next bank
will retain the already reliable Next-year/Last-year semantic frame while changing the
lexicon and surface construction, isolating lexical/construction transfer of the revised
H1/H4/H7/H0 hypothesis. New-cue transfer remains a later, separate test.

V8 supplies that controlled bank (29 jointly capable A1, 31 A2). It cleanly falsifies H0
as a reusable fourth component: H0 falls to 1.45%/1.93% of the all-head response and adding
it slightly worsens H1/H4/H7. The original triple alone retains 99.21%/100.56%, while the
remaining heads retain 2.89%/2.47%. Paired removal independently closes: the triple accounts
for 99.51%/100.49% of all H3 removal and essentially all behavioral removal, with a
2.41%/2.97% complement and zero instrument errors. The stable identification is therefore
an operational-equivalence class: H1/H4/H7 are the cross-construction core; small remainder
heads can be cue conditioned and are not part of the invariant unit.

The next circuit boundary is downstream of that core. The frozen rank-two H3 union must now
be installed as an actual H3 response—not merely used as a measurement projection—and
tested for behavioral sufficiency, orthogonal-complement inertness, and transport into the
weight-predicted L15H5/H1 reader pair on v8. This directly decides whether the DAS-derived
subspace is a manipulable circuit variable with a weight-readable downstream interface.

That actuation falsifies the stronger interpretation of the frozen rank-two union. Full H3
actuation is material, but the rank-two subspace carries only 74.72%/71.54% of its behavioral
effect and 66.28%/71.18% of its induced L15H5/H1 response; the orthogonal complement still
carries 25.43%/28.58% behavior. It is therefore a useful probe coordinate system, not yet a
selectively manipulable circuit variable. This is also a direct red team of complement-loss
DAS: a low-dimensional axis can look meaningful without making its complement causally inert.

A label-free rank-four SVD of the actual L8H1-induced H3 responses was then fit on v1/v2 and
sealed before v8. It captured 87.97% of training response energy, yet recovered only
73.06%/71.30% of full-H3 behavior and 63.92%/60.69% of L15H5/H1 transport—no improvement over
the frozen rank-two basis—and its complement retained 27.02%/28.75% behavior. This is a
wrong-object result, not evidence that optimization is inherently inferior to
difference-in-means: raw activation-response variance is not the downstream functional
metric. The next test retains the cross-environment rank-two semantic core and augments it
with two directions selected solely by exact H3-output-to-L15H5/H1 weight contractions. It
therefore tests the proposed tensor writer/reader geometry without fitting labels or a
task-specific scalar readout; held-out v8 behavior and complement remain decisive.

The checkpoint-only reader augmentation produces the first clear improvement in the
manipulable H3 variable. Relative to frozen rank two, rank-four behavioral coverage rises
from 74.72% to 87.19% in A1 and 71.54% to 82.66% in A2; L15H5/H1 transport rises from
66.28% to 84.56% and 71.18% to 82.50%. The behavioral complement falls to 13.08%/17.48%,
passing its selectivity gate. The basis itself captures 22.01% of the normalized static
reader-map energy versus 5.83% for rank two. This remains a registered near-miss, not an
identification: A2 behavior and both transport panels miss the frozen 85% floors (A1
transport by only 0.44 points). The contrast is nevertheless diagnostic. Exact weights
recover causal functional modes that both scalar/complement DAS and 88%-energy response
SVD missed, while the residual gap shows that static incidence alone omits
activation-conditioned attention routing and intervening transformations.

The efficient next screen is a nested rank sweep of the same label-free weight singular
modes, retaining the frozen semantic core and measuring the behavior/transport/complement
Pareto frontier. The smallest stable rank will then be sealed on a new capability bank.
Only after fixing this weight-derived hypothesis class should constrained DAS optimize a
small residual correction, with multi-environment fitting plus noise/KL/invariance
regularization; otherwise regularization can stabilize an objective that is still aimed at
the wrong scalar target.

The nested rank screen selects rank seven, not rank four or six. Rank seven recovers
93.55%/92.32% of full-H3 behavior, transports 95.98%/96.43% of the L15H5/H1 response, and
leaves only 6.60%/7.81% behavioral complement. Rank six remains at 87.55%/84.95% behavior,
85.88%/84.33% transport, and 12.75%/15.20% complement, so the selected boundary is not a
rounding artifact. Rank eight also passes but is dominated in simplicity. All bases are
nested and label/example-free, and rank four exactly replays the earlier receipt. Rank seven
is therefore the current highest-quality manipulable H3 candidate, subject to a new-bank
confirmation because v8 was used for rank selection.

The sealed v9 confirmation is a strict rank-seven near-miss. On a fully capable 32/32 +
32/32 bank, rank seven retains 92.99% A1 behavior but 89.33% A2 behavior, missing the frozen
90% floor by 0.67 points. Its orthogonal complement remains selective at 7.31%/11.07%, and
L15H5/H1 transport confirms strongly at 94.76%/95.78%; full H3 is material and every
instrument check is exact. Thus the tensor-derived modes and downstream interface transfer,
but rank seven is not promoted as a stable sufficient boundary. Rank eight is the robust
upper-bound candidate from the preregistered v8 sweep. The next compression test should use
it as a teacher and optimize only a tensor-anchored low-rank correction with full-distribution
KL, complement-inertness, noise consistency, and explicit multi-environment balance. This
directly tests regularization against memorization while avoiding the already falsified
task-scalar objective.

The tensor-anchored regularized compression is a valid null. All three discarded-normal
restarts converge near the checkpoint rank-seven initialization (minimum absolute cosine
0.9764), but every nonzero displacement worsens the odd-row held-out full-vocabulary KL
objective. Selection therefore returns the untouched weight rank seven at step zero, with
zero projector distance. V8 and v9 causal metrics replay to numerical precision; the apparent
v9 A2 gain is only 1.8e-6. Thus tangent noise, full-vocabulary sufficiency/complement KL,
tensor anchoring, and four-environment balance successfully prevent task memorization but do
not uncover a better rank-seven variable. The evidence now favors rank eight as the simplest
robust tensor-family candidate, requiring a wholly new-bank confirmation rather than further
regularization of the same rank-seven objective.

Rank eight then passes a wholly new v10 confirmation without fitting or outcome-dependent
adjustment. On all 31 jointly capable A1 and 32 A2 rows it carries 93.78%/91.29% of full-H3
behavior, leaves only 6.47%/8.89% in the exact orthogonal complement, and transports
96.75%/96.98% of the L15H5/H1 response. Full H3 remains material at 42.62%/30.05% of the
live L8H1 writer effect, and every reconstruction, identity, orthonormality, decomposition,
and price gate passes. The label-free checkpoint-derived H3 rank-eight subspace is therefore
the current stable manipulable circuit variable. The next boundary is causal downstream use:
remove the H3-induced L15H5/H1 response under rank-eight actuation and compare it with complete
L15 and seven-head-complement removals. Response transport alone is not yet evidence that the
pair mediates behavior.

That causal reader test rejects the L15 behavioral edge. Dynamically clamping all nine L15
head responses during rank-eight actuation changes behavior by only +0.007% in A1 and -1.00%
in A2; H5/H1 alone are slightly counter-mediating (-0.30%/-1.27%). Exact rank-eight replay,
base self-clamp, reconstruction, and composition controls pass. Thus the weight-predicted pair
is a reproducible representational reader but not a behavioral mediator. Large transported
response norms are insufficient to establish downstream circuit use. The next localization is
a complete dynamic-removal atlas of MLP11 and every attention/MLP module through block17 under
rank-eight actuation, followed by head splitting only inside causally material attention sites.

The complete removal atlas finds no such downstream module. The largest absolute singleton
removal is MLP17 at 5.95% in A1 and MLP13 at 2.19% in A2; no site reaches 10% in both panels.
Restoring all thirteen downstream module outputs jointly changes rank-eight behavior by only
+0.84%/-3.03%. This also explains the L15 paradox: downstream modules visibly transform the
H3 state, but those transformations are small signed corrections rather than the behavioral
readout. The dominant causal contribution remains in the residual skip stream. Consequently,
head splitting and greedy downstream search are not licensed by this subspace intervention.

The weight-level route is now explicit and executable. Projected H3 coordinates are mapped
through the actual attention11 output matrix with exact `F.linear` orientation, multiplied by
the frozen block12--17 residual coefficient product (1.51363148), and added directly to the
native final residual. This zero-fit program agrees with live rank-eight actuation plus all
downstream-module clamps within 2.86e-6 on answer/foil logits, and retains 99.16%/103.03% of
rank-eight behavior in A1/A2. The first implementation used the transposed convention for an
`nn.Linear` weight and is preserved as engineering-invalid; the source-verified orientation
repair changed no scientific condition. The next tensor property to extract is the explicit
eight-coordinate final reader: contract these residual modes with the analytic final
RMS-normalization and unembedding Jacobian, then test whether upstream H3 coefficients and
downstream weight-derived reader coefficients predict the per-row causal margin effect.

That analytic factorization passes with effectively exact predictive performance. For every
sealed v10 row, the upstream coordinate vector is the live H3 delta contracted with Q8; the
downstream coordinate vector is the eight weight-derived residual modes contracted with the
analytic final RMS-normalization and answer-minus-foil unembedding Jacobian. Their dot product
predicts the exact direct-route causal margin effect with cosine 0.99999985/0.99999991 and
relative RMSE 0.056%/0.047% in A1/A2. The largest coordinate supplies only 46.6%/44.5% of mean
absolute contribution (coordinate 0 in both), so the robust variable is genuinely distributed
over the compact eight-dimensional interface. This realizes the desired tensor use: actual
weights identify the downstream reader and, together with measured upstream coefficients,
predict intervention behavior row by row without a fitted task objective.

The first upstream Q8 compilation falsifies a serial reading of the earlier L9 necessity result.
L9H1/H4/H7 accounts for essentially all of the H3 response attributable to L9 heads, but removing
that triple removes only 14.46%/15.96% of the complete live H3-Q8 norm on v10. Within this minor
branch, the tensor account is strong: fixed-pattern H3 value weights predict the removed Q8 vector
at cosine 0.9989/0.9937 and relative RMSE 4.74%/11.98%; contracting it with the final reader
predicts the branch's causal margin effect at cosine 0.99987/0.99669. Thus the correct graph is
primarily parallel: L8H1 writes the subject state, L9H1/H4/H7 and L11H3 read it in separate routes,
with only a small L9-mediated contribution into H3. The next weight compilation targets the direct
L8-written subject-state -> H3-Q8 value path, which should account for the missing majority.

That simple direct-source compilation is a valid null on v10. Complete H3-Q8 norm is 6.92x/6.27x
the L9-mediated branch, confirming that L9 is minor, but base-pattern value changes at the two
subject-onset tokens leave 36.15%/45.99% of the Q8 norm unexplained. Coordinate cosine is
0.9617/0.8472 and relative RMSE 37.6%/53.2%; downstream composition consequently misses its
frozen error bars. The earlier bank's exact subject-source localization is therefore
construction-dependent rather than a stable source rule. This does not weaken the confirmed H3
rank8 downstream interface; it changes the upstream computation that generates its coefficients.
The licensed next object is a complete v10 source-region by pattern/value/interaction atlas in Q8
and causal-behavior space, followed by weight compilation of whichever source-factor union closes.

That complete upstream atlas passes and replaces the construction-specific subject-only rule with
a stable candidate operation. The causal suffix—subject onset, post-subject interval, and H3's self
position, including all three attention factors—reproduces 100.17%/100.17% of complete Q8 norm and
100.30%/101.06% of complete behavior on sealed v10. Across all sources, base-pattern value transport
is dominant but overshoots (109.69%/133.13% behavior); attention-pattern change supplies an opposing
-12.29%/-38.41%, and interaction is small at 2.72%/5.48%. Subject onset is material but incomplete,
while pre-subject value terms are exactly zero by causal order. All 18 partition cells close to
8.5e-6 or better, the compiled direct weight route replays to 4.2e-9, and no fitting or backwards
passes occur. The next falsification is a sealed v11 construction: capability first, then the frozen
causal-suffix/value-plus-opposing-pattern operation. This is also the relevant test of DAS
memorization—reuse of the operation and weight-defined Q8 interface, not training loss on another
answer/foil complement.

The sealed v11 falsification now passes without any optimization. On 63 capability-selected rows
from new reference/near and dispatch/behind constructions, frozen Q8 retains 99.93%/94.68% of the
full H3 causal behavior in A1/A2. The frozen subject+post-subject+self union retains 99.80%/100.16%
of complete Q8 norm and 100.03%/100.32% of its behavior. The operation signature recurs: value
transport contributes 105.21%/115.50%, pattern change opposes at -5.92%/-17.97%, interaction is
only 0.75%/2.58%, and all pre-subject value coordinates are exactly zero. Closure is below 8e-6;
there are no fits, gradients, or updates. This sharply reduces the memorization explanation for the
Q8 circuit even though it does not rescue constrained DAS as an identification method. The temporal
line is now a transferable Tier-4 component-level circuit: its remaining frontier is cross-task
state sharing and a finite causal-Hankel lower-bound/minimality test, not further within-task DAS.

The first cross-task weight-space screen is positive but is not yet causal identification. Mapping
temporal Q8 through the real H3 output matrix yields an orthonormal eight-dimensional resid18 write
space. The released selective `is/was` constrained-DAS axis places 13.32% of its squared norm in
that space: 19.18 times the isotropic 8/1152 expectation and above every one of 4,096 seeded Haar
controls. The shared component norm is 0.365 and the task-specific complement norm is 0.931, so
neither piece is degenerate. Q8 gauge rotation changes the projector by only 2.98e-8. This licenses,
but does not replace, a held-out causal test that installs the shared and specific write components
separately and together on `is/was` A/P/C families.

That causal test passes after one explicitly preserved sub-ULP float32 repair. The temporal-Q8
component carries 0.466/0.270 recovery on held-out v2 `is/was` A1/A2 and 0.480/0.272 on v3,
donorward on every row. Its P/C collateral is at most 0.0592/0.00261 on v2 and 0.0463/0.00202
on v3. The orthogonal `is/was`-specific component supplies the remaining 0.498/0.289 and
0.513/0.290 recovery; both pieces are positive and compose to the original cDAS intervention
within 2.86e-6 in final logits. Thus a small-norm temporal weight component explains roughly half
the cDAS behavior while remaining selective. This promotes cross-task Q8 reuse from geometry to
causal evidence. The next test is a finite 32x32 upstream-command by downstream-readout matrix
covering all four temporal/is-was quadrants, with zero-fit eight-state prediction and rank controls.

The finite shared-state test passes. A 32x32 matrix crosses 16 temporal and 16 `is/was` natural
upstream commands with the same 32 task-specific downstream contexts. The zero-fit contraction
`H_pred = C R^T` predicts all four exact intervention quadrants at cosine 0.9999966--0.9999997
and relative RMSE 0.084%--0.262%. Cross-task RMS effects are 0.116 and 1.250 versus a frozen
0.0546 observability floor. Commands span all eight coordinates, downstream readers resolve four
at the registered threshold, and rank eight captures effectively all exact matrix energy; thus this
finite family supports a state-dimension lower bound of four and upper bound of eight. Seeded source
permutation drops global cosine from 0.999997 to -0.236. This is a finite causal realization, not a
claim that the transformer's unrestricted Fliess/Hankel system is globally rank eight. The missing
cross-task circuit edge is now upstream: which native `is/was` modules write the shared state.

The complete 36-module atlas finds a distributed upstream and falsifies uncalibrated cross-type
weight ranking. MLP1 is the largest causal shared-Q8 writer (mean target-norm ratio 3.12, coordinate
cosine 0.675, behavioral recovery 0.740), followed by attention9 (1.98, 0.762, 0.543) and several
early MLPs. Eighteen modules clear the basic materiality screen, and the top norm is 4.87 times the
median. Raw output-weight incidence correlates with causal norm at Spearman 0.470 but ranks all nine
top-quartile sites as attention, missing causal-top MLP1; it is therefore not a calibrated selector
across module types. Per the registered distributed branch, the next test greedily composes complete
modules on the 16 discovery rows and confirms one frozen union on 48 untouched A rows before any
attention-head split.

That greedy branch is a valid null: MLP8 alone is the best discovery prefix, and every additional
module worsens both coordinate and behavioral objectives. On 48 untouched rows MLP8 retains strong
shared-effect prediction (behavior cosine 0.941, relative RMSE 0.342, mean effect 82.9% of target),
but its Q8 coordinate cosine is 0.778, just below the frozen 0.80 bar. The eight-module pool
overshoots to 276% mean behavior and coordinate RMSE 3.21. Thus the earlier multi-module singleton
responses are predominantly cumulative/overlapping rather than an additive distributed program.
No attention split is licensed. The next prospective v6 screen partitions MLP8's complete donor
output by source positions before considering its Down-weight/product modes.

That source-position screen passes on 29 jointly capable fresh-v6 rows. The post-cue interval is the
only material proper group: it carries 63.15% of complete-MLP8 mean absolute behavior, predicts the
shared target at behavior cosine .948, and improves Q8 coordinate cosine/relative RMSE from
.782/.672 for complete MLP8 to .839/.618. Prefix is exactly zero, the cue carries only 7.70%, and
subject determiner plus self carry the smaller remainder. The cue-through-query union exactly equals
complete MLP8, as required by causality. This licenses a literal weight split only at post-cue:
contract Q8 with MLP8 Down, causally test its rank-at-most-eight hidden-product modes and complement,
then decompose the activated product into left change, right change, and bilinear interaction.

The direct Down-mode hypothesis is a valid null after one float32-only closure repair. The exact
eight-dimensional row space of `S^T W_Down` carries only 19.48% of complete post-cue behavior;
its exact hidden-product complement carries 79.05% and 78.19% of the final Q8 coordinate RMS.
Rank-one through rank-four direct modes are essentially inactive. Within the small direct branch,
left plus right change retains 97.31% and bilinear interaction is only 5.76%, so product factorization
is not the failure. The topology is instead indirect: MLP8 writes a non-Q8 residual direction that
later computation converts into Q8. The next complete-module removal atlas conditions on precisely
that complement actuation and localizes its converter across attention/MLP9--17.

The conditional converter atlas passes and makes the indirect edge concrete. Removing attention9
under the frozen MLP8-complement actuation eliminates 86.10%/83.91% of complement behavior and
86.61%/81.42% of its final-Q8 norm in A1/A2, with Q8 cosine .9948/.9950. Attention11 and MLP9 are
smaller positive branches; MLP17 is a stable opposing correction. Clamping all attention/MLP9--17
responses removes exactly 100%, and every replay/self-clamp control is exact. Attention9 is therefore
the licensed dominant converter. The next complete nine-head removal atlas will identify its head
interface before source-factor and c_v/c_proj weight compilation.
