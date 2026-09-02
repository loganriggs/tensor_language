# Causal interactions between earlier writes and later bilinear layers

**Updated:** 2026-09-02 18:24 UTC

## Short answer

We have done several pieces of this analysis, including exact interaction decompositions and causal all-subset
experiments. We have **not** yet reached the level proposed here: for a later MLP such as MLP10, enumerate the
self- and cross-interactions among all earlier attention and MLP writes, determine which of those interactions each
of our known circuits uses, and then refine the useful source pairs into particular heads or bilinear units.

The closest successful results are:

1. an exact 3-by-3 source-interaction graph for the second attention layer in a toy model;
2. exact all-subset causal decompositions for a few selected sets of heads, MLPs, and paths;
3. an exact expansion of MLP1's response to an MLP0 branch over the named residual sources entering MLP1; and
4. a gradient screen of every bilinear product in MLP8, MLP9, and MLP12, followed by a real intervention on the
   selected products.

Those are real precedents, but none is a complete later-MLP interaction atlas indexed by our 62 circuits. My present
view is that the proposed atlas is one of the best next mathematical directions. It is much closer to our actual
interpretability objective than rank reduction: find reusable computations, split components that do different jobs,
predict their effects on held-out examples, and remove one computation without damaging unrelated ones.

The new paper, [*The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*](https://arxiv.org/abs/2606.27510),
strengthens the motivation. Vaidyanathan et al. show that the usual activation-patching effect of one component also
contains the interaction between that component and the rest of the model. They also show that multi-component
effects contain pairwise and higher-order interactions that single-component rankings cannot recover. Our factorial
experiments already measure this kind of interaction for small, chosen component sets. The tensor structure gives us
an additional advantage: inside a bilinear layer, the source-pair expansion is exact and cheap enough to use as a
map for deciding which expensive causal interventions to run.

## 1. The exact computation for MLP10

Let `r` be the residual vector immediately before MLP10's RMS normalization. The model's residual recurrence lets us
write it exactly as a sum of earlier writes, with the learned residual-mixing coefficients included:

`r = r_embed + sum_{j=0}^{10} r_attn_j + sum_{j=0}^{9} r_mlp_j`.

Here each `r_attn_j` is the output-projected attention write and each `r_mlp_j` is the output of an earlier bilinear
MLP, after multiplying by the scalar coefficients with which it reaches block 10. The repeated embedding/skip path
can be combined into one source. Thus a first coarse MLP10 analysis has 22 named sources: one embedding source,
11 attention writes, and 10 earlier MLP writes.

RMS normalization multiplies the whole vector at one token by a common scalar `g = 1 / rms(r)`. For a fixed forward
pass, define `z_s = g r_s`. Then the normalized MLP10 input is still an exact sum:

`z = sum_s z_s`.

The shipped bilinear MLP is

`MLP10(z) = Down[(Left z) * (Right z)] + bias`,

where `*` is elementwise multiplication. Substituting the source sum gives the exact ordered source-pair expansion

`MLP10(z) - bias = sum_s sum_t B_10(z_s, z_t)`,

`B_10(z_s, z_t) = Down[(Left z_s) * (Right z_t)]`.

This is the computation the user's proposal points to. It contains:

- a self-interaction `B_10(z_s,z_s)` for each earlier source;
- a cross-interaction between distinct sources, best reported as
  `B_10(z_s,z_t) + B_10(z_t,z_s)`; and
- an exact sum back to the native MLP10 write, apart from explicitly retained floating-point rounding error.

With 22 sources there are 484 ordered terms, or 253 terms after combining the two orders of every distinct pair.
This is not an exponential `2^22` experiment. The bilinear equation produces all pairs directly.

We can refine a selected pair to the native hidden units. If `l_u`, `r_u`, and `d_u` are hidden unit `u`'s Left
reader, Right reader, and Down decoder vector, its exact contribution is

`d_u (l_u dot z_s) (r_u dot z_t)`.

This tells us which decoder vectors and which compositions of their two input readers participate. It is useful
evidence even if the native hidden-unit basis is not the simplest semantic basis.

## 2. What “causal interaction” should mean here

There are three related measurements, and we should not conflate them.

### 2.1 Exact local interaction term

Subtract one exact `B_10(z_s,z_t)` term from the MLP10 output while leaving every other MLP10 term fixed, then let
MLP11 through the logits recompute. This answers:

> Does later computation use this particular source-pair contribution written by MLP10?

It is a clean intervention on a mathematically exact part of MLP10. It does not say that deleting source `s`
upstream would have the same effect, because an upstream deletion also changes attention patterns, normalization,
and every intervening layer.

### 2.2 Physical upstream-source intervention

Remove or replace an earlier attention/MLP write and recompute the network. This answers:

> What is the total causal effect of changing this actual earlier component in this background?

As the paper emphasizes, this effect includes how the changed component interacts with everything left in the
residual stream. It is therefore background-dependent rather than an isolated intrinsic importance score.

### 2.3 Factorial interaction between interventions

For two interventions `A` and `B`, run all four states and compute

`I(A,B) = Y(A,B) - Y(A,0) - Y(0,B) + Y(0,0)`.

`Y` can be negative CE on a circuit's positions, a named downstream write response, or another frozen behavioral
metric. A nonzero `I(A,B)` means the joint causal effect is not the sum of the two individual effects in this
intervention coordinate. For more components, the same inclusion-exclusion calculation gives every pairwise and
higher-order term. This is the Möbius/factorial computation already used in the repo, and is the discrete counterpart
of the paper's cross-component interaction.

The exact bilinear term in section 2.1 and the factorial effect in section 2.3 answer different questions. The first
locates where two inputs are multiplied inside MLP10. The second measures how two interventions combine through the
whole nonlinear suffix. A convincing circuit account should connect them: an exact local term predicts a selective
factorial effect, and the physical intervention confirms it on held-out data.

## 3. Relation to the multiple-mediators paper

The paper writes the ordinary noising effect as a pure path effect plus an interaction with bypass paths. Intuitively,
the effect of changing component `M` depends on what all the other residual-stream routes are carrying. For multiple
patched components, the joint effect is

`sum of individual patch effects + sum of pair and higher-order cross-interactions`.

This matters directly for our work:

- ranking one head or MLP at a time can miss backup or conditional computations;
- two modules with small individual effects can have a large joint effect;
- a component can look important only because of the background in which it was patched; and
- “remove each component and add the losses” is not a valid description unless the measured interaction terms are
  small.

The paper's local approximation says the interaction is a mixed Hessian term of the downstream function. That gives
us a useful screening computation:

`interaction(A,B) approximately equals D^2 Y[z](delta_A, delta_B)`.

But the paper deliberately uses exact patching rather than gradients for its causal conclusions. We should do the
same: use gradients and Hessian-vector products to search a large interaction space, then use finite interventions
and all required backgrounds for the claim.

Our tensor network makes the local part stronger than a generic Hessian approximation. The multiplication inside a
bilinear MLP gives the source-pair output terms exactly. The suffix from that MLP to CE is still nonlinear because it
contains later RMS normalizations, bilinear attention/MLPs, the logit cap, and CE, so behavioral interaction claims
still require finite tests.

## 4. What we have already done

| Existing result | What was computed | What it establishes | What it does not establish |
|---|---|---|---|
| `tn_gauge` F13 | The toy model's second-attention QK score was expanded exactly into all nine ordered pairs over embedding `E`, attention0 output `A`, and MLP0 output `M`. | `M x M` was 70.1% of score mass; the six largest of nine blocks recovered the model within `+0.0001` CE. This is a genuine source-interaction graph. | It is a toy attention-QK result, not a later-MLP or 62-circuit atlas. |
| `tn_gauge` F18 | On bilin18, remove `E`, `A`, or `M` only from attention1's QK input. | MLP0 output is causally essential there: `+0.6756` CE when removed, versus `-0.0106` for embedding and `-0.0002` for attention0. | It tests coarse single sources, not every source pair. Per-head QK normalization prevented simply copying the toy algebra. |
| Rung 457 | All 16 subsets of four equality/copy attention terms. | Individual importance is not additive; early and late terms partly substitute across depth. | Whole heads/terms were still too coarse to identify the shared computation. |
| Rung 466 | All 32 subsets of MLP8/9/12 and attention14/MLP17 under two matcher sources. | A task-shaped MLP group, a broad-suppression group, and a material cross-group interaction were causally separated. | It covered five selected public writes, not their input-source or hidden-unit interactions. |
| Rungs 467-468 | Gradient fingerprints for all 13,824 native bilinear products in MLP8/9/12, followed by replacement of 1,358 selected products. | The screen found a real held-out code-specific causal component: effect cosine `.885/.864` to removing the complete three-MLP group. | It failed on natural text: magnitude, controls, multi-MLP membership, and interactions did not transfer. This is the main warning that gradients are a shortlist, not identification. |
| Rungs 473-474 | Exact eight-state query-position factorial for MLP8/9/12, under two different removal definitions. | Pair and triple causal interactions are material. | Their assignment changes with the intervention definition. Higher-order terms are not automatically a unique semantic basis. |
| Rung 481 | All 16 subsets of exact MLP0 branches `T/C/I/S`, measured on the 62 circuit masks. | We did directly ask whether known circuits discriminate exact branches. | At this grain the branch damage was as large on controls and pair interactions were unstable. The 62-mask observation was not selective enough. |
| Rungs 484-486 | Exact finite path decompositions through attention1 (`A/B/V`) and through direct MLP0, attention1, and MLP1 (`D/A/M`). | The path profiles are extremely stable; T and I use attention1 differently; the MLP1 part dominates the full block-1 carrier. | No proper small path explained the effect, and token/bigram labels did not predict held-out downstream effects. |
| Rungs 487-491 | Exact polarization of MLP1's bilinear response, then expansion of its native-state term over embedding, attention0, MLP0 `T/C/I/S`, and attention1 sources. | This is our closest flagship later-MLP result. Attention1 was the unique named source necessary for both the T and I response on held-out intervention outcomes. | It expanded `B(delta_branch, native_state)`—one changed branch against every named source—not all native source-by-source interactions in MLP1. Rung492 also showed that subtracting attention1 only at MLP1 input was not a portable causal path. |
| Rung 500 | A finite L5H5-score-to-L8H4 substitution and MLP9's response, on prospective data. | MLP9 is a calibrated downstream reader of the copy score: response cosine `.835-.860`, versus about `.11` for payload and `-.80` for the wrong score. | It does not yet say which source pairs inside MLP9 compute that response. |

The detailed receipts are in [`basis_aligned/tn_gauge/GOALS.md`](basis_aligned/tn_gauge/GOALS.md),
[`basis_aligned/tn_gauge/SUMMARY.md`](basis_aligned/tn_gauge/SUMMARY.md), and sections 2584-2624 of
[`basis_aligned/bilinear_quotient/BILIN18_CONNECTION.md`](basis_aligned/bilinear_quotient/BILIN18_CONNECTION.md).

## 5. Have we established the proposed level of interpretability?

No. We have established the required algebra and several small examples, but not the complete causal object.

For a circuit `c`, the missing object is approximately

`F_c[s,t] = causal effect on circuit c of the MLP source-pair term (s,t)`.

It should be stored per example before averaging. Across the 62 circuits, each source pair then has a causal
fingerprint

`v[s,t] = (F_1[s,t], ..., F_62[s,t])`.

This could reveal exactly the structures we care about:

- two earlier writes used together by several circuits;
- one MLP that contains separate source-pair groups for separate jobs;
- several heads whose projected writes are interchangeable inputs to the same later computation;
- a backup computation visible only when its partner is absent; and
- a shared input feature that branches into different downstream outputs.

We should only merge terms when their downstream actions remain interchangeable on held-out data. We should split a
module when subsets have different circuit fingerprints and can be removed independently. This is downstream-defined
grouping, not grouping by parameter rank or activation cosine.

## 6. Gauge and basis issues

The coarse named sources are better posed than individual hidden vectors. A complete output-projected attention write
and a complete MLP write are actual addends in the residual stream. A change of coordinates inside an attention head
does not change its output-projected write. Similarly, the complete bilinear tensor and each source-pair output are
unchanged by the ordinary per-unit rescalings and permutations of the native bilinear factorization.

The individual native MLP units are less canonical. Each hidden unit has two input readers and one decoder vector;
their scales can trade off while leaving the product unchanged, units can be permuted, and alternative tensor
factorizations can sometimes express the same bilinear map. Unlike an ordinary linear hidden layer, arbitrary hidden
rotations are not generally a gauge because elementwise multiplication pins the product coordinates. But the native
unit list is still not guaranteed to be the sparsest semantic explanation.

Therefore the sensible order is:

1. find circuit-selective source pairs at the level of actual residual writes;
2. refine only the selected source pairs into heads, MLP branches, or native bilinear units;
3. group the refined pieces by downstream causal fingerprint and interchange behavior; and
4. compare alternative factorizations by held-out prediction, clean extraction, selective removal, and reuse—not by
   sparsity or storage alone.

This uses the native vectors as evidence without treating them as the true basis.

## 7. How gradients should be used

For circuit metric `Y_c`, let `g_c = dY_c / d(MLP10 output)`. The first-order effect score for a source-pair write is

`score_c(s,t) = g_c dot B_10(z_s,z_t)`.

This can score all 253 coarse interactions and all 4,608 hidden-unit contributions far more cheaply than separately
running the suffix for each one. We can also compute mixed Hessian-vector products when the important interaction may
be created after MLP10 rather than in MLP10 itself.

The safe workflow is:

- use gradients on a discovery split to identify a small set of source pairs or unit groups;
- freeze the groups and all thresholds;
- subtract or replace those exact terms and recompute the full suffix on confirmation data;
- test the effect both when likely partner components are present and absent;
- validate on held-out documents and, where available, a different task/register; and
- require low damage to unrelated circuit metrics.

Rungs 467-468 show why the last four steps are necessary: a gradient-selected product set was genuinely causal on
held-out code data but did not generalize to natural text.

## 8. Concrete next experiment

Start with MLP9 rather than scanning every later MLP at once. Rung500 gives MLP9 a calibrated semantic observation:
its output responds strongly to the correct copy-score substitution and rejects payload and wrong-score controls.
That makes the experiment identifiable before looking at source-pair outcomes.

### Stage A: exact MLP9 response decomposition

For the already calibrated L5H5-score-to-L8H4 action, record the actual sources entering MLP9 in the recipient-absent
and score-hybrid runs. Expand both MLP9 outputs into source pairs and subtract term by term:

`Delta MLP9 = sum_{s,t} [B_9(z_s^hybrid,z_t^hybrid) - B_9(z_s^absent,z_t^absent)]`.

This exactly answers which earlier-write interactions produce MLP9's known copy-score response. MLP9 has 20 coarse
sources—embedding, attention0-9, and MLP0-8—so there are 210 unordered source pairs. Keep per-token results for copy,
non-copy equality, and non-copy controls.

### Stage B: use circuit gradients only to shortlist

Contract each exact pair response with:

1. the final copy-task metric;
2. MLP9's own calibrated response direction;
3. supported metrics from the 62-circuit dossier; and
4. unrelated-position controls.

Do not select by norm or rank. Select pairs that explain the known score response, reject the payload response, and
have a stable circuit fingerprint across discovery halves.

### Stage C: finite causal confirmation

On confirmation documents, subtract each frozen pair group from MLP9's write and recompute MLP10-17. Run partner
present/absent backgrounds and all subsets for any small proposed group. Require:

- held-out prediction of the per-example effect;
- recovery or removal of the intended copy behavior;
- preservation of unrelated circuits;
- stable pair/group interactions across document halves; and
- a payload and position-shift control.

### Stage D: refine the successful pairs

If, for example, `attention5 x attention8` is selected, split those complete attention writes into output-projected
heads. If `MLP0 x attention8` is selected, use MLP0's existing `T/C/I/S` decomposition. If an MLP source remains too
broad, expand only that pair into the 4,608 native bilinear unit contributions and regroup them by their downstream
causal fingerprints. This turns the combinatorial problem into coarse-to-fine search without assuming heads or native
MLP units are the final basis.

### Stage E: generalize to MLP10 and other circuits

After the calibrated MLP9 case works, repeat the 22-source/253-pair calculation for MLP10. Use each circuit's own
task-conditioned metric from its dossier rather than assuming that every one of the 62 masks supplies an equally good
observation. Rung481 and rung499 already showed that a broad or low-support circuit battery can hide a real relation.

## 9. Does this change the current direction?

It changes the immediate follow-up, but it does not justify abandoning the current rung501 directed score experiment.
Rung501 asks a prerequisite question: which of four equality-score components can causally substitute for which
others according to both copy behavior and the calibrated MLP9 reader? That can identify reusable upstream score
computations below the whole-head basis.

The change is that the next step should not jump from a successful directed edge to rank, an SAE, or a generic
compression objective. It should explain the edge through the exact MLP9 source-pair response above. In other words:

`shared score action -> exact earlier-write pairs entering MLP9 -> selected hidden terms -> held-out circuit effects`.

This would connect the present cross-head grouping work to a literal compositional mechanism. If rung501 finds no new
edge, the calibrated L5H5-to-L8H4 edge is still sufficient to develop and validate the MLP9 interaction instrument.
The direction therefore survives either rung501 outcome.

## Bottom line

We have shown that interactions matter and have exactly decomposed them in several restricted settings. We have not
yet produced the full causal interaction basis for later MLPs or for the 62 circuits. The proposed source-pair atlas
is technically feasible because each bilinear MLP exposes its interactions algebraically. The strongest design is to
use gradients for search, exact source-pair terms for local specification, and finite held-out interventions for
causal claims. Starting from MLP9's now-calibrated copy reader gives this program a known positive and typed negative
controls before we search—a much stronger starting point than an unconstrained decomposition of MLP10.

## Update — 2026-09-02 23:26 UTC

The source-pair program has now been carried through MLP10 more deeply than the original proposal above:

- rung507 expanded MLP10 into all253 exact terms; its two gradient-selected terms failed finite causal confirmation;
- rung508 removed all21 fixed six-family sums, but zero was stable across document halves and score implementations;
- rung509's learned coupled dictionary failed planted ground-truth recovery before any model run; and
- rung510 finitely measured all1,012 `(score implementation, exact term)` nodes and tested all511,566 pairs without
  ranking. The instrument passed and716 nodes were active, but zero pair was downstream-equivalent.

This changes the next interaction experiment. Instead of searching another basis over253 terms, rung511 uses the
exact bilinear change from score-absent input `(L0,R0)` to score-present input `(La,Ra)`:

`dL*R0 + L0*dR + dL*dR`.

These left, right, and joint branches are fixed by the computation. All seven nonempty branch combinations are
removed, so inclusion-exclusion gives their singleton, pairwise, and three-way finite causal interactions. The same
combination is compared across all four score implementations, frozen before the other30 circuit families open, and
then physically substituted through layers11--17. If no global branch combination transfers, the registered route is
consumer-local measurement, starting with MLP11's already-known question-form reader. See the full numerical update
and computation in [explanation_2026-09-02_2326.md](explanation_2026-09-02_2326.md).
