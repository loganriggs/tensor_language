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
