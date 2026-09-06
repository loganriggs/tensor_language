# Hourly strategic circuit review — 2026-09-06 22:14 UTC

## Circuit interpretation target

The target remains a compact tensor program that predicts unseen behavior, composes across tasks,
supports selective causal intervention/removal, and can be evaluated directly from checkpoint
weights plus a small state. A fitted actuator, high response cosine, or a singleton patch atlas is
not sufficient evidence.

## What changed since 21:14

- The direct MLP8 Down-to-Q8 mode hypothesis was rejected. The exact rank-eight row space of
  `S_Q8^T W_Down` carries only 19.48% of the post-cue behavior; its hidden-product complement
  carries 79.05% behavior and 78.19% final-Q8 RMS. The natural route is indirect.
- Conditional complete-module removal localized the converter to attention9, which carries
  83.9--86.1% behavior and 81.4--86.6% Q8 norm. Attention9 H1/H4 then transferred prospectively
  on v10, retaining 80.9--86.3% of all-head behavior and 76.3--83.7% of Q8.
- Source-by-factor analysis found that the H1/H4 query write comes entirely from post-cue sources.
  Native-pattern value transport supplies 99.73%/105.42% behavior; pattern and interaction terms
  are small. The literal `MLP8 Down -> block9 gain/RMS -> L9H1/H4 c_v -> base pattern -> c_proj`
  compiler matches the captured intervention at >.99999989 behavior cosine and about 1e-5 error.
- The upstream MLP8 product cannot be linearized away: its isolated bilinear interaction carries
  19.8--21.2% absolute behavior and 26.7--33.5% Q8 norm. The compiled program retains left change,
  right change, and interaction exactly.
- A zero-refit v11 audit showed that tangent noise is the stable best rank-one regularizer but does
  not improve transfer: full-vocabulary error changes from .872/.949 unregularized to .895/.971.
  KL rotates toward DIM and is worse. Together with the rank7 null and rank8 transfer, this points
  to rank/target misspecification rather than a failed optimizer.
- Exact grouped downstream factorials corrected severe singleton overlap (47.5--50.8%
  nonadditivity). Attention9+11+15 alone accounts for 95.4--97.1% behavior and 95.5--96.0% Q8;
  the large-looking MLP singleton responses have only 1--2% conditional marginal contribution.
- Conditional head localization froze L11H1/H3 and L15H5. On prospective v10, the five-head union
  L9H1/H4 + L11H1/H3 + L15H5 retains 82.6--89.7% of complete three-attention behavior and
  80.8--89.0% of its Q8 norm, with behavior cosine >=.99937 and Q8 cosine >=.98298. Every layer
  contributes conditionally.
- A hash-bound end-to-end manifest now records the dominant L9H1/H4 tensor program, its gauge,
  exact checkpoint maps, causal coverage, and known omissions.

## DAS red team and confounds

- **Regularization versus identification:** noise is useful as an instability guardrail, but two
  distant zero-refit panels show no causal-vector improvement. KL on the same discovery bank does
  not close the “is this the requested subspace?” loophole.
- **Rank misspecification:** successful zero-fit rank8 transfer and failed regularized rank7/rank1
  compression are positive evidence that the interface is genuinely multidimensional.
- **Singleton overlap:** module responses that appeared 10--20% material individually collapsed to
  1--2% conditional marginals. Group factorials must precede head or weight splitting.
- **Head selection leakage:** L11H1/H3 and L15H5 were selected only on v6 and confirmed without fit
  on v10. The next weight decomposition may use v10 diagnostically, but any simplified auxiliary
  operation must be confirmed on another bank.
- **Global completeness:** the three-attention program explains about 96%, not 100%; the remaining
  3--4% is explicitly unresolved and cannot be silently assigned to residual transport.

## Throughput audit, 21:14--22:14

The hour progressed through source localization, direct-weight falsification, downstream module and
head localization, fresh transfer, exact source-factor algebra, literal weight compilation,
bilinear-factor necessity, a second regularization red team, grouped completeness, auxiliary head
selection, and prospective five-head confirmation. Each failure changed the next causal object;
none was rerun merely to seek a pass.

`CIRCUIT_FOCUS: PASS` — every run addressed causal localization, transfer, composition, tensor
compilation, completeness, or a live DAS confound.

`CEREMONY_BUDGET: PASS` — capability-only gates were reused, group tests replaced repeated singleton
scans, and all post-localization experiments were 7--32 forwards with explicit zero-fit prices.

`NOVELTY_LESSON_GATE: PASS` — the hour rejected direct-Q8 MLP modes, an interaction-free MLP,
rank-one regularization as repair, additive singleton accounting, and a single auxiliary head.

## Different next objects, ranked

1. **Auxiliary-head source/factor tensor.** For frozen L11H1/H3 and L15H5, localize exact source
   intervals and pattern/value/interaction terms conditioned on the existing L9H1/H4 program.
2. **Auxiliary c_v/c_proj compiler.** Translate only material terms through actual attention11/15
   value and output weights, then append them to the existing tensor manifest.
3. **Fresh complete five-head compiler confirmation.** Build one new capability-qualified is/was
   bank and test the fully compiled program without refitting or changing head membership.
4. **Residual 3--4% audit.** Only after the five-head weight program transfers, group the omitted
   small attention and MLP responses once; do not resume singleton scans.
5. **Further rank-one DAS fitting.** Demoted unless a new objective predicts an independent
   vector-valued causal object and is selected without using its final transfer bank.

The immediate highest-information object is the auxiliary-head source/factor tensor. It determines
whether all five heads implement one reusable value-transport operation or whether the apparently
compact head set hides multiple task-specific computations.
