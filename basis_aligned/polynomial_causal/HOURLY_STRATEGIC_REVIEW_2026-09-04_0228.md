# Hourly strategic review — 2026-09-04 02:28 UTC

## What counts as circuit progress

A useful decomposition must do more than lower rank or preserve average cross-entropy. It should:

1. state what information is read, what operation combines it, what is written, and who later uses it;
2. group pieces from different heads or MLPs when downstream computation treats them as one function, and split a
   native module when its pieces have different functions;
3. predict held-out examples and genuinely shifted text;
4. produce an executable sufficient computation, or an explicit computation plus background interface;
5. support selective removal, interchange, and editing while unrelated active controls remain intact;
6. compose or reuse the same identified computation across tasks; and
7. remain stable across documents, fitting restarts, and gauge rotations, or be defined by downstream operational
   equivalence.

The end goal is a smaller transparent tensor program that is predictive and manipulable. Rank, variance, reconstruction,
and cross-entropy are evidence or costs, not by themselves circuit identities.

## What changed this hour

- R592 candidate `3f44c224e` passed fresh review on every previous semantic and implementation blocker, including all
  50,304 logits, exact 639/322 call counts, fixed-width centered interventions, independent native reconstruction,
  nonfinite evidence, and receipt binding. It is blocked before execution only because its implementation would hold
  10,677,399,552 evidence bytes concurrently while the machine had less free space.
- A prospective storage amendment is frozen at `6663b7f02`. It writes canonical evidence in manifest order, hashes and
  fsyncs each slice, then removes only the verified current raw chunk. This lowers the exact data peak to
  7,839,996,928 bytes and requires 9,000,000,000 free bytes before model construction and again before SELECT. The
  model-free repair is active; execution remains forbidden pending a different-agent exact-byte review.
- The shared circuit framework reached a useful real red-team rather than nominal test coverage. Claude's 26 attacks
  execute with zero skips: 16 pass and 10 expose missing semantics. The failures cover arm roles and dead controls,
  exact split content, earlier-call nonfinite values, instrument-before-science precedence, projector purity, physical
  shapes, and scientific/diagnostic namespaces. Three attacks also revealed that Claude's glue discarded the very
  role/hash/evaluator inputs it expected the implementation to check. The implementation owner refused hacks and is
  paused while Claude mechanically preserves those inputs in the independent oracle.
- A standalone exact product-projector-to-quadratic-weight compiler landed at `7fe60459d`. For
  $g(x)=(W_Lx)\odot(W_Rx)$ and orthonormal $U$, it emits

  $$
  Q_\ell=\frac12\left(W_L^T\operatorname{diag}(u_\ell)W_R+
  W_R^T\operatorname{diag}(u_\ell)W_L\right),\qquad d_\ell=W_Du_\ell,
  $$

  and exactly reproduces

  $$
  W_DUU^Tg(x)=\sum_\ell d_\ell x^TQ_\ell x.
  $$

  The 25-test focused suite covers direct output, donor-minus-recipient interchange, planted and random cases,
  dependent columns, and invariance under $U\mapsto UR$. This closes the mechanical weight-translation problem for a
  fixed learned subspace. It does not discover the right causal subspace.
- Claude's separate late-layer analysis found that the 384-dimensional channel outside the late MLPs' shared
  768-dimensional frame is written mostly from core features and consumed mainly by the unembedding. Its activation,
  readout, and gate structures are all high-rank. That is a sharper functional description, but the registered small-
  interface and low-rank compression hypotheses failed.

## Mathematical diagnosis

There are now three cleanly separated problems:

1. **Identify a causal variable.** Construct donor/recipient examples where changing one semantic fact has a known
   downstream consequence and answer-preserving controls stay active.
2. **Find its minimal subspace.** Learn a projector using FIT only and require selectivity, held-out prediction,
   restart stability, and gauge-invariant subspace agreement. R540/R556 showed that simply optimizing answer movement
   can find a steering direction rather than an isolated variable.
3. **Compile the selected subspace into weights.** For a bilinear product-space projector this step is now exact and
   tested by `product_projector_quadratic_compiler.py`.

The third problem was never the main scientific bottleneck. The new compiler prevents us from repeatedly re-solving
its algebra and gives a strict equality test once a real learned $U$ exists. Work should now concentrate on the first
two problems and on downstream-use-conditioned definitions of sameness.

## Does the current route still have the highest information value?

Yes. R592 asks whether selector-like attention coefficients and copied-content factors have distinct causal roles
across four native sites. A selective hold would provide a meaningful variable on which to learn a smaller shared
subspace; a controlled null would reject that factorization. It changes our view of computation, not merely its rank.

The late-tail line is demoted as a compression route because every proposed small interface was falsified. It remains
useful as a model-architecture description and as evidence that high-rank channels can still have a simple *role*:
late MLPs write from the core and the unembedding reads the result. We should not convert that role description into a
claim of parameter reduction.

The framework remains worthwhile only as bounded infrastructure. Claude's attacks demonstrate that it can prevent
wrong circuit experiments from looking valid, but it cannot define the scientific counterfactual or select the right
subspace. Its adoption gate remains zero skipped/failing attacks, exact historical shadow parity, no escape hatches,
and less than 1,200 production lines.

## Different routes that remain live

1. **R592 registered factor interchange:** test selector versus copied-content roles directly, then learn a smaller
   subspace only if the hand-specified factor has a selective causal ceiling.
2. **Product-space DAS:** learn $U$ in the 4,608 bilinear product coordinates using several independently built
   counterfactual families, then use the new exact compiler and compare activation and weight interventions through
   the full model.
3. **Cross-head tensor blocks:** factor a tensor whose axes are donor selector feature, recipient content feature, and
   downstream response. Shared blocks, rather than head identities, are candidate computations.
4. **Downstream-response equivalence:** call two pieces the same when every registered downstream reader and
   intervention treats them the same. This directly represents “different internal outputs, same downstream use.”
5. **Predictive-state identification:** learn the smallest intervenable state that predicts future task responses
   across counterfactual families, with extraction/removal tests preventing a merely decodable state.
6. **MLP0 token/context decomposition:** exploit exhaustive token inputs and exact bilinear weights, but judge any
   token grouping by shared downstream behavior rather than activation proximity alone.

## Ranked next actions

1. Finish the R592 streaming repair, obtain fresh independent approval, verify the 9 GB capacity gate and all absent
   public namespaces, then enqueue exactly once through the managed GPU queue.
2. Have Claude correct only the lossy test adapter, freeze the revised oracle, then repair framework implementation
   bytes with a different agent. Require all adversarial and owner tests plus historical parity with no skips.
3. If R592 provides a selective factor ceiling, learn a minimal product-space projector on its FIT counterfactuals,
   compile that same $U$ to quadratic weights, and demand full-model equality plus SELECT/OOD/selective-removal gates.
4. If R592 is a controlled null, do not sweep ranks. Move to cross-head tensor blocks or downstream-response
   equivalence, using the same counterfactual authority.
5. Do not spend another rung compressing the late tail unless a new hypothesis predicts semantic grouping or selective
   manipulation rather than a smaller rank alone.

## Decision

Continue R592 as the primary scientific experiment, with the storage repair as a necessary execution-integrity fix.
Keep the Claude-tested framework and exact DAS weight compiler as supporting infrastructure. The next actual DAS claim
must concern a learned, selective real-model subspace; exact compilation of a fixed projector is now solved but is not
by itself circuit interpretation.
