# Current strategic update: bilin18 reverse engineering

Last updated: 2026-08-28 04:59 UTC after mapped suffix execution and the
program-context importance result.

## Bottom line

We do not yet have a full reverse engineering of bilin18. We have a complete
structural inventory and several real local mechanisms, but the strict whole-model
executable ledger is still at zero recovery against the current ship's remaining
cross-entropy damage. The project is now concentrating on the missing *interfaces*
between locally compressible computations, because those interfaces—not lack of
local low-rank fits—are the main obstacle to a predictive and manipulable tensor
program.

The current numerical ledgers are deliberately kept separate:

| Ledger | Explained | Remaining | Meaning |
|---|---:|---:|---|
| Structural inventory | 36/36 sites | 0 unlisted sites | Scope coverage only, not explanation |
| Named behavior | 32.1% ± 6.4% | 67.9% | Behavioral categories with names |
| Named causal recovery | 10.923% | 4.72714 nats / 89.077% | Causal interventions under its own denominator |
| Current-ship executable recovery | 0% | +0.8976 CE | The strict compositional target |
| Discovery 36-site ceiling | 55.038% | 2.50365 nats / 44.962% | Exploratory same-realization upper bound |
| Independent held-out ceiling | 53.694% [52.922, 54.387] | different 5.77495-nat stake | Replication of the ceiling, not ship credit |

Compiler-v2.1 remains a clean negative result: its joint MLP0/1 program recovered
33.692% of its teacher-KL stake and therefore earned no global executable credit.
That result is useful because it rejects the idea that a good local affine code is
automatically the code the downstream suffix needs.

## What we currently understand about MLP0

MLP0 is exactly a quadratic tensor map after RMS normalization:

$$
z=\operatorname{RMSNorm}(x),\qquad
m_0(z)=D\big((Lz)\odot(Rz)\big)+b.
$$

Its hidden products are continuous features. The strongest supported semantic
picture is not a hard partition in which all numbers or punctuation become the same
state. It is a shared lexical component plus token-specific and context-specific
continuous refinements. Downstream layers can distinguish individual tokens while
also reading a common number-like or punctuation-like component.

There is substantial compressibility evidence. A legacy 256-feature quadratic
surrogate reproduced roughly 97.8–97.9% under its own behavioral denominator, and a
C512 program compressed the native `Down` map by about 72% while retaining small
ordinary output errors. These are useful compiler upper bounds, not yet proofs that
the complete downstream computation reads a simple 64-dimensional semantic code.

The rank-64 MLP0 basis is currently best understood operationally: it is a compact
continuous output slice that carries measurable downstream causal effect. Its axes
do not yet have a unique semantic naming. Because the basis has an orthogonal gauge,
individual coordinate names would be arbitrary unless a downstream sparse program,
canonicalization rule, or intervention law selects a preferred gauge.

The next experiment is designed to discover that meaning from downstream use. It
compares, on identical fresh rows and initialization:

$$
L:\ \text{fit projected MLP0/1 coordinates locally}
$$

against

$$
R:\ \text{fit the same programs through suffix KL},
$$

then asks whether an explicit same-forward transport

$$
\widehat p_1(z_1,p_0)=\widehat p_1(z_1)+p_0A
$$

improves observational and edited responses. If suffix training or transport wins,
the 64-dimensional code has a principled suffix-relative functional meaning: it is a
compact state that the downstream suffix actually discriminates and transports on
the tested distributions and interventions. This would not make its gauge-dependent
axes uniquely semantic. If both fail, we will not
declare MLP0 uninterpretable; we will test whether the fixed output basis is wrong
using a prospective suffix-Fisher/oracle residual-rank basis assay.

For the fuller MLP0 account, including how the lexical and context decompositions are
computed, see [MLP0_CURRENT_UNDERSTANDING.md](MLP0_CURRENT_UNDERSTANDING.md).

## Semantic correction made this review

The physical student replacement had an important implicit type that is now explicit.
For orthonormal (B_l), state (P) means

$$
m_l^{P_B[N]}
=m_l^N+\left(\widehat p_l-m_l^NB_l\right)B_l^\top
=m_l^N(I-B_lB_l^\top)+\widehat p_lB_l^\top.
$$

Thus it installs the predicted rank-64 code while preserving the orthogonal
complement of the live frozen-ship surrogate (N). It does **not** preserve the
native-original MLP complement. The latter would be impossible with zero native
calls unless a separate complete compiler predicted it.

The runtime now rejects raw tensors and native-O handles on this path. A deployed-N
write is one-use and bound to site, current-state tensor, forward nonce, and broker
issuer. The fit state called `P/P/N` is precisely
`P_B0[N0] / P_B1[N1] / N2`. This remains a conditional slice correction, so the
standalone description length must include the deployed N producer.

This closes the mathematical ambiguity but not real-model provenance. The next
adapter must observe that N0/N1/N2 were recomputed exactly once at their live states,
that native MLP0/1/2 were called zero times in student scope, and that the real outer
forward and restoration ledgers closed.

## Active numerical result

The corrected middle-band native-feature sweep has completed its empirical ridge curve:

$$
k=0:\ 55.038\%,\quad k=512:\ 58.713\%,\quad k=1024:\ 60.619\%,
\quad k=2048:\ 63.378\%,\quad k=4608:\ 67.544\%.
$$

The respective discovery-row gains over k=0 are 3.675, 5.581, 8.340, and 12.506
percentage points. Ridge k=4608 closes 99.925% of the separately measured middle-band
headroom, but it uses the complete original 4608-feature bank and is not compression.

The run's separately labeled exact arm remains invalid as an identity: source audit
found that it omitted `Down_bias`, so its reported 68.059% is a joint zero-bias
ablation. The prospective repair has now completed. In one shared-object run, the
corrected bias-inclusive hooks and leaving MLP4–15 live both give
$5.098802047929132$ CE. Their pooled CE difference and maximum per-row loss-sum
difference are exactly zero; counts and an exact-arm replay are bit-identical. Under
the legacy descriptive denominator this is 67.5533%, not the invalid 68.059%. This
validates the hook identity only. It does not promote the empirical ridge family or
move any held-out, causal, executable, OOD, edit, or current-ship ledger.

Even the valid curve is not certified compression. Factor-complete feature-specific
standalone prices across the twelve sites are 21.234M reals at k=512, 42.467M at
k=1024, 84.935M at k=2048, and 191.103M at k=4608, before the existing 23.89M-reals
base program, indices, metadata, and runtime. The original factors also still execute.
On the common discovery stake, total structural prices per recovered nat are therefore
7.794M for the base and 13.801M, 19.658M, 30.836M, and 57.162M for k512 through k4608.
All five points are nondominated because both cost and fidelity rise. The ratios and
successive marginal prices nevertheless worsen monotonically in this fixed grammar.
The first held-out k512 attempt failed before scoring because the evaluation hook
omitted the compiled context features. The source-corrected rerun is complete. On
`skip7000`, k512 improves over k0 by $+3.675$ percentage points with 95% interval
$[3.514,3.841]$; on `skip11000`, it improves by $+3.811$ points with interval
$[3.671,3.957]$. The gain changes by only $+0.136$ points while the absolute level
drops by $1.208$ points. This is a held-out FineWeb document-split replication of the
incremental feature return, not OOD evidence and not promotion to an executable or
causally sufficient program.

## Genuine blockers

There is no fundamental blocker from missing RSPD or FineWeb data at this point. The
model implementation needed by the repository is available through the tracked
`jacclust.tt_model` loader path and model weights are cached; fresh FineWeb roles are
deliberately not frozen until the source closure authorizes them.

The current blockers are execution authorization and cost integrity:

1. The source-closed observed adapter is now implemented around the real frozen-ship
   dispatch surface. It counts all 18 attention and MLP dispatches, mints N0/N1/N2,
   applies corrections only at MLP0/1, poisons literal native MLP0/1/2 calls, restores
   exact instance-forward state on every exit, and returns only sealed capability
   objects plus an immutable receipt. It still needs independent source audit and one
   authorized full adapter/capability transaction before the suffix run is GO. A
   narrower production smoke of the checkpoint plus frozen dispatch path has passed.
2. The prohibitive full-logit content hashes have now been replaced prospectively by
   a source-bound one-use graph/storage identity. Ordinary mutation, replacement,
   detach, graph drift, and nonfinite values fail closed without transferring logits
   to CPU. Independent review gives GO-to-commit, but execution remains NO-GO until
   the observed adapter closes before aliases escape and enforces the checkpoint's
   50,304 logit width; that source path now exists but remains unaudited. Token IDs
   remain restricted to the tokenizer's 50,257 entries;
   slicing logits would change CE/KL normalization.
3. The graph-connectivity check currently performs an extra full suffix backward per
   fit batch. It needs a measured benchmark or a cheaper structural proof.
4. Document-shuffle and A-null controls remain intentionally unauthorized until a
   mapped-row capability binds the actual source/target document map.

## Pruned and ranked next actions

The ranking uses expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and redundancy with completed work.

1. **Audit and production-close real N-write provenance; remove the remaining backward cost trap.** This
   is the smallest boundary that turns the audited tensor algebra into a legal and
   feasible model forward. The typed N-write repair, cheap graph identity, full frozen
   dispatcher, and observed adapter are implemented; independent review, a production
   numerical closure, and the connectivity benchmark remain.
2. **Finish mapped controls and the collector/trainer/freezer/validator, then run
   matched L/R/T.** This is the direct discriminator between reconstructive geometry
   and a compact code downstream computation actually uses.
3. **Replicate one selected middle-band k on fresh documents with complete factor
   pricing and native-call poison.** This cheaply falsifies whether the promising
   58.71% curve is portable and genuinely simpler.
4. **Insert any admitted early program into a common-support
   attention × early-MLP × deep cube.** Only this can turn a local win into recovery
   against the real +0.8976 current-ship residual and expose interaction failures.
5. **Condition MLP2/3 on the admitted early state, then compile middle attention as
   distinct routing and value programs.** Existing pair effects are location-specific,
   so independent component fits and transported interaction coefficients are pruned.

Repeated hard clustering, wider Euclidean regressors on the same MLP0 inputs, more
untyped pair scans, and another unbounded k sweep are currently lower priority. They
would add local fit numbers without resolving the missing causal/compositional
interface.

The governing simplicity criterion is now consequence-tested rather than a bare
parameter ratio. `SIMPLICITY_CONSEQUENCE_VALIDATION_V1.md` specifies which capability
each complexity measure must predict—storage, compute, statistical generalization,
gauge invariance, composition, or edit locality—and how definitions will compete on a
common candidate bank. `params/nat` is retained as structural efficiency, not treated
as self-validating interpretability or literal MDL bits.

Claude's new `ops/circuit_audit.py` is retained as a useful bootstrap component-set
screen, but its initial labels are broader than its estimands. The static audit in
`CIRCUIT_CONSEQUENCE_HARNESS_AUDIT_2026-08-28.md` records that its two FineWeb splits
are held-out replication rather than OOD, constant ablation measures importance
rather than selective removal, and a token table is one extraction candidate rather
than a validation of a simplicity definition. Its first run completed in 144.7 seconds
for 16 of 55 certified registry entries. It validly finds a wide token-table recovery
spread (about $-19.3\%$ to $96.1\%$), but entries sharing the same native component set
necessarily receive the same result regardless of their different semantic claims.

The local-only model boundary now pins and validates the exact checkpoint, including
the 50,304-wide output head, and exposes every attention and MLP site through explicit
sequential dispatch. A separate frozen-ship loader validated the canonical 1.468 GB
artifact, its manifest, row receipt, and realization tree on CPU and reproduces its
attention and MLP dispatch formulas without importing historical runners. Synthetic
tests verify that deployed MLP1 reads the effective MLP0 write and MLP2 reads the
effective MLP1 write. The sealed observed adapter implementation now wraps that path.
It counts the full dispatch, mints typed N0/N1/N2 handles, applies P corrections only
at MLP0/1, poisons literal early-native calls, restores exact instance state under
`BaseException`, and returns no raw logits or dispatcher aliases. Independent source
audit and an authorized full adapter/capability transaction remain, so this does not
yet authorize the suffix experiment.

A production-shape implementation smoke then loaded and byte-validated the pinned
checkpoint and frozen ship, ran all-zero synthetic token IDs through the complete
explicit dispatch, and returned finite float32 logits of shape `[4,256,50304]` in
11.987 seconds including validation and load. The receipt is
`bilin18_observed_dispatch_smoke_receipt.json`. It used no corpus rows or scientific
score. This closes facade/ship numerical composability, not the adapter/capability
transaction or suffix hypothesis.

## Verification state

The focused runtime/capability/observed-adapter suite passes 59/59, and the complete CPU
suffix suite passes 134/134 after the typed-write and graph-identity changes. Independent
review gives exact-byte GO-to-commit
for the graph identity under the declared source-closed adapter boundary. No fresh
teacher or suffix scientific outcome was loaded by these repairs, so suffix-transport
execution remains explicitly NO-GO until the observed adapter is source-closed and
independently audited.

## 04:59 UTC hourly checkpoint: a real mapped teacher and a better global target

The explanatory balances have not moved: structural inventory remains 36/36, named
behavior remains $32.1\%\pm6.4\%$, named causal recovery remains $10.923\%$, and
strict current-ship executable recovery remains $0\%$ of the $+0.8976$ CE gap. No
scientific suffix row has been exposed. What changed is the executable boundary and
the evidence about which sites deserve the next compiler budget.

### Highest-priority action executed

Document-shuffled R/S training now has a source-closed execution path. A separately
constructed mapped broker:

1. binds its ledger to both the base run context and the exact document-block plan;
2. authorizes the source tokens and registered optimizer schedule before the P/P/N
   student forward;
3. validates the complete frozen fit tensor, exact mapped target indices, and exact
   target tokens before spending the one-use student trace;
4. runs the autonomous O/O/N teacher only on those target tokens; and
5. returns only a sealed loss capability, with teacher parameters detached and the
   native MLP0/1 call ledger closed exactly.

The ordinary broker cannot be repurposed for mapped execution, and a mapped broker
cannot enter the ordinary OON route. A mutated target batch is rejected before trace
consumption and can be followed by the valid target transaction. The fit owner now
runs the complete deterministic document-shuffled R/S optimizer schedule and returns
a distinct `MappedFitCandidate`; this type is deliberately ineligible for true-row
candidate selection. A-null/T remains closed because its false-paired object is an
upstream parent code, not a target-token O/O/N teacher. Shuffled-L remains closed
because it needs native coordinate labels at the mapped target trajectory states.

The full suffix/facade/ship boundary suite passes 170/170 in 46.27 seconds. This is
execution capability, not a numerical result; publication, calibration, the final
evaluator, mapped-L, and A-null/T still keep the scientific lifecycle NO-GO.

### New compositional evidence

The site-ranking problem is more severe than the earlier stack-level scale correction
showed. One-at-a-time ablation importance and leave-one-out importance have Spearman
correlation only 0.026/0.011 on the two large FineWeb roles. More importantly, a
program-context estimand—how much a live site improves over its own token table while
all other sites also use their tables—is strongly *anti-correlated* with one-at-a-time
importance: $-0.664/-0.687$. The stable top six are MLP17, MLP16, attention16, MLP15,
attention14, and attention17. Front MLP0--3 add approximately zero over their tables
in this particular all-tabled context, even though their one-at-a-time ablations are
large; the latter mainly measure how badly an isolated upstream removal corrupts the
live downstream stack.

This alternative complexity/importance definition has now passed a direct consequence
test. At a matched budget of six native sites, the program-context ranking recovers
1.016/1.038 nats from the all-tabled program on `skip7000`/`skip11000`, versus
0.519/0.521 for the one-at-a-time ranking and 0.815/0.840 for the best of eight frozen
random draws. The committed S1739 artifact is not strict current-ship recovery, but
it is concrete evidence that the definition supports a better constrained program
rather than merely producing a nicer description.

The subsequent discovery-only budget curve has also completed at
$K\in\{2,3,6,9,12\}$. Program-context allocation beats one-at-a-time allocation at
every budget on both roles. At K=2 and K=3 the conventional ranking is slightly
harmful while the program ranking recovers about 0.30 and 0.64 nats; at K=12 the
program ranking recovers 1.259/1.281 nats, about 31.0%/30.1% of the all-tabled stake,
versus 0.667/0.676 for one-at-a-time. There is no preregistered clean knee: the
largest marginal jump is from K=2 to K=3, then marginal return declines through
K=12. Because both large roles are spent for this family, the curve freezes a useful
allocation policy but does not confirm it on a new role.

A separate provisional SNR replication does *not* support SNR as a semantic-share
ranking: Spearman rho is 0.079 with $p=0.836$. It does reproduce the predicted
$1/\sqrt n$ shuffled noise-floor law (rho 0.988) and places all ten class deviations
above shuffled controls. The right interpretation is therefore “the deviations are
real but SNR does not order their semantic importance,” not “the semantic classes
vanished.”

### Largest remaining gaps

1. The suffix family still has no published true-row fit, validation numerator,
   calibrated final test, or strict current-ship contribution.
2. Shuffled-L target-state coordinate labels and A-null/T false-paired parent codes
   are not executable, so the strongest causal controls for the early interface are
   absent.
3. The all-tabled program loses roughly four nats, and the newly identified late
   residual sites do not yet have tensor programs adequate in program context.
4. Attention routing and attention value/content remain insufficiently separated;
   no typed late-attention compiler composes with the MLP programs.
5. There is no admitted cross-stack factorial result, genuine second-distribution
   OOD result, or selective extraction/removal result with collateral bounds.

### Pruned top five

1. **Finish the suffix lifecycle through mapped-L, A-null/T, publication,
   calibration, and final evaluation.** It is closest to a preregistered executable
   answer and directly tests whether a suffix-relative rank-64 gauge is more useful
   than local reconstruction. The remaining work is bounded and falsifiable.
2. **Confirm the frozen program-context allocation on a clean role, then use it for
   the next fixed compiler budget.** The advantage holds at every tested budget from
   2 to 12 on both spent roles, with diminishing returns after the K=3 jump; a new
   role is now more informative than another discovery-budget sweep.
3. **Compile MLP17/16/15 conditional on their actual upstream program interfaces.**
   They are the largest stable residuals under the consequence-validated ranking and
   are more relevant to whole-program fidelity than another front-MLP local sweep.
4. **Build typed routing/value programs for attention16/14/17.** These are the
   attention sites that remain valuable in program context; routing-only rank curves
   and one-at-a-time site scans are pruned as insufficient.
5. **Run an attention × early-MLP × late-program factorial cube on one common
   current-ship denominator.** The confirmed MLP redundancy and attention cooperation
   make additive accounting invalid; this is the first composition test capable of
   earning strict whole-model recovery.

Repeated hard token clustering, more SNR-derived semantic rankings, unconditioned
local reconstruction sweeps, and additional one-at-a-time ablation rankings are
pruned. They are now empirically redundant with weaker definitions and do not close
the causal or compositional interfaces.

## 04:24 UTC hourly checkpoint: from adapter closure to a numerical consumer

The model-wide explanatory balances remain unchanged. Structural inventory is
$36/36$, named behavior is $32.1\%\pm6.4\%$, named causal recovery is $10.923\%$
against its registered denominator, and strict current-ship executable recovery is
still $0\%$ of the $+0.8976$ CE gap. The best discovery and held-out ceilings remain
$55.038\%$ and $53.694\%$, respectively, on their separate denominators. None of
these currencies may be added.

One earlier blocker is now obsolete. The complete production-shape P/P/N student plus
coordinate-teacher transaction has run source-closed in 14.637 seconds: all 18
attention and MLP dispatches occurred once, N0/N1/N2 were each produced once, native
student MLP0/1/2 calls were zero, teacher MLP0/1 calls were one each, both 768-vector
moment batches were consumed, the broker ledger closed 1/1/1 with no outstanding
identity, and the hook/coordinator were inert. The immutable implementation-smoke
receipt is `bilin18_observed_adapter_transaction_smoke_receipt.json`. It uses synthetic
tokens and earns no scientific recovery, but real adapter provenance is no longer the
critical-path blocker.

The strategic review instead found a prospective source-closure hole: the row freezer
bound the protocol, runtime, and capability layers but not the real facade/ship/adapter
or any numerical fit, validation/program, or final consumer. Fresh fit rows could
therefore have been materialized before those consumers were frozen, allowing their
implementation to adapt to outcomes. The freezer now fails closed until the observed
stack and six named numerical source/test files exist, are tracked, and belong to the
same source closure. Fit data remain unexposed.

The first numerical pair now exists:
`early_mlp_suffix_transport_v1_fit.py` and its tests. It owns the sequential initialized-Q
denominator pass and exact true-row L/R/S0/S1 optimizer trajectories. Every batch is
bound before its forward to the full frozen fit tensor, registered permutation,
program snapshot, teacher kind, optimizer step, and P/P/N state; all outputs pass
through the observed adapter and one-use broker; Q moments use the registered float64
Chan/Welford merge; returned candidates are unselected CPU states with deterministic
transaction commitments. It has no row/model loader, selection, artifact publication,
or final scorer. Focused fit+lifecycle verification passes 16/16. The gate remains
closed on the still-absent program/selection and final pairs.

### Largest remaining gaps

1. No executable MLP0-to-MLP1 suffix-objective program has yet produced a held-out
   numerator; the rank-64 code therefore has operational but not semantic closure.
2. MLP2's dependence on the transported upstream state is not compiled.
3. Attention's relatively compact routing structure is not connected to an adequate
   value/content program; routing rank alone recovers little value behavior.
4. There is no current-ship, common-denominator cross-half composition result.
5. There is no genuine second-corpus/code OOD result or circuit-specific selective
   edit/collateral result for an admitted executable program.

### Pruned and ranked actions

The ordering jointly favors expected information gain, causal relevance, whole-model
composition, falsifiability, GPU cost, and nonredundancy.

1. **Complete the source-closed suffix transport numerical lifecycle.** Finish the
   validation/program freezer and final evaluator, then execute matched L/R/S/T. This
   is the nearest route to a legal held-out executable numerator and directly tests
   whether local geometry or suffix use defines the useful rank-64 code.
2. **Insert any admitted early package into a current-ship attention × early-MLP ×
   deep factorial cube on identical rows.** This prices interactions against the real
   $+0.8976$ residual rather than a legacy denominator.
3. **Compile conditional MLP2 after the admitted upstream interface.** Existing
   factorial evidence says its contribution is state-dependent, so an independent
   MLP2 fit is not a composable answer.
4. **Build a typed routing/value attention compiler.** Preserve the rank-compressible
   routing interface while giving values a richer, suffix-weighted content grammar.
5. **Run consequence validation only after an admitted executable candidate exists.**
   Compare simplicity definitions by matched fidelity, genuine OOD prediction,
   extraction, selective removal, and non-descendant collateral—not by token-class
   ablation aliases.

Additional clustering, fixed-lag scans, Euclidean MLP0 regressors, unpriced rank
sweeps, and corrected descriptive class replays are pruned for now: they cannot close
the missing executable interface or common-denominator composition gap.

## 04:34 UTC continuation: deterministic selection and lossless T freezing

The validation selector and program freezer now exist as the second numerical
source/test pair. They do not load rows or models. A candidate is admissible only if
its validation receipt binds the exact route-specific metric, one of each of the three
registered learning-rate trials, all $192\times192$ scored tokens, common support and
sufficient-statistic identities, zero student-native calls, inert restoration, and
copy worsening at most $0.01$. Selection is by the unrounded primary metric, then
smaller learning rate, then lexical tensor hash.

Selected dense $1152\times64$ maps are serialized by CPU float64 SVD with the
registered sign convention and a $2\cdot10^{-6}$ maximum replay bound. The freezer
round-trips fixed means, scales, biases, both affine weights, trainability, and route.
Selected L can now initialize T as an exact zero $64\times64$ cross map with only
$A$ trainable, and the fit owner accepts this post-selection T route on the identical
three-trial schedule and OON loss.

Inspection caught and repaired a consequential pre-test error: the first T freezer
shape omitted the trained `cross` tensor, which would have reconstructed every T
candidate as zero-$A$. A dedicated nonzero-$A$ test now proves exact fit restoration,
SVD freezing of the affine maps, and deployment round-trip without changing the cross
map. The expanded facade/ship/adapter/suffix suite passes 156/156 in 44.09 seconds.

This remains implementation progress, not a numerical result. The row gate is still
closed because the final source/test pair is absent, and it must remain closed until
the validation collector, document-shuffled and A-null mapped-row capabilities,
program artifact publisher, calibration bank, and one-shot final evaluator are all
complete. The next immediate implementation target is the validation collector and
mapped-control transaction boundary, not creation of a placeholder final file.

Separately, the corrected past-facing target-class hypothesis has now confirmed on
the previously untouched `skip1200` role: the joint attention-minus-MLP class-ratio
interval is $[0.0498,0.2193]$. This supports a stack-level induction/novel division of
labor and the negative novel damage of attention sites 14 and 16; attention 15 changed
sign and was correctly dropped. This is causal-specialization evidence, but it does
not change any executable, whole-model, OOD, or suffix-transport recovery balance.

## 04:45 UTC continuation: actual row shape, mapped controls, and raw validation receipts

An audit against the row freezer found a critical pre-execution incompatibility. The
canonical cache stores each role as `long[n,513]`, and the implementation amendment
requires truncation to the first 257 tokens: columns `0:256` are model inputs and
`1:257` are shifted targets. The initial fit owner instead required a hypothetical
`long[384,256]` role, so it could never bind the real row receipt. It now validates
and hashes the complete `long[384,513]` frozen tensor and derives inputs only after
that binding. The mapped context follows the same rule. No role was loaded while this
was wrong.

The document-control relation is now executable data rather than metadata. A pure
builder groups contiguous documents by equal row count, requires at least two
documents per stratum, draws the registered nonzero cyclic offset per stratum, moves
whole blocks while preserving their within-document row order, and proves the result
is a row bijection with no fixed document. It covers seed `2026083050` and all twenty
`2026083100+i` A-null plans. A sealed mapped context binds the global plan hash, exact
source schedule, exact mapped target indices, full fit-role hash, and target token
contents. The ordinary true-row broker still rejects mapped controls.

Validation selection can no longer trust manually supplied scalar metrics. Raw
float64 per-row sufficient statistics now reconstruct normalized local MSE or
token-weighted OON KL, global CE, copy CE, and matched baseline copy CE. The copy mask
is the frozen 64-token-history definition on shifted targets and exact positions
64:256. Support counts, zero native calls, restoration, common support, and the raw
statistic tree hash flow into the selector receipt. Both local and suffix pooled
values replay the registered runtime losses. The expanded source-closed suite passes
162/162 in 46.78 seconds.

The mapped **execution** capability is still incomplete: shuffled L needs target-row
student states before native-coordinate labels, shuffled R/S need target-token OON
teachers, and A-null T needs a mapped parent-code block without weakening the normal
same-forward T invariant. The final pair remains absent and the row gate remains
closed.

Claude's separate additivity confirmation materially strengthens the case for the
later common-denominator composition cube. On `skip11000`, the sum of one-at-a-time
MLP removal costs is $2.361\times$ the joint MLP-stack cost, while the analogous
attention sum is only $0.399\times$ the joint attention cost; intervals exclude the
registered nulls and the asymmetry survives all-position scoring. Thus independent
MLP importance is strongly redundancy-inflated and independent attention importance
is strongly cooperation-suppressed. This changes how local component evidence should
be interpreted, but not any executable recovery balance.
