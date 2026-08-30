# Current strategic update: bilin18 reverse engineering

Last updated: 2026-08-28 05:10 UTC after closing document-shuffled L execution.

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

## 05:10 UTC continuation: shuffled-L now uses a paired target trajectory

The model-wide explanatory balances remain unchanged: 36/36 structural sites,
$32.1\%\pm6.4\%$ named behavior, $10.923\%$ named causal recovery, and zero strict
current-ship executable recovery of the $+0.8976$ CE residual. This implementation
loaded no role and earns no scientific numerator.

The last document-shuffle execution gap is now closed. Shuffled-L uses two different
P/P/N trajectories under the same current L program:

$$
\text{source trajectory}\longrightarrow
(\widehat p_0(z_0^s),\widehat p_1(z_1^s)),
$$

with the exact autograd graph retained, and

$$
\text{mapped target trajectory}\longrightarrow
(m_0^O(z_0^t)B_0,m_1^O(z_1^t)B_1),
$$

under `torch.no_grad`. The target trajectory installs P0 before constructing its
MLP1 state, so the second label is taken at the mapped document's actual current
autoregressive state, not at a source state or an independent native trajectory.
The complete role, source schedule, document map, target indices, target tokens, and
program snapshot are validated before the one-use source trace is spent.

The mapped coordinate gateway permits MLP0 then MLP1 exactly once, checks detached
finite native outputs, records exact O-call and dispatch ledgers, and is revoked
before the local loss can be consumed. The observed adapter poisons all accidental
literal early-native calls during the target forward; the broker's two authorized
native calls use bound reviewed implementations captured before the poison. A test
proves the authorized calls work while a simultaneous ordinary module call fails.
Mutated target tokens fail before consumption. The deterministic fit owner now runs
same-budget shuffled L/R/S0/S1 trajectories and keeps all of them in the separate
negative-control candidate type. Full boundary verification passes 173/173 in 45.31
seconds.

### Updated gaps and priorities

1. **Implement A-null/T as false-paired parent-code transport.** This is now the only
   unimplemented mapped fit control. It must physically write the true source L0 code
   while feeding a sealed mapped-document L0 code only to the trainable cross map;
   reusing the target-token OON path would test the wrong hypothesis.
2. **Finish artifact publication, validation/calibration ownership, and the one-shot
   final evaluator.** Once A-null/T is source-closed, these are the remaining gates
   before rows may legally be materialized and the L/R/S/T question answered.
3. **Confirm the frozen conditional-greedy allocator on a clean role.** Greedy choice
   transfers within the discovery family and dominates a fixed program-context
   ranking, but both large roles remain spent for certification.
4. **Compile the conditional greedy residual set—MLP17 and attention16/14/11/17/13—
   using their actual program inputs.** This set captures cooperation missed by all
   per-site rankings; more front-layer local fitting is redundant.
5. **Run a common-denominator factorial composition test.** Only an interaction-aware
   attention × early × late lattice can earn strict recovery given confirmed MLP
   redundancy and attention cooperation.

Possible alternatives were pruned as follows. A direct target-token A-null is cheap
but causally wrong; another document-shuffle implementation is redundant; a new OAT,
LOO, SNR, or rank sweep has lower information gain than completing the admitted
lifecycle; and starting the late-site compiler before freezing the suffix source
boundary would leave two unfinished executable families rather than one testable
program.

### New program-level simplicity evidence during this implementation

S1741 replaces independent site ranking with conditional greedy selection: starting
from all 36 token tables, add the native site with the largest marginal recovery
given the sites already native. Selection used only `skip7000`; `skip11000` selected
nothing and served as a transfer check. The selected set is MLP17 plus attention
16/14/11/17/13. It recovers 29.65% on selection and 29.13% on transfer, versus
25.04%/24.37% for the fixed top-six program-context ranking.

This is an unusually direct validation of a simplicity definition. With published
module prices, the greedy program retains 55.741M native reals versus 71.667M for the
fixed ranking—22% less native structure—while recovering 4.8 percentage points more
held-out fidelity. The improvement arises because conditional marginal utility sees
attention cooperation that independent scores miss, and attention modules cost half
as much as MLP modules. It is discovery-only, not strict executable recovery, but it
supports a program-level Pareto objective

$$
\max_{S}\;\Delta\mathrm{CE}(S)\quad\text{subject to}\quad
\mathrm{DL}(S)\le C,
$$

with marginal gains evaluated after composing the current set, rather than “simple”
meaning low rank or low parameter count in isolation. The owning agent's fourteen-
budget Pareto run is currently using the GPU; this review used that interval for the
CPU-side shuffled-L implementation and tests.

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

## 05:26 UTC checkpoint: A-null/T is causally executable; allocation is non-submodular

The accounting currencies remain deliberately unchanged. Structural inventory is
still $36/36$ sites, named behavior is $32.1\%\pm6.4\%$, named causal recovery is
$10.923\%$, and strict executable recovery is $0\%$ of the paired current-ship
$+0.8976$ CE gap. The separate discovery/held-out constant-ablation ceilings remain
$55.038\%/53.694\%$. Completing a transaction boundary is not a scientific numerator.

The last mapped-fit execution primitive is now implemented. For each registered
`A_null_00` through `A_null_19`, a target document first runs through a detached,
native-free selected-L P/P/N trajectory and produces a sealed one-use parent code

$$
\widetilde p_0=\widehat p_0(z_0^{\mathrm{target}}).
$$

The subsequent source trajectory still writes its own executable code

$$
p_0=\widehat p_0(z_0^{\mathrm{source}})
$$

at physical MLP0. Only the MLP1 cross term is false-paired:

$$
\widehat p_1
=L_1(z_1^{\mathrm{source}})+\widetilde p_0A.
$$

The teacher is the true source O/O/N suffix, not the mapped target suffix. Thus the
control destroys only the learned parent/child pairing while preserving the source
task, source residual trajectory outside the cross read, local maps, and loss
support. The mapped parent is bound to the exact identity, program snapshot, issuer,
document plan, target indices, target tokens, and content hash; it cannot be copied,
serialized, mutated, replayed, or consumed by an ordinary route. Broker ledgers
separately record preparation and consumption, and all ephemeral gateways revoke.

Behavioral tests prove all three estimand-defining facts simultaneously: the physical
MLP0 output projects to the source code, the MLP1 correction projects to the mapped
target code when $A=I$, and the KL equals the true source teacher calculation. Only
$A$ receives gradients; both frozen local maps and native MLP0/1 parameters remain
gradient-free. Adapter tests prove the real target forward dispatches P/P/N exactly
once and that an accidental literal early-native call is poisoned. Fit orchestration
constructs parent, source student, and source teacher in that order under the frozen
schedule and returns A-null results in the negative-control candidate type. Focused
verification passes 48/48; the full suffix suite passes 157/157 in 45.28 seconds.

This removes a causal-interface blocker, but the lifecycle remains NO-GO for data
exposure: publication, calibration/validation execution ownership, and the one-shot
final evaluator are still absent. `rspd` is absent but irrelevant to this path. The
cached FineWeb roles needed by current discovery work are local; genuinely fresh
FineWeb/OOD data remain externally blocked because the instance has no HF token and
streaming bandwidth has repeatedly failed the registered time gate. That does not
block CPU implementation or testing.

### Allocation evidence and its limit

S1742 extends conditional greedy selection to fourteen native sites. On the transfer
role, the nested program recovers $12.1\%$ at $K=1$, $29.1\%$ at $K=6$, and $42.8\%$
at $K=14$ of its table-program stake. Nine of the first ten choices are attention;
for every nontrivial budget $K=2,\ldots,14$, it recovers more at no greater native
cost than the independent program-context ranking. This strongly supports measuring
simplicity by a *composed program frontier*—description/storage cost versus held-out
causal fidelity—not by rank, parameter count, or one-at-a-time importance alone.

But the greedy marginal gain rises four times. In particular, `attn7` becomes more
valuable after `attn9` is admitted. Therefore the utility is demonstrably
non-submodular and greedy has no $1-1/e$ approximation guarantee. The measured curve
is a constructive lower bound, not the optimal frontier. A preregistered two-start
swap search is currently using the GPU to measure whether this theoretical gap is
material at $K=6$; this checkpoint used the interval for the CPU-side A-null closure.

### Pruned top five

1. **Finish the suffix lifecycle and run matched L/R/S/T plus all mapped controls.**
   It is the shortest path to the first legal held-out executable numerator and
   directly decides whether the useful 64-D interface is local, suffix-defined, or
   transported. Publication/calibration/final ownership is the remaining blocker.
2. **Insert any admitted early program into a one-support current-ship factorial.**
   Score attention $\times$ early MLP $\times$ late/deep restoration against the
   actual $+0.8976$ residual, with Mobius interactions and document bootstrap. This
   is the first route from a local success to strict whole-model credit.
3. **Compile MLP2/3 conditional on the admitted MLP0-to-MLP1 state.** Existing
   off-distribution and interaction evidence rules out treating them as independent
   regressions; conditional compilation closes the next causal interface.
4. **Compile typed attention routing and value/content for the interaction-selected
   attention set.** Start with 16/14/11/17/13 rather than an OAT list, but use the
   completed local-search result to avoid canonizing a merely greedy set. Routing
   rank without a value grammar is not an executable attention explanation.
5. **Run consequence validation on admitted alternatives at matched fidelity.** Use
   genuine OOD prediction, circuit extraction, selective removal, collateral on
   non-descendants, and composition stability to decide which description-length,
   gate-count, interface-rank, sparsity, or program-grammar notion earns the word
   “simple.”

Further token clustering, isolated MLP0 regressions, OAT/LOO ranking, unpriced rank
sweeps, and additional spent-role allocation searches are pruned. They do not close
an executable interface, establish composability, or move a common denominator.

## 05:43 UTC continuation: negative-control selection and preflight identities

The next source-closed prerequisite is complete without creating the missing final
owner or exposing rows. Validation and canonical freezing now preserve the scientific
type of every mapped control. The selector requires exactly three learning-rate
trials *within* one immutable `(control, route, mapping_sha256)` family and can never
mix a document shuffle, an A-null, or a true candidate. The complete bank is exact:
four document-shuffled L/R/S0/S1 families plus twenty A-null/T families. All four
document-shuffle routes must share one plan, every A-null plan must be distinct, and
the resulting 24 frozen objects remain `FrozenMappedProgram`, not `FrozenProgram`.
Raw validation sufficient statistics bind the control and plan before any scalar is
allowed into selection.

The same mutable program stage now constructs the exact four signed-permutation and
four Haar gauge matrices, role-specific intervention positions/permutations with each
of 32 directions assigned exactly six times, and the teacher-only five-amplitude
calibration decision. These are deterministic program/preflight records; they do not
observe a candidate response.

The full test gate initially failed because these helpers were mistakenly added to
the prospectively frozen pure algebra contract. That failure was honored: the helpers
were moved into the already-declared mutable program stage, and both frozen contract
files were restored byte-for-byte before rerunning. Frozen-input verification now
passes and the expanded source-closure suite passes 172/172 in 55.20 seconds. The
semantic `final.py` pair remains absent, so the numerical source gate correctly stays
NO-GO; a partial placeholder has not been used to unlock rows.

S1743 adds a useful but limited allocation fact. The greedy K=6 set has no improving
single swap among all 180 neighbors, so the demonstrated non-submodularity does not
hide an immediate one-swap improvement. Its random-start arm was capped while still
improving and therefore cannot establish basin structure. A corrected three-start
run-to-convergence experiment currently owns the GPU. This does not alter the
compiler target or any scientific recovery currency.

The immediate ordering is now: (1) implement the real validation/final collector and
semantic terminal validator while keeping the file-existence gate closed until they
are complete; (2) publish the canonical true/mapped program bank, calibration and
gauge receipts with full fit/row/source/protected bindings; (3) execute matched
L/R/S/T; (4) insert any admitted program into the current-ship interaction cube; and
(5) compile conditional MLP2/3 and typed attention from that composed residual.

## 05:53 UTC continuation: validation assembly is now exact, not aspirational

The explanatory balance sheet has not changed: all $36/36$ sites have structural
inventory entries, named behavior remains $32.1\%\pm6.4\%$, named causal recovery is
$10.923\%$ under its own denominator, and strict executable recovery remains $0\%$ of
the paired $+0.8976$ current-ship CE gap. The largest missing interface is therefore
still an admitted early program that can be composed under the ship denominator; the
second is a typed routing-and-value account of the attention-heavy allocation; the
third is conditional MLP2/3 behavior after upstream replacement. Parameter compression
and site inventory do not move any of those currencies by themselves.

This interval closed a narrower but necessary gap in the suffix lifecycle. The program
stage already knew how to reduce one tensor batch to per-row local MSE, suffix KL, CE,
and copy CE. It now has an exactly-once collector which:

- hashes the exact 192-row validation tensor, shifted targets, frozen 64-token copy
  mask, and score interval into one common-support identity;
- consumes all 48 four-row batches in canonical order and rejects a replay, omission,
  row permutation, support-count change, nonfinite reduction, native student call, or
  non-inert hook closure;
- freezes a separately computed native copy baseline at construction and requires the
  candidate to reproduce its per-row copy-mask counts; and
- releases only the already-preregistered raw per-row sums/counts after complete
  assembly. It never receives or releases a logit, activation, label, or fit tensor.

This is not yet a scoring result and does not open a role. It removes an important
failure mode in which a launcher could average partial batches, change row order, or
compare copy loss on a different mask. The complete suffix source suite passes 183/183
in 54.11 seconds, and the prospectively frozen algebra files still have their exact
registered hashes. The final owner/test pair remains absent, so the row gate remains
NO-GO by construction.

The current candidate-action pruning, updated for this implementation rather than
copied from the prior checkpoint, is:

1. **Finish the sealed validation observation boundary and semantic final owner.** It
   has the highest information gain because it is the only action that can turn five
   fitted objective/transport hypotheses and 24 mapped controls into falsifiable,
   common-support evidence. The CPU assembly half is now complete; the remaining half
   must compute reductions inside the observed adapter so raw held-out tensors cannot
   escape.
2. **Publish the canonical true/mapped program and preflight bank before any final
   load.** This is low-GPU and high-integrity: it binds selected programs, all 21 row
   maps, eight gauge tensors, intervention geometry, native baseline, and protected
   snapshots into the lifecycle unlock. Without it, final replay and causal edits are
   not composable or auditable.
3. **Execute the matched L/R/S/T experiment once the source gate is genuinely GO.**
   This is the first action that can distinguish local-coordinate objective failure
   from missing executable parent-code transport. Its document shuffles and finite
   A-null bank make both claims causally falsifiable rather than reconstruction-only.
4. **Insert any admitted early program into the common-support attention × early ×
   deep factorial.** This alone can move strict current-ship recovery and reveal
   cross-half interaction failure. S1742/1743 make a greedy attention prefix a useful
   lower bound, not an optimum: rising marginals disprove submodularity even though the
   K=6 greedy set is one-swap locally optimal.
5. **Compile the residual conditionally: MLP2/3 first, then typed attention routing
   plus value/content.** This exploits polynomial/tensor structure where it buys a
   closed downstream interface. Attention rank alone and independent MLP fits are
   pruned because neither supplies an executable consumer/producer contract.

Additional token clustering, isolated MLP0 semantic regressions, unpriced rank curves,
more OAT/LOO site rankings, and new descriptive allocation searches remain below the
cut. They are cheaper, but redundant with known structure and cannot currently change
a causal, OOD, edit, composition, or ship-denominator decision. The active GPU basin
run is still useful as a bounded robustness audit; its first random start has climbed
from $0.4261$ to $1.0873$ through three sweeps but has not yet converged, so no basin
claim is recorded.

## 06:13 UTC continuation: held-out reduction boundary and deployed baseline closed

Validation can no longer masquerade as fitting or accept caller-reduced model outputs.
The trace schema now admits a selection-only role with a canonical 48-batch schedule,
separate validation tensor identity, true teacher map, frozen program hash, trial,
control provenance, and exact batch tokens. Fit and validation run contexts reject one
another in both directions. Document-shuffled and A-null candidates retain their
training-control identity, but evaluate on true validation rows: A-null T uses the
true same-forward executable L0 parent at evaluation, exactly as preregistered, rather
than incorrectly replaying its false fit parent.

For every candidate batch, the observed adapter owns the complete no-gradient P/P/N
student and exact coordinate or O/O/N teacher transaction. Local MSE or suffix KL, CE,
and copy CE are reduced inside the one-use teacher result. Only six float64/integer
four-row vectors and tensor-free ledger hashes cross the boundary; logits, codes,
labels, states, callbacks, and deployed-write handles do not. A mixed-support bug found
by the new tests was fixed: the OON teacher is already sliced to 64:256 while the sealed
student logits remain length 256, so the reduction now canonicalizes each tensor to
the scored support before comparing shapes.

The copy admissibility baseline is also no longer caller-provided. A separate exact
identity is minted from the complete validation role, shifted-target/copy-mask support,
and canonical batch. The adapter runs deployed N/N under literal-native poison,
reduces CE/copy internally, and feeds an exactly-once 48-batch baseline collector.
Candidate collectors now require that completed support-bound baseline object; the old
raw-vector construction path was removed.

Adversarial tests cover cross-role authority, noncanonical schedules, changed role
tokens, illegal control/route pairs, mixed full/sliced KL tensors, incomplete/replayed
baseline and candidate batches, support mixing, native-call attempts, closure drift,
and tensor escape. The complete suffix closure passes 198/198 in 55.18 seconds;
frozen algebra verification passes; the semantic final pair is still absent and the
row gate remains NO-GO. No explanatory currency moves.

The next highest-priority unit is now narrower: implement the semantic final owner and
canonical program/preflight artifact publisher together, including validation receipt
and broker-ledger completeness checks. That source must exist in full before the
file-existence gate can open. After publication code is frozen and re-audited, matched
L/R/S/T execution remains the first numerical action. On the independent allocation
run, random start 1 converged after six sweeps to the same 1.2037 K=6 value as greedy;
start 2 is still running, so a common-basin statement remains premature.

## 06:24 UTC continuation: canonical program bank is now a complete object

The balance sheet is unchanged and its denominators remain separate: structural
inventory is $36/36$ (scope, not explanation), named behavior is
$32.1\%\pm6.4\%$, named causal recovery is $10.923\%$, and strict executable
recovery is still $0\%$ of the paired $+0.8976$ current-ship CE gap. The separate
constant-ablation ceiling is $55.038\%$ on discovery and $53.694\%$ on held-out
rows. Thus the largest gaps remain an admitted producer/consumer program for the
early stack, its interaction with the ship stack, conditional MLP2/3 behavior, and a
typed routing-plus-value account of attention.

This interval completed the pure canonical assembly which the eventual publisher
will freeze before final scoring. It now requires all 87 validation candidates—five
true routes and 24 mapped families, each at three learning rates—plus exactly 48
batch receipts and one broker-ledger hash for every candidate, and exactly 48
receipts for the deployed N/N baseline. A selected program cannot enter the bank
unless its sufficient-statistics hash occurs in that complete common-support
manifest. The four shuffled programs must retain one shared map, the twenty A-null
transport controls must retain twenty distinct maps, and a mapped object cannot be
silently converted into a true arm.

The bank also freezes a tensor-native edit geometry from the entire selected-L0 fit
trajectory: the exact $73{,}728\times64$ code support, float64 mean and covariance,
a sign-canonical eigensystem, a trace-relative clipped spectrum, natural code RMS,
and 32 reproducible covariance-shaped Rademacher directions normalized to unit RMS.
It binds this geometry to the selected L tensor hash, and includes eight exact
orthogonal gauge transformations, balanced role-specific intervention assignments,
teacher-only amplitude calibration, the native validation baseline, all selected
true/mapped tensor programs, and one recursive payload hash. This turns “try an edit
in the 64-D code” into a replayable tensor intervention rather than a basis-dependent
probe.

The added adversarial tests reject an 86-of-87 manifest, a 47-of-48 candidate
execution, mixed support, duplicated null maps, changed calibration, and geometry
from a different L program. Focused tests pass 4/4; the repository-discovered suffix
suite passes 183/183 in 59.50 seconds, and frozen-input verification remains intact.
No role was loaded and the final source/test pair remains absent, so the lifecycle is
still correctly NO-GO.

The pruned next-action ranking is now:

1. **Implement the semantic final owner and create-only program publisher.** This is
   the only remaining source/interface closure before legal execution. It must
   validate fit, row, source, protected-file, complete-validation, payload, and
   one-shot final bindings on deserialization—not merely trust the in-memory builder.
2. **Execute matched L/R/S0/S1/T with all mapped controls.** This has the highest
   immediate scientific information gain: it distinguishes a local 64-D coordinate
   code, suffix-demanded refinements, and learned parent-to-child transport using
   causal negative controls on one support.
3. **Run the admitted early program in the current-ship interaction cube.** Mobius
   terms across early MLP, interaction-selected attention, and deep restoration are
   required before any local reconstruction earns whole-model causal credit.
4. **Compile MLP2/3 conditionally on the admitted early state.** Use tensor/polynomial
   low-rank factorizations only where they preserve the producer/consumer interface;
   independent fits are already known not to compose reliably.
5. **Compile attention as typed routing and value/content, then validate practical
   simplicity.** Compare equal-fidelity alternatives by OOD prediction, extraction,
   selective removal, non-descendant collateral, gauge stability, and composition;
   parameter count or rank alone is not a validated simplicity notion.

Further token clustering, isolated semantic probes, rank-only sweeps, and spent-role
allocation scans remain pruned because they cannot presently change a common-support
causal or executable decision. The independent GPU allocation audit has now shown
starts 1 and 2 both converge to 1.2037, but start 3 has only initialized; a shared
basin claim remains withheld.

## 06:32 UTC continuation: the canonical artifact now has a trusted inverse

The strategic denominator still does not move: 36/36 sites are inventoried, named
behavior is $32.1\%\pm6.4\%$, named causal recovery is $10.923\%$, and no admitted
program yet recovers any of the paired $+0.8976$ current-ship CE gap. The separate
constant-replacement ceiling remains $55.038\%$ discovery and $53.694\%$ held-out.
The limiting uncertainty is executable composition, not another local fit statistic.

The highest-priority CPU action this interval closed the missing inverse of the
canonical program builder. A `weights_only=True` artifact load is now reconstructed
into typed `FrozenProgram`, `FrozenMappedProgram`, baseline, execution-manifest, and
transport-geometry objects. Validation checks exact schemas; three-trial identities;
metric and copy gates; source/statistics/tensor hashes; the two factorized site states;
T's $64\times64$ cross map; dense tensor replay; all 87 candidate and 48-batch
commitments; native baseline support; the covariance eigensystem; the exact eight
gauges; balanced validation/final assignments; and replay of the teacher calibration
rule. It then deterministically rebuilds the whole bank and requires recursive
tensor-hash identity with the deserialized artifact.

This matters because an outer file receipt alone authenticates bytes but does not
show that those bytes still denote the program selected by the mathematical contract.
The new inverse supplies the semantic half of that check; the create-only publisher
will supply the row/fit/source/protected-file provenance half. Tests cover a literal
tensor mutation under the old hash and a changed gauge under a recomputed outer hash.
Both fail. The focused bank gate passes 4/4, and the combined suffix, observed-model,
and frozen-ship suite passes 202/202 in 63.13 seconds with the repository root bound
explicitly on `PYTHONPATH`. The first combined invocation failed at collection because
that path was absent; it ran no test bodies and is retained as an invocation failure,
not hidden as a code failure.

A separate agent result also narrows the semantic-probe branch. On ten registered
classes, the earlier post-hoc SNR-versus-share correlation did not replicate:
$\rho=0.0788$, $p=0.8356$, versus the discovery reference $0.7333$. Frequency likewise
had $\rho=0.0182$. The shuffled noise-floor law did replicate
($\rho=0.9879$), and all ten class deviations exceeded their shuffled controls. These
are reused curated rows, not genuine OOD, so the result falsifies the proposed class
ordering while preserving only “signal exceeds shuffle.” It strengthens the decision
to prune more isolated semantic correlations until an executable interface supplies
consequences.

Updated priority order:

1. **Implement the create-only publisher and semantic final owner.** The typed inverse
   makes this lower-risk now. Publication must bind row receipt, fit ledger/manifest,
   complete validation evidence, source closure, protected artifacts, and the exact
   bank before any one-shot final-role load.
2. **Execute the matched L/R/S0/S1/T experiment with all mapped controls.** It is the
   first falsifiable test of whether a local code, suffix objective, or cross-site
   transport provides an admitted early-stack interface.
3. **Measure whole-ship composition with a preregistered Mobius cube.** Cross early
   MLP, interaction-selected attention, and deep restoration on one support to locate
   the unexplained CE and prevent local fidelity from being mistaken for recovery.
4. **Compile MLP2/3 conditionally on the admitted early state.** Favor tensor and
   polynomial factorization only when producer/consumer replay and gauge invariance
   survive; independent local fits are redundant with prior failures.
5. **Compile attention as routing plus value/content, then certify simplicity by
   consequence.** Equal-fidelity candidates must predict unseen distributions,
   extract circuits, support selective edits/removal, limit collateral, and compose.

Rank-only sweeps, token clustering, SNR/share variants, OAT/LOO rankings, and further
spent-role allocation searches stay below the cut: they are cheap but do not close a
causal interface or move a whole-model denominator. The GPU allocation job remains
active; starts 1 and 2 converged to 1.2037, while start 3 has reached 0.9659 after two
sweeps. No common-basin claim is made until that run terminates.

## 16:18 UTC continuation: local tensor geometry is subordinate to causal transport

The empirical fourth-moment branch and suffix-transport branch are no longer treated
as competing plans. Empirical M4/Wick can select a compact polynomial grammar on
natural states; suffix transport is the causal test of whether that grammar produces
a state which MLP1 and the live suffix can actually use. This corrects a drift toward
another standalone local reconstruction project.

Three CPU-only commits advanced the source gates without opening data or a model.
`53af848f` implements deterministic population geometry, exact bias-free bilinear
residuals, streamed empirical Grams, blocked noncentral Wick, Spearman ties, and
document bootstrap (9/9 tests). `73d8288e` implements a create-only, outcome-blind
three-role freezer for 2,084 documents per role and exact FIT100/FIT200/FIT400 masks
(19/19 tests), but has not published any role identity or read a parquet column.
`b22832b2` implements the missing suffix semantic owner: canonical bank publication
and semantic reload, route/null recomputation, result->manifest->last-authority
ordering, integrity failure, and a hard no-global-credit boundary. The combined suffix
suite passes 201/201.

The real suffix run is still mechanically NO-GO until the declared observed final
execution adapter and adversarial test exist, are independently audited, pushed, and
included in a new source-closed authority. The empirical real-data run is also NO-GO
until exact-factor versus cached-write replay, PCA boundary degeneracy, FIT-first
validation sealing, implementation independence, and bootstrap authority are closed.
These are local implementation gates; FineWeb, checkpoint, cache, `rspd`, GPU, and
network access are not blockers.

Claude's S1813 result also changes the simplification target. Rank-1 all-site tables
retain 77/79/78% of rank-64 top-1 fidelity at 5.628M versus 20.531M stored reals, but
the shared embedding-to-row map is then 94.3% of the bill. The next cheap whole-program
test is therefore a frozen map-rank sweep at rank-1 tables with CE/OOD/causal scoring,
not another table-rank sweep. This is fifth behind suffix final execution, L/R/S/T,
the current-ship macro cube, and conditional MLP2/3.
