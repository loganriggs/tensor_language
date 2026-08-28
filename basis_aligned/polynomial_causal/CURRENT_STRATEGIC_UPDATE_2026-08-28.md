# Current strategic update: bilin18 reverse engineering

Last updated: 2026-08-28 after the corrected middle-MLP identity control.

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
omitted the compiled context features. A source-corrected rerun is currently active;
until it finishes, the family supplies no held-out replication or promotion.

## Genuine blockers

There is no fundamental blocker from missing RSPD or FineWeb data at this point. The
model implementation needed by the repository is available through the tracked
`jacclust.tt_model` loader path and model weights are cached; fresh FineWeb roles are
deliberately not frozen until the source closure authorizes them.

The current blockers are implementation and cost integrity:

1. The sealed capability still needs a source-closed bilin18 adapter around the real
   frozen-ship forward. It must distinguish literal dispatcher calls at all three
   early sites from effective rank-64 correction calls at MLP0/1.
2. The prohibitive full-logit content hashes have now been replaced prospectively by
   a source-bound one-use graph/storage identity. Ordinary mutation, replacement,
   detach, graph drift, and nonfinite values fail closed without transferring logits
   to CPU. Independent review gives GO-to-commit, but execution remains NO-GO until
   the observed adapter closes before aliases escape and enforces the checkpoint's
   50,304 logit width. Token IDs remain restricted to the tokenizer's 50,257 entries;
   slicing logits would change CE/KL normalization.
3. The graph-connectivity check currently performs an extra full suffix backward per
   fit batch. It needs a measured benchmark or a cheaper structural proof.
4. Document-shuffle and A-null controls remain intentionally unauthorized until a
   mapped-row capability binds the actual source/target document map.

## Pruned and ranked next actions

The ranking uses expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and redundancy with completed work.

1. **Finish real N-write provenance and remove the two per-batch cost traps.** This
   is the smallest boundary that turns the audited tensor algebra into a legal and
   feasible model forward. The typed N-write repair was executed in this review;
   the observed adapter, cheap graph identity, and connectivity benchmark remain.
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
than a validation of a simplicity definition.

## Verification state

The focused runtime/capability suite passes 48/48, and the complete CPU suffix suite
passes 134/134 after the typed-write and graph-identity changes. Independent review
gives exact-byte GO-to-commit for the graph identity under the declared source-closed
adapter boundary. No fresh row, model, teacher, or scientific outcome was loaded by
these repairs, so suffix-transport execution remains explicitly NO-GO until the
observed adapter is source-closed and independently audited.
