# Correlative transfer to new combinations and its two value sources

10 September 2026.

**The frozen circuit transfers to new reporter–noun combinations, but neither
value source can replace the full computation.** The complete26-head/14-layer
interface recovers96.9%of the donor-directed effect in the first construction
and89.9%in the second. The first-layer value branch alone recovers about62%;
the contextual-value branch recovers26–36%. Both fail the registered sufficiency
test. Their full-vocabulary effects also interact substantially.

This advances prediction beyond replaying the original panels. It also closes
two proposed simplifications while preserving a useful full conditional circuit.
The original model still supplies its contextual routing, inputs and background.
We have not extracted a smaller standalone program or established all four goals.

The preceding [interface and folded-program report](correlative_replay_and_folded_program_2026-09-10.md)
explains the saved artifact, exact weight folds and native intervention checks.
This experiment uses that same artifact with no refitting or changes to its
selected heads, directions, mixing coefficients or intervention strength.

**What is new about these inputs.** Inspecting the old generator revealed that
changing its seed only changes the order of32fixed lexical cases. Such a rerun
would not test new combinations. We instead paired each reporter with a different
noun: offsets5and13in the two target constructions replace the original11and19.
The disjoint control pairs each reporter with a different object using offset3.

The four panels contain16preselected groups each:

| Panel | Controlled change | Expected circuit behavior |
|---|---|---|
| A1 | both ↔ neither in a short praise frame | Transfer the and/nor preference |
| A2 | both ↔ neither in a report frame | Transfer the same preference |
| P | Change the reporter while retaining the cue | Preserve the and/nor preference |
| C | either/or ↔ not/but with disjoint answer tokens | Little effect from the target circuit |

The selected rows contain112unique tokenized prompts, with zero exact overlap
against384unique prompts in the complete original source banks. Tokenization,
semantic positions, matched lengths and construction checks pass. No examples
were discarded based on model performance. The vocabulary and templates are
familiar: the shift is in their combinations relative to the circuit's source
data. This does not establish absence from model training or novelty relative
to every experiment in the repository.

**The two computations being separated.** At each selected layer, the saved
scalar is the sum of a first-layer-value contribution and a local-value
contribution:

\[
s_l=s_l^{\mathrm{first}}+s_l^{\mathrm{local}}.
\]

Each contribution sums across sources and selected heads. The first reads the
shared first-layer value stream. The local reads values computed from that
layer's current contextual state. Both use the same complete native routing,
including query/key normalization and positional rotation. The first branch
therefore still depends on context through routing; it is not a token-only
attention program.

For a first-only swap, the residual edit is

\[
\Delta r_l=w_l\left(s_l^{\mathrm{first,donor}}
                         -s_l^{\mathrm{first,live}}\right),
\]

where \(w_l\) is the frozen residual writer. The local-only swap has the
analogous formula. Every later layer recomputes from the edited state, so its
live component can differ from its untouched value. Joint first-plus-local
replacement is the complete scalar swap. We preserve signed value-mixing
coefficients, including negative values and values above one.

We reused the existing compiled executor. Component maps zero one of the two
reader vectors while leaving the output writers unchanged. This provides the
new intervention through the already tested primitive. Native base outputs and
the full swap reconstructed from the sum of source scalars agree exactly in
this run. All registered numerical and counting checks pass:36body forwards,
576sequence evaluations,1.90executor seconds, no model fitting or weight updates.

**Prediction and source sufficiency.** Recovery is the average fraction of the
native base-to-donor answer-margin change achieved by the patch. It is not
classification accuracy or a percentage of the entire model explained.

| Program | A1 recovery | A2 recovery | A1 full-effect error | A2 full-effect error |
|---|---:|---:|---:|---:|
| Complete scalar | 0.9693 | 0.8988 | 0 | 0 |
| First-layer values only | 0.6167 | 0.6223 | 0.4531 | 0.3824 |
| Contextual values only | 0.3584 | 0.2563 | 0.6189 | 0.6788 |

Full-effect error measures the difference from the complete swap over all50,304
next-token logits at the scored position, after subtracting each row's mean
logit. It is normalized by the size of the full intervention effect. The frozen
single-source criterion requires recovery>=0.8and error<=0.15in both target
constructions, plus the registered control limits. Both source-only candidates
fail clearly. There is no subsequent gain, mixture or head search.

The complete circuit passes its registered new-combination gate. The native
model answers correctly on both endpoints of every row, and complete-swap
control movements are0.0206for P and0.0222for C, below the frozen0.23limit.
The movement unit is the first target panel's median native answer separation.
The source-only controls also remain small, but that cannot compensate for
their inadequate target effects.

A post-result4,000draw bootstrap over the16authored groups gives recovery
intervals[0.949,0.989]and[0.885,0.913]for the complete program. First-only
intervals are[0.570,0.663]and[0.591,0.655]. These describe this finite panel;
they are not a certificate of broad language generalization. Related panels
share lexical group construction and should not be pooled as independent tests.

**Adding scalar contributions differs from adding behavioral effects.** The
scalar identity is exact. But if \(z_0,z_F,z_L,z_{FL}\) are logits for no edit,
first-only, local-only and complete edits, the interaction is

\[
I=z_{FL}-z_F-z_L+z_0.
\]

An additive prediction of the two separate effects would require this vector
to be small. Its relative full-vocabulary size is0.3133in A1 and0.2859in A2,
failing the registered0.10limit. Bootstrap intervals are[0.301,0.327]and
[0.275,0.296]. The complete jointly recomputed program remains valid; the
failed hypothesis is addition of the separate endpoint effects.

Average task recovery partly hides this interaction: full recovery minus the
sum of the two partial recoveries is only-0.0057and+0.0201. Even the vector of
per-row task-margin interactions has relative sizes0.139and0.093. Averaging
across examples can cancel some errors, and reading only two answer logits
can hide changes elsewhere in the vocabulary. The experiment does not localize
the interaction to one normalizer or layer; the entire downstream computation
can contribute.

**A mathematical bound from the saved effects.** Let \(f\) be the centered
full intervention vector and \(a\) one partial vector, each concatenating the
panel's rows and vocabulary coordinates. The smallest possible error when
predicting \(f\) by a constant multiple of \(a\) is

\[
\min_\alpha\frac{\|f-\alpha a\|}{\|f\|}
=\sqrt{1-\frac{(f^Ta)^2}{\|f\|^2\|a\|^2}}.
\]

This follows by minimizing the quadratic in \(\alpha\), whose solution is
\(\alpha=(f^Ta)/\|a\|^2\). The saved norms and squared errors determine the
dot product via \(2f^Ta=\|f\|^2+\|a\|^2-\|f-a\|^2\), so no new model run
or fitted direction is needed.

The first-only endpoint vector still has a minimum relative error of0.332in
A1and0.274in A2after optimal rescaling. For the contextual vector the floors
are0.444and0.491. Allowing an arbitrary constant linear combination of both
observed endpoint vectors reduces the floors to0.290and0.254, still above0.15.
The corresponding two-vector Gram systems are well conditioned enough for this
calculation, and an independent planted-vector oracle agrees within8.9e-16.

These are bounds on predicting an already observed output effect from constant
linear combinations of two other observed effects. They do not bound changing
the physical intervention dose, a context-dependent adapter, or an undiscovered
nonlinear program. They support retaining explicit joint recomputation rather
than treating the current partial effects as an additive response library.

**Decision.** Keep the complete14-scalar conditional interface and its new-
combination evidence. Close the first-only and contextual-only value-source
simplifications for this frozen interface. Both sources and their interactions
remain in the explanatory program. The hypothetical46,080coefficient single-
source core is not adopted; the76,032coefficient full port core and all
545,902,902native model parameters remain charged.

The next useful distinction is whether the circuit follows a grammatically
relevant correlative cue or merely the cue in the familiar frame. An outer
correlative with a completed inner correlative would separate those possibilities:
the most recent cue and the still-open cue can disagree. That would test a
different computational claim using the frozen interface, instead of tuning
the value branches that failed here. Such a structural test has not yet run.

Receipts: [data and provenance](../../CORRELATIVE_RECOMBINED_ROWS_V1.json),
[frozen protocol](../../CORRELATIVE_VALUE_SOURCE_V1_PREREGISTRATION.md),
[native result](../../CORRELATIVE_VALUE_SOURCE_V1_RESULT.json), and
[post-result mathematical audit](../../CORRELATIVE_VALUE_SOURCE_AUDIT_V1_RESULT.json).
