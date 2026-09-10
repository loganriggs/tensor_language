# Moving from mixed-input importance to an exact attention support rule

**Update, 10 September, 04:32 UTC:** the first-attention native test has now
completed. Its token-derived implementation is faithful, but its behavioral
carrier criterion fails. The mathematical audit below quantifies why an exact
implementation of this interface still leaves most of the behavior unexplained.
The earlier sections preserve the reasoning that preceded this result.

The latest native test rejects a simple localization of MLP4's mixed cue/context
input to either attention4 or the earlier residual stream. Neither group alone
reproduces the effect. More importantly, this mixed correction is small on the
is/was A2 panel, so continuing to decompose it would not directly explain that
panel's main cue response.

The next completed mathematical step concerns the first attention layer. Its
queries, keys and values are still functions of individual tokens. Combined
with the model's absence of softmax, this gives an exact rule for which
query–source edges can carry a cue change or a joint cue/context effect.
The rule and its implementation pass controls with the actual tiny-model
normalization and positional rotations. Trained-model causal relevance of the
resulting message program remains untested.

## What the raw-input native test found

The raw input to MLP4 contains a new attention4 write plus an incoming residual
stream. Using each four-text square, we measured how the cue change differs
between the two contexts. Call that mixed raw-input difference D, its
attention4 contribution A, and the remaining contribution P=D-A.

We compared the full target input with three edited raw inputs: remove D,
retain only A, or retain only P. For each input we recomputed actual RMS
normalization, MLP4, and the entire downstream model. All four observed text
corners were used, so this was source localization, not missing-corner
prediction or independent execution of either producer.

| Retained mixed contribution | Full-vector error against the complete mixed-input correction |
|---|---:|
| Attention4 only | 77.1–85.6% |
| Earlier residual only | 44.1–57.0% |

Both hypotheses fail the10% threshold. The ranges cover16panel/direction cells.
The answer-margin tests are also required; no alternative arm is promoted.
Earlier-residual-only performs better in this comparison, but remains far
from sufficient. The actual attention4 write still depends on earlier
computation, even when its direct residual counterpart is removed from an
input edge. This cannot establish that attention4 runs independently.

The size of the complete mixed-input correction also differs across panels:

- has/had:43.6–59.7% of the full source cue-effect norm in the vocabulary frame,
  and12.3–29.0% in the answer-margin frame.
- is/was A2:5.2–6.7% in the vocabulary frame and about0.9–1.0% in the margin frame.
- is/was held-out:7.3–14.5% in the vocabulary frame and4.2–11.5% in the margin frame.

These are ratios of norms, not additive causal percentages. The registered
requirement that this correction be material in both frames on every panel
fails. A complicated decomposition of a small correction would not explain
the main is/was A2 response.

The run is valid:112forwards on980sequence instances,64local MLP evaluations,
2.29seconds of executor time. Native raw-input recurrence and direct-embedding
mixed cancellation are bitwise correct. All MLP, identity, direct-source and
parent replay discrepancies are zero. The FP64 source fold has maximum
absolute discrepancy.000411 and relative discrepancy5.76e-8; it passes its
registered relative numerical check. All545,902,902native weights remain
charged, with zero demonstrated structural savings.

## Why the first attention layer is a different mathematical object

Before the first attention operation, a position's input depends only on its
own token. Embedding normalization, head normalization and positional rotation
all preserve that locality. For one head, the read at query position i is

    read_i = sum_(j <= i) score(i,j) * value_j
    score(i,j) = (query_i · key_j)(query2_i · key2_j) / head_width².

A query is the vector used to select source information; keys describe a
source for matching, and values are the information it writes. Here the score
is a product of two dot products. There is no softmax coupling scores across
different source positions.

Consequently each summand is a function of the token at i, the token at j,
and their fixed positions. This restricts where an intervention can act.

For a single cue edit, an edge can change only if its query or source token
changed. Computing the difference using just those edges gives the same
attention-read difference as subtracting the two full attention reads.
The source functions still use the actual learned weights; this does not
identify which allowed edges have substantial behavioral effects.

For a cue-by-context square with disjoint edited positions, the restriction
is stronger. Let C be the cue positions and X the context positions. A joint
mixed term can occur only when

    (i in C and j in X) or (i in X and j in C), with j <= i.

If neither endpoint changes, the term is constant. If both depend on the same
variable, or only one endpoint changes, the term depends on only one variable;
its mixed difference cancels. Only an edge connecting opposite edit types can
contain both variables. This proves the support rule without fitting weights,
choosing a rank, or looking at model outcomes.

In particular, a query position unchanged by either edit has zero mixed read
in the first attention layer. This statement concerns the attention read,
not the whole transformer block: the following normalization and bilinear
MLP can combine separate cue and context effects. It also does not extend
automatically to later attention layers, whose input vectors are contextual.

## Executed checks and dataset consequences

The tiny-model test uses actual RMS normalization, the implementation's RoPE
rotations, and captured native first-layer factors. The independent full-read
formula agrees exactly. Restricting a cue change to the allowed single-edit
edges agrees within9.72e-17; restricting the mixed difference to opposite-edit
edges agrees within5.56e-17. The mixed signal is live, with norm.4954.

In that fixture the cue changes position0 and context changes position3.
Causality leaves exactly one possible mixed edge: source0 to query3.
All other mixed read positions vanish to numerical precision. A separate
softmax example produces a nonzero mixed response at an unchanged query,
showing why the no-softmax assumption matters.

The deterministic token audit covers all46existing squares. Per attention
head, their full causal graphs contain1,412query–source edges. The exact
single-cue difference needs at most326edges, and the mixed difference at most82.
For the35evaluation squares alone, the corresponding counts are1,247,271and63.
These count possible terms in counterfactual read differences; they are not
whole-model parameter or runtime savings. All46semantic query positions have
at least one possible mixed edge, so this rule does not itself explain the
small is/was mixed effect by declaring its query structurally disconnected.

## What this changes in the research plan

The next trained-model question should be the causal relevance of the complete
first-layer cue-message program, using the exact token-derived edge rule.
We should establish that before concentrating on its smaller mixed component.
The program would read token identities and positions, apply the learned
normalized product-matching rule, and supply a source write to the native
suffix. Its weights, token parser, background and all adapters must be charged.

The rule is an execution and localization tool, not discovery of a learned
tense algorithm. No first-layer trained circuit, new OOD generalization,
selective removal, independent smaller executor or composition result is
claimed here. The native protocol for that message test has not yet been
registered. The CPU support implementation and row audit are the concrete
continuation already completed after the raw-origin null.

## Evidence

- [Native raw-origin result](../BILIN18_MLP4_RAW_INTERACTION_ORIGIN_V1_RESULT.json)
- [Frozen raw-origin protocol](../BILIN18_MLP4_RAW_INTERACTION_ORIGIN_V1_PREREGISTRATION.md)
- [Exact first-attention support implementation](../token_local_attention_support.py)
- [Executed native-factor controls on the tiny model](../TOKEN_LOCAL_ATTENTION_SUPPORT_V1_CONTROLS.json)
- [Token-based support and edge-count audit](../TOKEN_LOCAL_ATTENTION_SUPPORT_V1_ROW_AUDIT.json)

## Completed first-attention test and mathematical reassessment

The concern about slow circuit progress is justified. Recent work has produced
faithful local algebra and several useful falsifications, but no new bilin18
computation satisfying all four requested properties. The original
[handoff](bilinear_circuit_reconstruction_codex_handoff.md) and
[pilot report](bilinear_reconstruction_pilot_report.md) already distinguished
an exact executor from discovery. The pilot's proposed successor was a shared
read–route–write operation: compute a match, select information, and deliver it
to several consumers. Accumulator rank or local reconstruction was insufficient.

The latest test computed the complete first-attention cue-write difference
directly from tokens and weights. It swapped that write into the receiving
model while preserving the receiving first-value stream. It used all nine
heads and all valid positions on 72 previously opened pairs, in both directions.
There were 40 model forwards, 720 sequence evaluations and eight independent
token-factor productions; execution took 1.63 seconds. Instrument and compiled
effect-fidelity gates pass; the across-panel behavioral carrier gate fails.
There is no head, position, gain or first-value follow-up to rescue this result.

### The distinction can be quantified exactly

Let `t` be the natural output change between a pair of texts, `d` the output
change caused by swapping the native component, and `d_hat` the change caused
by its compiled replacement. All three use the same receiving baseline and
output coordinates. Define

    r = ||d|| / ||t||
    p = <d,t> / ||t||²
    e = ||d_hat-d|| / ||d||.

Here `r` measures effect size, `p` measures signed alignment with the natural
change, and `e` measures compiler fidelity. Expanding a squared distance gives

    E = ||d-t|| / ||t|| = sqrt(1 + r² - 2p).

The triangle inequality then gives the interval

    max(0, E-e*r) <= ||d_hat-t|| / ||t|| <= E+e*r.

These are derived identities/bounds, not a fitted correction or new success
threshold. They show why improving `e` cannot repair a large `E`: an exact
compiler faithfully reproduces the component's limited behavioral effect.
For example, a component producing exactly 1% of the target vector has perfect
compiler fidelity but 99% error against the target.

The executed CPU audit reconstructs `E` from the saved native norms and signed
projections. No model was loaded, no population was changed, and no new native
outcomes were obtained. A separate two-coordinate calculation checks the
identity and bound.

| Existing panel | Error versus natural full-vocabulary change | Error versus natural answer-margin change |
|---|---:|---:|
| has/had A2 | 94.2–96.2% | 99.2–99.7% |
| has/had held-out | 93.2–94.6% | 98.9–99.6% |
| is/was A2 | 60.9–62.0% | 70.4–71.2% |
| is/was held-out | 75.4–77.6% | 93.3–94.1% |

Ranges cover the two swap directions. Full-vocabulary vectors have their mean
logit removed; an answer margin is the preferred-token logit minus its foil.
Errors are Euclidean relative errors, not percentages of decisions explained.
Compiler uncertainty changes any listed relative error by at most 0.00000473.
Thus numerical compilation error is much too small to explain this failure.
Previously opened held-out panels are not fresh OOD evidence.

### What weight folding should identify instead

For a bilinear MLP with linear maps `L`, `R`, output map `D`, bias `b`, and
raw input `x`, a scalar downstream reader `c` gives the exact local expression

    cᵀ f(x) = xᵀ Q_c x / (||x||²/d + epsilon) + cᵀ b
    Q_c = sym(Lᵀ diag(Dᵀ c) R).

The reader specifies which output is consumed; `sym(A)=(A+Aᵀ)/2`. This is
the weight-defined quadratic-form approach developed by
[Pearce and colleagues](https://arxiv.org/html/2410.08417v2), with the actual
input normalization kept explicit here. It is already implemented in this
project; deriving it again would not constitute new circuit progress.

For two consumers we need to compare their functions jointly. Shared inputs,
shared products and shared routing are different claims. For example,
`a*b` and `a*c` share the calculation of `a` but have distinct products.
Identical values with different matching rules share payload computation,
but need not share routing or independently editable memory. Attention makes
this explicit: a consumer reads `sum_s routing(t,s)*payload(s)`. Reassociating
that sum into a recurrent state is established
[linear-attention algebra](https://proceedings.mlr.press/v119/katharopoulos20a.html);
it does not identify what the learned matcher computes.

There is also a precise module-splitting test. If candidate input groups have
linear maps `U_A,U_B`, the quadratic numerator contains the mixed term
`2*aᵀ U_Aᵀ Q_c U_B*b`. It separates additively across these groups for every
input exactly when that cross matrix vanishes for every relevant reader.
The normalization may still couple the groups and must remain an explicit
shared operation. Nonzero cross terms identify a required interaction in
this coordinate/domain proposal; they do not by themselves name its semantics.

The useful research target remains a shared operation with explicit producers
and multiple actual consumers, tested through independently recomputed and
joint interventions. A fold into a later weight matrix cannot cross omitted
nonlinear layers. A good local tensor fit cannot supply missing inputs or
explain the native background. The preceding response-ladder and shared-product
nulls already demonstrate these limits; they should not be rerun under new names.

The broader ledger reinforces the denominator issue: the old v23 direct-carry
projection of 0.833 is relative to its conditional joint intervention, not the
natural text-pair change, and its selectivity gate failed. It is not an 83%
solution to the user's goal. The smaller associative-retrieval model's earlier
endpoint-field positive also later failed broader layout/field tests; it is a
different model and cannot fill the bilin18 evidence gap.

This reassessment closes the weak local write branch and supplies an executed
effect-geometry audit for comparing future candidates. It does not claim a new
trained algorithm, selective extraction, OOD success or structural savings.
All 545,902,902 native parameters remain necessary in these executions.

- [Native first-attention result](../BILIN18_FIRST_ATTENTION_CUE_MESSAGE_V1_RESULT.json)
- [Frozen first-attention protocol](../BILIN18_FIRST_ATTENTION_CUE_MESSAGE_V1_PREREGISTRATION.md)
- [Executed geometry audit](../CAUSAL_EFFECT_GEOMETRY_AUDIT_V1_RESULT.json)
- [Audit implementation](../causal_effect_geometry_audit_v1.py)
- [Old direct-carry result](../../bilinear_quotient/circuits/followups/temporal_iswas_v23_direct_residual_readout_factorial_v2_result.json)
- [Later endpoint-field limitations](forward_endpoint_field_circuit.md)
