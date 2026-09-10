# Moving from mixed-input importance to an exact attention support rule

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
