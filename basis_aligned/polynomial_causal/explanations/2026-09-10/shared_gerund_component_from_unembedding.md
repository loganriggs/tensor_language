# A shared grammatical component from unembedding structure

**10 September, 15:49 UTC.** We found a partial grammatical component using
unembedding weights alone to choose its direction. It transfers about 59–62%
of the tested bare-verb versus -ing cue effect across held-out verbs. Taking
the cue change from a different verb works almost as well. Removing the
component damages the grammatical task while largely preserving the tested
unrelated correlative behavior.

This is useful causal evidence for shared structure, but the registered 80%
sufficiency requirement failed. We have not extracted its upstream producer or
identified a complete independent circuit. Most of the scalar change arrives
at the last MLP from earlier computation.

The sequence was: check prior module work; derive a shared grammatical direction
from eight matched token pairs; fold it through the last bilinear MLP; test 16
different verbs in two sentence frames, plus two controls; then audit uncertainty
and the split between incoming state and MLP contribution on CPU. All 64 sentence
pairs preferred the specified answer to its paired foil at both native endpoints. The managed job used 12
body forwards over 144 sequence instances and took 1.22 seconds of executor time.

## Shared structure within different token functions

The previous screen asked whether entire token quadratic functions were nearly
the same. They were not. Here a token can contain a shared grammatical component
alongside its word-specific content.

For eight pairs—work/working, walk/walking, talk/talking, play/playing,
help/helping, look/looking, wait/waiting and learn/learning—we compute

    v = mean(U_ing - U_bare),       e = v / ||v||.

U_t is the unembedding vector for token t. This direction uses trained weights,
not activations or the test outcomes. The matched lexical pairs are supplied
explicitly; this is not unsupervised discovery of the grammatical relation. It proposes a shared distinction between
bare and -ing verb forms. It is not assumed to be a pure linguistic variable.

The 16 test verbs, including read, cook, sing, build, write and swim, do not enter
that mean. Their average contrast cosine with e is .349, versus .574 for the
eight construction pairs. All test contrasts have positive alignment. The
direction also has cosine .635 with the old gerund-like minus base-verb-like
cluster means. That connects the two unembedding views descriptively; the old
clusters did not select or replace e.

## Folding the scalar backward

Let r be the residual entering MLP17, u its RMS-normalized version, m the MLP
output, and h=r+m the final residual. The shared scalar splits as

    s_h = e^T h = e^T r + s_m,
    s_m = e^T m = u^T Q u + e^T bias,
    Q = sym(L^T diag(e^T D) R).

Thus we can distinguish information carried into the last MLP from information
its bilinear products contribute. The folding identity is established algebra
from the earlier dossiers; the new evidence concerns this grammatical reader
and its interventions. The complete native background remains necessary.

For a donor/recipient pair, a scalar swap adds e*(s_donor-s_base) to the final
residual. An MLP-only swap uses s_m; an incoming-state swap uses e^T r. A
cross-verb intervention takes the scalar cue change from the next verb in the
fixed list and applies it to the recipient. This transfers a donor cue delta,
not an independently computed or donor-free command.

We also remove the component by replacing h with h-e*(e^T h). Every edited
state goes through the original final RMS normalization, unembedding and tanh
softcap. Scalar contributions add before these nonlinear operations; changes
in logits or cross-entropy need not add.

## The causal results

The first frame includes, for example, “The workers often cook. the workers
can still” versus “... are still,” comparing “ cook” with “ cooking.” The
second adds “According to the report,” before the final clause. The final
input token and token count match within each pair. These are new authored
tests for this component, not a claim of absence from model pretraining.

| Swap | First frame | Report frame |
|---|---:|---:|
| Entire last MLP output | 15.0% | 21.1% |
| Shared scalar in the last MLP output | 7.1% | 10.0% |
| Shared scalar carried into the last MLP | 51.5% | 52.2% |
| Shared scalar in the final residual | 58.6% | 62.1% |
| Final scalar cue change from a different verb | 58.4% | 62.0% |

Recovery is the average of (base_margin-patched_margin)/(base_margin+
donor_margin), with each native margin oriented toward its own grammatical
answer. It measures the cue effect on the answer contrast, not the fraction
of the whole model explained.

Final-scalar recovery 95% paired-bootstrap intervals are 52.9–64.6% and 55.9–68.8%.
Cross-verb intervals are 53.2–63.8% and 56.2–68.1%. Neither supports the registered
80% sufficiency criterion. The complete last MLP swap itself has a small effect,
so the weak MLP-only scalar does not imply that the carried variable is absent.

Zero removal raises correct-token cross-entropy by .565/.686 nats averaged
across both grammatical endpoints. Intervals are [.418,.714]/[.504,.875]. The
damage is asymmetric: bare-form endpoints average+.074/+.079, whereas -ing
endpoints average+1.056/+1.293 nats. Positive CE change means worse prediction.

The unrelated either/not correlative control has mean absolute CE change .0232
under removal, with interval [.0170,.0297], below the registered .10 limit. Its
scalar swap has mean absolute CE change .0139 and cue recovery .0022. The
answer-preserving can/may control has scalar-swap mean absolute CE change .0443,
also below .10. Thus the preservation portions of the swap predicate passed;
its target sufficiency portion failed.

Removal of the can/may states has mean absolute CE change .221. It was reported,
not part of the registered unrelated-control removal gate: those states still
use the bare/-ing distinction being removed. Nevertheless, this limits any
claim of broad preservation. One unrelated control family cannot establish
universal selectivity.

## What this changes and what remains missing

The native write projection and its weight-folded scalar agree, and four online
interventions replay the terminal formulas within 2.82e-5 maximum logit error.
All registered instrument and capability tests pass. The selective-removal
predicate passes; sufficient scalar transfer and sufficient last-MLP production
fail. The failures stay recorded without changing thresholds or fitting a gain.

On the raw scalar, about 84–88% of the cue-delta norm is in the incoming-state
term, and 12–16% in the MLP term. These are accounting ratios, not additive
percentages of behavioral causation. Their sum reconstructs the actual scalar
change within .00083 absolute error against changes of order 14,000–18,000.

The next useful target is the earlier producer of that incoming scalar. Tracing
it backward must retain trained residual mixing, attention and normalization;
the current result does not justify replacing them with a cue lookup. Broader
construction transfer, stronger unrelated controls, independent extraction and
composition with another identified computation remain open.

The component stores e, its product reader, Q and bias: 1,332,865 additional
coefficients. All 545,902,902 native parameters remain. This is progress on
shared causal structure, with no structural model reduction yet.

Evidence: [preregistration](../../GERUND_SHARED_READER_V1_PREREGISTRATION.md),
[native result](../../GERUND_SHARED_READER_V1_RESULT.json),
[paired audit](../../GERUND_SHARED_READER_AUDIT_V1_RESULT.json),
[lexical weight audit](../../GERUND_SHARED_READER_LEXICAL_AUDIT_V1_RESULT.json),
and [MLP17 dossier](../MLP17_CURRENT_UNDERSTANDING.md).
