# Relevant products are not yet a sufficient shared circuit

The new trained-model test found a shared set of128MLP4 products that is much
more relevant than a random set of the same size. However, it still fails badly
as a replacement for the complete source computation. Task-specific sets also
fail. This is useful localization evidence, not a recovered circuit or a reason
to weaken the success criterion.

I am stopping this particular sparse-product proposal. The next mathematical
object is a cue-by-context square: four inputs that let us distinguish an
interaction created by MLP4's multiplication from one already present in the
representation reaching MLP4. The exact partition and dataset construction have
now been implemented and checked on CPU; the trained-model square test has not
yet run.

## What we tested on the GPU

MLP4 contains4608 products of two learned linear input measurements. A selected
product contributes its scalar value times a learned output-weight column.
We selected128products, about2.8% of the module, using the original24fitting
pairs only. The selection combined both has/had and is/was tasks.

The selection used a derivative of the final answer-minus-foil logit score with
respect to MLP4's output. A derivative measures sensitivity to a very small
change. Multiplying that sensitivity by each product's paired change proposes
which products may matter for the task. We averaged absolute contributions at
both native endpoints, normalized each task's scores, and took the smaller of
the two task scores for each product. The top128formed the shared proposal.

This did not assume that derivatives predict a complete finite intervention.
The earlier receiving-tangent test already rejected that assumption. Every
selected subset was tested by actually swapping its product values and
recomputing the whole downstream model. We compared it with a completeMLP4
source swap in the same receiving context. Both swap directions were tested.
The source bias and every unselected product remained in the background.

The native screen used80forwards on1,392sequence instances and4backward
passes. Its executor took2.07seconds on the managed GPU lane. All instrument
checks passed, with maximum reported oracle error2.17e-5. The four backward
passes computed input sensitivities; no weights were trained.

## Results

The error below is the norm of the difference between a subset's output effect
and the full source-swap effect, divided by the full effect's norm. The vector
contains all50,304centered vocabulary logits. Lower is better; the registered
sufficiency threshold was10% in every panel and direction.

| Product set | Full output-effect error across evaluation cells |
|---|---:|
| Shared128 | 70.0–82.0% |
| Own-task128 | 56.6–73.1% |
| Other-task128 | 72.0–92.4% |
| Fixed random128 | 97.3–99.0% |

The shared set beats random by the registered margin in both the full-vector
and answer-margin tests in every cell. It nevertheless fails sufficiency in
all eight cells. Own-task subsets are diagnostic comparisons, not alternative
promoted winners. We will not increase the budget or change the scoring rule
after seeing these results to rescue this hypothesis.

The task-specific top128sets overlap in26products. The shared set overlaps the
has/had set in52and the is/was set in47. Such overlap is not an identified
shared semantic operation: it could reflect common context processing,
parameterization, or correlated task sensitivities.

Removing the shared products without a donor leaves82.8–89.4% of the native
paired answer-margin contrast by signed projection. The removal therefore
does not eliminate either behavior. This result does not establish selectivity:
we did not include an unrelated behavior family in this basic screen.

All evaluation families had previously been opened. The fitting and evaluation
token sequences are disjoint, but this is not pristine task-family OOD evidence.
All545,902,902native parameters remain charged, and actual structural saving
is zero. The selected source slices alone would contain442,368weight entries
plus128indices; their opaque contents and native input generation are not
explained by listing them.

## The next question: who constructs a cue/context interaction?

Consider four texts:

    cue0, context0        cue1, context0
    cue0, context1        cue1, context1.

For is/was, a cue can be this versus that in an otherwise matched prefix.
The context change can replace lexical background words. In the existing
has/had rows it can change several fields together, such as event and person;
we explicitly call this a lexical-background bundle, not a single subject
feature.

Let n00,n01,n10,n11 be the observed normalized MLP inputs for these four
corners, with context indexed first and cue second. Define

    u = n10 - n00                 context change at cue0
    v = n01 - n00                 cue change at context0
    w = n11 - n10 - n01 + n00     input nonadditivity
    a = n00 + u + v.

If w is zero, the cue and context changes add at the MLP input. A nonzero w
means their combination is already represented differently before the MLP's
multiplication. Earlier computation or input normalization can both cause it.

Write the module's bilinear map as

    B(x,y) = Down[(Left x) * (Right y)]
    m(n) = B(n,n) + bias.

The four-corner output difference is exactly

    m(n11) - m(n10) - m(n01) + m(n00)
      = B(u,v) + B(v,u)
        + B(a,w) + B(w,a) + B(w,w).

The first two terms are the interaction created by the MLP between the
separately observed cue and context changes. The remaining terms account for
nonadditivity inherited at the MLP input. Bias cancels. This gives a concrete
way to ask whether the multiplication constructs a conjunction or responds
to one represented upstream.

The decomposition is an anchored local identity. It is not a unique
attribution of final behavior, and the inherited term does not by itself tell
us which earlier layer or normalizer created w. All components must eventually
be tested through the actual downstream model.

It also gives a falsifiable three-corner prediction. If input changes compose
additively, the missing fourth MLP output is m(a). Its exact error is the
inherited term. The actual fourth input is needed to diagnose that error;
using it cannot be counted as predicting an unseen corner. The synthetic a
need not be a realizable normalized native input, another reason to separate
algebraic prediction from extraction of a token-to-output circuit.

## What has already been executed for this next step

The new CPU controls verify the identity on arbitrary four-corner inputs,
under factor rescaling, and after actual RMS normalization. They include one
example with only a locally created product interaction and another with only
inherited input nonadditivity. Maximum identity error is1.46e-13 in FP64.
A normalization example produces nonzero w even when the raw input changes
add; apparent cue/context cooperation therefore cannot automatically be
assigned to MLP multiplication.

A deterministic audit of the existing96paired rows constructs46disjoint
squares:11from fitting rows and35from evaluation rows. Fourpaired rows have
no remaining compatible context partner and are left unused. Pairing uses
only token lengths, answer orientation, cue positions and cue-token identities,
then row-ID ordering; it never consults model outputs or the new saliency scores.
Cue-change positions and lexical-context-change positions are disjoint.
These remain previously opened texts, not new OOD samples.

The earlier contextual-source factorial separated Left and Right operand
changes for one paired edit. This next construction instead uses two distinct
input variables across four natural text corners, and explicitly separates
inherited input nonadditivity. It is not a rerun of that operand-factor test.

The gradient tool also now has live tiny-model controls. Its first fixture
had a zero-initialized output head, so its gradient-liveness check correctly
failed. Initializing that planted head to nonzero weights before any native
run produced a central-difference error2.03e-12. Native primal values,
padding, hook cleanup, gradient mode and parameter flags all check correctly,
including after a deliberate downstream exception.

## Evidence and code

- [Frozen native protocol](../../BILIN18_MLP4_SHARED_PRODUCTS_V1_PREREGISTRATION.md)
- [Native result, scores, subsets and controls](../../BILIN18_MLP4_SHARED_PRODUCTS_V1_RESULT.json)
- [Native gradient helper](../../source_margin_gradient.py)
- [Semantic-square algebra](../../semantic_square_bilinear.py)
- [Executed square controls](../../SEMANTIC_SQUARE_BILINEAR_V1_CONTROLS.json)
- [Deterministic square row audit](../../SEMANTIC_SQUARE_ROWS_V1_AUDIT.json)

The research goal remains active. There is no adopted circuit and no queued
owned GPU successor at this boundary. The square algebra and row audit are
the next concrete work already completed; native square intervention criteria
still need to be registered before that experiment.
