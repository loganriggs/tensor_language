# Third noun and animacy: prospective generalization of the observed heuristic

Frozen after SUBJECT_OBJECT_CONTROLLER_V1, before this population's native outputs.
Parent is valid: controller capability fails, the predeclared second/nearest-noun
rule matches128/128 decisions. Per-world additive approximations preserve all
these signs, with interaction/nonconstant RMS ratios .062–.158. Those approximations
use opened whole cubes; they are descriptive, not held-out predictors.

The parent cannot distinguish grammatical object, nearest noun, nearest human,
or the noun two tokens before readout. This test changes that ambiguity directly.
No parent phrase is repaired, no old outcome is relabeled.

## Frozen domain

Eight worlds: four lexical triples × defend/introduce. Triples are
king/brother/crate, man/boy/box, father/son/cart, actor/prince/book.
Use the first noun type in both subject and object positions, independently
singular/plural. The third noun is either the human or inanimate member of its
triple, independently singular/plural:

    The kings promised the king near the brothers to defend
    The kings promised the king near the crates to defend

Cross c(control verb promised+1/persuaded-1), s(subject number), o(object
number), a(third noun number), h(human+1/inanimate-1):32 corners/world,
256 total prompts. All eight worlds are structural held-out tests relative
to the parent; lexical items partly overlap. No calibration or fitting.
All token-length and single-token-factor checks pass;64 opposite subject/object
number reversals preserve the token multiset. No full prompt overlaps the parent.

## Four predictions and opposing mechanisms

Each rule predicts the sign of m=logit(themselves)-logit(himself):

1. Controller: (s+o+c*(s-o))/2.
2. Object: o, regardless of verb or third noun.
3. Nearest noun: a, including an inanimate noun.
4. Nearest human: (o+a+h*(a-o))/2.

The fourth is another explicit bilinear selector; it changes the selector
variable from grammatical control to the third noun's human/inanimate category.
These are competing input-output programs, not claimed native implementations.

A. Instrument: row/builder/source hashes, exact32-term Boolean-basis orthogonality,
native answer-margin versus full-logit capture maxabs<=1e-3 ANDrelFrob<=1e-5,
finite outputs, coefficient closure maxabs<=1e-9, hooks restored, exactly16
forwards/256 sequence evaluations, no fitting/backward passes.600s watchdog.

B/C/D/E. For controller/object/nearest-noun/nearest-human respectively, require
in EACH of32 corner cells at least75% correct signs among the eight worlds
and mean signed margin>=1. Same bars for every rule; no pooling across corners.
Report conditional-on-h subgroup verdicts as diagnostics without promoting a
failed overall rule. No outcome-based world filtering or rule/threshold changes.

Record full centered vocabulary coefficient norms and margin coefficients for
the32 terms. The exact finite-grid transform is a descriptive identity, not a
global native polynomial or an extracted program. In particular, a category-
conditioned sign rule need not predict native confidence or other vocabulary.

If one rule passes, freeze it for neural localization and a further independent
family. If none passes, report the structured failure; do not alter the tail,
lexical triples, thresholds or selector grammar to rescue this population.
The grammatical controller hypothesis remains failed on the parent even if
new contexts support it. A passing nearest-human rule is limited to these
lexical/structural populations until broader category transfer and neural edits.

## Price and requirements left open

Sixteen native batches of16 prompts; full256×50304 FP64 output matrix is
98.25MiB, below256MiB per analysis tensor. All545902902 parameters retained.
No weights/axes fit, no compression or independent extraction. Save compact
JSON statistics, not the dense output tensor. Native capability and structural
generalization of a predefined behavioral rule are not neural circuit discovery.
Removal, input/operation/consumer identification, full-output prediction and
compositional reuse still require causal, weight-grounded evidence.
