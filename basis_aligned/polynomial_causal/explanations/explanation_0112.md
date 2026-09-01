# Plain-English research plan — 2026-09-01 01:12Z

(Yardstick: damage = extra prediction error above the real model; **LOWER IS BETTER**. A “certificate”
= one of 62 measured behaviors kept close to the original under its registered test. A useful compiled
model must also reproduce interventions; matching ordinary predictions is not enough.)

## Our goal

We are trying to compile the 546M-parameter bilin18 language model into a substantially smaller,
transparent **tensor program** that still behaves like the original.

“Behaves like” has four parts:

1. **Predictive:** it preserves the real model’s next-token distribution on fresh and out-of-distribution
   text, not only the rows used to build it.
2. **Composable:** its simple pieces still work when installed together. A collection of good local
   approximations that fails as a whole is not a compiled model.
3. **Manipulable:** deleting, swapping, or changing a named part produces the same causal change as doing
   so in the real model. This is what makes the program an explanation rather than an opaque imitation.
4. **Actually simpler:** every stored value, input edge, state, multiplication, and routing decision is
   priced. Lookup-table memorization or a large hidden correction does not count as understanding.

In the Theseus metaphor, the goal is not merely a ship that floats. It is a mostly glass ship whose
visible parts preserve both sailing behavior and the effects of removing a plank.

## Where we are now

The best-value registered model keeps a mixed set of attention-score directions: the largest 96 and the
smallest 8 at every replaced head. It costs about 180M stored values, adds `0.0573` CE, and preserves
11 of 62 circuit-behaviors. A compute-sparse version uses one quarter of the MLP units per token and adds
an almost perfectly repeatable `~0.016` surcharge.

Two structural facts now dominate the plan:

- **MLPs are conditionally sparse.** The useful neuron set changes per token. Static pruning is expensive,
  while per-token top-k selection is nearly free. This is an executable conditional-computation result.
- **Attention score maps have a global floor.** Removing almost any part of their fine singular band—at
  the front, tail, half the heads, or even a quarter of the heads—costs roughly `0.052–0.062`, with almost
  the same circuit-damage fingerprint. Value maps do not do this; their compression cost is smooth.

The intervention result was also corrected today. After subtracting the compiled model’s pre-existing
damage, knockout collateral rankings transfer at Spearman `0.86–0.94` for five tested components. The
compiled model usually identifies **which circuits** an intervention affects, but understates **how much**
it affects them: own-effect scale is about `0.37–0.80` of the real model. Attention 16 is the one clear
anomaly. The earlier “large versus small component” law was an accounting artifact and is retired.

## The current plan: test whether the floor is one repairable functional mode

The circuit fingerprints suggest that many different score-map deficiencies may create the same error
as a function of text position. Write the signed per-position loss change of compressed configuration
`i` as a vector

$$
d_i = \mathrm{CE}_i - \mathrm{CE}_{\mathrm{real}}.
$$

The strongest version of the hypothesis is

$$
d_i \approx \alpha_i v,
$$

where every deficiency changes only the amplitude `α_i` of one shared function-space direction `v`.
Rungs 275/276 are currently saving two such vectors. Their registered first check is cosine similarity
at least `0.95`.

That check is a useful screen, but it is not yet enough to license a repair. Common hard tokens can make
two loss vectors look parallel even when the model errors differ. The robust version of this plan is:

1. Collect signed **logit-change** and CE-change vectors from several genuinely different deficiencies,
   not only two nearby rank settings.
2. Fit the shared modes on some documents and compute the singular spectrum on held-out documents after
   removing document, token-frequency, and behavior-class means.
3. Ask whether one mode predicts a new deficiency’s vector and amplitude without refitting that mode.
4. Only then build a legal correction from activations available inside the compiled model. The correction
   must not store position IDs, target tokens, or evaluation rows.
5. Re-evaluate aggregate CE, all 62 certificates, fresh/OOD text, and the knockout-transfer battery.

A strong win is not “cosine passed.” It is: one small shared primitive removes at least half of the
`~0.055` floor on held-out documents, preserves or increases the 11 certificates, and works across more
than one deficient configuration. If the first singular mode does not dominate out of sample, we keep
the measured rank `k` modes rather than forcing a rank-one story.

## Independent path 1: compile the tangent behavior, not just the ordinary output

The current replacements were mainly selected to match unperturbed behavior. That can preserve predictions
while shrinking intervention effects. A more direct compiler would match both a module and its derivatives
along the interventions we care about.

For native program `f`, compiled program `g`, reachable state `x`, and intervention direction `u`, fit in a
Sobolev-style metric:

$$
\mathcal L = \mathbb E\|f(x)-g(x)\|^2
+ \lambda\,\mathbb E\|J_f(x)u-J_g(x)u\|^2.
$$

The intervention directions need not be arbitrary: use the registered mean ablations, circuit-removal
directions, and activation swaps. Random directions are negative controls. This path targets the observed
weakness directly: causal destinations transfer, causal magnitudes are damped.

Success would mean matching the present model’s CE while moving own-effect ratios toward 1 and retaining
collateral Spearman at or above `0.9` on held-out components. Failure would show that local derivative
matching does not survive joint composition, which is itself a clean reason to prefer a global compiler.

## Independent path 2: learn the minimal predictive state, rather than compressing every native module

The present program follows the native architecture closely: replace each attention or MLP operation with
a cheaper relative. A more radical route is to identify states only up to what the future can observe.

Two hidden states are equivalent if every tested continuation and intervention gives the same future
logits. This is a causal quotient or predictive-state representation. Prefix/continuation Hankel matrices
give a measurable lower bound on its dimension; controlled Hankel blocks extend the test to interventions.
If those matrices have stable low numerical rank across documents, their singular coordinates become the
state of a new small tensor program, with polynomial transition maps fitted between layers or token steps.

This could bypass the score-map floor entirely: the native network may use 148 delicate coordinate systems
to maintain a much smaller downstream state. The decisive tests are held-out continuation prediction,
intervention prediction, and rank stability under new documents. If rank grows with sample size or changes
under small corpus shifts, there is no small quotient at that grain and this path closes honestly.

## Independent path 3: replace the distributed cancellation with one shared invariant

The floor may not be “missing information” at all. It may be a repeated numerical correction—normalization,
centering, or a signed cancellation—implemented redundantly across heads. The head-subset experiments say
that preserving arbitrary pieces does not help; that is exactly what one would expect if a global invariant
must be satisfied everywhere.

Instead of retaining more singular directions, measure candidate invariants of the exact score matrices:
row means, centered Gram moments, trace/Frobenius terms, entropy, and signed fine-band bilinear forms. Regress
the common functional error mode on these quantities across heads and documents, then implement the smallest
successful statistic once as a shared library primitive.

The falsifier is stronger than correlation: intervening on the proposed invariant must move the common
damage amplitude while matched-energy perturbations orthogonal to it do not. A win would explain why a
quarter-head deficiency and an almost-full deficiency pay the same toll, and would replace many tiny
directions with one named operation.

## Independent path 4: choose coordinates by causal response, not weight energy

Ordinary SVD orders directions by matrix norm, but the project repeatedly finds that norm and causal price
differ—the smallest score directions can be indispensable. We can instead solve a generalized eigenproblem
whose numerator is preservation of signed circuit/logit responses and whose denominator is storage or
activation energy.

Concretely, build a response matrix whose columns are signed changes under the registered circuit and
knockout battery. Find directions maximizing

$$
\frac{w^\top C_{\mathrm{causal}}w}{w^\top(C_{\mathrm{storage}}+\epsilon I)w}.
$$

Fit the basis on one half of circuits/documents and score it on the other half. This is different from the
failed “make a few high-damage blocks exact” strategy: it optimizes a shared basis for response preservation,
not a hand-chosen allocation of native heads. It wins only if it Pareto-dominates norm-SVD at equal stored
values on both aggregate CE and held-out certificates.

## How these paths fit together

The paths answer different questions and can be run in this order:

1. **Function-space spectrum:** is the `0.055` floor one mode, a few modes, or many?
2. **Shared invariant:** if it is one/few, can we name and compute the amplitude cheaply?
3. **Causal-response basis:** if the mode is not expressible by a simple invariant, can a supervised but
   held-out causal basis preserve it more cheaply than SVD?
4. **Tangent compiler:** whichever predictive representation wins, make intervention fidelity part of the
   fitting objective instead of hoping it emerges afterward.
5. **Predictive-state quotient:** in parallel on CPU/small GPU slices, test whether following the native
   architecture is unnecessary and a much smaller state machine exists.

Every branch uses the same adoption gate: fresh/OOD prediction, joint composition, intervention transfer,
certificate count, and literal complexity price. A result that improves only one ledger is a useful finding,
but it does not complete the project.

## What we should not do next

The existing evidence closes several tempting loops: more arbitrary head/direction subsets, uniform rank
sweeps, Tucker/HOSVD of raw MLP weights, static deep token tables, larger context codebooks, and targeted
exactness at a few “important” blocks. Repeating them with new hyperparameters would add volume rather than
information. The next experiments should distinguish the mathematical objects above.

The project succeeds when the final artifact is small enough to state, accurate enough to trust, and causal
enough to edit—and when its failures are predicted by the program rather than discovered after deployment.
