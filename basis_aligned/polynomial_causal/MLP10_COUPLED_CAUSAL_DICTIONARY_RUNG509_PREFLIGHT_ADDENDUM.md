# Rung509 preflight addendum: exact call price and held-out response forecast

Status: frozen during CPU implementation, before any rung509 CUDA/model outcome.

The original registration omitted three required call classes from its price: one direct-native replay in each exact
singleton batch, the absent/intact factor captures needed to construct fitted atom changes, and the two separate
discovery/confirmation passes needed to measure pair composition. It also priced confirmation exact-term collection
without explicitly saying how those unopened responses are scored. This addendum corrects both issues without
changing the eight-atom model, data split, fitting settings, stability bars, physical interventions, or routes.

## Held-out exact-term forecast

When the stability and discovery-removal gates leave two through eight atoms, collect all253 exact-term finite
responses on confirmation documents without refitting. Align the six discovery fits as registered, then use the
arithmetic mean of their assignment tensors and shared response vectors; no seed is selected. The fixed dictionary
must predict the 4-by-253-by-34 confirmation response tensor using the discovery coordinate scales.

Against a one-response baseline that predicts every exact term by the discovery mean response, the dictionary's
confirmation standardized mean squared error must be at most `.75` times baseline. It must also be at most `.80`
times a fixed seed509 permutation control that independently permutes the253 term assignments within each score
implementation while preserving every atom's marginal assignment mass. These are response-prediction controls, not
activation reconstruction or adoption criteria. Failure makes C false even if weighted group removals happen to be
large.

After alignment, all physical atom interventions use the arithmetic-mean assignments. This fixes the deployed
candidate independently of confirmation and prevents favorable-restart selection.

## Correct conditional price

For each248-document exact-singleton phase, every62-batch block runs one direct replay, one score-absent capture, and
under four sources one intact capture plus253 term removals:

`62 * (1 + 1 + 4*(1+253)) = 63,116` forwards.

Each eight-atom physical phase reruns one absent capture and, under four sources, one intact capture plus eight atom
removals:

`62 * (1 + 4*(1+8)) = 2,294` forwards.

For `q` confirmed atoms, each joint-removal phase costs

`62 * (1 + 4*(1+choose(q,2)))` forwards.

Both discovery and confirmation joint phases are required: the former freezes the composition rule and the latter
tests it. Therefore the maximum complete price at `q=8` is

`2*(63,116 + 2,294) + 2*62*(1 + 4*(1+28)) = 145,328` forwards.

The run stops at63,116 if the fitted dictionary is not stable, at65,410 if fewer than two atoms pass discovery
physical removal, and at130,820 if fewer than two atoms pass confirmation or the held-out exact-term forecast. There
are exactly six CPU fits: three seeds on each of two discovery halves. Backwards and deployed parameter changes
remain zero.

Pred A now includes exact equality between observed and the applicable conditional call count. Pred C includes both
the fixed held-out exact-term forecast and at least two physical atom confirmations. No scientific threshold was
weakened; the added forecast makes identification stricter.
