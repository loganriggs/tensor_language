# MLP4 shared native products selected by fit-only behavioral saliency

Registered before native fitting or evaluation. Authority: original bilinear
handoff/pilot with appended structural-success criterion. The previous
response-approximation ladder is closed. This screen changes the SOURCE
product support and retains the entire native downstream computation.

Question: can the same128 of4608 native MLP4 products reproduce its complete
source-interchange effect for both has/had and is/was? Native product labels
are candidate computational terms, not identified semantic features. Fixed128
is a single sparse-library hypothesis, not a tuned budget or minimality claim.
No budget, source site, score rule, gain or subset-family rescue after outcome.

Use unchanged L9_SHARED_QUERY_ROUTER_V1_ROWS:24fitpairs,72evaluation pairs in
has_heldout/has_a2/is_heldout/is_a2. All original native errors remain. These
families are previously opened; no pristine task-family OOD claim.

For each task and each fit pair, compute the gradient g of the actual final
answer-minus-foil logit margin with respect to the FULL source MLP4 output,
separately at both native endpoints. Freeze weight requires_grad flags during
these four reverse-AD forwards and restore them afterward. Source hook enables
recording after MLP4 inside the otherwise unchanged native no_grad backend.
No weight fitting, alternate forward executor or donor-conditioned derivative.
Actual native source products phi are inputs to Down. Let dphi=phi_d-phi_b.

    score_t,j = mean_fit_rows sum_valid_tokens
                (abs(dphi_j * (g_base · Down_j))
                 + abs(dphi_j * (g_donor · Down_j)))/2.

The two endpoint answer/foil orientations may reverse; absolute values make
the saliency independent of that sign convention. All valid source tokens
contribute; padding does not. Normalize each task score to sum1. Shared score
is the elementwise MINIMUM of the two normalized task scores. Select its top128,
using ascending native index to break ties. Own-task and cross-task arms use
the corresponding task's top128. Random control is the first128 entries of a
CPU torch.randperm(4608), generator seed910320, shared across all panels.
Save all scores and masks before evaluating. This derivative saliency is only
a proposal: prior tangent and cancellation failures remain valid. We make no
finite-effect prediction by summing derivative attributions.

For each evaluation panel, capture both native endpoints. For BOTH receiving
sides evaluate eight arms, all valid source positions:

- full: native_output + Down(all paired product changes);
- identity: no product change;
- shared: same edit restricted to the shared128 products;
- own: task-specific128 products;
- cross: the other task's128 products;
- random: fixed random128 products;
- direct_oracle: replace with the complete donor MLP4 output;
- remove_shared: native_output - Down(shared native products), no donor.

Keep source bias, upstream generation and full native suffix. The removal
matches a static Down-column deletion at this sole source layer; it does not
establish selectivity without unrelated controls. No value-port/head restriction.

Count:8fitforwards (4native primal,4gradient) on96sequence instances,4backwards;
8evaluation-native forwards on144sequence instances;64armforwards on1152;
total80forwards/1392sequences,4backwards.600second watchdog, managed GPU only.

A instrument: frozen source/row/backend/checkpoint/primitive hashes; no fit/eval
exact token-sequence overlap; no native refiltering; tiny gradient centraldiff
and cleanup controls; fit gradient/native full-logit and phi replay; full versus
direct oracle; identity native replay; all maxabs<=1e-3 ANDrelFrobenius<=1e-5.
Check source masks, allfinite, exact counts, restored hooks/gradmode/parameter
flags, no parameter grads. Prior full-source margin/effectnorm replay uses
same numerical bars. Native full-source effects must be live>1e-8.

B shared sufficiency: relative error of shared effect versus full source effect
<=.10 in BOTH centered50304-logit and answer-minus-foil margin frames, all8
panel/direction cells. Effects always subtract the receiving native baseline.
C matched-control advantage: random relativeerror minus shared relativeerror
>=.05 in BOTH frames, all8cells. D live reference: all full-source full-vector
and margin norms>1e-8. Instrument miss invalid; scientific miss is a null.
Own/cross arms are diagnostic, not alternative promoted winners. Report shared
static-removal cue-contrast retention separately, without a promotion predicate.

This is a basic screen, not semantic identification, whole-task sufficiency,
independent extraction, selective removal or proven composition. Those require
follow-up after a pass. All545902902native parameters and source/background
execution remain charged. Shared128 nominal source program has442368 learned
weight entries (Left/Right/Down slices) plus128indices; no actual saving while
full native model/background remains. Do not claim the subset explains its
input quantities merely because its weights are explicit.
