# HOURLY STRATEGIC REVIEW — 2026-09-01 14:15 UTC

## Full goal

Compile bilin18 into a tensor program that is predictive on fresh and shifted text, physically composable with
other accepted replacements, manipulable under preregistered signed interventions, and simpler under literal
storage, compute, edge, state, and program accounting. For MLP0 specifically, explain the exact computation as
token-only, token-by-context, and context-only parts; identify shared token features by common downstream action,
not merely by embedding proximity; and retain an exact token-identity term if that is the simplest faithful account.

## What changed since 13:23

- The honest factored MLP16 program passed physical composition and a lower-fidelity signed tier, closing that arc.
- Exhaustive length-one MLP0 measurements rejected a universal sparse token code: response R² was about .33 and no
  high-response/low-write equivalence pairs appeared. Attention1's response was compact; MLP1's was much broader.
- A linear token-predictable write preserved joined response at R² .9717 and decoded token identity at 90.4%, but
  the follow-up shuffle control exposed a mean/intercept confound in the causal interpretation.
- The exact degree-one token modulation reaches 97.6% token retrieval at rank64, but a mean-preserving shuffled map
  already reaches joined R² .9253. Rank64 raises this to .9640, recovering about52% of error left after the mean;
  for attention1 it raises .7203 to .8860, recovering about59% of remaining error.
- The exact degree-one write curve saturates near .397 R² in both normalized-MLP0-input and raw-token coordinates.
  The registered strong null fired, so rank tuning and immediate token-by-context transfer are closed.

## Confound audit

- **Shared mean/intercept:** absolute injection R² is invalid as evidence for token-specific content when a fitted
  component restores the common write. All future arms need a mean-only baseline and shuffle-subtracted increments.
- **Nonlinear consumers:** `response(full) - response(component)` is not an additive attribution. Exact factorial
  arms and Möbius interaction terms are required for attention1 and MLP1 separately.
- **Frame consistency:** each arm must enter the same native block1 state, with raw-token reinjection, attention0,
  MLP0 bias, layer norm, and block1 weights held fixed. Do not compare effects computed in different baselines.
- **Leakage/postselection:** token-mod5 fitting and heldout evaluation remain fixed. The rank64 observation is now
  descriptive and cannot be selected as the next confirmatory rank. Use the complete canonical degree-one component.
- **Dead knobs:** verify each replacement changes the actual block1 input and include mean-only, shuffled-token, and
  full-native controls. Report tensor checksums and intervention norms.
- **Precision/accounting:** use source-precision values for discovery. Compression is not a claim in the next rung;
  literal program price only matters after causal roles are separated.
- **Length-one scope:** these measurements identify the token-fed boundary, not context behavior. They must not be
  generalized to token-by-context or context-only terms without a separately licensed contextual experiment.

## Independent approaches and falsifiers

1. **Exact mean-conditioned causal factorial.** Split the exhaustive token-only write into constant mean `M`, the
   complete canonical degree-one modulation `L`, and exact residual `Q = F-M-L`. Evaluate all eight subsets through
   attention1 and MLP1, then compute Möbius main effects/interactions. Highest information because it directly fixes
   the confound. Kill a distinct-role claim if `L` or `Q` has negligible shuffle-subtracted contribution or effects
   are entirely interaction-dependent.
2. **Consumer-aware quadratic spectral decomposition.** Pull back the local Jacobian/Hessian of attention1 and MLP1
   to the exact quadratic token kernel, producing separate reader-weighted eigenfunctions. This may explain why the
   same write looks compact to attention1 and broad to MLP1. Kill if crossvalidated reader-weighted bases do not beat
   the canonical full degree-one/quadratic split at equal literal rank.
3. **Downstream token equivalence classes.** Cluster tokens only after subtracting the common write, using the paired
   `(attention1 effect, MLP1 effect)` metric, and validate by interchange. Kill if within-cluster interchange is not
   better than frequency/embedding-matched controls. Do not revive activation-only SAE clustering.
4. **Why generate a common write?** Compare the exact MLP0 common arm with an equivalent downstream bias at each
   consumer and with deletion. If later normalization or bilinear interaction makes the MLP0 placement essential,
   the effects will differ. Kill the functional-common-write story if a cheaper fixed bias is causally identical.
5. **Exact token×context/context-only ANOVA on natural contexts.** Once the token-only components have valid causal
   currency, use the already exact TT/X/CC algebra and crossed token/context resampling to separate identity,
   token-by-context, and context-only effects. Kill transport if the mean-conditioned token modulation does not
   generalize beyond length one.
6. **Toy identifiability benchmark.** Train controlled bilinear models with known flat, block, hierarchy, and DAG
   token features, then test which weight/kernel methods recover them under rotations and redundant embeddings.
   Use this to distinguish a failed method from absent structure in bilin18. Kill methods that fail known ground truth.

## Ranked next actions

Run the exact `M/L/Q` eight-arm causal factorial first, with no rank choice and with mean-only and shuffled-token
controls frozen before execution. Its primary output is the separate and interaction-dependent contribution of `L`
and `Q` to attention1 and MLP1 after conditioning on `M`; no live contextual transfer is licensed merely by a high
absolute R². If `L` has a reproducible positive increment, pursue downstream token equivalence classes in the
consumer-effect metric. If `Q` dominates or only works through `L×Q` interaction, move instead to the consumer-aware
quadratic spectral route. In parallel conceptually—but not as an unregistered GPU search—specify the toy benchmark
so later claims about hierarchy or DAG recovery have a positive control.
