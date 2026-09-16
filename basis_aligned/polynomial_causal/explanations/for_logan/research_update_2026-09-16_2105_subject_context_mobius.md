# Research update: the missing subject-number transform has a low-rank context interaction

## Bottom line

The DCT briefing's warning not to average away context was directly useful. The
small causal path found previously is now localized more sharply as

`embedding-number node -> subject-dominant transform + low-rank subject×context interaction -> rank-one L11H3 write`.

On the corrected frozen 32-subject × 4-context factorial panel, a
subject-plus-context additive model misses the L11H3 response scalar by
`0.19480` relative L2. The interaction has RMS `22.9110`; one interaction mode
captures `0.77931` of its energy and two capture `0.91460`. This is evidence for a small explicit
interaction node rather than a token-only scalar or an unrestricted donor head
state. It is not yet an extracted native circuit: subject and context labels
were used only as factorial indices.

## Frozen assay

The already-opened 32 subject forms from the coordinate-removal authority were
crossed with four contexts: `near`, `behind`, `under`, and `above`. A corrected
authority explicitly selected and asserted an atomic opposite-number noun
attractor in all 128 rows. Every 32×4 subject/context/attractor combination in
this corrected panel was unopened when its authority and preregistration were
frozen.

For every cell we repeated the same-norm removal of the frozen embedding-number
coordinate and measured

`alpha(subject, context) = axis^T (L11H3_base - L11H3_removed)`.

We then used the unique zero-sum two-factor/Möbius decomposition

`alpha = grand + subject_main + context_main + subject_context_interaction`.

No answer logits, gradients, or parameter fitting were used.

## Result

The centered sum-of-squares allocation is:

| term | centered energy fraction | RMS |
|---|---:|---:|
| subject main effect | `0.95401` | `105.599` |
| context main effect | `0.00108` | `3.555` |
| subject×context interaction | `0.04491` | `22.911` |

The interaction is modest in variance but causally important at component
precision: removing it gives raw relative L2 `0.19480` and centered relative L2
`0.21192`. Its singular values are `228.83`, `95.34`, and `75.75`; rank one
captures `0.77931`, and rank two captures `0.91460`.

The corrected result rejects a tempting stronger claim. Its additive error is
at the `0.5625` lower-tail percentile among 256 within-subject column
permutations: context labels do not align a meaningful shared additive main
effect. However, a clearly marked post-hoc 10,000-permutation red-team found
that the interaction's rank-one energy is unusual: observed `0.77931`, null
median `0.61365`, null 99th percentile `0.76597`, and upper-tail fraction
`0.0044`. Thus the interaction has a coherent dominant mode, while its factor
still needs native-state extraction and prospective confirmation.

The four corrected context-conditioned scalar means are `near=-46.63`,
`behind=-41.00`, `under=-51.02`, and `above=-46.54`. Those label-level means are
descriptive and are not licensed graph inputs.

## Symmetric red-team and the initially invalid run

The first crossed run is deliberately retained with terminal `invalid`. Its
64 replay cells differed from the earlier 64-row capture by up to `0.001007`,
above the preregistered `1e-4` gate. We did not loosen that gate.

Two predeclared positive fixtures had already shown that the decomposition code
was not silently manufacturing a negative result:

- a known additive matrix produced interaction at `3.55e-15`;
- a known rank-one interaction was recovered within `6.11e-15`.

The remaining hypothesis was batch-size arithmetic: the crossed run used 128
rows, while the earlier assay used 64. A separately registered precision audit
rebatched `near`/`behind` in the exact original 64-row order and put
`under`/`above` in a second 64-row batch. It passed every gate:

- exact replay maximum and RMS error: `0.0`;
- 128-row versus 64-row alpha drift: RMS `0.0003088`, maximum `0.0014648`;
- additive-error drift after rebatched capture: `1.94e-7`;
- rank-one-energy drift: `8.29e-8`;
- removal geometry error: `2.89e-15`;
- interaction-to-exact-replay RMS ratio: effectively infinite because replay
  was bit-exact.

Thus the initial gate failure was a real and correctly caught numerical
batching effect, not a miscoded interaction or a reason to discard the
substantive decomposition. The original invalid terminal remains unchanged;
the audit terminal is `context_mobius_precision_qualified`.

### Inherited attractor-indexing bug

A second red-team found a genuine design bug inherited from the earlier
removal authority. Its expression `attractor_pair[2-number_index]` indexed the
three-tuple `(singular, plural, stratum)`. Consequently, singular-subject rows
used the literal token `regular` or `irregular` as the attractor, and
plural-subject rows used a plural noun. Those results were real for their actual
texts, but they did not test the claimed opposite-number noun attractor.

We preserved the flawed artifacts and narrowed their interpretation. A new
authority uses `attractor_pair[1-number_index]` and asserts noun atomicity,
position, and opposite number for all 128 rows. It also duplicates every model
capture. The corrected assay passed exactly: duplicate alpha maximum error
`0.0`, geometry error `2.89e-15`, and both synthetic fixtures near `1e-15`.
Its additive error differs from the flawed panel by only `0.00169`, and its
rank-one energy by `0.02898`, so the qualitative interaction result survives
the fix. The corrected terminal is
`corrected_context_rank1_interaction_candidate`.

### Corrected behavioral removal and mediation

We then froze and ran that missing replication before opening any behavior
logits from the corrected panel. It combined the target coordinate removal, 16
same-token/same-norm/same-distance embedding controls, full and rank-one L11H3
restoration, and 16 matched output-axis controls over all 128 rows. Every
registered gate passed:

- native agreement and the frozen embedding decoder were correct on all 128
  rows and in every number, context, and regularity cell;
- coordinate-removal damage was positive on all 128 rows, RMS `3.27177`, or
  `0.62030` of native-margin RMS;
- equal-distance embedding controls had median RMS `0.08076` and maximum
  `0.12683`; target/median ratio was `40.51`;
- unrelated `can`/`will` collateral was `0.05586` of target damage;
- full L11H3 restoration recovered `0.09945` of damage with cosine `0.88363`
  and positive effect on `0.82031` of rows;
- the frozen rank-one restoration recovered `0.09480` of damage;
- rank one reproduced the full-head rescue with cosine `0.99919`, relative L2
  `0.05800`, and norm ratio `0.95743`;
- matched random output axes had median effect RMS `0.00699`, versus `0.35256`
  for rank one, and their best relative error was still `0.97941`;
- repeated removed-head capture error was `0.0`, and maximum intervention
  geometry error was `5.77e-14`.

The terminal is `corrected_removal_and_rank1_mediation_held`. This repairs the
behavioral evidence under genuine opposite-number nouns. It confirms selective
removal and a reusable small rank-one edge, but the full-head/native-base value
remains an explicit donor port and the corrected nouns are not a fresh lexical
panel.

## What this changes in the sparse graph

The previous graph boundary was

`causal embedding-number node -> unresolved lexical/context transform -> small rank-one L11H3 edge`.

It is now narrower:

`causal embedding-number node -> large subject term + rank-1/2 subject×context correction -> small rank-one L11H3 edge`.

This advances localization and compression, but extraction is incomplete. The
factor scores are currently observational coordinates on the opened factorial
panel. The next experiment should attempt to predict the subject and context
factor scores from native checkpoint states, with leave-noun-pair-out and
leave-context-out tests. Only a frozen native factorization should proceed to a
new-vocabulary/new-context causal test. A token or template lookup table would
not satisfy reusability.

## Artifacts

- `SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_V1_RESULT.json`
- `SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_PRECISION_AUDIT_V1_RESULT.json`
- `SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_RESULT.json`
- `SUBJECT_NUMBER_CONTEXT_INTERACTION_RANK_SPECIFICITY_V1_RESULT.json`
- `SUBJECT_NUMBER_CORRECTED_REMOVAL_MEDIATION_V1_RESULT.json`
- their preregistrations, bindings, frozen authority, and managed runners
