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

## Exact one-layer source-factor decomposition

We next opened L11H3 itself using the existing exact attention factor primitive,
with no regression. For base versus coordinate-removed states, every source
term was decomposed into the full 31-term Möbius expansion over native
`Q`, `K`, `Q2`, `K2`, and output-projected effective `U`, then projected onto
the frozen writer axis. The six causal source positions were kept open.

The instrument passed strongly:

- exact head source identity error: `0.0`;
- maximum complete Möbius closure error: `3.43e-5`, relative `9.12e-7`;
- independent corrected-scalar replay error: `6.10e-5`;
- synthetic five-factor closure error: `4.44e-16`.

This localizes the L11H3 edge sharply. The contextualized subject source alone
reconstructs the scalar response to `0.03367` relative L2; adding the second
determiner source reaches `0.02666`. The other four source positions each have
RMS below `2.16`, versus subject-source RMS `118.91`. Their small terms partly
cancel: subject aligned recovery is `1.0105`.

At the factor level, `U` alone reaches `0.19816` relative L2. Greedy exact
addition of `K`, `K×U`, and `Q×Q2×U` reaches `0.10759`. Eight unscaled
source×factor atoms reach `0.09728`; the selected atoms are:

1. `subject:U`
2. `subject:K`
3. `subject:K×U`
4. `attractor:Q`
5. `subject:Q×Q2×U`
6. `subject:Q×K2`
7. `subject:K×K2×U`
8. `subject:K×K2`

The registered matched-readout specificity gate is a valid null. The target
writer was more compressible than all 16 orthogonal random readouts, whose
median eight-atom error was `0.18238`, but the absolute advantage was `0.08510`
rather than the preregistered `.10`. Therefore the exact decomposition and its
source/factor compression are accepted, while the eight-atom set is not yet
licensed as behavior-specific or OOD-stable.

Mechanistically, this moves the unresolved transform upstream: L11H3 reads the
effect almost entirely from the subject position's already-contextualized value
state. Direct context-token and attractor-token source writes are small. The
next clean boundary is therefore the construction of that contextualized
subject value before L11, not an unrestricted six-source attention donor.

### Upstream value decomposition: false negatives separated from the result

We attempted the next five-port Möbius split of the subject value into embedding
recurrence, layer 0–3 writes, layer 4–7 writes, layer 8–10 writes, and the
inherited first-value bus. Neither attempted result is licensed.

V1 failed because of a genuine implementation bug: string-prefix grouping made
`attn_10`/`mlp_10` match both the layer-1 prefix and the explicit layer-10
prefix, double-counting layer 10. Its component reconstruction error was
`1542.23`; terminal `invalid` is preserved.

V2 replaced all component keys with exact integer layers and removed the double
count. It still failed the preregistered absolute component-reconstruction bar:
independently propagated float32 components differed from the natively ordered
residual sum by `0.0014648`, above `1e-5`. Other checks nearly closed—native
subject-value replay `9.54e-5`, Möbius closure `3.05e-5`, relative closure
`6.56e-8`—but the registered instrument is conjunctive, so V2 also remains
`invalid`.

V3 prospectively performed that audit without changing the panel, five ports,
32 corners, readouts, greedy rule, scientific thresholds, or price. It formed
the residual groups in float64 and defined the late group as the exact native
remainder, while separately comparing that closed group with the independently
propagated late group. Closed reconstruction error was exactly `0.0`; the
required correction had maximum absolute size `0.0015831` but relative L2 only
`2.79e-7`. Native value replay (`9.16e-5`), Möbius closure (`4.58e-5` absolute,
`6.90e-8` relative), and the synthetic fixture (`4.44e-16`) also passed. Thus
V2's apparent negative was float32 association, not a missing path. V1 and V2
remain invalid; V3 is the licensed result.

The five-port decomposition is compact for the frozen writer readout. Its
target RMS is `122.03`. Four exact Möbius terms reduce relative L2 error to
`0.01298`: late writes 8–10, middle writes 4–7, the middle×late interaction,
and the early×late interaction. The late main effect alone has RMS `101.95`
and aligned recovery `0.83290`. Middle writes contribute `0.14242` aligned
recovery, the embedding recurrence `0.10413`, early writes `0.03389`, and the
inherited first-value bus contributes `-0.14478`.

The registered positive red-team also matters. The target's four-term error is
below the random-readout median (`0.05546`), but its advantage is only
`0.04248`, below the registered `0.10`. Therefore this is a valid, compact
decomposition of the selected writer response, not yet evidence that these
four terms are behavior-specific or reusable. The failed specificity null is
not explained away by the numerical repair.

### Late-writer graph: sparse terms transfer, specificity needs a harder audit

The dominant late band was then opened into eight ports: the propagated layer
0–7 state, the six individual attention/MLP writes in layers 8–10, and the
first-value bus. The exact 256-corner / 255-dividend Möbius instrument passed:
closed reconstruction was `0.0`, the MLP10 gauge correction was `1.07e-6`
relative L2, native replay was `9.16e-5`, and closure was `3.81e-5` absolute /
`9.47e-8` relative.

Selection was restricted prospectively to 36 main or pair terms. Six unscaled
terms were chosen on only the first eight noun pairs in `near`/`behind` and
then frozen:

1. `mlp_8`
2. `upstream_0_7`
3. `mlp_10`
4. `attn_9`
5. `attn_9×mlp_9`
6. `mlp_9×mlp_10`

The same sum achieved relative L2 `0.13469` on discovery, `0.12226` on held-out
contexts, `0.15576` on held-out nouns, and `0.15276` on the joint holdout. No
coefficients were fit. Aligned recovery remained positive on every panel
(`0.8546`–`1.0116`). This is direct evidence that a sparse intermediate formula
transfers across both lexical and context axes rather than merely compressing
the pooled authority.

The registered random-readout median gate also passed: random median joint-OOD
error was `0.29530`, giving the target an advantage of `0.14254`. A post-hoc
rank check, however, exposes fragility in that positive result: 8 of the 16
random readouts had *lower* joint-OOD error than the target. The random errors
are bimodal, so the median-advantage threshold overstates specificity. The
registered pass is retained, but behavior-specific/reusable status should wait
for a larger null panel with a rank/quantile gate and effect-magnitude audit.

The first 64-direction specificity/pruning audit is retained as `invalid`.
It exactly reproduced the six selected masks, had closed component error `0.0`,
and had relative Möbius closure `2.00e-7`, but its float32 Möbius subtraction
reached absolute closure error `1.53e-4`, above the registered `1e-4` bar when
maximized over 65 readouts. Its pruning and specificity outputs are provisional
and are not used here. The corrective audit must change only Möbius arithmetic
to float64 while preserving the panels, 64 new nulls, masks, and scientific
gates.

## What this changes in the sparse graph

The previous graph boundary was

`causal embedding-number node -> unresolved lexical/context transform -> small rank-one L11H3 edge`.

It is now narrower in two stages:

`causal embedding-number node -> mostly late-layer subject-position writes + small cross-band corrections -> subject U atom -> small rank-one L11H3 edge`.

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
- `SUBJECT_NUMBER_L11H3_SOURCE_FACTOR_MOBIUS_V1_RESULT.json`
- `SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V1_RESULT.json`
- `SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V2_RESULT.json`
- `SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V3_RESULT.json`
- `SUBJECT_NUMBER_L11H3_LATE_WRITER_OOD_MOBIUS_V1_RESULT.json`
- `SUBJECT_NUMBER_L11H3_LATE_WRITER_SPECIFICITY_AUDIT_V1_RESULT.json`
- their preregistrations, bindings, frozen authority, and managed runners
