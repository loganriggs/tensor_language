# Current project update — 2026-08-28 18:30 UTC

## UPDATE START

## The short version

We have an executable low-dimensional description of the first two MLP writes, but
we do **not** yet have a faithful explanation of the whole model or even a complete
causal explanation of MLP0. The strongest current MLP0 description is still a
roughly 64-dimensional continuous code with substantial shared lexical structure and
token/context refinement. Most individual coordinates do not yet have stable English
meanings.

The immediate experiment is designed to answer a more useful question than “does the
compressed MLP reproduce its own activation?” It asks whether the compressed
MLP0-to-MLP1 program reacts to controlled edits in the same way as the exact model,
and whether those reactions survive the real downstream network. This is the first
test that directly connects compression to prediction, extraction, and selective
editing.

The experiment has not run on the final role. The scientific result count remains
**0 of 68 actions**. Most recent work has been source-closure and adversarial testing
so that those 68 comparisons cannot be silently mislabeled or computed from the
wrong physical model.

## What the “68 things” are

They are **68 controlled whole-program evaluations**, not 68 discovered circuits.
There are 34 early-MLP program arms, and every arm is tested with two choices for
MLP2:

- `N`: use the deployed/compiled MLP2;
- `E`: restore the exact/native MLP2.

Thus 34 × 2 = 68. The paired MLP2 backgrounds tell us whether MLP2 can compensate
for an MLP0/MLP1 approximation, or whether an apparent success depends on keeping the
exact downstream component.

The 34 early-program arms are:

| Count | Plain-language purpose |
|---:|---|
| 1 | inherited older compiler (`QQ`) |
| 1 | local-loss MLP0 + local-loss MLP1 (`LL`) |
| 2 | independently fitted singleton hybrids (`S0/L1`, `L0/S1`) |
| 1 | jointly suffix-fitted MLP0/1 pair (`RR`) |
| 2 | co-adapted removal hybrids (`R0/L1`, `L0/R1`) |
| 1 | local pair plus the fitted parent-to-child transport map (`LT`) |
| 1 | the same pair with that transport map set to zero |
| 20 | false transport maps used as a finite null comparison |
| 2 | shuffled-teacher controls |
| 2 | fully deployed and fully native early-MLP baselines |
| 1 | a simple newly fitted mean program |
| **34** | **total** |

Only 22 arms receive causal edit tests: `LL`, `LT`, and the 20 false transport maps,
all under the `N` background. The `E` background is observational CE only. This is
intentional: exact MLP2 tells us whether MLP2 compensation matters, but the registered
edit question concerns the executable compiled suffix.

## What is actually implemented

All 68 observational action names now resolve to explicit physical programs. Every
four-row batch is bound to the complete 513-token rows (context plus shifted targets),
the selected program bank, the exact hybrid or null components, and the common scored
support. The expected observational workload is fixed at
68 × 48 = 3,264 student forwards.

The four program-free baselines and all 64 program-bearing actions have adversarial
tests. Changing the target token, swapping a hybrid, using the wrong null map, or
calling a forbidden native early MLP causes failure before a score can be published.

This review also closed two finite-edit defects before final execution:

1. The preregistration requires an output-KL response ratio, but the typed final
   boundary previously carried only response-vector inner products. Output KL is now
   a separate typed per-row numerator/denominator statistic, pooled inside each
   document bootstrap draw.
2. The runtime applied an MLP0 code edit to the executable parent and physical write,
   but captured the code immediately **before** the edit. It now captures the edited
   executable code, and an adversarial test checks that equality.

The focused affected suite passes 73/73. The complete suffix plus observed-adapter
regression passes 265/265 in 108.44 seconds.

## What remains blocked

There is no data, checkpoint, `rspd`, cache, GPU, or FineWeb blocker for the current
CPU work. The blocker is an implementation boundary, not external infrastructure.

The missing piece is one sealed finite-response transaction. For each four-row batch
it must:

1. run one exact teacher baseline, positive edit, and negative edit;
2. for each of the 22 response arms, run its own student baseline, positive edit, and
   negative edit;
3. apply the identical MLP0 physical edit to teacher and student;
4. recompute exact or compiled MLP1 from the edited state;
5. reduce MLP1-code response, centered-logit response, and output KL internally;
6. return only arm-bound scalar sufficient statistics and receipts.

With 48 batches this freezes 144 shared exact-teacher forwards and
22 × 144 = 3,168 student response forwards. Sharing the teacher is legitimate
only inside that sealed transaction. Raw teacher states, codes, writes, or logits may
not escape or be cached for the semantic result owner.

The current aggregate final callback is not sufficient: it can accept already-made
“baseline”, “candidate”, and “null” records without proving that they came from
`LL/N`, `LT/N`, and the indexed null actions. The next implementation must derive
those labels from the canonical action bundle, not trust caller-supplied names.

## What the latest model evidence says

The recent attention-calibration arc rules out three tempting simple explanations for
deep composition failure.

- Scalar gain correction can recover about 61–65% of the shallow B0 deficit, but only
  about 12% at the deeper B3/B5 prefixes. Magnitude mismatch is therefore not a
  sufficient deep explanation.
- Average direction is mostly preserved, but this average hides a sharp exception:
  attention layer 9 reverses direction, with cosine -0.134 at B3 and -0.628 at B5.
  A direct follow-up shows that this striking reversal is a symptom, not the
  bottleneck: compiling L9 recovers only 0.8–1.7 points against a registered 10-point
  bar, negating it loses 6–7 points, and zeroing it gains only 1–2 points.

This increases the value of finite response tests. Local MSE, norm matching, mean
cosine, and even one conspicuous bad layer can all misidentify a distributed
interaction failure.

## How much is explained

Different numbers answer different questions, so they should not be collapsed into
one percentage.

| Question | Current answer |
|---|---:|
| Every module has some tested structural surrogate? | 36/36 |
| Whole-program storage certified removable? | 5.3481% |
| Older behavior covered by human-readable labels? | 32.1% ± 6.4% |
| Strict named causal CE headroom recovered? | 10.923% |
| Strict named causal CE still unexplained? | 4.72714 nats |
| Current compiler's +0.8976-nat gap newly recovered? | 0% |
| Final coupled early-MLP actions evaluated? | 0/68 |

The 36/36 number is coverage, not understanding. The 5.35% number is certified
executable compression. The 10.92% number is causal CE recovery under an older strict
ledger. None establishes full reverse engineering.

## Did the mathematical reviews help?

Yes, but mostly by changing the experiment and ruling out misleading definitions of
simplicity. They have not yet delivered a semantic dictionary for the 64-dimensional
code or a large whole-model CE gain.

Concrete value already obtained:

1. **Gauge-aware simplicity.** Rotating a 64-dimensional latent basis, rescaling a
   bilinear gate and inversely rescaling its output, or exactly folding a constant
   does not change the function. Such changes are true zero-cost equivalences and
   must not be rewarded as discoveries. This is why cost is measured after quotienting
   exact reparameterizations.
2. **An algebraic compression bound.** If a token table has rank $r$ plus a mean,
   its learned downstream row map has rank at most $r+1$. This exposed a previously
   fictitious rank-64 capacity: the rank-1 table needs only a rank-2 map. That is a
   real pricing improvement, although it does not explain semantics.
3. **Composition, not local rank, became the target.** The typed observable/Koopman
   work produced an exact two-step error identity and a tested CPU solver. It says a
   proposed small state must predict its next state and edit response through MLP1,
   not merely reconstruct MLP0 locally. The current finite-edit experiment is the
   source-closed version of that principle.
4. **Hierarchy got an operational test.** “Mean + lexical class + token + context” is
   useful only if progressively adding levels approaches independently optimized
   causal distortion without forcing later levels to redefine earlier ones. This is
   the successive-refinement criterion. It has not yet been run because it needs the
   trusted downstream response boundary.
5. **Task-specific CE is not universal sufficiency.** Blackwell/Le Cam deficiency
   supplied a later test: can a cheap decoder from the compressed observation reproduce
   the native responses across a bank of interventions? This would justify extraction
   and editing beyond one CE task. It is a proposed validation layer, not a current
   result.

Mathematical ideas that were useful mainly by being falsified or pruned:

- generic token-splice Hankel/automata state failed badly out of distribution;
- independently fitted tangent frames were unstable;
- coefficient Tucker/HOSVD for MLP1/2 was dense at full numerical mode rank;
- generic joint diagonalization concentrated weights without meaningful causal gain;
- local norm matching and scalar calibration did not compose at depth.

These negative results save us from spending the final-role budget on attractive but
non-composable simplifications.

## Current ranked plan

1. **Build the sealed 22-arm finite-response transaction.** Highest information gain,
   directly causal, and required for every edit/extraction claim. Bind geometry,
   amplitude, row, position, direction, sign, physical edit, program, and call ledger.
2. **Complete the 18 downstream consumer-norm measurements.** They are now integrity
   and localization diagnostics, not a proposed explanation by themselves.
3. **Complete nine frequency bins, teacher agreement/KL, and observational
   aggregation for all 68 actions.** This tests rare-token failure and MLP2
   compensation on common support.
4. **Make the final owner derive LL/LT/null response summaries from canonical action
   receipts.** Remove the remaining arm-swapping/fabrication path.
5. **Run the full regression and independent audit, then execute the final role once.**
   Admit a simpler program only if whole-program CE/KL, edits, controls, and
   composition pass together.

The project is not stuck on MLP0 alone. We have a compact executable MLP0/1 language;
what is missing is proof that its causal interface is the same interface the rest of
the model uses. That is the shortest path from compression to actual reverse
engineering.

## UPDATE END
