# Hourly circuit and systems review — 2026-09-04 15:30 UTC

## Outcome this hour

The Task 14 subject–verb agreement screen found a strong causal handle for the final ` is` versus ` are` choice.

- Replacing the residual stream at the output of block 11 recovers 33.0% of the answer change; replacing it after block 12 recovers 93.7%.
- Replacing attention block 11 alone recovers 61.3%.
- Inside that attention block, head 11.3 alone recovers 60.4%. Every held-out A1 and A2 example moved in the intended direction.
- The two answer-preserving controls moved only 3.0% and 3.9% of the normal donor effect.

This does **not** yet identify a grammatical subject-number variable. The A1 and A2 tests used two sentence constructions separately, not literal donor swaps between the constructions, and the controls do not exclude a generic output-side `is`/`are` signal. The result is therefore a strong causal screen for a swappable agreement-output signal at head 11.3.

## Where the time went

Model execution was not the bottleneck:

| Work product | Approximate wall-clock interval | Model time | Outcome |
|---|---:|---:|---|
| pronoun candidate | commit 14:43, result 14:55 | 2.17 s, 8 forwards | honest null: construction confounded |
| quote candidate | commit 14:47, result 14:56 | 2.07 s, 8 forwards | honest null: endpoint tokenization invalid |
| disposable wording probe | around 15:04 | 1.60 s, 1 forward | stopped synthetic prompt iteration |
| Task 14 first screen | adapter around 15:11, result 15:22 | 6.23 s, 228 forwards | valid 55-site evidence; head extension skipped by a control-flow bug |
| corrected Task 14 screen | fix 15:23, result 15:28 | 7.11 s, 264 forwards | attention 11 and head 11.3 pass |

The total model time was about 19 seconds. The dominant cost was candidate design, prior-art checking, review, and correcting orchestration. The engineering target remains one decision-quality screen or honest null per ten serial minutes.

## Systems corrections made

1. A candidate lint now catches two failure modes before a GPU job: answer/foil endpoints that merge into a different token, and mention order that perfectly predicts the answer.
2. The fast-screen framework now supports an active control that preserves the correct answer. Its score is

   $$
   E_C = \frac{|m_{\mathrm{patched}}-m_{\mathrm{native}}|}
   {\operatorname{median}(\text{native A1/A2 donor effect})},
   $$

   where $m$ is the answer-token logit minus the foil-token logit. Small $E_C$ means the site ignores a meaningful change that should not alter the answer.
3. Task 14 reuses the existing frozen dataset rather than creating another prompt family.
4. Its prior-work receipt binds the task record, main dossier, module dossier, method-failure log, exact authority files, and the relevant earlier results.
5. The first screen exposed a generic orchestration bug: residual-stream sites and attention modules were ranked together, so the trivial final residual state won and prevented the head stage from opening. The corrected compiler chooses the best passing attention module for head expansion. The original 55-site measurements remain valid; only its conditional head extension was incomplete.

## Duplicate-work prevention

The Task 14 JSON already records several later experiments that the older prose dossier still calls “next.” File freshness or a matching hash is therefore not enough to decide novelty. Before each candidate we now require:

1. read the canonical task JSON;
2. read the relevant module dossier and method-failure record;
3. bind the exact sources in a machine-readable novelty receipt;
4. label the candidate as a replication, extension, contradiction test, or genuinely new claim;
5. append the result to the task record immediately after interpretation;
6. at the hourly checkpoint, compare the latest event in the task record with the prose dossier’s claimed frontier.

## Three-hour mathematical review

The combination of weak removal and strong interchange is expected in a model with multiple interacting mediators. Earlier work found that mean-replacing heads 11.3 and 15.5 barely changed natural agreement accuracy; the new donor interchange strongly steers `is` versus `are` through head 11.3. These are compatible if another path compensates during removal while head 11.3 still carries a swappable signal.

The useful localization measurements around block 11 are

$$
R(\mathrm{resid}_{11})=0.330,
\quad R(\mathrm{attn}_{11})=0.613,
\quad R(\mathrm{MLP}_{11})=0.064,
\quad R(\mathrm{resid}_{12})=0.937.
$$

Here $R$ is the fraction of the native donor-induced logit-margin change recovered by patching. These four numbers are **not** cells of one additive factorial. Patching `resid:11` replaces the incoming state and then recomputes attention and the MLP. Patching `attn:11` or `mlp:11` substitutes a cached module output, and patching `resid:12` replaces the complete state after the block. Combining those heterogeneous interventions would give meaningless interaction coefficients.

A valid small factorial can still test whether a cached incoming residual, attention output, and MLP output combine additively, but it must rerun all eight cells through one explicitly fixed replay equation. For two components measured under that common equation,

$$
I_{AB}=R(A+B)-R(A)-R(B),
$$

with the native recipient as the zero point. A nonzero $I_{AB}$ is a causal interaction: the effect of one component depends on whether the other was also replaced. The three-way term is the usual inclusion–exclusion extension. This factorial is secondary to the clean cross-syntax identification test.

This is localization, not the final basis. Head 11.3 is a 128-dimensional implementation-defined slice. The sharper mathematical path is:

1. establish a syntax-general variable with literal cross-syntax interchange;
2. find the smallest rotated subspace inside the head output that preserves that interchange behavior and passes unrelated controls;
3. contract that subspace through the exact output projection and the exact downstream linear/bilinear weights;
4. test whether the resulting writer–reader path predicts held-out behavior and can be removed without damaging unrelated circuits.

Rank or activation energy is useful only as a cost or diagnostic. It is not evidence of an interpretable circuit by itself.

## Next serial decisions

1. Run literal PP↔relative-clause donor interchange at attention 11 and head 11.3 using the already-frozen validation pairs.
2. Add an unrelated task with the same ` is`/` are` endpoints to distinguish grammatical number from a generic output-token signal.
3. Run the small block-11 component factorial to measure redundancy and interaction, but do not let it delay the two identification tests above.
4. Only if the site survives those tests, fit a minimal causal subspace and translate it into exact weight paths.
