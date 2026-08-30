# Plain-language project update — 2026-08-28 18:00 UTC

This document supersedes the 17:30 update as the easiest entry point. It separates
new work from the stable picture so that engineering progress is not confused with a
new scientific explanation.

## UPDATE AFTER 18:00 — every program action now owns its runtime trace

All 64 program-bearing actions now derive the actual runtime route, control, program
hash, MLP2 background, and batch schedule from their sealed physical materialization
and complete scored row. The four remaining actions are the already implemented
native/deployed baselines. An exhaustive test found and closed one real missing
interface: the deterministic zero-cross transport program existed but its `zero_A/T`
final runtime identity was not licensed. Final-only controls remain forbidden during
fit and validation.

This closes the highest-priority source-to-execution gap, but final scientific actions
remain **0/68** until consumer norms, paired edits, frequency/agreement aggregation,
complete action aggregation, and independent audit are implemented. The detailed
hourly reasoning is in `HOURLY_STRATEGIC_REVIEW_2026-08-28_1815.md`.

A concurrent fixed-point scalar experiment now closes output magnitude as a sufficient
explanation for the deep partial-composition failure: it recovers only about 12% at
B3/B5. A subsequent direction test also rejects a globally rotated-stream account,
although it finds localized sign reversals at attention layer 9. This moves paired
finite response tests ahead of norm summaries for explanatory value; the 18 registered
norms remain required integrity diagnostics.

---

<!-- UPDATE START -->

## UPDATE SINCE THE PREVIOUS PLAIN-LANGUAGE EXPLANATION

### Short version

We still have **no newly admitted whole-model replacement** and we have not opened the
final 68-action result. The scientific score therefore has not moved. What did move is
the reliability and completeness of the experiment that will decide whether the
coupled MLP0/MLP1 programs are genuinely useful.

The other genuinely new result concerns cheap scalar corrections. A scalar can repair
an isolated compiled/live interface almost perfectly, but the same correction rule can
harm a different partial composition. Scalars remain cheap candidate components; they
are not automatically function-preserving or universally reusable.

### 1. The 68 actions now have correct physical call accounting

The experiment contains 34 choices for the coupled MLP0/MLP1 program, each evaluated
with two choices for MLP2:

- `N`: the deployed simplified MLP2;
- `E`: the exact original MLP2.

This gives \(34\times 2=68\) actions. These are controlled model configurations, not
68 discovered circuits.

Each action now has a mechanically derived ledger of which computations it is allowed
to call. For example, `RR/E` uses replacement programs for MLP0 and MLP1 but must call
the exact MLP2 once per batch. `O/O/E` must call all three original MLPs. A missing,
extra, or hidden original call now causes failure. The ledger covers 48 common
observational batches per action, or \(68\times48=3264\) scored student forwards.

This corrected an important design error: the old global rule said that the entire
experiment should make zero original MLP calls, which is logically impossible for the
`E` and `O/O` controls.

### 2. A target-only row substitution is now rejected

Each scored example has 512 input tokens and one next-token target. The previous
identity check fixed the inputs but did not independently bind the target. It now
hashes all 513 tokens in every role row. Changing only the answer token therefore
fails before a model forward, rather than silently changing CE.

Here **CE**, or cross-entropy, is

\[
  \operatorname{CE}(p,y)=-\log p(y),
\]

where \(p(y)\) is the probability assigned to the true next token. Lower is better.

### 3. All four program-free baselines can now run through the observed adapter

The four combinations are:

| MLP0/MLP1 | MLP2 | Name |
|---|---|---|
| deployed | deployed | `N/N/N` |
| deployed | exact | `N/N/E` |
| exact | deployed | `O/O/N` |
| exact | exact | `O/O/E` |

For an `N` MLP2 action we also compare its output distribution with the exact
`O/O/N` teacher using KL divergence,

\[
  D_{\mathrm{KL}}(p\Vert q)
  =\sum_t p(t)\log\frac{p(t)}{q(t)}.
\]

KL is zero only when the two predicted distributions are equal. It measures more than
whether their top token matches. `E` actions are scored with CE but do not invent a
teacher comparison.

The current suffix/observed CPU suite passes **258/258 tests**. No final evaluation
row, model outcome, or winner was opened while adding these checks.

### 4. The scalar-repair result became more precise—and less universal

In one isolated partial model, attention layer 5 produced roughly the same directions
as the native model but with badly wrong output scale. One scalar for the whole layer
removed about 98--100% of that *local compiled-baseline accuracy cliff*. The same was
true at layer 6. This is useful because one multiplication is almost costless compared
with storing a large matrix or token table.

It did **not** recover the live model. It returned the damaged configuration to the
roughly 13% top-1 compiled baseline, while the live model is around 39--42% on those
slices.

The scalar was then tested inside bottom-up partial compositions. A scalar measured in
one global configuration did not transfer uniformly:

- in some compositions it recovered 10--63% of the local deficit;
- in another it made the result worse;
- the corrected depth curve was still non-monotone.

So there are now two different claims:

1. **Exact folding or a gauge change** is truly free: it changes coordinates or moves
   a factor into an adjacent weight without changing the model function.
2. **A fitted scalar calibration** is cheap but function-changing. It must be selected
   without final/OOD data, counted in the program, and tested as part of the complete
   composition.

A corrected experiment has now finished. It measured each gain inside the exact
partial composition where that gain would be used, rather than borrowing gains from
the fully compiled stream. That correction helps the shallow `B0` composition: its
gap recovery rises from 37--40% without correction to 53--57%. It still fails at the
harder boundaries. At `B3`, recovery is about -23% with depth-matched gains versus
+10--12% with the earlier global gains; at `B5` it is about -14% versus +11--12%.
The curve remains non-monotone. All three registered predictions therefore fail.

The clearest reading is that matching individual output norms is neither necessary
nor sufficient for compositional compatibility. The useful scalar depends on the
joint downstream state, not merely on the local layer's native-versus-replacement norm.
This makes a small fitted affine correction a legitimate candidate on a simplicity
frontier, but not an explanation or certificate by itself. This was a diagnostic job,
not the final 68-action experiment.

### 5. What remains before the final early-MLP result

The exact current blocker is internal runner completeness, not missing FineWeb data,
`rspd`, a checkpoint, a cache, or a GPU.

The next implementation step is to bind every materialized program—joint, local,
hybrid, removed, shuffled, null, and mean—to the same full-row action identity already
used by the baselines. After that remain:

1. nine predeclared target-frequency bins;
2. output-norm measurements at all 18 later consumers;
3. paired edited and unedited forwards with their own call ledger;
4. common-row aggregation of all 68 actions; and
5. an independent audit before the final rows are opened once.

This is why several hours of work can leave the scientific score unchanged: the work
is closing ways that an apparently good result could be caused by a swapped program,
a changed target, an uncounted native call, a different row set, or a post-hoc control.

<!-- UPDATE END -->

---

## CURRENT BEST UNDERSTANDING

### What MLP0 appears to compute

The best current description is a **continuous, shared lexical code**, not a hard
partition of tokens into one class each.

A useful schematic is

\[
  z_0(x,c) \approx \mu + \ell(x) + \delta(x,c),
\]

where:

- \(x\) is the current token;
- \(c\) is its context;
- \(\mu\) is an average output;
- \(\ell(x)\) is a token-dependent lexical contribution shared across contexts; and
- \(\delta(x,c)\) is a continuous context-dependent refinement.

The useful output lives close to a roughly 64-dimensional subspace. This means we can
store and execute a lower-rank code. It does **not** yet mean that we know 64 clean
human concepts. A token such as “Paris” can combine several properties—place name,
capitalized word, entity, and so on—and context can move the representation. Hard
clusters are therefore too restrictive unless downstream computation genuinely treats
all members as interchangeable.

The principled notion of a “cluster” is downstream equivalence: two states belong to
the same class only if every downstream experiment we care about gives the same result.
In practice that equivalence can be approximate and task-relative. If the downstream
model distinguishes every token, hard clustering gives no compression; a sparse shared
dictionary or low-dimensional continuous code may still be simple.

### What MLP1 and MLP2 add

MLP1 reads a state already changed by MLP0, so an MLP1 approximation fitted on native
inputs need not work on compressed MLP0 inputs. MLP2 can compensate for errors created
by the pair. This creates large interactions: separately good replacements can fail
together, and exact MLP2 may repair a failure that deployed MLP2 cannot.

That is the motivation for treating MLP0, MLP1, and the MLP2 background as a coupled
program rather than declaring success from local reconstruction error.

### What “rank 1 is decent” currently means

The rank-1 model's next-token accuracies are approximately 9.90%, 10.65%, and 10.07%
on the three roles. Rank 64 obtains 12.88%, 13.49%, and 12.89%; the live model obtains
39.32%, 42.35%, and 38.88%. Thus rank 1 retains about 77--79% of **rank 64's task
accuracy**, not 77--79% agreement with the live model.

Rank 1 can still be an interesting extracted rule if the objective is “build a tiny
predictor that performs some behavior.” It is not yet a faithful probabilistic copy,
a validated removal mechanism, or evidence of the same OOD generalization.

## HOW SIMPLICITY IS BEING USED

There is no single adequate scalar definition. We use a Pareto frontier: a program is
interesting when no alternative is both cheaper and better on all required outcomes.

### Cost side

We record separately:

- stored numbers or bits;
- effective matrix/tensor rank;
- number of executable operations;
- number and dimension of learned tables or dictionary atoms;
- number of exceptional cases;
- small calibration terms such as a scalar, bias, diagonal, or low-rank correction;
- whether a change is an exact fold/gauge equivalence with zero incremental runtime.

### Consequence side

We require the cost reduction to buy something measurable:

- next-token CE and top-1 accuracy;
- agreement and KL relative to bilin18;
- transfer to held-out roles and frequency/OOD slices;
- prediction of finite interventions;
- selective removal with low collateral damage;
- stable composition with later layers; and
- actual storage and runtime reduction.

This distinguishes four valid but different goals:

1. **Rule extraction:** a small program performs behavior \(X\).
2. **Functional faithfulness:** a small program matches the model's distribution.
3. **Causal editability/removal:** program edits predictably change \(X\) without
   damaging unrelated behavior.
4. **Mechanistic faithfulness:** the program preserves the same internal interfaces or
   causal organization.

A rank-1 rule may be good for the first and bad for the other three. A cheap scalar may
help functional faithfulness while changing the internal implementation. We should
report those wins under the correct heading instead of forcing one notion of
interpretability to stand for all of them.

## WHAT THE THREE-HOURLY MATHEMATICAL REVIEWS CONTRIBUTED

They have changed the design, although they have not yet produced a newly admitted
whole-model score.

The main useful move is to define simplicity through **predictive equivalence**, close
to minimal realization or bisimulation: retain exactly the state distinctions needed
to predict declared downstream observations and interventions. Operationally, two
early states may be merged only when their downstream logits, consumer responses, and
finite edit responses are sufficiently close on held-out data.

The second useful move is a **typed polynomial/observable closure**. Because the model
has linear and bilinear pieces, we can search for a small set of observables—constants,
low-rank coordinates, selected products, inverse-RMS scale, and edit directions—that
remains approximately closed when propagated from MLP0 through MLP1, MLP2, and the
suffix. This is more compositional than minimizing MLP0 output MSE alone.

The third is **gauge-aware pricing**: first remove exact coordinate freedom and foldable
affine transformations, then price the remaining program. This prevents a rotated or
rescaled copy of the same computation from appearing artificially complex.

These ideas directly motivated the downstream KL, consumer-norm, edit, composition,
and cost gates in the 68-action experiment. Their status is “promising operational
definitions being implemented,” not “proved semantic decomposition.”

## HONEST SCORECARD

| Quantity | Current value |
|---|---:|
| Module-level structural surrogates | 36/36 |
| Certified whole-program stored values removed for the registered consequence | 5.3481% |
| Older behavior carrying human-readable labels | 32.1% \(\pm 6.4\%\) |
| Strict named causal CE headroom recovered | 10.923% |
| Newly admitted recovery of the current replacement's \(+0.8976\)-nat CE gap | 0% |
| Final early-MLP actions defined and physically call-accounted | 68/68 |
| Final early-MLP scientific actions evaluated | 0/68 |
| Program-free baseline execution paths implemented | 4/4 |

The most important line is 0/68. We have candidates and a much stronger judge, but no
final winner yet.

## IMMEDIATE PLAN

1. Finish the source-closed bridge from each named action to the program actually run.
2. Implement frequency, all-consumer norm, and paired-edit reductions.
3. Audit the complete bundle independently.
4. Run all 68 configurations once on identical final rows.
5. Plot separate cost-versus-performance frontiers for extraction, functional
   faithfulness, editability/removal, and mechanistic faithfulness.
6. If a program passes, install it into the current whole-model compiler and measure
   how much of the real \(+0.8976\)-nat gap it removes.

The main risk is no longer that MLP0 has no low-dimensional structure. It is that a
locally simple code is not closed under the computations performed by MLP1, MLP2, and
later attention. The coupled 68-action experiment is designed specifically to decide
that.
