# MLP2 rank-512 arbitrary-factor refit v1 — preregistration

## Question

Every tested program that retained 512 **native** MLP2 products failed final
consequences, and partial native writes were worse than deleting MLP2.  This assay
tests the smallest nonduplicative change of atoms: can 512 freely refitted bilinear
products recover MLP2 substantially better at the same executable price?

The candidate is

$$
\widehat f(x)=\widehat b+widehat D
  \left((\widehat Lx)\odot(\widehat Rx)\right),
$$

with $\widehat L,\widehat R\in\mathbb R^{512\times1152}$,
$\widehat D\in\mathbb R^{1152\times512}$, and
$\widehat b\in\mathbb R^{1152}$.  It stores 1,770,624 floating values and computes
512 bilinear products per token, exactly the scalar price of the failed native K512
programs but with arbitrary product and output directions.

This is an activation-distribution student generator.  Success is decided only by
held-out final-model CE, teacher KL, centered-logit error, and top-1 agreement—not by
training MSE.

## Fresh roles

Freeze 384 previously unused FineWeb source documents, one 257-token row per
document, after recursive registry exclusion.  The first 192 documents are `TRAIN`;
the second 192 are `EVALUATION`.

- `TRAIN[0:160]` fits parameters.
- `TRAIN[160:192]` selects checkpoints and stopping time using native MLP2 write
  NRMSE only.
- `EVALUATION` remains unopened until both candidate bundles and their hashes have
  been serialized.  It is opened exactly once for final consequences.
- Scored/captured positions are 64 through 255 inclusive.

No prior MLP2 FIT_SELECTOR, VALIDATION, or sealed REPLICATION row may be used.

## Frozen candidates and controls

All candidates start from the already frozen, separate-role LOCAL512 physical
program, including its folded bias.

1. `NATIVE`: exact model.
2. `ZERO`: no MLP2 write.
3. `LOCAL512`: frozen native-factor K512 program, no refit.
4. `DOWN512`: hold Left/Right fixed and refit Down plus bias.
5. `FULL512`: initialize from selected `DOWN512`, then refit Left, Right, Down, and
   bias jointly.
6. `RANDOM512`: initialize from a deterministic random native 512-product support,
   including its correctly folded omitted-product mean, then use the identical full
   refit schedule. This is a nonpromotive optimization/initialization control.

`DOWN512` distinguishes output-basis correction from genuinely new bilinear atoms.
All three 512-product arms have the same stored-scalar and product-count price.

## Frozen optimization

- Capture native pre-MLP2 states and native MLP2 writes once on `TRAIN`.
- Optimize float32 token-mean write MSE with Adam, deterministic seed 2026082921,
  batch size 1,024.
- `DOWN512`: at most 600 steps, learning rate $10^{-3}$.
- `FULL512`: initialize from best `DOWN512`; at most 1,200 steps, learning rate
  $3\times10^{-4}$.
- `RANDOM512`: at most 1,200 steps with the same learning rate, batching, checkpoint
  cadence, stopping rule, and train/dev rows as FULL512.
- Evaluate TRAIN-dev every 25 steps.  Stop after eight evaluations without at least
  0.1% relative improvement.  Retain the minimum-dev checkpoint.
- Publish the full convergence curve.  Training MSE has no decision authority.

After fitting, put every active bilinear factor into the minimum-squared-norm scaling
gauge.  For one factor with row norms $l,r$ and Down-column norm $d$, first equalize
$l,r$, then scale both input rows by $a=(d/\sqrt{lr})^{1/3}$ and the Down column by
$a^{-2}$.  Verify function preservation on a deterministic canary before saving.
This canonicalization changes neither outputs nor executable price. Afterward,
report the maximum/minimum active factor-norm ratio and the ratio of summed singleton
component energy to actual centered write energy on TRAIN-dev. A factor-norm ratio
above $10^4$ or cancellation ratio above 100 is a pathological factorization failure.

## Evaluation metrics and decisions

On all 192 `EVALUATION` documents, report candidate-minus-native CE, native-to-
candidate KL, centered-logit NRMSE, native top-1 agreement, and prefixes 48/96/192.

`FULL512` is an **absolute whole-model pass** only if at 192 documents:

- $|\Delta\mathrm{CE}|\le0.02$ nat;
- teacher KL $\le0.02$ nat;
- centered-logit NRMSE $\le0.10$;
- top-1 agreement $\ge90\%$; and
- all four metrics change by at most 0.01 absolute (one percentage point for top-1)
  between the 96- and 192-document prefixes.

Arbitrary product refitting is **materially better than output-only refitting** only
if FULL512 reduces both KL and absolute dCE by at least 20% relative to DOWN512.
It is **materially better than native selection** only if it reduces both relative
to LOCAL512 by at least 50%.

Using a source-document bootstrap with 10,000 deterministic draws and Bonferroni
lower bounds over the four comparisons (KL and absolute pooled dCE versus LOCAL512
and ZERO), FULL512 must also have a positive simultaneous improvement lower bound
over both controls before any stronger “better” wording is used.

A local-write NRMSE above 0.25 on TRAIN-dev is an optimization failure rather than a
scientific rejection.  Otherwise failure of final gates rejects rank-512 refitting
under this activation-MSE generator; it does not reject consequence-trained CP or
conditional block-term models.

This prospective experiment can establish an executable in-distribution MLP2
candidate.  It cannot move the strict ledger without an independent replication,
composition with upstream compressed programs, exact price replay, and terminal/OOD
tests.
