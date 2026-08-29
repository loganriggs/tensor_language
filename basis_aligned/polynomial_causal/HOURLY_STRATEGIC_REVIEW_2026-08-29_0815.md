# Hourly strategic review — 2026-08-29 08:15 UTC

## The update

This review produced one completed whole-program experiment and resolved one apparent
contradiction in the compiled-program story.

The contradiction was S1898: restoring a late native MLP changes about 3.5% of top-1
predictions, even though the context-free derivation said the native and compiled MLP
receive the same length-one stream. S1899 measured that stream directly at all 18 MLP
entries. The relative difference is only $1.15\times10^{-7}$ at MLP0 and remains below
$4.14\times10^{-7}$ everywhere. Thus the context-free premise survives. The changed
top-1 tokens are downstream effects of extremely small numerical differences, with
small logit margins the leading hypothesis now being checked independently. This does
not make the late MLPs behaviorally irrelevant: a tiny perturbation can still flip a
near tie. It does mean that the previously proposed hidden-state/dataflow failure is
not real.

The completed experiment redistributed the **same storage budget** across all 36
compiled sites. A site rank $r$ is the number of singular directions retained from
that site's 5,419-token by 1,152-output lookup table. Its learned fallback map has rank
$\min(r,512)$. The literal site price was

$$
C(r)=r(5419+1152)+2(1152)+2(1152)\min(r,512),
$$

and the sum across sites could not exceed the uniform rank-512 price of 163.666944
million stored real values. Therefore any improvement is a real simplicity/prediction
frontier improvement, not merely a larger model.

The preregistered allocator maximized the fraction of each site's table energy retained.
It failed decisively: it was worse than uniform rank 512 by
`0.01944 / 0.02053 / 0.02309` nat on the three roles, and worse than the rank-shifted
null by `0.01356 / 0.01436 / 0.01553` nat. All controls passed. Local *relative*
compressibility is therefore not a valid whole-program importance measure.

A non-promotive diagnostic allocation using **raw**, rather than per-site-normalized,
spectral energy improved over uniform on all roles by
`0.00772 / 0.00675 / 0.00447` nat. It missed the frozen 0.005 margin on the third role
and was not the primary hypothesis, so it is only a lead. Its allocation spends heavily
on most MLPs and early attention 2--4, while heavily compressing several attention
sites. The result suggests that absolute write scale contains useful causal information,
but it does not yet show that this allocation generalizes to fresh documents.

## What fraction is actually explained?

There is no honest single percentage, because the project has four different ledgers:

| Meaning of “explained” | Current strict result | What is missing |
|---|---:|---|
| Every residual-writing component can be intercepted/replaced | 36/36 sites | autonomous semantic interfaces between replacements |
| Original parameter storage has a downstream removal certificate | 5.3481% | 94.6519% has no certified safe deletion |
| Native-model CE damage has a named causal attribution | 10.923% | 4.72714 nat of measured damage remains unnamed |
| A behavior/path has extraction, removal, and OOD evidence | 0/68 cells | every terminal action cell |

The 68 cells are not 68 hidden mechanisms. They are entries in an evidence matrix:
candidate behaviors or paths crossed with actions such as prediction, extraction,
selective removal, and OOD transport. A cell is complete only when the action was
actually run with controls. This is why a compiler can imitate a substantial amount of
behavior while the strict terminal-cell count remains zero.

The strongest current positive is functional: at fixed vocabulary coverage the
context-free program reproduces the model's particular wrong answer about 6--7 times
more often than a marginal-matched null. The largest gap is composability: local table,
map, tensor-rank, or MSE improvements often do not improve end-to-end CE after residual
addition and RMSNorm.

## Largest gaps and confusing results

1. **No certified terminal circuit.** We still lack one behavior for which a smaller
   executable path predicts held-out examples, survives OOD templates, and can be
   removed with limited collateral damage.
2. **The best MLP3 decoder is not the best local reconstruction.** Family F with native
   Down obtains KL `0.05772`, better than refitted Down's `0.08476`, although native
   Down has worse local NRMSE (`0.86957` versus `0.70275`). This is the clearest evidence
   that local reconstruction is the wrong optimization target.
3. **The standalone state interface is not solved.** A rank-512 map works well when fed
   the native length-one stream, but the fully recursive compressed stream loses
   `1.09--1.27` nat and direct closed-stream refitting loses about `5.5` nat. S1899 does
   not contradict this: it concerns the settled context-free table program's exact
   length-one premise, whereas the failed map experiment asks a learned fallback to
   carry uncovered-token state autonomously.
4. **Shared dictionaries are budget-dependent.** Global sharing helps against an
   equal-storage independent baseline at tight rank-64/128 prices, but a shared trunk
   loses at the rank-512-scale budgets. A universal output language is not established.
5. **A simplicity proxy can optimize the wrong thing.** Normalized spectral energy now
   has a direct falsification at equal executable price. Raw energy has a small lead,
   but needs a prospective causal/OOD confirmation.

## Candidate actions considered and pruned

- Another normalized-energy allocator is pruned: it failed uniform and shifted-null
  comparisons on all roles.
- Direct-sum/HOSVD canonicalization with the existing rank-64 early-MLP projectors is
  pruned: those projectors mix more quadratic energy than Haar controls.
- Large-budget global or shared-plus-private output dictionaries are pruned by the
  rank-512 results. Only the tight-budget regime is still open.
- More local MSE/NRMSE fitting without a downstream loss is pruned by Family F's native-
  Down/refitted-Down reversal.
- Generic activation SAEs or sparse rotations without a causal projector remain low
  priority: sparsity alone does not solve composition, intervention, or gauge freedom.
- Repeating the older native-module budget allocator is redundant. It selected which
  modules remain native, whereas today's completed test allocated ranks while all 36
  sites remained compiled.

## Top five after pruning

1. **Native-Down, causally balanced Family-F successor.** Highest expected information
   because the measured KL reversal directly isolates the local-fit/causal-fit mismatch.
   It can test a smaller MLP3 program on finite edits and downstream CE. It is currently
   blocked by its own required independent row audit and the missing CUDA measurement
   adapter/semantic validator, not by GPU availability.
2. **One terminal copy/continuation extraction-removal-OOD cell.** This is the shortest
   path to validating whether any simplicity definition is practically useful. The
   contract exists, but fresh four-role rows, reviewed head/product adapters, bootstrap
   authority, checkpoint binding, and a create-only result lifecycle are still required.
3. **Finite L8 $\rightarrow$ L11 $\rightarrow$ L14 transport triangle.** The 384 unique
   documents now exist. Predicting a sealed finite response through
   $T_{8\to11}T_{11\to14}$ is a direct test of composable state rather than reconstruction.
   Source closure, create-only lifecycle, and full null/control families remain to be
   completed before launch.
4. **Prospective causal-weighted uneven ranks.** Combine fit-only spectra with separately
   frozen suffix sensitivity, then compare against uniform, raw-energy, and shifted
   allocations at identical storage. Today's raw-energy diagnostic justifies this; the
   failed normalized allocator prevents using compressibility alone.
5. **Tight-budget rank-64/128 shared-plus-private factorization.** This is the only
   untested regime where sharing has already beaten an equal-storage independent
   comparator. Its expected return is below the causal and terminal tests because it
   still risks producing rotationally arbitrary compression rather than editable states.

## Action executed in this review

The highest-priority source-closed and unblocked action at the start of the review was
the exact-budget uneven-rank experiment; the higher native-Down action was authority-
and implementation-blocked. The run completed in 244.1 seconds. The primary hypothesis
failed, the controls passed, and the raw-energy diagnostic supplied the narrow successor
described above. This is a real whole-program outcome, but it does **not** complete an
E1--E4 evidence cell and does not move any strict ledger.

Static evidence:

- `uneven_table_rank_allocation_results.json`
- `UNEVEN_TABLE_RANK_ALLOCATION_PREREG_2026-08-29.md`
- `basis_aligned/bilinear_quotient/ops/context_free_premise_results.json`
