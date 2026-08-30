# Hourly strategic review — 2026-08-30 06:25 UTC

## Bottom line

The project has substantially better **mechanism candidates** than it did yesterday,
but the strict whole-model balance has not moved:

- certified storage explained: **5.348245316%**;
- named causal CE explained: **10.923302467%**;
- unexplained CE: **4.72714 nat = 89.076697533%**;
- desired actions jointly passing extraction, selective removal, and OOD transport:
  **0/68**.

The strongest current progress is qualitative but real: we now have exact tensor
programs for previous-token and equality-copy services, a nearly pairwise-complete
interaction description of MLP0's three token/context branches, and dozens of localized
behavior slices.  The principal remaining failure is **composition**.  A primitive can
be exact and extractable while still serving several behaviors, so deleting it causes
collateral damage.  Conversely, geometric subspaces can look simple without predicting
which behavior another direction affects.

This hour executed a new CPU analysis rather than launching another local fit.  Its
result is sharp:

> Cosine/norm geometry is useful for proposing shared factors, but it does not
> transport as a causal or hierarchical simplicity metric.  Shared/private tensor
> structure must be selected using held-out intervention responses.

## State inspected

- HEAD before this review: `97bd0c98`, with successor terminal-race repairs immediately
  below it at `6c30b798`.
- The RTX 5090 was idle (`1 MiB`, `0%`).  The only live scientific prerequisite was a
  CPU newline row-preparation test; old shell watchers are not active experiments.
- Bracket lifecycle repairs pass their focused/full tests but require a fresh
  independent re-audit before execution.
- Successor v3's two terminal races are repaired but the immutable old audit remains a
  NO-GO; a fresh independent audit is still required.
- Newline has an active prospective amendment and row-preparation work.  Concurrent
  files were not edited here.
- The expired eight-hour plan was audited but is no longer the controlling schedule.
  Its substantive results remain: Family F failed its reconstruction gate; recursively
  closed stream maps failed badly; one global/typed/shared-private 36-site basis failed
  at the rank-512-scale price; the rank-64 finite transport triangle failed destination
  sufficiency; and the original E4.1–E4.3 terminal cells remained incomplete.  Plans,
  scaffolds, and row artifacts were not counted as outcomes.

## New result: geometry does not transport to causal effects

### What was computed

For circuits `i` and `j` localized to the same component, two already measured
quantities exist:

1. **absolute cosine similarity**

   \[
   C_{ij}=|\langle d_i,d_j\rangle|,
   \]

   where each `d_i` is the fitted unit direction for circuit `i`;

2. **cross-circuit causal concentration** `A_ij`, measuring how strongly ablating
   direction `i` affects the positions belonging to circuit `j`, relative to its
   off-slice effect.

If direction geometry were itself a reliable hierarchy or routing code, larger `C_ij`
should predict larger `A_ij`, with the relationship transporting across components.
The analysis computed off-diagonal Spearman rank correlation, within-target rankings,
and a 20,000-draw matrix-label permutation test.  The permutation relabels both axes of
the cosine matrix together, preserving its symmetry and spectrum while breaking its
alignment to named causal effects.

### Results

| component/phase | shared variance | off-diagonal Spearman | permutation p | top causal source chosen by cosine |
|---|---:|---:|---:|---:|
| a8 full | `0.9161` | `+0.6611` | `0.0343` | `3/5` |
| a8 residual | — | `+0.0362` | `0.8274` | `1/5` |
| a16 full | `0.4887` | `+0.4198` | `0.00035` | `1/13` |
| a16 residual | — | `+0.1340` | `0.1866` | `0/13` |
| m16 full | `0.9567` | **`−0.5411`** | `0.0348` | `1/6` |
| m16 residual | — | `−0.0976` | `0.6665` | `2/6` |

M16 is the decisive counterexample.  It has the strongest common geometry of all three
components—`95.67%` shared variance and mean full-direction cosine `0.9473`—yet more
similar directions predict **less**, not more, cross-circuit causal concentration.
Removing the common direction leaves geometry nearly unrelated to causal effects in
all three components (maximum absolute off-diagonal correlation `0.1340`).

This does not say tensor factorization is useless.  It says that HOSVD, cosine, norm
minimization, or a pretty DAG cannot validate the structure alone.  The causal response
tensor must be part of model selection.  In particular, a block can count as a shared
parent only if it predicts held-out intervention cells and improves selective edits per
stored/executed price.

The implementation and three synthetic contract tests are:

- `component_geometry_causal_transport.py`;
- `test_component_geometry_causal_transport.py`;
- `component_geometry_causal_transport_receipt.json`.

The receipt is retrospective discovery evidence using already-opened summaries.  It
opens no model, rows, or protected outcomes and is not a confirmatory p-value.

## Largest remaining gaps

1. **No complete shared-service/use-branch interface.** Equality copying is a broad,
   exact service, but induction-only removal fails because code and other copy uses are
   real consumers.  We have not separated matcher, payload, and behavior-specific use.
2. **No predictive composition grammar.** MLP0 shows large pair interactions; MLP2
   compensates for upstream compression; independent replacements interact.  The new
   Möbius machinery is tested only on a toy and the completed three-branch MLP0 cube,
   not on a library of executable circuits.
3. **Too few terminally executable independent circuits.** Previous-token and equality
   copy can be run now.  Bracket, successor, and newline are close but still behind
   independent lifecycle/audit gates.  Four-circuit interaction tomography is therefore
   not yet well-defined at uniform quality.
4. **Sparse MLP1 remains conditionally priced.** Its dictionary has useful sparse codes,
   but the present router still computes native gates.  Representation sparsity has not
   become executable simplicity.
5. **No certified whole-model error bound.** Local exactness and selected CE assays do
   not provide an RMSNorm/residual-aware bound on accumulated logit or CE error.

## Candidate actions and pruning

The candidates were judged by expected information gain, causal relevance,
whole-model composability, falsifiability, GPU cost, and duplication.

### Kept

1. **Fit component-conditioned shared/private blocks to causal response tensors.**
   Geometry alone is now falsified as the selector.  Use intervention mode × source
   circuit × affected circuit tensors at a8/a16/M16; train on held-out cells and require
   prediction of causal cross-effects.  This is low-GPU/CPU-first and directly tests
   whether a shared block is useful.
2. **Finish independently audited successor, bracket, and newline endpoints.** These
   are the shortest path to additional executable, behavior-diverse circuits and hence
   to a real composition study.  Infrastructure-only work earns no scientific credit,
   but the remaining audits are necessary and nearly complete.
3. **Four-circuit sparse Möbius intervention tomography.** Once four comparable programs
   are executable, fit main/pair/selected triple effects on a subset of arms and predict
   withheld combinations.  This directly measures composability and selective removal.
4. **Factor equality copying into matcher, payload, and use branches.** The terminal OOD
   result already proves the matcher is real and broadly useful.  A use-conditioned
   payload decomposition is more likely to fix collateral than narrowing the target
   mask post hoc.
5. **Circuit-specific finite-Hankel rank-growth tests.** Quote parity and successor may
   admit small state programs; bracket depth and equality memory may show systematic
   rank growth.  This cheaply decides which program class to attempt before fitting it.

### Pruned or demoted

- **Cosine/norm/HOSVD as a standalone simplicity definition:** directly fails to
  transport from a8/a16 to M16 causal effects.
- **One universal shared-parent DAG:** a16 already falsifies it, and M16 shows that even
  stronger common geometry need not yield selective leaves.
- **More flat rank sweeps:** rank-64 DAS remains far below causal completeness and is
  redundant without a new response-aware objective.
- **Another local-MSE SAE/dictionary fit:** MLP1 already supplies the positive sparse-code
  result; router price and composition, not convergence, are the bottlenecks.
- **Immediate ten-circuit powerset:** costs 1,024 combinations and mixes circuits at very
  different readiness levels.  The four-circuit predictive gate must pass first.
- **Raw CE-only optimization:** retained as an ultimate prediction metric, but rejected
  as the sole structure selector because it cannot distinguish a reusable shared service
  from collateral behavior coupling.

## Top five priority order

1. **Causal-response-selected block structure** — best immediate information/price and
   directly revised by this hour's falsification.  Existing a8/a16/M16 data support a
   CPU-first held-out analysis.
2. **Close three near-terminal circuit pipelines** — highest near-term contribution to
   usable mechanisms; also unblocks a fair interaction experiment.
3. **Four-circuit Möbius prediction gate** — strongest direct test of whole-program
   composition, but blocked until at least four circuit programs have comparable
   executable interventions.
4. **Equality matcher/payload/use factorization** — directly targets the best terminal
   mechanism's only failed property, selective collateral.
5. **Finite-Hankel program-class triage** — cheap and falsifiable, but downstream of
   having verified response functions for each service.

## Action executed this hour

The highest-priority safe unblocked portion of item 1 was executed:

- wrote and tested the geometry-to-causality transport analyzer;
- ran it on all off-diagonal a8/a16/M16 full and residual response cells;
- preserved 20,000-draw label-permutation statistics and synthetic aligned/reversed
  controls;
- converted the result into a design constraint for the next block-term fit:
  **causal held-out cells, not cosine reconstruction, choose the decomposition**.

No strict ledger credit is added.  The next nonredundant step is a response-tensor
model-selection baseline—flat, independent, and shared/private—using held-out causal
cells and literal storage price.  It should remain CPU-only until one model predicts
the held-out cross-effects better than the others.
