# Hourly strategic review — 2026-08-30 02:00 UTC

## Bottom line

The strict native-model ledger did not move this hour:

- certified stored structure: **29,196,288 / 545,904,054 = 5.348245316%**;
- named causal cross-entropy: **0.57968 / 5.30682 = 10.923302467%**;
- still unexplained: **4.72714 nat = 89.076697533%**;
- circuits passing extraction, selective removal, and OOD transport: **0 / 68**.

The useful outcome was a pruning result.  The MLP1 program that looked like a cheap
rank-64 candidate is not a standalone rank-64 replacement.  It is a rank-64 correction
to a much larger frozen program.  Charging the complete dependency changes its literal
storage from the advertised correction price of 153,920 numbers to **60,707,648
numbers**, or **3.812 times the entire native MLP1**.  We therefore must not use it as
the “simple MLP1” axis in the planned composition experiment.

This catches a genuine interface error before a GPU factorial.  The next MLP1 candidate
will instead be a self-contained hard-TopK decomposition of MLP1's `Down` map.

## What was inspected

- Current head was `de676bd3`; recent shared commits concern the compiled table/map
  allocation, not a new native-model explanation.
- The current static explanation and previous 01:45 review were reread.
- No Codex GPU job is running.  The shared runner's last `fresh_window_certification`
  job exited with code 1; that namespace is concurrent Claude work and was not edited or
  claimed here.
- The compiler-v2.1 program artifact, canonical frozen ship, exact native facade,
  C512 code, CONTINUE512 loader, and state-complete runtime were traced at call level.
- The 2026-08-29 eight-hour deadline has expired, so its checklist is historical rather
  than an active mandate.  Its failures and unchecked cells remain preserved.

## The computation performed this hour

At MLP1, the historical base program computes

$$
N_1 = T_1[\mathrm{token}] + \mu_y
      + \left([a_1,m_0]-\mu_x\right)W_1.
$$

Here:

- $a_1$ is the **live layer-1 attention write**;
- $m_0$ is the **effective live MLP0 write**, so it can be native or C512;
- $T_1$ is a token-indexed table with one 1,152-dimensional row for every vocabulary
  item;
- $W_1$ is a linear map from the concatenated 2,304-dimensional vector $[a_1,m_0]$
  to the 1,152-dimensional MLP1 write.

The rank-64 compiler then predicts only the coordinates of $N_1$ in a frozen
64-dimensional output basis $B_1$.  If $P_{B_1}=B_1B_1^T$, its output has the form

$$
Q_1 = (I-P_{B_1})N_1 + B_1\,\widehat c(z_1).
$$

Thus the orthogonal complement $(I-P_{B_1})N_1$ still requires the complete base
program.  The correction cannot execute by itself.

The exact literal storage audit is:

| object | stored real numbers |
|---|---:|
| token table $T_1$, shape $50{,}257\times1{,}152$ | 57,896,064 |
| ridge $W_1,\mu_x,\mu_y$ | 2,657,664 |
| complete base $N_1$ | 60,553,728 |
| rank-64 correction including basis | 153,920 |
| **complete historical candidate** | **60,707,648** |
| complete native MLP1 | 15,926,400 |

The candidate is storage-heavy but compute-light: its registered dense products are
about 2,879,488 multiplies per token, 18.08% of native MLP1's three dense maps.  This is
an interesting time-memory trade, but not the storage simplification that the proposed
factorial claimed.

Receipt:
[`historical_mlp1_candidate_price_audit.json`](historical_mlp1_candidate_price_audit.json).
The exact-byte and A→B→A race tests passed `2/2` in 9.04 seconds; the complete focused
suite passed `8/8` in 9.33 seconds.  An independent re-audit returned GO at committed
source `45b2d975`.  Model weights, evaluation rows, and model outputs were never opened.

## What this means scientifically

The historical candidate **can** technically be run after C512: the exact facade passes
the effective MLP0 write to MLP1, and the frozen $N_1$ formula consumes it.  It can also
be crossed with CONTINUE512.  But that run would answer only:

> Does a ship-trained, table-heavy MLP1 interface transport to native/C512 states?

It would not answer:

> Do three independently simpler native early-MLP programs compose?

Those are different questions.  The first remains a possible diagnostic, but it is no
longer the highest-return experiment for the reverse-engineering objective.

There is a possible rehabilitation: the source that originally built $T_1$ used a
rank-64 token table plus 2,000 exact row exceptions.  Those compact factors were not
serialized in the canonical artifact.  Recovering them and proving exact replay could
make the table much smaller.  Until that is done, charging only the rank-64 correction
is false pricing.

## Largest remaining gaps

1. **No jointly executable MLP0/1/2 simplification.**  C512 and CONTINUE512 exist, but
   the MLP1 axis is still missing.
2. **No predictive residual-state quotient.**  The Rayleigh/Fisher attempt failed badly
   on held-out documents, so local error geometry still does not predict finite
   composition.
3. **Sparse descriptions are not yet fully sparse tensor executors.**  MLP1's old
   hard-TopK Down result recovered about 0.938 CE at 32 active atoms, but its program
   was not serialized, composition-tested, or given semantic atom names.
4. **Minority/OOD contracts remain fragile.**  The compiled fallback map can improve
   pooled CE while hurting uncovered or unseen-target rows.
5. **No terminal circuit success.**  Named effects have not yet jointly passed
   extraction, selective removal, and OOD transport.

## Candidate pruning

- **Reject the historical compiler-v2.1 Q1 as the “simple MLP1” factorial axis.**  Its
  complete literal storage is 3.812 times native MLP1.
- **Do not train MLP2 with the failed scalar Rayleigh objective.**  Held-out document
  prediction and curvature calibration both failed by large margins.
- **Do not repeat raw HOSVD or unweighted rank sweeps.**  Every native polarization
  slice is full-rank with a smooth tail, and those ranks did not predict causal knees.
- **Do not optimize pooled compiled CE alone.**  Covered and uncovered groups have
  replicated sign/magnitude conflicts.
- **Keep, but demote, compact recovery of the historical $N_1$ table.**  It may produce
  a valid diagnostic candidate, but it is trained under the wrong background for the
  main native-composition question.

## Top five next actions

### 1. Train and freeze a genuinely standalone sparse MLP1 `Down`, then run the cube

Fit the already toy-validated hard-TopK weight-action form

$$
\widehat D_1(g)=c+A\,\operatorname{TopK}_{32}(Eg)
$$

on fresh FIT documents, select without FINAL access, and run all eight
C512 × sparse-MLP1 × CONTINUE512 arms.  This is now the highest priority because it
directly fills the missing compositional interface, has a clear storage/runtime price,
and is falsifiable by full CE and Möbius interactions.  The frozen specification is
`MLP1_SPARSE_C512_CONTINUE_FACTORIAL_V1_PREREGISTRATION.md`.

### 2. Compute a sparse-router oracle bound for the full bilinear interaction

The Down-only program still computes all 4,608 native bilinear gates.  Before training a
hierarchical/DAG router, allow an oracle to choose a small number of complete product
atoms per position and measure the best possible CE/composition frontier.  If the oracle
cannot pay for itself, prune the entire routed-tensor family; if it can, compare flat,
tree, and DAG routers by prequential code length and executed products.

### 3. Build a signed downstream consumer bank and common reducing blocks

Replace scalar attention-response norms with signed coordinates from several frozen
consumers.  Common invariant subspaces of their pullback quadratic forms give a
gauge-invariant definition of state blocks.  The known-answer commutant toy already
works; the real test must predict unseen finite interactions and cross-block edit
additivity.

### 4. Add distinct late causal endpoints

Verify capitalization, induction/copy, question, and lexical/BPE endpoints with
specificity controls.  Then define early MLP atoms jointly by sparse writing to and
reading from those endpoints.  This provides the extra equations needed to identify
MLP0/1 structure, rather than merely adding circuit anecdotes.

### 5. Repair the compiled fallback contract on fresh documents

Restore or condition map capacity while reporting covered, uncovered, and unseen-target
CE separately.  This can improve the executable compiled program quickly, but ranks
below native early-layer composition because it explains less of the original network.

## Action executed and immediate continuation

The complete historical-candidate price/interface audit was implemented, run, tested,
and preserved.  It converted the former top action from a misleading compression test
into a lower-priority transport diagnostic.  The next execution path is therefore the
fresh, standalone hard-TopK MLP1 program and its eight-arm composition cube.  No user
decision is required; no scientific blocker or GPU blocker is presently active.
