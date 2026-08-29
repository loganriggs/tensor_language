# Hourly strategic review — 2026-08-29 19:30 UTC

## Outcome first

The highest-priority safe diagnostic was preregistered, committed, and run.  It asked
whether the newly discovered common finite-response mode across six MLP2
replacements can be predicted by a very small context interface: the document-level
mean and diagonal second moment of the RMS-normalized state presented to MLP2.

It cannot.  On 191 supported held-out documents, two-fold out-of-fold Pearson
correlation was **0.22587**, versus the frozen 0.50 bar.  The state model reduced MSE
by only **4.6436%** relative to token/count baselines, versus the frozen 20% bar.
Both fold correlations were positive (0.3871 and 0.2058), and the label-permutation
control passed at 0.1714, so this is not complete noise; it is simply much too weak
to be the proposed gate.

The exact run used 48 native full-model forwards and took **18.35 seconds wall
time** (**15.72 seconds inside the runner**).  Result SHA-256 is
`f47eb2529b7f3872dd361f04169d4dd4f0c66b0ba7dd9281fa7e282d09d10afe`.
No raw state, target, logits, or per-document prediction was published.

This changes the interpretation of the preceding 97.6% response-rank result.  The
common mode is a regularity of the *effects* of partial MLP2 writes, but it is not a
simple document-level property of the input state.  Any useful gate must be
tokenwise/nonlinear, or it must be learned jointly with the mixed write it controls.

## What fraction of the whole model is explained?

The strict balance sheet does not move:

| Currency | Explained | Remaining gap |
|---|---:|---:|
| Structural intervention surfaces | 36/36 | Interfaces exist, but most do not have semantic programs |
| Consequence-certified whole-program storage removal | 5.348245316% | 94.651754684% uncertified |
| Named causal CE | 0.57968 / 5.30682 = 10.923302467% | 89.076697533% unnamed |
| Unexplained strict causal CE | — | 4.72714 nat |
| Complete extraction/removal/OOD actions | 0/68 | Entire terminal chain remains open |

The 97.6% low rank of a six-arm response matrix is not counted here.  It compresses
variation in one intervention assay; it does not yet compress MLP2, identify a
semantic circuit, or predict unseen interventions.

## Largest remaining gaps and confusing evidence

1. **MLP0--MLP2 compensation is unmeasured.**  MLP0 C512 is a strong physical
   compression, while MLP2 attenuated earlier MLP0 discrepancies.  We still lack the
   four-arm composition term that says whether their simplifications coexist.
2. **MLP2 atoms are wrong.**  Every native K512 selector fails, and every tested
   partial write is worse than deleting MLP2.  SUFFIX helps copy-positive positions
   slightly while damaging ordinary positions badly.  This is coordinated
   cancellation/conditional routing, not independent useful channels.
3. **The common MLP2 response mode lacks an input coordinate.**  Six very different
   supports share one dominant effect direction, but document-level state moments
   predict little of it.  The large difference in error scale between the two outer
   parity folds is also a warning that a small number of contexts or a distribution
   split may dominate the aggregate.
4. **The semantic consumer bank is narrow.**  Copy/induction is localized, but
   capitalization, numeric progression, syntax, and entity continuation are not yet
   a causally verified late-consumer basis for interpreting upstream writes.
5. **No terminal utility is closed.**  The copy mean-replacement screen found a
   strong four-head effect but failed collateral damage; extraction, selective
   removal, and OOD transport remain uncertified.

There is no checkpoint, FineWeb, `rspd`, cache, or hardware blocker.  The GPU was
idle for this short run.  The special eight-hour window ended at 12:00 UTC and is
treated as historical evidence, not silently reopened.

## Candidate pruning

Pruned now:

- more native MLP2 K sweeps or support selectors;
- document-level mean/second-moment context gates;
- degree-1 through degree-3 small-edit extrapolation to full replacement;
- independent Frobenius HOSVD/CP/Tucker or a weight SAE without final consequences;
- loosening the failed copy collateral gate or reopening its selection role;
- treating response-matrix rank as storage compression.

Retained only inside better objectives:

- gauge/norm minimization as conditioning before a causal-metric factorization;
- MDL as a price for choosing among executable programs, not as the program
  generator;
- polynomial/tensor algebra for the exact bilinear block after the relevant mixed
  coordinates are identified.

## Top five next actions

The ranking uses information gain, causal relevance, whole-model composability,
falsifiability, GPU cost, and duplication.

1. **Run the MLP0-C512 × MLP2-ZERO composition telescope.**  Measure native,
   C512-only, ZERO-MLP2-only, and their combination on the same 192 documents, then
   compute
   $$
   \Delta_{\mathrm{interaction}}
   =\Delta\mathrm{CE}_{C512+ZERO2}
    -\Delta\mathrm{CE}_{C512}
    -\Delta\mathrm{CE}_{ZERO2}.
   $$
   This is cheap, directly tests the suspected compensation interface, and tells us
   whether independently simpler components can compose before fitting a new MLP2.
2. **Fit a tokenwise, arbitrary-mixture balanced MLP2 correction from ZERO.**  Use
   FIT products for reachability, finite downstream logit/consumer responses for
   observability, and refit both the product mixture and output write.  Compare
   ranks 32/64/128 by held-out final CE/KL and executable price.  This is the
   highest-return new atom family after native channels failed.
3. **Build and causally verify multiple late consumers.**  Add capitalization,
   numeric succession, syntactic closure, and entity continuation with sufficiency,
   necessity, and collateral controls.  Their joint response matrix is the missing
   target for Hankel/causal-state and simultaneous-factorization work.
4. **Replace copy mean-ablation with input-conditional interaction tests.**  Test
   the known four-head copy bundle with tokenwise patching/gating and powerset or
   Shapley-style interaction resolution on a new prospective role.  The objective is
   a first terminal extraction/removal/OOD action, not another head label.
5. **Compose independently validated MLP0, MLP1, MLP2, and attention programs.**  Use
   a factorial/telescope design and charge interaction CE explicitly.  This becomes
   worthwhile as soon as MLP2 has a nontrivial mixed candidate; doing it with the
   failed K512 selector would be redundant.

## Action executed

The completed diagnostic is defined by
`MLP2_FINITE_RESPONSE_CONTEXT_PREREGISTRATION.md`, implemented by
`run_mlp2_finite_response_context.py`, and recorded in
`mlp2_finite_response_context_result.json`.  It is a real negative outcome, not an
unrun runner.  Its failure narrows priority 2 to tokenwise/joint gates and promotes
priority 1 as the next cheap whole-model experiment.  No strict ledger quantity is
changed.
