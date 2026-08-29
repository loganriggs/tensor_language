# Hourly strategic review — 2026-08-29 13:20 UTC

## UPDATE PART — what changed during this review

There are two real numerical updates and one execution failure/recovery update.

1. **E4 attempt 1 did not measure the circuit.** It failed before the first model
   forward while hashing a scalar bfloat16 model buffer. The frozen authority is
   spent and the exact failure is preserved in
   `terminal_copy_selection_v1_failure.json`. There is no ledger, result,
   scientific-negative receipt, or passer receipt. Therefore this is an
   infrastructure failure, not evidence for or against the eight copy-circuit
   candidates.

2. **The failure has been repaired without changing the experiment.** The hash
   now flattens a tensor before viewing its bytes. Attempt 2 has a distinct
   create-only namespace and binds the first authority, first failure, all six
   absent first-attempt outcome/lock paths, and the historical first-attempt git
   source. It explicitly records that rows, masks, the fit bank, and checkpoint
   were already deserialized, but no model forward or outcome occurred. The
   complete selection assurance suite passes: **66/66 tests in 133.66 seconds**.
   A fresh independent byte-level audit is in progress. Until that passes and a
   new authority is frozen, E4 remains **NO-GO**.

3. **The MLP-heavy storage allocation transfers, but its value is small.** At a
   fixed price of 360,723,456 stored real numbers, moving table rank from
   attention sites to MLP sites improves all-position cross-entropy by
   **0.01578 / 0.01518 / 0.01363 nat** on the three document roles. This is an
   exploratory, overwrite-style artifact, not a receipt-backed result. It moves
   no strict explanation ledger. The evidence supports using an MLP-heavy split
   as a build default; it does not certify a universal 12.5–25% attention rule,
   because one registered optimum prediction failed and one best arm is at the
   boundary of the tested range.

4. **The best combined context-free build preserves the same accuracy shape but
   remains far from the model.** On 18,432–36,864 scored tokens per role, the live
   model has **38.88–42.35%** next-token top-1 accuracy and the compiled program
   has only **14.01–14.75%**. For targets seen at least 125 times in fit data, the
   program retains **53.7–54.4%** of live accuracy. For rare-target buckets it
   retains only roughly **2.4–8.3%**. The combined cheaper build is nearly tied
   with the older deployed build, so its compression improvement is real but it
   does not repair the missing contextual/long-tail computation. This result is
   also discovery-only and currently has no receipt.

5. **A same-coverage check turns the combined build from a strict win into a
   trade.** At the deployed 5,419-token coverage, the combined build is 29%
   cheaper and improves overall top-1 by 0.11–0.20 percentage points on all
   three roles. It improves the unseen-target bucket on all three roles, but
   reduces the 125+-frequency bucket by 0.65–1.20 points on all three. Thus the
   compression redistributes accuracy from common to rare targets at deployed
   coverage; it is not uniformly better. This strengthens the need for a
   frequency- and context-conditional residual rather than a single global rank
   allocation rule.

6. **E4 completed as a receipt-backed scientific negative.** The four-head copy
   bundle is causally important: replacing its output by the registered position
   mean raises copy-position CE by **0.44870 nat**, with simultaneous lower bound
   **0.26700**, and its specificity point estimate is **0.46352 nat**, lower bound
   **0.28182**. But the same intervention raises off-target CE by **0.02441 nat**.
   The frozen collateral budget was 0.01 nat, so the collateral margin is
   **-0.01441**, with simultaneous lower bound **-0.19611**. No candidate passes
   all three gates. The negative receipt forbids final/OOD opening. This rejects
   the eight registered position-mean replacements; it does **not** show that the
   four heads are unrelated to copy behavior.

## What the percentages mean

The strict ledgers have not moved:

- **Structural coverage: 36/36 sites.** Every attention and MLP site has an
  executable replacement interface. This says we can intervene everywhere; it
  does not say the replacements explain the behavior.
- **Certified parameter storage removed: 5.348245316%.** This is the fraction of
  the original stored weights removed by replacements that passed the current
  strict certification rules.
- **Named causal cross-entropy explained: 10.923302467%.** Cross-entropy is the
  model's average negative log probability for the correct next token. Of the
  5.30682-nat gap used by the ledger, named interventions account for 0.57968
  nat. **4.72714 nat, or 89.076697533%, remains unnamed.**
- **Terminal causal actions: 0/68.** None of the 68 fine-grained circuit actions
  has yet passed the complete selection, fresh-data, OOD, extraction, and
  selective-removal chain.

The honest whole-model answer is therefore: interfaces are complete, but only a
small minority of behavior has a certified causal explanation.

## Largest remaining gaps

1. **No selectively removable terminal circuit.** The exact eight-candidate copy
   screen is complete. It found a large causal four-head copy bundle, but its
   registered mean replacement exceeds the off-target collateral budget. The
   missing object is a more precise, input-dependent decomposition that preserves
   those heads' non-copy work.
2. **The current compiler deletes most context.** A context-free table maps the
   current token to a stored site output. The rare-target accuracy collapse and
   the 25–28 percentage-point live/program top-1 gap show that token identity
   alone cannot express most of the model's program.
3. **Local reconstruction and downstream behavior disagree.** Family F's
   rank-512 native-Down form gives downstream KL 0.05772, better than the locally
   refitted decoder's 0.08476, even though its local normalized error is worse.
   This says the native downstream geometry matters and local mean-square error
   is not a sufficient simplicity objective.
4. **Component replacements have not been composed and attributed exactly.** We
   need an error telescope: measure the change from replacing MLP0 alone, then
   MLP1, MLP2, attention, and their combinations on the same documents. Without
   this, compensation and interaction can make individually good fits fail as a
   whole program.
5. **No certified long-tail/context residual.** The remaining program likely
   needs a small context-conditioned correction on top of the shared lexical
   map. We do not yet know whether that correction is sparse, low-rank,
   hierarchical, or circuit-specific.

## Candidate actions considered and pruned

The ranking uses expected information gain, causal relevance, ability to compose
into the whole model, falsifiability, GPU cost, and duplication of finished work.

- **More attention/MLP allocation sweeps:** pruned. Two budgets and coverages
  already show a small 0.014–0.019-nat free gain. Further tuning will not explain
  the missing 4.72714 nat.
- **Refitting Family F's decoder for lower local MSE:** pruned. The existing
  behavioral diagnostic shows that this can make downstream KL worse.
- **A rank-64 pointwise state as the whole causal state:** pruned by prior
  negative evidence.
- **An unconditioned SAE, HOSVD, or weight-only factorization:** demoted. These
  remain useful proposal generators, but a lower tensor error or sparser weight
  code is not evidence of explanation until it predicts fresh behavior,
  composes, or enables selective edits.
- **More small 32-document descriptive fits:** pruned unless used for debugging.
  Promising claims now require the frozen larger roles and document-level
  uncertainty.

## Top five actions now

1. **Run the rank-512 native-Down behavioral port on fresh rows.** This directly
   tests the strongest lesson from Family F: preserve the model's downstream
   decoder instead of optimizing local reconstruction. Score CE, native-to-port
   KL, top-1 agreement, ordinary/error secants, and registered edits.

2. **Resolve the four-head copy bundle's interaction and collateral.** On a new
   prospectively frozen role, decompose the selected heads' output into a
   copy-conditioned component and a complementary component, then intervene on
   only the copy component. The target is to retain the measured 0.44870-nat copy
   effect while bringing off-target damage below 0.01 nat. Do not relax E4's gate
   or reuse its selection role for promotion.

3. **Build the exact MLP0/MLP1/MLP2/attention composition telescope.** Use the
   same documents and measure singleton replacements, pairs, and the combined
   program. The interaction remainder tells us whether MLP2 compensates for
   MLP0 simplification and whether independently learned components compose.

4. **Fit a small context-conditioned residual only where the context-free build
   fails.** Start with rare-target/error positions and compare sparse dictionary,
   shared-trunk/private-residual, and low-rank bilinear corrections at matched
   executable cost. Require improvement on held-out documents and OOD roles,
   not just activation MSE.

5. **Turn the resulting frontier into certified simplicity.** For each candidate,
   report stored real numbers, multiply-adds, sparse nonzeros, causal edit count,
   CE/KL/top-1 behavior, OOD transport, and selective-removal collateral. A
   representation is called simpler only when its lower price predicts a useful
   capability beyond reconstruction.

## Action executed in this review

The highest-priority action was executed end to end. The E4 attempt-1 failure was
preserved; the scalar serializer and recovery lifecycle passed two independent
audits and **79/79** tests; a new authority was frozen; and the exact 576-forward,
192-document transaction published receipt-last. The outcome is a scientific
negative with no selected candidate. Final/OOD roles remain unopened and are now
forbidden by the negative receipt.
