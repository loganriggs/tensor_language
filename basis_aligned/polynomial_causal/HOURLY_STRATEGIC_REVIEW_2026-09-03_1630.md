# Hourly strategic review — 2026-09-03 16:30 UTC

## Circuit interpretation targets

1. **Specify the computation:** say what information is read, what operation combines it, what is written, and what
   later computation uses the result.
2. **Group and split at the right boundaries:** combine pieces from different heads or MLPs when they implement the
   same variable, and split one native module when its parts do different jobs.
3. **Predict held-out and shifted cases:** predict activations and behavioral effects on examples and distributions
   not used to find the circuit.
4. **Extract a sufficient computation:** an isolated executable circuit, or a circuit plus an explicit background,
   reproduces the intended computation or signed causal effect.
5. **Manipulate selectively:** removal, interchange, or editing changes the intended behavior without changing
   unrelated behaviors; redundancy and interactions must be measured.
6. **Compose and reuse:** shared computations work across tasks and combine predictably with task-specific pieces.
7. **Identify stable units:** the claimed units survive dataset splits, fitting restarts, and harmless changes of
   basis, or are defined directly by downstream causal equivalence.

The full goal is a smaller executable description of bilin18 that predicts fresh and OOD text, composes when several
replacements are installed, supports selective causal edits, and is simpler under literal storage, computation,
edge, state, and program costs. Compression or lower rank alone does not meet this goal.

## What changed since 15:30

- R546 and the independent R548 audit established a fresh three-value pending-opener site. The native model was
  correct in all 24 FIT/SELECT cells. Complete L13H8 swaps moved every answer-changing row in the donor direction,
  and all answer-preserving edits had large enough complete-head effects to be useful controls. This is a live causal
  site, not yet an identified pending-opener subspace.
- R549 was frozen and queued to measure the L13H8 swap's exact effects on 41 later head and MLP writes. FIT alone
  selects a response; SELECT tests it. R550 independently audits the result.
- R551 was frozen before R549 runs. It rejects a later response as an independent second target when most of that
  response lies in the two-dimensional span that directly reads out the three closer tokens. This closes a shortcut
  left open by R549's original selection rule.
- R552 built an outcome-blind induction selector-by-payload dataset: 180 groups, 720 factorial conditions, 1,800
  paired edits, and 1,440 unique prompts. R553 independently verified the token-level selector, payload, necessity,
  and invariance semantics, split disjointness, and hashes. No model output has been opened.
- The generated experiment ledger now separates a scientific protocol hash from an exact execution hash. It had no
  unexplained repeated protocols or repeated executions at its last rebuild.

## Confound audit

- **Answer-readout shortcut:** R540 learned a closer-logit steering direction. R551 now prevents the same shortcut
  from licensing a supposed independent downstream target.
- **Shared token difficulty and leakage:** the induction variable-token banks are disjoint across splits, and exact
  sequences never cross groups. Reuse inside one factorial group is declared rather than counted as independent.
- **Interactions:** a one-factor patch can include its interaction with other components. The 2×2 induction design
  exposes the selector effect, payload effect, and their finite interaction separately.
- **Post-selection:** R549 candidate selection is FIT-only. SELECT cannot change the winner; FINAL_TEST and OOD stay
  closed.
- **Dead controls:** pending-opener answer-preserving complete-head swaps and induction relation-preserving edits are
  verified to make real input changes. The pending-opener controls also produce measurable downstream effects.
- **Loss nonlinearity and baseline subtraction:** current decisions use signed logits and internal response vectors,
  not sums of separate CE changes. Any later CE test must compare the joint intervention directly with the native
  model rather than add marginal losses.
- **Precision and inactive settings:** exact forward counts, artifact hashes, split openings, and configuration values
  remain audit requirements. No rank or reconstruction statistic can promote either circuit.

## Is this still the best route?

Yes. The main uncertainty is now causal identity, not whether a low-dimensional approximation exists. Pending opener
tests whether one late representation can satisfy multiple semantic edits and independent consequences. Induction
tests a different and more explicitly compositional object: selection of a source and transport of its payload.
Running both prevents one difficult circuit or one long queue item from defining the entire research direction.

The most informative next moves are:

1. **Audit R549 and apply R551 immediately.** This changes targets 1, 4, 5, and 7 by deciding whether there is a
   reproducible later consequence beyond the final closer readout. Kill the multi-output route if no candidate passes
   SELECT or if the winner is mostly a direct closer readout.
2. **Run induction native capability on the frozen FIT/SELECT factorial rows.** This changes targets 1, 3, 5, and 6:
   the model must solve all four selector/payload combinations, tolerate the joint answer-preserving and nuisance
   edits, and depend on the selected match. Kill the synthetic route if those behavioral requirements fail before
   searching for sites.
3. **Measure separate selector and payload complete-state ceilings only after capability.** This changes targets 2,
   4, 5, and 6 by testing whether the two factors can be manipulated independently and whether their joint effect is
   predicted. Kill a proposed site when the full state cannot produce the required factor-specific effect.
4. **Use causal-response fingerprints across modules.** Group pieces only when they substitute for one another across
   the registered edits and downstream readers. Kill a grouping when it fails held-out interchange or has different
   joint effects. This is a genuinely different basis from native heads or weight similarity.
5. **Derive a lower bound when interchange remains non-identifying.** Treat two activation descriptions as equivalent
   when every registered downstream computation responds identically, then ask how many distinguishable causal
   states remain. This can show when the evidence cannot identify a finer basis, rather than forcing another fit.

Rank-reducing the heads or MLPs is deliberately not selected: it would measure storage or reconstruction but would
not tell us which pieces implement selection, payload transport, pending-opener state, reuse, or selective control.
