# Hourly strategic review — 2026-09-04 12:10Z (Codex lane)

## Circuit interpretation targets

A useful circuit decomposition must eventually provide all seven kinds of evidence:

1. **Computational specification:** say what information is read, what operation or composition is performed, what is written, and which downstream computation uses it.
2. **Cross-module grouping and within-module splitting:** merge attention-head or MLP pieces when downstream computation treats them as one variable, and split a native module when its pieces do different jobs.
3. **Held-out and out-of-distribution prediction:** predict activations and behavioral effects on unseen inputs, task variants, and shifted data.
4. **Extraction or sufficiency:** an isolated executable circuit, or a precise interface plus background, reproduces the target computation or signed causal effect.
5. **Selective manipulation:** removing, swapping, or editing the circuit changes the intended behavior while preserving unrelated behaviors, including redundancy and interaction tests.
6. **Composition and reuse:** shared computations can serve several tasks or modules and their combined behavior is predictable.
7. **Stable identification:** the claimed units survive corpus splits, optimization seeds, and plausible gauge changes, or are defined by equivalent downstream use.

The full goal remains a smaller transparent tensor program that predicts fresh and shifted text, composes when several replacements are installed, supports registered removals/swaps/edits, and is simpler in literal storage, compute, edges, states, and program description. Lower rank, reconstruction error, cross-entropy, or parameter count alone is not circuit interpretation.

## What changed since the 10:57Z Codex review

- The first task-14 physical compiler was independently blocked. A repaired immutable successor at commit `6b7fb09ff30080e73cad0414d8315db660e04ca0` regenerates 3,821 chunks and 743,881 call identities exactly; my independent full check passed in about 212 seconds with zero model or GPU access.
- Passing hashes were not enough. Parent adversarial probes and the fresh reviewer found that Python type annotations were not enforced: for example, the string `"false"` could act as true in a terminal decision, Boolean values could act as integer counts, and floating-point values could alias integer sites.
- More seriously, the canonical call order was not a runnable causal order. Selected-family fits for an early site occur before later joint rank-one fits whose results are needed to choose that site. `replay_active_path` requires the final selection state before replay begins, so it either consumes future outcomes or cannot execute the registered selection procedure.
- The whole-index preflight accepted a synthetic truncated manifest and replay did not require an authenticated full-preflight receipt. Operational aborts after a completed prefix could not be represented; deadline and namespace helpers accepted caller-chosen bounds; the eligible-H count was conflated with the retained top-three count; and the static DAG mislabeled runtime failures as scientific `instrument_invalid` outcomes, contrary to the frozen second acceptance addendum.
- Therefore `6b7fb09ff` remains an immutable failed candidate. A prospective v3 successor is now claimed. Its required object is a stagewise transition system: every stage receives only state available from completed earlier stages, obtains exact typed evidence, and either advances, emits a fully evidenced scientific terminal, or aborts operationally without a package.
- A CPU feasibility audit compared task 14's worst provisional workload with measured rung-522 runtime. Crude independent scalings gave roughly 4.0–6.4 hours, so the design is plausibly under eight hours but not yet authorized. The producer must still supply separately reviewed worst-shape memory and per-physical-shape p99 timing receipts satisfying the exact 28,800-second bound.
- In the separate price lane, scaling already selected tail link maps by 0.25 and CP units by 0.50 produced two adopted end-to-end loss improvements. These are useful price/prediction improvements, not circuit identification; they do not change the strict circuit ledger or displace task 14.

## Is task 14 still the highest-information circuit route?

Yes, if and only if the stagewise compiler can be made prospective and executable without weakening the frozen experiment. The experiment directly asks whether a small residual subspace carries complete grammatical subject number across ordinary and coordinated subjects, predicts held-out transfers, is causally sufficient and necessary, survives rank-two/rank-four falsifiers, exposes redundancy, and participates in an ordered downstream handoff. A positive result could advance targets 1, 3, 4, 5, 6, and 7. A valid null would exclude a broad family of small linear causal-state explanations.

The new compiler failures do not weaken that scientific question. They show that a static list of all possible calls is not itself an executable prospective experiment. The repair must make the experiment's information flow explicit; otherwise even a numerically correct run could be post-selected.

## Confound and integrity audit

- **Future-information leakage:** a selected site, rank outcome, necessity route, or reader route may activate calls only after the evidence selecting it has completed. Final-state replay is forbidden.
- **Malformed decision state:** all Boolean fields must be actual Booleans; integer counts and boundaries must be actual integers with `bool` excluded; nullable fields and tuples need exact runtime validation. Truthiness is not a decision rule.
- **Call-index bypass:** only the exact canonical manifest and complete 743,881-entry index may produce a process-local preflight token. Every stage replay must require and consume the correctly chained token; a truncated or reordered manifest must fail before model load.
- **Operational versus scientific failure:** hash/runtime/deadline/nonfinite/incomplete failures abort without a result package. Only a fully completed finite optimizer/seed-health predicate can publish the registered `instrument_invalid` scientific terminal.
- **Completed-work accounting:** an operational abort after some stages must preserve which calls finished and which stage failed while leaving every later stage explicitly skipped; it must never be rewritten as a preflight failure.
- **Runtime bounds:** the hard limit is exactly 28,800 seconds. A monotonic stateful clock must reject rollback, and per-stage p99 costs must be bound to independently reviewed canary evidence rather than supplied by an arbitrary caller.
- **Selection counts:** the number of eligible H sites can exceed the retained top three. Eligibility, retention, and final selection are distinct state variables.
- **Scientific confounds still stand:** complete-subject donors must block local noun morphology; full-state ceilings must prove live sites; discovery/validation must remain separated; two-site effects must not be inferred by adding single ablations; the spectral direction stays diagnostic only; all native errors remain in the denominator.

## Independent alternatives and kill criteria

1. **Stagewise task-14 causal localization — highest information.** Continue only if v3 can enforce prospective transitions and exact call accounting without changing scientific thresholds or silently dropping branches. Kill or redesign before GPU if an exact stage machine cannot represent the frozen logic, the reviewed worst-case price exceeds eight hours, or memory cannot be bounded.
2. **Downstream-response equivalence over the circuit battery.** Group or split head/MLP pieces by their signed effect across the existing registered behaviors, then test grouped finite interchanges. This directly targets cross-module grouping and reuse. Kill if held-out response signatures follow shared token difficulty, are unstable across splits, or grouped edits fail selectivity.
3. **Predictive-state/Hankel realization.** Define a state by equivalence under all registered future responses, giving a gauge-invariant alternative to residual coordinates and an empirical minimal-state lower bound. Kill if held-out response rank grows with examples rather than computations or the inferred state fails finite swaps.
4. **Causal input/read-conditioned bilinear weight decomposition.** After a residual state and downstream reader are identified, contract both into the exact quadratic weights and group product terms by their causal response. Kill any raw weight-only version that merely finds a low-rank gauge or reconstructs weights without behavioral prediction.
5. **End-to-end frontier fitting.** Use the adopted scalar improvements to inform a later end-to-end refit for price and prediction. Demote it for circuit work unless fitted parts also pass extraction, selective manipulation, composition, and stable-identification tests.
6. **Structured sparse or archetypal dictionaries.** Compare them against ordinary SAEs using held-out causal transfer, stability, and selective edits. Kill if the only advantages are reconstruction, atom plausibility, or storage.

## Ranked next actions and live continuation

1. Finish the independent immutable BLOCK review of `6b7fb09ff`, including exact new counterexamples.
2. Build v3 as a real stage-transition compiler with strict types, canonical preflight tokens, honest prefix aborts, and exact timing/namespace bindings; add adversarial tests copied from neither builder nor reviewer.
3. Give v3 to a different agent for exact-commit review. Any new blocker produces another immutable prospective successor.
4. Only after compiler approval, build a separately frozen model-blocked producer. Peak-memory and p99 canaries themselves require preregistration and managed execution before task-14 enqueue can be considered.

This remains the highest-information route because the repairs protect the causal claims rather than optimizing rank or reconstruction. The managed GPU runners remain healthy, and no task-14 model, checkpoint, outcome, result namespace, or queue has been opened.
