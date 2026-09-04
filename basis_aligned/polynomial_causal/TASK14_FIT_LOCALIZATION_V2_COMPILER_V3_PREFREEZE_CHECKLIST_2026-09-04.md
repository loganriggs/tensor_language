# Task-14 localization-v2 compiler v3: pre-freeze correctness checklist

**Recorded:** 2026-09-04 13:10 UTC. **Status:** blocking checklist for a mutable CPU-only draft, not an
approval. The compiler, producer, model, checkpoint, GPU, task outcomes, and queue remain unauthorized.

Compiler v2 commit `6b7fb09ff30080e73cad0414d8315db660e04ca0` remains blocked by independent review commit
`60892e3994250b7f58330f4b2a84f8ed4126c928`, review SHA-256
`3131fffd0b6c8cd18789b69e4909b0002ca3e90f2c965391c07444f56b63756a`. Compiler v3 is a new
stagewise repair under the prospective claim in
`TASK14_SUBJECT_VERB_AGREEMENT_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_PREREGISTRATION_V3_2026-09-04.md`.
It must not be frozen until every item below is implemented and tested.

## Frozen science that must be restored exactly

1. **Initialization.** Section 8 of the approved v2 localization preregistration defines each entry by

   ```text
   SHA256("task14-localization-v2-init" + "|" + seed + "|" + rank + "|" + site
          + "|" + objective_name + "|" + d + "|" + j)
   ```

   The entry is `+1` when the low bit of the first digest byte is zero and `-1` otherwise, followed by reduced
   QR in increasing column order and a sign choice making the corresponding diagonal of $R$ positive. The
   earlier compiler's counter-block label is a different initializer and is forbidden.

2. **Finite invalid screen denominators.** A fully completed, finite gradient denominator at or below
   $10^{-12}$, or natural-margin denominator at or below $10^{-6}$ nat, reaches the first scientific terminal
   `instrument_invalid`. It must not be confused with an empty eligible-site set or with an operational fault.
   Nonfinite values remain an operational abort with no package under the controlling second producer-acceptance
   addendum.

3. **Eligibility relation.** Every eligible Q site must also satisfy every H eligibility condition. Therefore the
   Q set must be a subset of the full eligible-H set, before H is truncated to its top three sites.

4. **Spectral diagnostic.** Spectral nonfiniteness is an operational abort, not a publishable scientific terminal.
   The diagnostic may be completed before selection, but its numerical values cannot choose H, Q, rank, or family.

## Causal execution protocol

5. The executable chain is

   $$
   C_s\longrightarrow R_s\longrightarrow E_s\longrightarrow C_{s+1},
   $$

   where $C_s$ is the one-use capability for stage $s$, $R_s$ is the exact physical replay receipt, and $E_s$
   is the validated stage evidence. A successor must bind the predecessor completion receipt, including replay
   identity, active-path root, executed-call count, work ledger, and evidence digest. It cannot skip $R_s$.

6. Every public field of a capability, replay receipt, child chunk receipt, completion receipt, timing token, and
   terminal must be recomputed against its sealed internal core. Changing a public stage, root, count, ledger, or
   predecessor must fail.

7. A completion attempt is consumed before validating caller-supplied payload or replay data. A missing, wrong, or
   malformed receipt may be followed only by an operational abort, never by a corrected retry.

8. The terminal identifier must bind the full causal capability/completion chain and therefore all stage evidence
   and replay roots. Two executions with different evidence cannot share a terminal identifier merely because their
   terminal names match.

## Exact dataflow and calls

9. The machine-readable DAG must freeze exact `reads(stage)`, `writes(stage)`, transition reads, guards, and
   `calls(stage)`. The call declaration must bind the stage's chunk identities/root, template census, physical-call
   shapes, and shape multiplicities. Self-consistency is insufficient: tests need an independent expected matrix.

10. At minimum, gradients bind native cache evidence; discovery ceilings bind native and gradient evidence; joint
    fits bind the native cache and discovery-ceiling evidence; the spectral diagnostic binds joint-fit projector
    evidence; selection binds completion of the spectral stage without using its numerical values; selected fits
    bind native and selection/projector evidence; validation ceilings bind native and selected-site evidence; and
    validation, necessity, redundancy, and reader stages bind the fitted projectors and earlier evidence they
    actually consume.

11. Stage guards may use only frozen inputs and predecessor evidence. Selected family/rank fits occur only after all
    eligible-site joint rank-one fits are complete. No early stage can accept a final state assembled from later
    outcomes.

## Work and failure accounting

12. Every inactive chunk receipt records its activation guard, the evaluated false state,
    `template_call_count`, and `executed_call_count=0`. A positive template count must never be presented as incurred
    work. Ordered child receipts must be retained and bound into the stage root.

13. An operational abort records the exact failed stage, active chunk and call offset, completed slices, attempted
    and completed calls, partial completed-call root, and incurred forward/backward/graph/update/example/token ledger.
    It distinguishes refusal before a call from failure after a completed call and never permits a scientific package.

14. Count an attempt before entering the producer callback. Count completion only after a valid call-completion
    return. If the callback raises, preserve the attempted-call position and a conservative incurred-work account;
    do not silently lose the call from the ledger.

## Timing, namespace, and source closure

15. Compiler v3 may schema-check candidate timing receipts but must have no timing-authorization issuer. GPU execution
    remains blocked until a later independently reviewed artifact binds exact producer/model/checkpoint/runtime/device
    receipts, complete physical-shape coverage, peak memory, fixed overheads, and

    $$
    t_{\mathrm{bootstrap}}+C(\mathrm{start})+t_{\mathrm{publication}}\le 28{,}800\ \mathrm{s}.
    $$

    A caller-supplied shape-weight table is a mathematical helper, not authorization.

16. Preflight hashes the source file actually executing and all frozen closure files before model access, captures
    immutable manifest/index/input bytes, and does not reopen them during replay. Namespace preflight accepts no public
    path override and treats files, directories, and dangling links at any reserved destination as occupied.

## Required test evidence before freeze

- Real stage replay for every conditional branch, including active and inactive joint fits, selected fits, validation,
  singleton necessity, two-site redundancy, ordered reader, and future-information rejection.
- Operational aborts at every reachable DAG prefix and at a physical call prefix, including callback failure.
- The complete 743,881-entry global preflight, source mutation after import, captured-input replay, and all coherent
  manifest/index mutation attacks.
- Exhaustive exact-type attacks over every Boolean, nullable field, count, site, score, tuple, SHA-256, capability,
  replay receipt, completion receipt, and terminal field.
- Explicit terminals for both finite invalid denominator cases; Q-not-H rejection; exact initialization known-answer
  vectors; evidence-dependent terminal identifiers; wrong receipt followed by valid receipt rejection.
- An independently specified DAG read/write/guard/transition/call matrix, shape-multiplicity closure, and tests showing
  that execution-equivalent calls share a timing class while every compute-relevant change separates classes.
- No permanently skipped materialized-artifact tests. Full regeneration must be byte-identical under
  `PYTHONHASHSEED=0`, `1`, and `999`; broad predecessor/authority suites, `py_compile`, and `git diff --check` must pass.

Only after the exact source, manifest, binary call index, dry-run report, and tests are immutable and pushed may a
different agent begin independent review. That review licenses at most construction of a model-blocked producer; it
does not itself authorize model or GPU access.
