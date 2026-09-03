# Parallel circuit bootstrap playbook

This document turns one deeply investigated circuit into better datasets, tools, and prompts for the next parallel
wave. The goal is not to produce many component scores. It is to identify behavior-level computations that:

1. predict held-out and out-of-distribution behavior;
2. can be extracted as an executable computation;
3. can be removed without damaging unrelated computations; and
4. expose reusable subcomputations when multiple behaviors share them.

Low rank, reconstruction error, parameter count, and storage are possible descriptive measurements. None is a success
criterion by itself.

## Work pattern

Use one parent track and initially two subagent tracks.

- The parent investigates one circuit deeply and owns canonical records, GPU scheduling, integration, and promotion.
- Each subagent owns one different behavior and produces CPU-checkable inputs: counterfactual rows, semantic positions,
  a preregistration, implementation, tests, and a dry-run receipt.
- Subagents do not edit canonical circuit records or enqueue GPU jobs unless the parent explicitly delegates that
  authority.
- Reserve a unique rung on `AGENT_BOARD.md` before naming files. A task-local label does not make a duplicate global
  rung safe.

The parent reviews every deliverable before execution. A subagent result is a proposal, not canonical evidence.

## Iterative two-agent waves

Use two agents until the dataset, intervention, and audit interfaces have survived a complete build-and-review cycle.
The first wave may put both agents on the same circuit with deliberately different information:

- **Builder:** sees the frozen hypothesis and dependencies, implements the counterfactual rows, exact intervention,
  result schema, tests, and deterministic dry run.
- **Independent critic:** sees the frozen hypothesis and dependencies but not the builder's live implementation. It
  tries to invalidate the causal meaning, denominator, split boundary, intervention algebra, provenance, and terminal
  decision using its own fixtures and tests.

The parent then reconciles both packages before any model run. A criticism becomes useful shared infrastructure only
when it is converted into at least one of: an invariant enforced by code, a planted failing fixture, a prompt clause
with an explicit failure example, or a change to the frozen experiment. Prose advice alone is not considered a
bootstrap improvement.

After this first interface-hardening wave, give the two agents **different circuits**. Each receives the accepted
helpers, tests, prompt clauses, and failure examples from all earlier waves. The parent keeps one circuit as a deep
reference investigation and compares all three tracks under the same evidence ladder:

1. native behavior on held-out examples;
2. a full-state causal ceiling at a plausible site;
3. factor-level transfer with opposing predictions;
4. active, selective removal on target and unrelated examples;
5. reuse or composition with another circuit;
6. untouched out-of-distribution confirmation; and
7. independent audit of row membership, computations, and decisions.

Do not let later agents inherit scientific outcomes for unopened splits. They inherit **methods and negative design
examples**, not answers. Keep discovery, selection, and out-of-distribution groups disjoint, and use a new critic or a
fresh outcome-blind audit when promoting a claim.

### Wave handoff contract

Every agent returns one machine-checkable packet containing:

- the exact behavioral claim and at least two meaningful counterfactual constructions;
- all plausible interpretations of those counterfactuals and matched active controls;
- the smallest tensor term being exchanged or removed, written as an equation and checked by reconstruction;
- row-level outputs plus exact split, family, semantic-position, and provenance identifiers;
- held, failed, and ambiguous bars without deleting nulls;
- reusable helpers and invariant tests, separated from circuit-specific code;
- a short list of newly discovered failure modes and the exact prompt/test change each caused; and
- the next discriminating experiment, including the observation that would kill the hypothesis.

The parent keeps a versioned prompt/tool manifest for each wave. A new wave starts only after that manifest records
which lessons were accepted, rejected, or remain untested. This gives later agents strictly better instruments while
preserving outcome blindness.

## Circuit investigation stages

Each circuit advances through the same stages, while failed stages remain recorded:

1. **Define the behavior.** State the causal variable and output comparison in ordinary ML/interp language.
2. **Build counterfactuals.** Include more than one answer-changing construction and matched answer-preserving
   controls. List every valid interpretation if a prompt admits multiple counterfactuals.
3. **Verify native behavior.** Freeze FIT/SELECT groups before searching model sites.
4. **Find a complete-state ceiling.** Ask whether replacing the full activation at a site can transfer the intended
   change. A null here blocks finer decomposition at that site.
5. **Split the computation.** Compare semantic factors such as selector, source value, relation, and output payload.
   Do not assume an attention head or MLP is the correct atomic unit.
6. **Translate to weights.** Express a held activation factor using embeddings, normalization, and exact model
   matrices. Verify equality numerically before interpreting it.
7. **Test active removal.** Delete the proposed factor on target rows and on unrelated rows where the deleted tensor is
   demonstrably nonzero.
8. **Test reuse and OOD.** Separate a genuinely shared subroutine from task-specific routing. Open untouched OOD data
   only after the earlier gates hold.
9. **Audit independently.** Recompute row membership, aggregates, bootstrap intervals, split openings, hashes, and
   terminal decisions without model calls.

## Counterfactual checklist

A row family is not ready merely because the expected answer changes. Before any localization run, record:

- the intended latent change;
- everything intended to remain fixed;
- all plausible alternative latent changes caused by the edit;
- the exact source and query token positions;
- whether base-to-donor and donor-to-base interventions are both meaningful;
- the grouping unit used for train/selection/bootstrap splits;
- a matched answer-preserving family;
- a second answer-changing construction that reaches the same claimed variable by a different surface edit.

If two interpretations remain valid, keep them as separate labeled families. Do not force one label and call the other
noise.

## What every experiment must save

- Every row-level causal measurement, not only a mean.
- Stable row, group, family, split, endpoint, and semantic-position identifiers.
- Native and intervened answer margins and cross-entropy.
- Full-vocabulary logit change when selectivity matters.
- Intervention norm on controls, so a zero-effect result cannot be a zero-intervention result.
- Exact input, code, preregistration, checkpoint, and result hashes.
- Literal forward/backward counts and which splits were opened.
- Frozen pass and null decisions, including failed cells.

Before a result is eligible for interpretation, validate its saved boundary with
`ops/result_contract.py` (or a stricter experiment-specific wrapper). Run the contract separately for every arm or
direction whose evidence table has the same authority row IDs. It must check exact authority membership, split closure,
finite standard JSON, declared field types, the model-call/mutation envelope, and required provenance hashes. A passing
scientific score does not override a failed result contract.

## Knowledge packet returned after each track

The parent extracts five reusable objects from every completed track:

1. **Dataset pattern:** which edits produced unambiguous causal variables and which were confounded.
2. **Semantic coordinate mapper:** code for locating source, query, answer, and nuisance positions.
3. **Intervention primitive:** the smallest exact activation or weight term tested.
4. **Control pattern:** how the intervention was made nonzero on unrelated behavior.
5. **Failure diagnosis:** behavior failure, site-ceiling failure, factor failure, selective-removal failure, OOD failure,
   or bookkeeping/implementation failure.

These objects are added to the next prompt. This is the bootstrap step: later agents inherit tested code and concrete
failure examples, not just prose conclusions.

## First-wave lessons that are mandatory in later prompts

The first two-agent wave added four checks that are now part of the shared template:

1. **Treat output structure as audited data.** Tests must validate field types as well as values. In particular, a
   scalar decision string must not serialize as a one-item JSON list. A scientifically held result with a malformed
   frozen envelope remains a failed audit and must be repaired by a prospective rerun, not by editing result bytes.
2. **Freeze crossed factors before intervening.** When testing score-only versus value-only attention computations
   across several layers, cache the native recipient and donor factors first. At each later layer, remove the live
   recipient term and insert the requested frozen combination. Otherwise an earlier intervention can change the
   supposedly untouched later factor, so the named arms no longer isolate score from value.
3. **Use semantic positions across unequal prompt lengths.** Source, payload, and query coordinates come from row
   metadata. Do not require donor and recipient prompts to have equal tensor length, and never use padded absolute
   indices as semantic coordinates.
4. **Pair causal transfer with an opposing prediction.** A factor is not identified merely because it transfers an
   answer. Score-only and value-only arms must make different predictions on selector-only and payload-only edits;
   active answer-preserving controls must rule out a broad contextual write. Save donor-answer CE and full-vocabulary
   change so generic damage cannot masquerade as transfer.

The second wave added four more:

5. **Evaluate every denominator from row metadata before model access.** A formula written for answer-changing rows may
   become identically zero on an answer-preserving match break or diagonal. The model-free suite must enumerate every
   cell, substitute the frozen answer identities and direction signs, and reject an impossible or ambiguous
   denominator as a specification error rather than a scientific null.
6. **Do not mix units when normalizing selectivity.** Use residual-stream intervention norms only to establish that a
   control intervention is active. Normalize a logit-margin change by a frozen logit-margin target scale and a
   full-vocabulary logit RMS by a frozen vocabulary-logit RMS target scale. The prompt must include the complete
   control-to-target scale lookup before outcomes are opened.
7. **Write the literal factor sum and an independent reconstruction.** If a factor has semantic-role and site indices,
   every intervention formula must retain both. Replaying a tensor obtained from the same hook is circular; also
   reconstruct the isolated term from the canonical hash-pinned attention computation and prove that it plus the
   remainder equals the native head output.
8. **Separate pre-outcome freezing from post-outcome execution admission.** An auditor may require the future result
   namespace to be absent in its original dry run. If the managed queue later reruns dry-run after that result exists,
   use a tiny hash-pinned execution adapter: its dry run verifies the frozen auditor bytes and source-pair presence
   without parsing outcomes, and its real path executes those exact bytes. Never weaken the frozen auditor after seeing
   a result merely to satisfy queue admission.
9. **Freeze machine-readable names and their census, not just prose labels.** Every family, variant, condition,
   direction, arm, metric, and bootstrap cell must have one canonical serialized token. The model-free package must
   independently enumerate the literal cells, assert their counts, and hash their ordered identities. Display names
   are documentation only: an implementation must not infer authority membership from a human-readable label.
10. **Make post-outcome screens say exactly what they establish.** A threshold chosen after outcomes are visible may
    triage candidates, but neither passing nor failing it identifies or rules out a circuit. Saved terminal labels must
    say `passed/failed the recorded post-outcome filter`, not `held`, `stable`, `absent`, or another confirmatory claim.
    Require semantic metadata—not only row IDs—to match across compared arms, and bind analysis/test hashes in output.
11. **Validate counterfactual meaning from tokens, not labels.** For every recipient/donor row, recompute the variables
    claimed changed and held fixed from token IDs and semantic positions. Test both physical directions when their
    meanings differ, reject identical endpoints, and require several group-disjoint lexical, layout, and length
    realizations. Reversing one prompt pair is not a second counterfactual construction.
12. **Publish one atomic evidence package.** Evidence arrays, row tables, result, and receipt are one scientific object.
    Write and validate them in a unique same-filesystem staging namespace, inject crashes during evidence writing and
    between every publication boundary, and make retry recoverable. After any exception, either the complete mutually
    hash-bound package exists or no final artifact exists.
13. **Keep the evidence ladder machine-readable.** Counterfactual validity, interaction isolation, held-out prediction,
    OOD prediction, sufficiency, selective removal, reuse/composition, and stable identification are separate claims.
    Every handoff must mark each as held, failed, or not tested and provide its own required evidence. Interchange
    success must never silently promote removal, OOD, or uniqueness claims.
14. **Keep builder and critic information asymmetric.** The builder receives the frozen scientific hypothesis and
    repair contract. The critic independently reconstructs semantic changes, manifests, metrics, and crash behavior
    from pinned authorities and must not adapt to the builder's current working file. The parent reconciles their exact
    artifacts; neither agent can waive the other's failed contract.
15. **Audit scientific nulls as deeply as positive results.** Every terminal produced after model execution must carry
    the complete evidence for the phases it opened. A FIT-only null has a smaller exact row and array census than a
    FIT+SELECT hold, but it may not replace those rows with summaries or fixture data. A missing-evidence null is not a
    conservative result; it is an unauditable result.
16. **Bind evidence membership to the frozen authority, not to itself.** Internal row-order hashes only show that files
    agree with their own declared order. Independently derive the exact authorized endpoint IDs, directed
    intervention-by-arm IDs, and endpoint-by-site factor IDs, then require literal equality for the opened phases. A
    self-consistent invented dataset must fail.
17. **Exercise crash recovery through the real managed entry point.** Staging and quarantine helpers are insufficient
    if an earlier unused-path guard makes them unreachable. Test a hard-crash partial package through the same adapter
    the queue invokes. Recovery may move only marker-bound, recognizable experiment bytes; it must leave a complete
    prior outcome or arbitrary occupied path untouched and fail closed.
18. **Recompute the meaning of saved evidence.** Authorized IDs, shapes, and hashes do not prove that the payload under
    those IDs is correct. Join endpoint tokens and semantic coordinates to the frozen row; join each direction to its
    exact recipient and donor; reconstruct every arm's inserted tensor from saved factors and require
    `hook_delta = inserted - live_removed`; and recompute margin, cross-entropy, and vocabulary-change identities from
    primitive logits. Include a complete, internally hash-consistent negative fixture for each mismatch.
19. **Derive failure reasons as well as positive decisions.** An `invalid instrument` terminal is not self-justifying.
    Rebuild its exact phase-specific clause list from the retained endpoint, factor, intervention, and structural
    evidence, use one canonical order, and reject invented, omitted, duplicated, or mis-prefixed reasons. Exercise both
    a FIT stop and a SELECT stop with complete hash-consistent evidence packages.
20. **Do not serialize an unauditable implementation check as a scientific null.** If a check uses a full intermediate
    tensor that the evidence package intentionally does not retain, failure must abort before publication. Examples
    include full-attention reconstruction, replay/native full-logit equality, incomplete capture, the live-factor
    reconstruction at an intervened state, the observed hook write, and nonfinite tensors. The completed-package
    validator must reject a result that tries to reintroduce any such failure as matching text. Retained algebraic
    checks may still produce an invalid terminal, but their values and clauses must be recomputable from saved arrays.
21. **An auditable proxy supplements rather than replaces a preregistered end-to-end check.** If the preregistration
    requires two interventions to match in full-vocabulary logits, continue to run that comparison even when the logits
    are too large to retain. Make a mismatch a pre-publication hard failure. A saved equality of their inserted tensors
    can independently explain why the identity should hold and can be rechecked during audit, but it must not silently
    substitute for the registered downstream test.

The current machine-readable contract is
`basis_aligned/bilinear_quotient/ops/circuit_causal_validity_next_wave_handoff_rung585.json`; its human audit checklist
is `basis_aligned/polynomial_causal/CIRCUIT_CAUSAL_VALIDITY_NEXT_WAVE_CRITIC_CHECKLIST.md`. Later versions may supersede these paths only
prospectively and must preserve their negative fixtures.

The prospective phase-evidence and managed-recovery additions are machine-readable in
`basis_aligned/bilinear_quotient/ops/circuit_causal_validity_next_wave_handoff_rung585_v2_addendum.json`. Apply it on top
of the base contract for every later wave; it does not alter the frozen R585 scientific thresholds.

## Subagent prompt template

> Investigate `<circuit id>` as a behavior-level computation. Read its canonical record and all named negative events
> before designing anything. Your bounded deliverable is CPU-only: (a) at least two answer-changing counterfactual
> families with group-disjoint FIT/SELECT roles, (b) active matched controls, (c) semantic source/query coordinates,
> (d) a preregistration with fixed pass and null outcomes, (e) implementation, tests, and deterministic dry run. Treat
> attention heads and MLPs as candidate containers, not the true basis. Do not optimize rank or reconstruction unless
> it directly supports extraction, selective removal, OOD prediction, or reusable composition. Record multiple valid
> counterfactual interpretations instead of hiding ambiguity. Do not touch canonical registries or enqueue GPU. Claim a
> unique rung on the board first, preserve all frozen nulls, and return exact paths, hashes, tests, and unresolved risks.

> Before handoff, run the generic `ops/result_contract.py` checks on planted held and planted null artifacts. Declare
> scalar/list/dict field types explicitly, validate exact authority membership, and use strict finite JSON. If several
> arms reuse the same row IDs, validate each arm independently rather than concatenating them into a duplicate table.
> Include the canonical serialization token for every family/variant/condition/direction/arm/metric and a separately
> generated fixture that asserts the exact cell census plus a hash of the ordered cell IDs.
> State whether every threshold was prospective or chosen after seeing outcomes. For a post-outcome screen, use a
> literal non-confirmatory terminal label and name the fresh, group-disjoint intervention that would test the lead.

## Scaling beyond two subagents

Increase breadth only after one full parent-plus-two-agent wave has produced reviewable knowledge packets. Five agents
are useful when datasets and intervention primitives are already stable; before that point they tend to reproduce the
same ambiguity five times. Add circuits in batches, compare failure classes, update this playbook and the prompt, then
launch the next batch.
