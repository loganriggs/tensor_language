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

## Subagent prompt template

> Investigate `<circuit id>` as a behavior-level computation. Read its canonical record and all named negative events
> before designing anything. Your bounded deliverable is CPU-only: (a) at least two answer-changing counterfactual
> families with group-disjoint FIT/SELECT roles, (b) active matched controls, (c) semantic source/query coordinates,
> (d) a preregistration with fixed pass and null outcomes, (e) implementation, tests, and deterministic dry run. Treat
> attention heads and MLPs as candidate containers, not the true basis. Do not optimize rank or reconstruction unless
> it directly supports extraction, selective removal, OOD prediction, or reusable composition. Record multiple valid
> counterfactual interpretations instead of hiding ambiguity. Do not touch canonical registries or enqueue GPU. Claim a
> unique rung on the board first, preserve all frozen nulls, and return exact paths, hashes, tests, and unresolved risks.

## Scaling beyond two subagents

Increase breadth only after one full parent-plus-two-agent wave has produced reviewable knowledge packets. Five agents
are useful when datasets and intervention primitives are already stable; before that point they tend to reproduce the
same ambiguity five times. Add circuits in batches, compare failure classes, update this playbook and the prompt, then
launch the next batch.
