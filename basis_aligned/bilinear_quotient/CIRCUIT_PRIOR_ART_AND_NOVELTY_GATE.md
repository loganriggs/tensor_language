# Circuit prior-result and novelty gate

## Purpose

The experiment index catches repeated protocols that are already represented as version-2 circuit
events. It does not catch an old result written only in the historical ledger, a module fact stored
under another name, or a previously failed discovery method applied to a new tensor. This gate is
required before freezing any new circuit preregistration.

## Required review receipt

Every preregistration must include or link a receipt with these fields:

```yaml
canonical_objects:
  - module.attention.5
aliases_searched:
  - attn5
  - attention block 5
  - head 5.7
  - induction gate
  - content gatherer
method_families:
  - task_conditioned_causal_direction
matched_prior_claims:
  - results.section_877
relation: extension
novelty_delta: "Quantify whether the whole block's output is a single transported direction on held-out natural text and code."
decision_changed: "Determines whether later interventions can replace the block write by one fixed vector."
reviewer: pending
```

The allowed relations are `replication`, `extension`, `contradiction_test`, and `new_question`.
`new_question` is valid only when no matched prior claim answers the proposed question. Renaming a
module, script, metric, or tensor does not make the question new.

The reviewer checks the generated circuit dossier, experiment index, ownership registry,
specialist-head notes, and the historical results ledger. Searches must include module numbers,
dotted head names, functional descriptions, and old terminology. A prior match without an explicit
relation and novelty delta blocks the preregistration.

## Initial normalized prior results

### `module.attention.5`

Aliases: `attn5`, `attention block 5`, `L5 attention`, `head 5.7`, `induction gate`, `copy
head`, `content gatherer`, `pooler`, `constant write`, `sink`.

Known before 2026-09-04:

- Sections 877 and 882: block 5 is the main induction/copy gate in the front induction circuit.
- Sections 998, 1006, and 1007: content gathering is concentrated in layers 3–5 and largely in
  layer-5 head 7, with a distributed cooperative remainder.
- Sections 1039, 1043, 1044, and 1047: block 5 is a broad residual-routing/pooling component for
  which simple token or local-window stand-ins were insufficient or limited.
- Circuit registry entry `Sink (5.7)`: head 5.7 has a constant-vector, mean-replacement-free
  legacy result.

The 2026-09-04 whole-block measurements—98.1% in one direction, cross-corpus cosine 0.997, and
about 95% fixed-vector recovery—are an **extension/replication of geometry**, not discovery of a
new attention-5 circuit.

### `method.unsupervised_energy_or_variance_as_circuit_basis`

Aliases: `activation energy`, `output variance`, `PCA basis`, `top singular directions`, `energy
recovery`, `low-rank basis`, `magnitude ranking`.

Known before 2026-09-04:

- Section 617: principal components order magnitude, not functional class discrimination; the
  compact subspace and interpretable rotation are different questions.
- Section 836: a large class-and-position keep effect disappeared against a matched-rank
  shuffled-label control.
- Section 1056: low-variance directions can remain causally privileged, so variance share does
  not order causal importance.

Rule: energy, variance, reconstruction error, and rank may be controls, lower bounds, or prices.
They do not identify a circuit by themselves. A discovery claim additionally needs a
task-conditioned causal measurement, a matched-capacity null, and held-out intervention evidence.

## Three searchable record types

The long-term registry must expose separate generated views for:

1. behavior circuits and shared computations;
2. module/component facts, including whole modules and sub-head structure; and
3. method failures and known invalid inference patterns.

Until those views are generated automatically, this file is the canonical bridge for module facts
and method failures. Every new result must update its matching record before the next experiment is
opened.

The current human-readable views are `circuits/MODULE_DOSSIERS.md` and
`circuits/METHOD_FAILURES.md`. They are mandatory prior-art inputs, not optional narrative notes.
