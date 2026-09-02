# Task-conditioned cross-module decomposition

Status: research-direction correction, 2026-09-02 00:47 UTC. This is a design document, not a preregistration and not
authorization for a model run.

## Goal

Find computations that can cross native head and MLP boundaries:

- group pieces that calculate or carry the same downstream variable;
- split a head or MLP when different pieces serve different tasks or compositions;
- expose shared parents, task-specific children, and reuse;
- predict the computation and its behavioral effect on held-out and OOD inputs;
- extract a sufficient task computation or well-specified task interface;
- remove, swap, or edit the target computation with low damage to unrelated behavior; and
- predict how shared pieces combine with task-specific branches.

Compression, rank, and parameter count are not the target. They are controls against trivial extra capacity and,
after a causal decomposition is found, possible evidence that it is a simpler executable program.

## Evidence already in the repository

1. The 62 behavior tags localize broadly, but they are not 62 independent mechanisms. Attention8 is the strongest
   location for16 tags and attention16 for13.
2. Attention8 supports a shared-parent-plus-children organization: one direction explains91.61% of five fitted
   circuit directions, and removing it makes four residuals selective. Attention16 does not share that organization.
3. Small learned subspaces are enriched for task effects but incomplete: rank4 recovered8–23% of full-component
   effects and representative rank64 fits recovered only about25–35%.
4. An exact equality/copy service generalizes, but its removal is not induction-selective. This suggests one shared
   matcher with several task-specific output/use branches.
5. Existing whole-component and fitted-direction response tensors are valuable measurements, but pooled low-rank
   fits were dominated by one large MLP16 interface and did not establish a general shared/private hierarchy.

Together these results argue for task-conditioned grouping at a finer grain than components, without treating a
low-rank basis as semantic.

## Candidate computation graph

Use an overcomplete set of algebraic terms. These are probes, not assumed final components.

### Attention terms

For each head, describe a routed write as

`query feature × key feature × relative-position factor × OV write`.

Allow query features, key features, and OV writes to be shared independently across heads. This can represent two
heads sharing a matcher but writing different outputs, or different attention patterns producing downstream-
equivalent outputs.

### MLP terms

For each bilinear MLP, describe a term as

`left input feature × right input feature -> output write`.

Native product channels, learned mixed products, and exact token-conditioned terms can all propose candidates. A
shared input feature may participate in several products; several products may write a downstream-equivalent output.

### Cross-module graph

Keep three types of node distinct:

- producers: input features available to a computation;
- compositions: Q×K or left×right products; and
- consumers: output writes and later readers.

This distinction prevents merging successive steps merely because they affect the same task.

## Task fingerprints

For each proposed term, measure signed effects across:

- registered task-member and matched off-task positions;
- natural contexts and controlled synthetic variations;
- later modules/readers;
- removal, amplitude changes, and interchange; and
- fresh documents and a second corpus.

Regress out the common global-damage direction before clustering. The 62-tag vector alone is insufficient because
old rank sweeps showed that it can be almost one-dimensional. Include targeted behavior axes for the first selected
families rather than relying only on all-purpose CE damage.

## First two behavior families

1. **Equality/copy/induction.** Reuse the known exact equality matcher. Ask which Q/K terms calculate matching,
   which OV/MLP branches carry copied identity or other outputs, and which downstream readers turn the shared match
   into induction-specific versus other copy behavior.
2. **Ordered successor.** Identify terms that represent an ordered relation separately from terms that carry the
   successor value or choose where it is used. This provides a different composition structure and guards against a
   copy-specific method.

## Discovery and validation

1. Collect fine-grained task fingerprints on fitting documents.
2. Propose a sparse producer–composition–consumer graph with shared nodes and task-specific branches. Compare a flat
   independent graph, shared-parent-plus-child graphs, and a mixture that chooses structure separately by module.
3. Freeze groups before evaluation.
4. On held-out documents, test whether within-group interchange is substantially closer than between-group
   interchange and whether the graph predicts unseen task effects.
5. Remove or transplant one proposed path. Require the expected task effect with low damage to unrelated tasks.
6. Test the same grouping on a shifted corpus or synthetic task variation.

Success is not reconstruction or low rank. It is a grouping that predicts what computation will transfer under
interchange, what task will break under removal, and what unrelated tasks will remain intact.

## Cheapest first computation

Before a GPU collection, audit the exact existing copy/equality and ordered-successor artifacts and map every stored
intervention to the producer–composition–consumer schema. Determine which fine-grained Q/K/OV and MLP terms can be
reconstructed without refitting. Then freeze the smallest missing response collection. Do not rerun whole-component
localization or another generic rank sweep.
