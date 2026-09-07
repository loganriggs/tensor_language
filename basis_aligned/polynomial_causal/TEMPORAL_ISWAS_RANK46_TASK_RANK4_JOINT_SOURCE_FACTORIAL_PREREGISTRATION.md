# Temporal/is–was rank46 task-rank4 joint source factorial

## Question and frozen source sets

The corrected edge atlas shows exact, task-typed weight edges but also proves that
singleton effects cannot be added: the additive response errors are `2.50` and `4.71`.
This experiment asks whether the atlas nevertheless selects a jointly executable source
set, or whether serial/contextual interactions invalidate singleton ranking itself.

The shortest source prefixes reaching 80% of positive singleton incidence are frozen:

- temporal: `MLP1, MLP0, MLP2, MLP3, L0H3, MLP4, MLP8, MLP7, MLP6`;
- is–was: `MLP1, MLP2, MLP0, MLP3, MLP4, MLP5, L2H2, MLP6, MLP7, MLP8, L4H7`.

Also test their eight-source intersection, twelve-source union, each set's complement
inside rank46, and full rank46.  Patch complete cached module/head outputs jointly into
the otherwise live base run.  Rows, rank-four response bases, endpoint positions, and
physical reader stay frozen.  No greedy reselection follows this factorial.

## Measurements and predictions

Measure task-rank-four response signed projection and relative squared error across all
eight response sites, final answer/foil behavioral projection, centered-logit controls,
and top-1 flips.

1. Authority, source lists, row disjointness, self patch, full-rank46 replay, finiteness,
   and the bounded forward price pass exactly.
2. Each task's own top-80 source set jointly gives response projection at least `.70`,
   response error at most `.30`, and behavioral projection at least `.60`.
3. Own source sets beat cross-task source sets by at least `.10` in response projection
   for both tasks.  Failure with preserved task-typed weight modes means physical writers
   are shared and typing arises inside their downstream weight functions.
4. The union jointly gives response projection at least `.80`, response error at most
   `.20`, and behavioral projection at least `.75` for both tasks, without control
   top-1 flips or median centered-logit KL above `.02`.
5. Each physical complement gives at most `.50` response projection for its task, while
   full rank46 remains at least `.80`.  This distinguishes a sufficient/necessary source
   program from redundant distributed support.

Five passes yield `joint_task_typed_source_program`.  A failure is preserved as
`joint_source_interaction_boundary` and selects a serial layer-band or signed-interaction
factorial, not a new rank threshold.  Maximum price is 32 model forwards, zero fitting,
zero model updates, and zero transformer backwards.
