# Multi-environment, multi-reader constrained DAS preregistration

## Motivation

The current evidence supports the user's overfitting diagnosis in a qualified form. A learned rank-one
axis can optimize its own scalar or complement objective while losing full-vector transfer. Noise and
KL are useful regularizers but have not been uniformly sufficient. Multicue pooled aligned DAS is the
one optimized method that beats pooled difference-in-means on both held-out panels, while its sizable
complement effect still shows task conditioning. The next test must prevent the optimizer from winning
by learning the identity of one task/readout.

## Frozen causal object

Optimize a rank-one or rank-two projector at the already validated temporal auxiliary writer. Do not
change the writer site, examples, downstream intervention implementation, reader definitions, or rank
after observing outcomes. Use the component-resolved is/was/temporal shared-Q8 route to define readers:

1. the local writer response;
2. the validated intermediate L11/L15 attention readers;
3. resid18 shared-Q8 coordinates;
4. centered full-vocabulary logits and answer/foil margin.

Readers are divided into construction readers and sealed evaluation readers. At least one downstream
reader and the centered full-vocabulary logits are never used for early stopping.

## Objective

For environment `e`, reader `r`, base state `x_b`, donor state `x_d`, and orthogonal projector `P`,
define subspace and complement interventions

`x_P = x_b + P(x_d - x_b)` and `x_C = x_b + (I-P)(x_d-x_b)`.

Minimize the panel-balanced worst-environment loss

`max_e sum_r w_r [ D(F_er(x_P), F_er(x_d))/s_er + D(F_er(x_C), F_er(x_b))/s_er ]`

plus a small projector-stability penalty across independent minibatch/noise views. `D` is squared error
for causal coordinates and margin, and centered logit KL plus normalized squared error for the full
vocabulary. Scales `s_er` are frozen from native base-to-donor effects. This makes complement inertness
one term of a multi-reader causal operator objective, not the sole training label that the axis can
memorize.

## Methods and regularization arms

Use matched rank and identical data partitions for:

- pooled difference in means;
- plain aligned DAS;
- Gaussian activation-noise DAS;
- centered full-vocabulary KL DAS;
- DIM-anchored projector DAS;
- multi-environment worst-case DAS;
- worst-case DAS with noise plus KL.

Freeze regularization strengths by A1 validation only. A2 constructions, the held-out downstream
reader, and the final full-vocabulary evaluation remain sealed. Use at least three deterministic starts
per learned method and report projector principal-angle stability rather than selecting the luckiest
restart.

## Acceptance and falsification

A learned projector graduates only if, on every sealed panel:

- its multi-reader joint objective beats matched-rank pooled DIM;
- the subspace intervention preserves at least the DIM causal effect at each independent reader;
- the complement effect is no larger than DIM at each independent reader;
- centered full-vocabulary KL and normalized vector error both improve;
- restart/bootstrapped projector stability clears the frozen principal-angle bar;
- cross-task composition with the shared-Q8 circuit does not add destructive interaction.

If optimization improves trained readers but not sealed readers, classify task memorization. If noise/KL
improves stability but not causal transfer, retain it as regularization evidence, not a better circuit.
If no optimized arm beats DIM, the conclusion is that the objective/reader family is still wrong—not
that optimization cannot in principle find a better subspace. The next response would be to optimize
the finite causal operator or Hankel block identified by the component-resolved circuit, rather than
adding more penalties to a single readout.

## Efficiency

Cache native base/donor states and reader targets once. Share each forward pass across all reader losses.
Screen rank one first; run rank two only if rank one passes sealed stability but leaves structured reader
residual. Stop dominated regularization arms after the validation gate. This keeps DAS subordinate to
circuit discovery rather than allowing a large hyperparameter sweep to consume the queue.
