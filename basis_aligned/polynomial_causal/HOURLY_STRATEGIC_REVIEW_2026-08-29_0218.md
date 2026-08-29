# Hourly strategic review — 2026-08-29 02:18 UTC

## Bottom line

The highest-value unanswered question is now narrow and executable:

> Can 256 or 512 shared native Block-3 product gates reproduce the **sum** of the
> four RMS-polarized terms, and does that replacement preserve the autonomous
> computation through Blocks 4--17?

The fit stage found a real but modest selection signal.  It did **not** answer this
question because it scored the four terms as separate stacked targets.  A causal
validation transaction has now been implemented and its CPU contract passes 51 tests.
Its first independent audit identified three launch blockers; all three have been
repaired and a second audit is in progress.  The RTX 5090 is currently occupied by the
parallel context-free fallback rank sweep, so the safe action during that interval is
to finish and freeze this validation transaction rather than contend for the GPU.

## Honest current accounting

These quantities answer different questions and must not be added together.

| Ledger | Current credit | What remains |
|---|---:|---:|
| Structural inventory | 36/36 modules represented | semantics and causal consequences of the representations |
| Whole-program storage certified removable | 5.3481% | 94.6519% has no consequence certificate |
| Older named-behavior estimate | 32.1% ± 6.4% | not a whole-model faithfulness fraction |
| Strict named causal CE recovery | 10.923% | 4.72714 nats, or 89.077%, remains unexplained |
| Final extraction/removal/OOD actions closed | 0/68 | the complete final action ledger |

No global ledger moves in this review.  The Block-3 fit is search evidence, not causal
or semantic credit.

## What is actually known about the Block-3 candidate

For the exact RMS decomposition

$$
z = u+v,
$$

the native bilinear MLP write expands into four typed terms:

$$
B(z,z)=B(u,u)+B(u,v)+B(v,u)+B(v,v).
$$

One shared subset of native product gates and one decoder were fitted across those four
terms.  On 92,160 fit positions, the stacked typed-term normalized errors were:

| Program | K=256 | K=512 |
|---|---:|---:|
| activation-selected | 0.75659 | 0.68491 |
| matched random | 0.79042 | 0.71050 |
| label permutation | 1.00412 | 1.01371 |

This says selection contains signal beyond a random subset, and the label control
destroys it.  It does not say the deployed sum has error 0.68--0.76: errors in the four
terms may cancel or reinforce.  The executable programs cost 3,545,600 bytes and 256
products per token at K=256, or 7,086,592 bytes and 512 products at K=512, versus 4,608
native products.  Literal simplicity is therefore already defined; functional
faithfulness is still unknown.

## Largest remaining gaps

1. **The Block-3 interface is unvalidated.**  We have no held-out summed-write NRMSE,
   autonomous suffix KL/CE, mirror response, or typed singleton recovery.
2. **The final causal action ledger is still 0/68.**  Physical routing exists, but the
   final semantic reducer and full consequence replay do not yet provide extraction,
   selective-removal, and OOD evidence.
3. **Independent early-MLP simplifications have not earned composition credit.**  A
   replacement can look good alone because later MLPs compensate; joint replay is the
   test of whether interfaces compose.
4. **Most strict CE remains unnamed.**  The causal ledger has recovered only 10.923%,
   leaving 4.72714 nats.  Local tensor reconstruction cannot close this account.
5. **The context-free program still mishandles uncovered tokens.**  Parallel work now
   estimates 0.82--0.88 nat of fallback-specific loss on uncovered positions; its rank
   sweep is already running and should not be duplicated here.

## Candidate moves considered and pruned

The ranking uses expected information gain, direct causal relevance, whole-model
composability, falsifiability, GPU cost, and duplication of already completed work.

### 1. Run the audited Block-3 native-gate validation

Highest priority.  It directly joins tensor structure, the exact polynomial expansion,
native-gate gauge invariance, literal executable cost, and autonomous causal replay.
It has sharp failure conditions and matched random, permutation, omission, singleton,
and mirror controls.  The first wave uses 192 rows from 79 source documents; all
repeated rows are accumulated into their source document before a 2,000-draw document
bootstrap.  The replacement is made at all 256 positions and Blocks 4--17 are rerun
without teacher-state reuse.

### 2. Follow the frozen branch from the Block-3 result

If K=256 or K=512 passes the local, KL, CE, control, singleton, and mirror screen,
complete its 16-replacement/15-omission interaction cube on validation and then final.
If activation-fitted gates fail but candidate and mirror errors decay downstream,
complete the cube only as a downstream-null test.  Otherwise stop that fit family and
fit the same finite gate grammar to downstream consequences.  This keeps a failed
optimizer/objective from being confused with a failed mathematical grammar.

### 3. Close the 68-action extraction/removal/OOD interface

This is the largest direct project-level measurement gap.  The scorer must distinguish
internal reconstruction, functional extraction, selective removal, and OOD transport.
A simpler program is useful only if its complexity predicts one or more of those
capabilities better than an equally faithful but less structured control.

### 4. Run independent and joint MLP0/MLP1/MLP2 replacements

First measure each replacement alone, then all pairs and the triple on identical rows.
The interaction residual

$$
\Delta_{ij}=E_{ij}-E_i-E_j
$$

distinguishes composable interfaces from downstream compensation.  Strong negative or
positive interaction tells us which downstream module defines the correct quotient of
an upstream representation.  This is more informative than another isolated MSE fit.

### 5. Build a shared sparse dictionary only after defining its consumers

A weight SAE, nonnegative/sparse dictionary, hierarchical dictionary, or DAG can be
useful if the same atoms sparsely parameterize MLP0 writes **and** the MLP1/MLP2 reads
that consume them.  The objective must include downstream consequences and literal
description/execution cost.  A weight-only SAE is not prioritized because basis changes
can make it look sparse without improving any intervention or OOD task.

## Mathematical ideas retained versus deferred

- **Retained now:** exact bilinear polarization, native-gate subset rank, product-gauge
  invariance, document-clustered uncertainty, autonomous causal substitution, and
  literal bytes/products.  These all make predictions beyond local MSE.
- **Retained as the next fit family:** consequence-fitted finite native gates, which is
  a small system-identification problem with the downstream suffix as the observation
  map.
- **Deferred:** norm minimization followed by HOSVD.  Toy work showed scalar product
  gauge balancing is exact and useful for conditioning, but coefficient HOSVD alone
  does not establish a downstream interface.
- **Deferred:** generic SAE/dictionary learning, MDL, and information bottlenecks until
  the 68-action harness can test whether their claimed simplicity buys extraction,
  removal, OOD transport, or executable savings.
- **Pruned for this step:** full invariant-ring calculations, Hankel/automata methods,
  and more local coefficient probes.  They currently lack a defined composable state or
  duplicate information already supplied by the gate assay.

## Action executed in this review

The validation harness was repaired rather than launched prematurely:

1. corrected 192-row uncertainty accounting to 79 source documents;
2. accumulated repeated rows before document bootstrap;
3. separated the physical call receipt by wave, exact arm, causal family, native typed
   Down call, candidate typed decoder call, and direct deployed program call;
4. explicitly certified zero outer full-model forwards/returns and zero native MLP3
   calls on student arms;
5. closed the source hash over every modified and directly relied-on test;
6. added a known-answer document-clustering test and an adversarial cross-family call
   substitution test.

The scoped CPU suite passes **51/51**.  Independent re-audit is in progress.  Launch is
blocked only by that scientific gate and the currently occupied GPU, not by data access,
checkpoint access, row caching, or implementation ambiguity.
