# MLP1 native-gate assay: what the negative result means

Date: 2026-08-28

## Result in one sentence

The experiment did **not** find an admissible context-independent subset of 32,
128, or 512 of MLP1's 4,608 native multiplication gates that reliably preserved
the measured downstream response better than the registered controls.

This rejects the tested native-gate pruning rule. It does not say that MLP1's
quadratic function needs 4,608 products in every basis.

## What was measured

For a residual input $z$, MLP1 has the form

$$
y(z)=b+D\big((Lz)\odot(Rz)\big).
$$

Each coordinate of $(Lz)\odot(Rz)$ is one native multiplication gate. The assay
asked whether the same small set of those gates mattered across independent
contexts and independent downstream probes. Candidate sets were fitted on one
document cohort, frozen before validation, and scored on a disjoint cohort.

The protected run performed 512 backward passes and completed in 253.15 seconds.
The fitted bundle was serialized before validation, and validation did not modify
it. Raw logits, targets, residual gradients, and response tensors were not
published.

## What worked

The selected gate identities were repeatable across the two independent probe
halves:

| gate budget | support Jaccard overlap | support-rank Spearman correlation |
|---:|---:|---:|
| 32 | 0.939 | 0.956 |
| 128 | 0.766 | 0.958 |
| 512 | 0.784 | 0.960 |

Jaccard overlap is the fraction of the union of two selected sets that appears in
both sets. Spearman correlation measures whether gates receive a similar ordering,
without requiring the score scale to match. Thus the negative result is **not**
explained by noisy or arbitrary gate selection.

At budget 128, the selected gates also formed numerically well-conditioned fitted
systems. The failure is downstream: the primary selector did not dominate simple
response-energy, activation/Down, deranged-factor, and random controls on both
validation halves and both execution modes.

## Why the candidate was not admitted

The preregistered rule required all of the following:

1. stable selected support;
2. at least 5% lower held-out loss than every control, with a simultaneous lower
   confidence bound above that threshold;
3. no document harmed by more than 2%;
4. positive results at budgets 32, 128, and 512;
5. success both when the chosen gates used context-specific fitted coefficients
   and when they were simply switched on with a shared coefficient.

Only the support-stability requirement passed.

At budget 32, the primary and response-energy programs were effectively identical
on all registered validation cells. At budget 128, the primary context-specific
fit was slightly worse than response energy on both validation halves
($-0.114\%$ and $-0.168\%$ relative improvement). The shared all-on version beat
response energy by only $2.46\%$ and $1.13\%$, below the required 5%, and its
worst-document harm reached 7.35% in that comparison. Across all controls, some
cells favored the primary subset and others strongly favored a control; the full
observed improvement range was approximately $-10.47\%$ to $+13.08\%$.

At budget 512, every registered selector was rejected because the fitted support
coefficients exceeded the frozen norm gate. Therefore only 32 of the planned 48
comparisons existed, and the simultaneous bootstrap certificate was correctly not
computed rather than silently conditioning on the successful fits.

## The mathematical conclusion

The checkpoint's hidden-unit basis is not a privileged simple program under this
test. Stable gate importance exists, but selecting native gates does not produce a
uniformly better small executable across contexts and execution modes.

The correct next object is the basis-independent folded quadratic tensor

$$
T_{oij}=\sum_{n=1}^{4608}D_{on}
\frac{L_{ni}R_{nj}+R_{ni}L_{nj}}{2},
$$

with the bias $b$ stored separately. This tensor represents the same quadratic
map after quotienting out hidden-unit permutations and reciprocal rescalings.
Its mode ranks can reveal shared input and output subspaces that native gate
deletion cannot see. A Tucker or CP factorization could then define **new** products
that are fewer than the checkpoint's 4,608 products.

HOSVD energy is only a screening statistic. It can establish multilinear
compressibility, but not minimal product count, causal preservation, semantic axes,
or final cross-entropy. Any promising tensor curve must next be made executable and
tested at the MLP0-to-MLP1-to-MLP2 interfaces.

## What is now pruned and what remains open

Pruned:

- claiming that a fixed top-32 or top-128 subset of native MLP1 gates is the
  simple downstream code;
- increasing the same native-gate budget without changing the representation;
- treating stable gate rankings as sufficient evidence for an executable circuit.

Still open:

- a new low-product-rank basis obtained from the folded tensor;
- Tucker/block-term structure with a sparse symmetric core;
- a context-routed or hierarchical gate dictionary rather than one global support;
- a joint MLP0/MLP1/MLP2 factorization whose interfaces are sparse in the same basis;
- overlapping lexical features plus continuous context variables, if their
  downstream readers and whole-model consequences are also simple.

The source result is
`tensor_bilin18_mlp1_global_gate_results.json`; its final status is
`no_admitted_support`.
