# MLP2 trajectory-robust rank-512 v1 — preregistration

## Question

The frozen MLP2-FULL512 student has extra CE `0.052876` on native states, but
installing MLP0-C512 increases its marginal CE by `0.008739` nat. Is this positive
interaction caused by train/deploy trajectory shift at the pre-MLP2 interface?

The candidate grammar and literal price stay unchanged:

$$
\widehat f(x)=\widehat b+widehat D
  \left((\widehat Lx)\odot(\widehat Rx)\right),
$$

with 512 products and 1,770,624 floating coefficients. No context gate, native MLP2
call, token table, rank increase, or private C512 correction is allowed.

## Training backgrounds

For each frozen training document, capture paired pre-MLP2 states:

- (x_N): the native trajectory;
- (x_C): the trajectory with frozen MLP0-C512 physically installed.

Targets are the exact native MLP2 function evaluated at the corresponding state:

$$
y_N=f_{\mathrm{MLP2}}(x_N),\qquad
y_C=f_{\mathrm{MLP2}}(x_C).
$$

The same token positions and documents appear in the two backgrounds, preventing a
document-composition confound. Train/dev/evaluation source documents remain disjoint.

## Frozen first-stage objective

Let (v_N,v_C) be the fit-only mean centered target energies per scalar for the two
backgrounds. The balanced loss is

$$
\mathcal L_{\mathrm{balanced}}
=\tfrac12\frac{\mathbb E\|\widehat f(x_N)-y_N\|^2}{v_N}
+\tfrac12\frac{\mathbb E\|\widehat f(x_C)-y_C\|^2}{v_C}.
$$

Each background receives one half of the objective regardless of row count or target
scale. Checkpoint selection minimizes the worse of the two normalized development
MSEs, breaking ties by their mean. This prevents a nominal improvement obtained by
sacrificing the C512 trajectory or the native trajectory.

Initialize from frozen FULL512 and use the same rank, parameterization, optimizer
family, batch size, and maximum 1,200-step budget as its parent. Publish separate
native/C512 learning curves and the paired state-shift statistics. Local loss has no
scientific decision authority.

## Physical evaluation

After the program bytes are frozen, open a fresh evaluation role and run:

1. native;
2. MLP0-C512 only;
3. frozen original MLP2-FULL512 only;
4. frozen original C512 + FULL512;
5. robust MLP2-R512 only;
6. C512 + robust MLP2-R512.

Use extra CE, teacher KL, centered-logit NRMSE, top-1 agreement, task accuracy, and
48/96/192-document prefixes. Recompute the factorial interaction for both MLP2
students with a paired 10,000-draw source-document bootstrap.

## Decisions fixed before data

The trajectory-shift hypothesis passes only if the robust student, relative to frozen
FULL512:

- reduces the C512 composition interaction by at least 50%;
- has interaction 95% CI wholly inside `[-0.005, +0.005]` nat;
- reduces combined-arm CE and KL by at least `0.005` nat each;
- increases standalone native-background CE and KL by no more than `0.005` nat;
- retains the same prefix-stability standard; and
- has a positive paired-bootstrap lower bound for combined CE improvement.

If development normalized MSE on either background exceeds the original FULL512
value after all 1,200 steps while still improving, report optimization failure rather
than rejecting the hypothesis.

If the balanced local fit passes local optimization but fails the physical criteria,
trajectory coverage alone is rejected. The next prospective grammar is the same-price
student trained with suffix-logit/Fisher consequence weighting. It is not licensed as
a retrospective rescue in this experiment.

This experiment cannot move the strict ledger without replication, OOD transport,
and a terminal extraction/removal use.

