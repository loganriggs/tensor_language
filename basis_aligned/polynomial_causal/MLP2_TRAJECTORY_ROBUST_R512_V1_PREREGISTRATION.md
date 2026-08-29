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

For each frozen training document, capture paired pre-MLP2 states on the same token
positions and common document support:

- (x_N): the native trajectory;
- (x_C): the trajectory with frozen MLP0-C512 physically installed.

Here (x) is exactly the post-attention RMS-normalized input passed to MLP2, and
(y) is the complete native MLP2 residual write, including its bias. Targets are
the exact native MLP2 function evaluated at the corresponding state:

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
scale. Predictions, targets, and loss are float32; target-energy aggregation is
float64. A checkpoint is eligible only when both normalized development MSEs are no
more than 2% above their step-0 FULL512 values. Among eligible checkpoints, selection
minimizes the worse background, breaking ties by their mean. This prevents a nominal
minimax improvement obtained by sacrificing one trajectory.

Two programs initialize from identical frozen FULL512 bytes and use the same rank,
parameterization, Adam optimizer, learning rate, 1,024-token total batch, maximum
1,200-step budget, development cadence, and checkpoint opportunities:

- `CONTINUE512` duplicates the native-background half-loss, using two independently
  sampled 512-token native batches per step;
- `ROBUST512` uses one 512-token native batch and one paired C512 batch per step.

This matched continuation control separates trajectory exposure from the extra data
and optimization that the still-improving frozen FULL512 parent did not receive.
Publish separate native/C512 learning curves and paired state-shift statistics. Local
loss has no scientific decision authority.

The exact unopened row parent is
`mlp0_c512_mlp2_full512_composition_v2_rows_receipt.json`, TRAIN file SHA256
`efb7daed052009df10e1619f90c5648977144a8f549304c3bc56dbbd0f2130d8`, with
documents 0–159 for fit, 160–191 for development, and positions 64–255. Evaluation
must use a later registry-disjoint source-document family.

## Physical evaluation

After both program byte families are frozen, open a fresh evaluation role and run:

1. native;
2. MLP0-C512 only;
3. frozen original MLP2-FULL512 only;
4. frozen original C512 + FULL512;
5. continued-native MLP2-CONTINUE512 only;
6. C512 + continued-native MLP2-CONTINUE512;
7. robust MLP2-ROBUST512 only;
8. C512 + robust MLP2-ROBUST512.

Use extra CE, teacher KL, centered-logit NRMSE, top-1 agreement, task accuracy, and
48/96/192-document prefixes. For each MLP2 program (P), compute the same-wave
factorial interaction

$$
I_P=CE(C+P)-CE(C+\mathrm{native2})-CE(N+P)+CE(N+\mathrm{native2}),
$$

where (C) means C512 is installed and (N) means native MLP0. Run one paired
10,000-draw source-document bootstrap over all registered contrasts. Historical
`0.008739` is context only and is never substituted for the fresh old-FULL arm.

## Decisions fixed before data

Use Bonferroni simultaneous two-sided 95% percentile intervals across eight
registered document-level contrasts (per-contrast quantiles 0.003125 and 0.996875).
The narrow trajectory-exposure hypothesis passes only if:

- fresh frozen-FULL interaction has a lower bound above zero;
- the lower bound of (0.5|I_{FULL}|-|I_{ROBUST}|) is above zero;
- the simultaneous upper bound of (|I_{ROBUST}|) is at most `0.005` nat;
- ROBUST combined-arm CE and KL improve over frozen FULL by at least `0.005` nat
  each at the lower bound;
- ROBUST standalone CE and KL are noninferior to frozen FULL within `0.005` nat at
  the lower bound;
- ROBUST combined-arm CE improves over CONTINUE by a positive lower bound;
- the same prefix-stability standard passes; and
- all call, byte-price, precision, and no-native-MLP2 controls pass.

The eighth bootstrap contrast is ROBUST combined-arm KL improvement over CONTINUE;
it is reported and simultaneously bounded but is diagnostic rather than an additional
pass gate.

Report `optimization_inconclusive` rather than a scientific failure if, after all
1,200 steps, the best worst-background NRMSE exceeds `0.25`, the last four development
points are strictly decreasing, and the step-1200 worst loss improves by at least 1%
relative to step 1100. This replaces any visual “still improving” call.

If the balanced local fit passes local optimization but fails the physical criteria,
paired trajectory exposure under this optimizer is rejected. Even a pass supports
only “paired trajectory exposure, beyond matched continued training, reduces
composition brittleness”; it does not prove covariate shift uniquely caused the whole
interaction. The next prospective grammar is the same-price student trained with
suffix-logit/Fisher consequence weighting. It is not licensed as a retrospective
rescue in this experiment.

This experiment cannot move the strict ledger without replication, OOD transport,
and a terminal extraction/removal use.

The fit-only implementation is
`train_mlp2_trajectory_robust_r512_v1.py`. It is licensed to open only the previously
unopened `TRAIN` role from the MLP0-C512 × MLP2-FULL512 composition transaction. It
must freeze the candidate bundle and publish a receipt with
`evaluation_opened=false`. A separate source-closed physical evaluator must freeze
new registry-fresh evaluation rows; neither the earlier composition evaluation nor
the parent FULL512 evaluation may be reused for scientific decisions.
