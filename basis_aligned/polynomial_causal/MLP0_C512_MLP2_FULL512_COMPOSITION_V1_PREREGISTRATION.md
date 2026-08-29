# MLP0-C512 × MLP2-FULL512 composition v1 — preregistration

## Question

The frozen MLP0 `C512` program is nearly lossless by itself, and the frozen MLP2
`FULL512` program recovers 68.27% of the CE damage of deleting MLP2 on the native
trajectory. Do they remain useful when installed simultaneously, or does one depend
on live-model compensation that the other removes?

No parameter is fitted in this assay. Both programs, all arms, metrics, thresholds,
and parent hashes are frozen before fresh evaluation rows are opened.

## Physical arms

Run the full sequential model with native attention and all unmentioned native MLPs:

1. `NATIVE`: native MLP0 and native MLP2.
2. `C512`: frozen C512 at MLP0 and native MLP2.
3. `FULL512`: native MLP0 and frozen FULL512 at MLP2.
4. `BOTH`: frozen C512 at MLP0 and frozen FULL512 at MLP2.

C512 receives the exact native MLP0 product vector and replaces only `Down` plus its
folded intercept and native `Down_bias`. FULL512 receives the actual RMS-normalized
pre-MLP2 state produced by its arm. Native MLP0/MLP2 calls must be zero in the arms
that replace them.

## Frozen parents

- C512 program SHA-256:
  `3ecf43b485d343bc5413e817dbd4236e5ce6cdaa7a3e0e653214e812b84ce470`
- C512 fit receipt SHA-256:
  `79d0069864e9df521a99fc36531dd86c7ed31106f58f029d681fb1788a269f82`
- FULL512 bundle SHA-256:
  `d0ad8aedcfec5097e2791d64281f5cc4b644af450968456fc64dc7312123078e`
- FULL512 receipt SHA-256:
  `3578a68b4e8c20ea95f55a62cf9ff4e59e628bd69dbbad995f17a20f5265a7b2`

## Fresh role

Freeze a new registry-disjoint 192-document `EVALUATION` role, scored at token
positions 64 through 255. The paired source document is the inference unit. The
freezer may also create an unopened 192-document `TRAIN` sibling because it reuses a
tested two-role freezer; no model or optimizer may open that sibling in this assay.

## Measurements

For each arm and prefixes 48, 96, and 192, report candidate-minus-native CE,
native-to-candidate teacher KL, centered-logit NRMSE, native top-1 agreement, and
next-token accuracy.

Let (d_C,d_F,d_B) be the document-level CE changes for `C512`, `FULL512`, and
`BOTH`. The factorial interaction is

$$
I=d_B-d_C-d_F.
$$

Also report both conditional marginal changes:

$$
d_{F\mid C}=d_B-d_C,\qquad d_{C\mid F}=d_B-d_F.
$$

Use 10,000 deterministic source-document bootstrap draws for a two-sided 95% CI on
(I), and one-sided 95% upper bounds on `BOTH` dCE and KL.

## Decisions

`composition_compatible` requires, on 192 documents:

- `BOTH` dCE no greater than `FULL512` dCE plus 0.01 nat;
- `BOTH` KL no greater than `FULL512` KL plus 0.01 nat;
- absolute interaction at most 0.01 nat and its 95% CI wholly inside
  [-0.02, 0.02]; and
- dCE and KL each change by at most 0.01 between prefixes 96 and 192.

`positive_synergy` requires the upper endpoint of the interaction CI below zero.
`incompatibility` requires the lower endpoint above +0.01 nat. Anything else is
`interaction_inconclusive`.

This assay can establish in-distribution composability of two frozen physical
programs. It cannot certify either parent, attach semantic labels, establish OOD
transport, or move the strict ledger without independent replication and terminal
tests.

