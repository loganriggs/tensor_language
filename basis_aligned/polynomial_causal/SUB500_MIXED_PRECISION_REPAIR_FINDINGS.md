# Rung 436 findings — the +29,968-byte repair does not remove the position-level failure

Date: 2026-09-01T21:47Z

## Registered verdict

Prediction A holds: the shipped artifact has exactly 495,847,230 scalar values and 991,724,428 bytes. Source-native
tensors use the registered BF16 treatment, all 440 QK64 factor pairs are FP16, all 26,544,384 generated MLP0/4
values are BF16, and the exact 14,984-value four-tensor/no-dense MLP16 program is FP32 and tensor-identical to the
rung392 source. Every hook and role check is live.

Predictions C and D hold. Census damage is `+.05296855` nat with 17 of 62 certificates; additive composition tax is
`1.03409`, normalized effect-vector cosine is `.999959`, and certificate-count difference is zero. On the untouched
WikiText-103 segment `[470824,501664)`, full original-native mean/p95/max damage is
`.065277/.111963/.161361` nat; conditional fresh maximum is `.0125`.

Prediction B fails. Mean per-position CEV difference from the adopted fp32-source rung392 parent is `.0067535`,
inside the `.010` bar, but maximum difference is **`.1164665`**, above the registered `.050` limit and above half
of rung414's `.1176414`. The strong null is false because aggregate behavior and identity remain good, but full pass
was required. Rung437 is not licensed and was not executed. No mixed-precision sub-500M adoption is claimed.

## What the failed repair identifies

The pre-registered diagnosis was wrong. Restoring the quadratic MLP16 coefficients from BF16 to FP32 changes the
rung414 CEV by mean/p95/max `.001239/.003896/.022341`. That coefficient-cast difference accounts for only `3.91%`
of rung414's total squared deviation from rung392. Its cosine with the deviation that remains under FP32 MLP16 is
`-.2506`, so it partly cancels rather than causes the larger error. Removing coefficient rounding slightly worsens
overall RMSE (`.009301 -> .009590`) while reducing the single largest deviation by only `.001175`.

The corrected statement is therefore:

> the degree-two MLP16 program amplifies upstream source-BF16, QK-FP16, and generated-MLP-BF16 perturbations;
> rounding the quadratic program's own 14,984 coefficients is not the main source of the tail deviation.

Rung412 showed that the same upstream two-byte families without the MLP16 replacement differ from their parent by
only `.02416` at the worst position. Their interaction with the nonlinear MLP16 surrogate creates the `.116` tail.
This is a composition/conditioning failure, not a storage-accounting failure.

## Decision

Preserve rung414 and rung436 as near-misses. Do not sweep FP16, selected FP32 tensors, or another position threshold.
The adopted physical byte frontier remains the 511,758,646-scalar / 1,023,517,292-byte 43-certificate tier. The
495,847,230-scalar rung392/393 program remains adopted only under its honest 1,867,449,228-byte source-format bill.

For future simplicity learning, this is a useful adversarial case: a 29,968-byte local repair looks compelling under
parameter count, mean CE, certificate count, and aggregate composition, but fails a sealed position-level consequence
because nonlinear composition amplifies perturbations elsewhere. A learned simplicity objective must retain tail and
composition consequences rather than reward local precision or bytes alone.
