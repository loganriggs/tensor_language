# Rung 414 preregistration — physical all-two-byte sub-500M tier

## Decision

Rungs392/393 adopted a 495,847,230-value composite containing QK64, MLP0/4 p768, and the tiny layer16 quadratic
program, but its honest source-format price is 1,867,449,228 bytes. Rungs412/413 show that the QK and generated
MLP tensors survive physical two-byte storage. This rung asks whether the full lower-fidelity composite survives when
its layer16 program is also physically BF16.

BF16/fp16 are storage encodings, not interpretability or scalar compression. Runtime hooks explicitly convert to
float32 before arithmetic. No latency or low-precision-kernel claim is in scope.

## Frozen object and price

- Source-native tensors: frozen source-aware BF16 treatment.
- QK: rank64,440 factor pairs, all retained factors fp16.
- MLP0/4: p768 programs, all five retained tensors per layer BF16.
- MLP16: output directions [4,1152], form vectors [4,2,1152], form values [4,2], and constant [1152], all BF16;
  exactly14,984 values, no dense quadratic forms.
- Whole program: exactly495,847,230 values and991,694,460 bytes, or .92359 GiB.

## Frozen populations and predictions

Use the unchanged census/certificate population, additive comparator, and fresh windows. The full native-relative
shifted check uses untouched WikiText-103 [439984,470824), after the prior segment ending at439984. FINAL stays closed.

- pred_a: every source/QK/MLP0/4/MLP16 dtype, shape, count, no-dense, live-hook, fit, selection, and byte identity
  holds exactly.
- pred_b: census damage at most .070 nat, at least10 certificates, and mean/max absolute per-position CE change from
  the saved fp32-source rung392 artifact at most .010/.100.
- pred_c: census-to-additive damage ratio in [.90,1.35], normalized certificate-vector cosine at least .95, and
  certificate-count difference at most7.
- pred_d: full original-native shifted mean/p95/max at most .075/.140/.220 and conditional fresh maximum at most .040.

Strong null: physical identity failure, census at least .10, at most5 certificates, shifted mean at least .10, or an
inert layer16 hook.

A complete pass licenses one original-native signed gate for this exact physical object. No dtype, layer, rank,
population, or threshold tuning.
