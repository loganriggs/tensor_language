# Rung 412 preregistration — physical BF16 derived-MLP byte-frontier repair

## Decision

Rungs 368/369 established the semantic behavior of the QK64 + MLP0/4-p768 program after global source-aware BF16
rounding and fp16 QK-factor storage. A shipped-object audit found that the two generated MLP programs were still
float32. This rung asks whether storing those generated tensors in BF16 preserves the frozen 43-certificate tier and
therefore restores the literal all-two-byte bill.

This is storage encoding, not an interpretability or scalar-compression claim. Runtime computation remains fp32 after
explicit dequantization by the existing hook.

## Frozen object and price

- Identical source model, source-aware global BF16 rounding, QK rank 64, 440 QK maps, fp16 QK factors.
- Identical selected MLP layers {0,4}, fit rows [24,48), and p768 reduced-rank programs.
- Cast each shipped MLP program's encoder, left, right, down, and bias tensor to BF16 on CPU before it is retained by
  the hook.
- Required stored shapes per layer: [768,1152], two [4608,768], [1152,4608], and [1152].
- Exact two-layer generated-program count: 26,544,384 values = 53,088,768 bytes.
- Exact whole-program price: 511,758,646 scalar values = 1,023,517,292 bytes.
- The old executed mixed-dtype artifact is priced as
  2*(511,758,646-26,544,384) + 4*26,544,384 = 1,076,606,060 bytes.

## Frozen populations and predictions

The census, certificates, and eight fresh windows remain unchanged. The shifted check uses the untouched WikiText-103
segment [409144,439984), disjoint from the prior [378304,409144) composition segment. FINAL remains unopened.

- pred_a: every source/QK/MLP identity holds; all generated MLP tensors are BF16; QK factors are fp16; exact tensor
  counts and the 1,023,517,292-byte bill hold.
- pred_b: census CE damage is at most .015 nat and at least 43 of 62 circuit certificates remain valid.
- pred_c: shifted mean/p95/max are at most .025/.060/.120 nat and maximum fresh-window damage is at most .030.
- pred_d: against the saved mixed-dtype rung-368 per-position CE vector, mean absolute change is at most .003 nat and
  maximum absolute change is at most .050.

Strong null: any physical identity fails, census damage is at least .025, or certificates are at most 35.

A complete pass with the null false licenses one original-native signed intervention gate for this exact physical
artifact. It does not by itself restore adoption and does not license dtype, layer, rank, or threshold tuning.
