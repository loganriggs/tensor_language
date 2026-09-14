# Paired counterfactual transport through O framing sources

Use all 24 frozen British/American pairs from the fresh panel. First run every
row natively and capture the normalized block-9 attention input plus inherited
first values. Then rerun each row while replacing only its framing source
positions with the paired donor tensors inside the exact O source computation.
The recipient final query, other O sources, S/R branches, all other heads and
the native suffix stay unchanged. This is 48 native captures plus 48 swaps,
96 body forwards under a 180-second cap.

- `pred_a`: native target/work-jobs scores replay the frozen fresh artifact
  within `1e-5`; exact original O source recomposition is at most `1e-10`;
  tensors outside the framing mask are unchanged exactly; all values are finite
  and exactly 96 forwards execute.
- `pred_b`: native target cue is positive for at least 10/12 pairs and the
  paired swap target-effect RMS is at least `1e-5` in each template.
- `pred_c`: the bidirectional swap paired-cue effect differs from twice the
  frozen framing-removal paired-cue effect by at most 35%, with cosine at least
  `.8`, in each template.
- `pred_d`: each of cat/dog, red/blue, Monday/Tuesday and apple/orange swap
  effect RMS is at most 50% of target swap-effect RMS in each template.

The factor two is frozen before execution: under an additive exchanged source
write, swapping both members doubles the paired effect of removing each member's
own write. Failure rejects that transport law without changing the removal
receipts. This is conditional interchange at head9.8, not upstream necessity,
corpus OOD, a universal semantic mask, static compression, or quantization.

