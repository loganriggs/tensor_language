# The grand-combined arm: layer 0, fully codebooked

All three component codebooks active simultaneously in one forward pass (546M model):
**QK** = per-head 256-token-class factor tables · **OV** = per-head sparse dictionaries
(512 atoms × 16 signed coefficients/token) · **MLP-0** = self/cross blocks from 256-class
inputs (pair exact). Assignments/supports frozen from weights-only fits; 9.9M continuous
table values.

| stage | ΔCE |
|---|---|
| components alone (L2-fit): QK / OV / MLP | +0.008 / +0.055 / +0.166 |
| all three together, L2-fit | **+0.455** (sum of parts: 0.230 — superadditive) |
| jointly CE-trained, 1500 steps / 65k tokens | +0.194 (under-trained + data-limited) |
| jointly CE-trained, 4500 steps / 65k tokens | +1.624 (overfit: train CE fell to 1.1) |
| **jointly CE-trained, 4500 steps / 2.1M tokens** | **−0.019** |

Findings:
1. **Coarse components compound superadditively** through the bilinear structures
   (+0.455 vs 0.230 summed) — the same non-composition seen in head redundancy and the
   cross-block sides, now at the component level.
2. **Joint behavioral training fully repairs the composition** given adequate data:
   the fully-codebooked layer ends slightly BETTER than the original.
3. Training-protocol calibration (logged for the program): ~1M trainable table params
   generalize fine from 65k tokens; ~10M require ~2M tokens (the 4500/65k run memorized,
   train CE 1.1, held-out +1.62).

Honest DL framing: the 9.9M table values describe the layer's vocab-space computation
~30× more compactly than the folded tables they replace (and in interpretable objects:
token classes, atom combinations), but the raw weight parameterization of layer 0 is
itself only 24M params — the win is a STRUCTURED, readable description at zero behavioral
cost, not raw parameter count.

## The sqrd12 contrast column

Same treatment on the 162M squared-attention model (QK vq256 + OV sparse 512×16 — its
ReLU² MLP has no bilinear block structure, so MLP is untouched), same training protocol
(5.6M table floats, 4500 steps, 2.1M disjoint tokens):

| stage | bilin18 (546M) | sqrd12 (162M) |
|---|---|---|
| components alone, L2-fit | +0.008 (QK) / +0.055 (OV) | +0.116 (QK) / +0.221 (OV) |
| together, L2-fit | +0.455 (QK+OV+MLP; superadditive) | +0.275 (QK+OV; SUB-additive) |
| jointly CE-trained | **−0.019** | **+0.188** |

Three contrasts: (1) sqrd12's component errors compose sub-additively — the per-row
pattern normalization appears to absorb part of the joint error, where bilin18's
unnormalized product amplifies it; (2) CE training recovers only ~32% of sqrd12's L2
error vs >100% on bilin18; (3) the ~15× compressibility gap seen at the per-head level
(file 05) survives behavioral training — it is a property of the model, not of the L2
fitting stage. Fewer heads and no second branch means less redundancy for the codebook
to lean on. `../sqrd12_grand.py`, `../sqrd12_grand.json`.
