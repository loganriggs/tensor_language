# Tier-2 results: Logan's Elriggs models (layer-0 QK, ΔCE-audited MDL)

Models (verified from configs + state dicts + jacclust/tt_model.py):
**bilin18** = gpt2-bilinear-sqrd-attn-18l-9h-1152embd (546M; TWO QK branches, pattern =
(q1·k1)(q2·k2)/D² unnormalized). **sqrd12** = gpt2-sqrd-attn-12l-6h-768embd (162M; one
branch, pattern = (q·k/D)² row-normalized). Gates: reference forward ≡ tt_model (exact);
layer-0 factor folding exact to 1e-15. Frozen conventions per mdl_accounting.py; ΔCE
binding (Logan 2026-07-15); eval pile-10k T=512.

## CE gate (Logan: "verify 3–4")
| model | CE @256 | CE @512 | CE @1024 |
|---|---|---|---|
| bilin18 | 3.63 | **3.23** ✓ | 5.50 ✗ (per-position CE explodes past ~512: 3.4 → 10.9) |
| sqrd12 | — | **3.37** ✓ | 3.50 ✓ |

**Model property (flag for Logan):** bilin18's unnormalized score-product attention
degrades sharply beyond ~512 context (row mass grows with key count); all audits at T=512.

## bilin18 layer-0 joint frontier (884 MiB full)
| compression | DL ratio | joint ΔCE |
|---|---|---|
| all 9 heads → vq256 (256 token-classes/branch) | 6.1e-3 (165×) | **+0.0084** |
| H3,H6 → vq256, rest vq16 | 2.0e-3 (500×) | +0.0177 |
| all vq16 | 8.1e-4 (1240×) | +0.0420 |
| zero 7 "free" heads, keep H3,H6 exact | 0.223 | +0.534 (!) |

- Per-head marginals: 7/9 heads individually zeroable at |ΔCE| ≤ 0.011 — but marginals
  do NOT compose (+0.53 jointly): heads are individually expendable, collectively
  load-bearing. Coarse VQ (not zeroing) is the right compression for redundant heads.
- Pattern-MSE is useless as a behavioral predictor: vq16 points with pattern-MSE
  0.14–0.95 cost |ΔCE| ≤ 0.011; H3's vq16 IMPROVES CE (−0.011).
- The vq classes are readable token types: digits, punctuation, sentence-initial
  (In/It/We/This), uppercase splits, suffix morphology (ion/ter/ers), semantic nouns
  (people/government/police), determiners.

## sqrd12 layer-0 (single branch, 6 heads)
All-heads vq256: ΔCE **+0.116** at the same 6.1e-3 DL ratio — ~15× less compressible
than bilin18. No head zeroes for free (H3 ablation +0.356, yet svd16 ≈ free: a genuinely
low-rank, load-bearing head). Fewer heads → less redundancy; and row normalization makes
patterns sensitive to fine score differences.

![frontier](fig_tier2_frontier.png)

Caveats: single eval distribution; VQ codebooks fit under factor-L2, not CE-trained (the
basis_aligned e7 lesson suggests CE-in-the-loop would push the frontier further); layer-0
only (path-folded deeper layers = Tier 3); ε reported as curve points, not one number.
