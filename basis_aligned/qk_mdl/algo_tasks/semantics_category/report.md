# Semantics verification: the six CATEGORY-CODE directions at block 3 of bilin18

**Name under test:** "d_k carries evidence that the NEXT token is category k" (k in
subword / punct / capital / digit / funcword / other), for the six decision axes of a ridge
probe fit on the residual after block 3.

**Verdict in one line:** the directions are verified as causal *write-channels* — adding
alpha·d_k moves next-token probability mass into category k, dose-dependently, diagonally,
and beyond random-direction controls — but they are **falsified as a load-bearing code**:
projecting the probe subspace out of the block-3 residual costs the model essentially
nothing (dCE +0.0003 ± 0.0003), and the damage that does occur from the raw 6-dim span is
*exactly* category-proportional, not category-targeted. The block-3 code is also rotated
away by block 8 (probe transfer decays to near-majority while freshly-fit accuracy rises).
The honest name is therefore: **"d_k is a steering dial for the model's next-token
category prior; the natural category code is redundantly distributed and does not live in
this 6-dim slice."**

Setup: probe and all directions fit on exploration data (cooc rows 0-2400); every causal
number below is from the held-back audit set (FineWeb rows 448:600, 152 rows x 127
next-token predictions = 19,304 paired tokens; T=128). dCE = per-token paired delta;
quoted +- is the row-clustered SE (per-token SEs in the JSONs). Baseline held-back CE
3.4946 = categoryCE 0.6744 + withinCE 2.8202 (exact split). Model forward is the verified
no-softmax bilinear forward (unnormalized (q1.k1)(q2.k2) pattern).

## 1. Probe and directions (s1_probe.json, probe_blk3.pt)

Ridge to one-hot on the block-3 residual, 57,150 exploration tokens; val accuracy **0.598**
vs majority 0.441 (matches the prior 0.60 at this depth). Per-category recall: subword
0.70, other 0.81, punct 0.45, digit 0.37, funcword 0.36, capital 0.34. Decision axes
d_k = probe column minus column-mean, unit-normalized. Geometry: pairwise cosines are
small-to-moderate (max |cos| 0.14-0.44; funcword~other -0.44); the six centered axes span
exactly 5 dims (they sum to zero), and even the six *raw* columns are effectively rank 5
(6th singular value 30x below the 1st). "Six directions" is really a 5-dim decision
subspace plus a shared-mean component.

Median block-3 residual norm r_med = 32,151; all steering strengths alpha below are in
units of r_med (a unit direction times alpha*r_med).

## 2. Steering gate — 6x6 dose-response (s2_steering.json)

Add alpha*d_k to the block-3 residual at all positions; measure mean change in predicted
probability mass of each category (columns) on all held-back positions.

alpha = +0.50 (collateral dCE +0.05 to +0.09 per token):

| steer \ mass | subword | punct | capital | digit | funcword | other | dCE +- SE |
|---|---|---|---|---|---|---|---|
| subword  | **+.0044** | +.0012 | +.0009 | +.0001 | -.0047 | -.0019 | +0.077+-0.003 |
| punct    | -.0015 | +.0013 | +.0023 | +.0013 | -.0025 | -.0009 | +0.059+-0.003 |
| capital  | -.0009 | +.0018 | **+.0121** | +.0007 | -.0016 | -.0122 | +0.059+-0.003 |
| digit    | +.0001 | +.0021 | +.0009 | **+.0031** | -.0047 | -.0015 | +0.050+-0.003 |
| funcword | -.0014 | -.0004 | -.0002 | -.0006 | **+.0100** | -.0074 | +0.086+-0.004 |
| other    | -.0012 | -.0030 | -.0020 | -.0003 | -.0030 | **+.0094** | +0.064+-0.003 |

alpha = +1.00 (collateral dCE +0.42 to +0.74):

| steer \ mass | subword | punct | capital | digit | funcword | other | dCE +- SE |
|---|---|---|---|---|---|---|---|
| subword  | **+.0122** | +.0017 | +.0092 | -.0001 | -.0155 | -.0075 | +0.742+-0.013 |
| punct    | -.0042 | **+.0191** | +.0033 | +.0028 | -.0157 | -.0052 | +0.421+-0.009 |
| capital  | -.0039 | +.0065 | **+.0274** | +.0006 | -.0061 | -.0245 | +0.427+-0.010 |
| digit    | -.0010 | -.0014 | +.0053 | **+.0095** | -.0090 | -.0034 | +0.569+-0.012 |
| funcword | -.0056 | -.0036 | +.0049 | -.0013 | **+.0253** | -.0197 | +0.609+-0.012 |
| other    | -.0046 | -.0154 | -.0043 | -.0008 | -.0046 | **+.0296** | +0.599+-0.013 |

- **Diagonal dominance:** the diagonal is the largest entry in 11/12 named runs at
  |alpha| in {0.5, 1.0} (exception: punct at +0.5 is rank 2 behind a +0.0023 capital leak;
  rank 1 at 1.0). Negative alpha flips the sign of the diagonal in 12 of 12 runs at
  |alpha| <= 1.
- **Monotone:** the diagonal increases monotonically in alpha across
  {-1, -0.5, -0.25, +0.25, +0.5, +1} for all six categories. Beyond |alpha| = 2 the
  relationship breaks (capital at -2 gives +0.098 — the model is broken there, see section 6).
- **Random-direction control** (6 random unit dirs, same alphas): produces *comparable
  collateral dCE* (+0.068 to +0.096 at 0.5; +0.50 to +0.65 at 1.0) but max |d_mass| of only
  0.0017-0.0057 at 0.5 and 0.0045-0.0101 at 1.0, with no reproducible diagonal structure.
  Named diagonals exceed the largest random-control entry for 5/6 categories at alpha 1.0;
  digit (+0.0095) is only at the top of the random range — digit is the weakest verified dial.
- Reading the off-diagonals: steering any non-funcword category drains funcword/other
  (the two highest-baseline-mass categories) — mass conservation, not cross-category
  confusion; the one real leak is subword->capital (+0.0092 at alpha 1).

**Gate: PASS at |alpha| <= 1 for 5/6 directions, digit marginal.** Note the honest cost
framing: the collateral dCE is what *any* equal-norm perturbation costs; the
category-specific effect rides on top of a generic degradation, and at alpha 1.0 that
generic degradation (+0.4-0.7 nats) is an order of magnitude larger in CE terms than the
category mass moved.

## 3. Subspace ablation — the load-bearing test (s3_subspace.json)

Project the probe subspace out of the block-3 residual at all positions; exact split
CE = categoryCE + withinCE. Prediction of the name: categoryCE takes the hit
disproportionately (vs the baseline share categoryCE/withinCE = 0.674/2.820 = 0.239, and
vs random 6-dim subspaces).

| ablation | dCE | d_categoryCE | d_withinCE | ratio cat/within |
|---|---|---|---|---|
| probe span, raw 6-dim | +0.0363+-0.0024 | +0.0070+-0.0009 | +0.0293+-0.0021 | **0.239** |
| probe span, centered 5-dim | +0.0003+-0.0003 | +0.0002+-0.0001 | +0.0000+-0.0002 | 6.0 |
| random 6-dim (mean of 5 seeds) | +0.0005 | +0.0001 | +0.0004 | 0.36 |

- The **centered 5-dim decision subspace — the actual code axes — can be deleted almost
  for free** (dCE +0.0003, within 1 SE of random-subspace cost). The tiny damage it does
  cause is category-shaped (ratio 6.0, d_categoryCE +0.0002+-0.0001, ~2 SE), so the *sign*
  of the direction-level prediction is right, but the magnitude is ~0.0002 nats —
  negligible against baseline categoryCE of 0.674.
- The raw 6-dim span costs more (+0.036) but its ratio is 0.239 — **numerically identical
  to the baseline category/within split**, i.e. perfectly category-neutral damage. That
  extra damage comes from the shared-mean (6th) component, which behaves like a generic
  residual direction, not like category code.
- Consistency check (s5): the natural block-3 residual puts a median 6.3% of its norm in
  the centered 5-dim subspace — *slightly less* than the 6.6% a random 5-dim subspace
  would get. The code amplitude in the natural residual is tiny.

**The direction-level disproportionality claim fails in any meaningful magnitude.** This
extends the prior layer-level falsification (MLP0-3 ablation ratio 0.27 vs control 0.36)
down to the direction level: the model does not *rely* on this subspace to carry category
information — the linearly-decodable code is a redundant, low-amplitude readout, and the
downstream computation reads category information from elsewhere (or re-derives it).

## 4. Dial demo — funcword vs punct (s4_dial.json, s7_addenda.json)

1,349 held-back positions are ambiguous (funcword and punct are the top-2 predicted
category masses, both >= 0.15). Flip rate of the model's category argmax at those positions:

| condition (alpha=1.0) | funcword->punct | punct->funcword |
|---|---|---|
| steer +d_punct | **0.438** | 0.082 |
| steer +d_funcword | 0.183 | **0.379** |
| zero | 0.000 | 0.000 |
| random dir | 0.319 | 0.160 |
| placebo (+d_digit) | 0.315 | 0.155 |

Named steering flips in the named direction above both controls (punct: 0.438 vs 0.32;
funcword: 0.379 vs 0.16) and suppresses the reverse flip (0.082 / 0.183 vs control 0.16 /
0.32) — the *asymmetry* is the real signature: control conditions flip ~2:1 toward punct
regardless of identity; named steering sets the direction of the asymmetry. Example flips
(steer +d_punct): "...No. 21 by Scout.com" fw/punct 0.52/0.46 -> 0.13/0.81 (actual next
"."); "...your favorite sit-in restaurant, and item" 0.65/0.19 -> 0.29/0.55 (actual "?").

**Honest limitation:** at alpha 0.5 (where collateral is 7x lower) the named advantage at
these ambiguous positions disappears entirely (named flips 0.08-0.10 vs placebo/random
0.04-0.16; steering +d_punct even *lowers* mean punct mass at ambiguous positions by
-0.010). Ambiguous positions are fragile to any perturbation; the dial only separates
from placebo at strengths that already cost ~0.5 nats globally. The dial works as a
*population-level prior shift*, not as a precise per-position switch.

## 5. Persistence (s5_persistence.json)

Block-3-fit probe applied to later residuals (held-back set; majority = 0.445):

| depth | transfer (raw) | transfer (RMS-matched) | fresh-fit probe | principal cosines vs blk3 subspace |
|---|---|---|---|---|
| blk3 | 0.595 | 0.595 | 0.595 | 1.0 (self) |
| blk4 | 0.570 | 0.572 | 0.599 | 0.79-0.70 |
| blk8 | 0.456 | 0.331 | 0.646 | 0.57-0.39 |
| blk12 | 0.463 | 0.462 | 0.660 | 0.53-0.35 |
| blk16 | 0.467 | 0.482 | 0.663 | 0.36-0.14 |

The block-3 code is **consumed/rewritten between blocks 4 and 8**: transfer accuracy
collapses to within ~2 points of majority while fresh-probe accuracy *rises* to 0.65+.
The category subspace rotates steadily away (principal cosines 0.7 -> 0.14 by blk16).
Category information is not destroyed — it is re-encoded in new directions with growing
strength toward the readout. The block-3 directions are an early, transient basis for it.

## 6. Red-team (s6_redteam.json, s7_addenda.json)

- **"Is it just a few high-frequency tokens?"** Top-5 share of the positive
  within-category probability gain at alpha 1.0: subword 9%, capital 3%, other 5%, digit
  14% — genuinely distributed (219, 645, 434, 30 tokens to reach half the gain). Funcword
  60% and punct 75% *are* concentrated — but their *baseline* category mass is equally
  concentrated (top-5 share 53% and 86%): these categories are inherently a handful of
  tokens. In every case the gain profile tracks the natural within-category distribution,
  i.e. steering acts like a category-level prior multiplier, not a top-5-token hack.
- **"Is it really 6-dimensional?"** No. Centered directions are exactly rank 5; the
  steering Jacobian (d mass / d alpha from the +-0.25 runs) has singular values
  [0.032, 0.021, 0.009, 0.008, 0.0006, 0.0002] — participation-ratio effective rank
  **3.1**, i.e. roughly four usable steering axes; the digit axis is the weakest dial
  (smallest diagonal, only at the top of the random-control range).
- **Breaking alphas:** |alpha| = 2 already costs +1.8-2.4 nats (matrix entries become
  sign-inconsistent, e.g. capital at -2); +4 costs ~+4 nats; +16 saturates near +6 nats
  (soft-capped logits). The valid dial range is |alpha| <~ 1, and the clean-diagonal range
  is |alpha| ~ 0.5-1.0.

## Files

- `common.py` — model forward (block-3 edit hook), category defs, paired stats
- `s1_fit_probe.py` / `s1_probe.json` / `probe_blk3.pt` — probe + directions
- `s2_steering_matrix.py` / `s2_steering.json` — 6x6 dose-response + random controls
- `s3_subspace_ablation.py` / `s3_subspace.json` — subspace ablation, CE split
- `s4_dial_demo.py` / `s4_dial.json` — funcword/punct dial at alpha 1.0 + examples
- `s5_persistence.py` / `s5_persistence.json` — probe transfer, principal angles
- `s6_redteam.py` / `s6_redteam.json` — concentration, effective rank, breaking alphas
- `s7_addenda.py` / `s7_addenda.json` — dial at alpha 0.5, baseline concentration control
