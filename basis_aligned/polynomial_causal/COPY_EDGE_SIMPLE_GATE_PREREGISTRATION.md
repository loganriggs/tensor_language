# Simple input-side gates for the exact copy edge

Status: **exploratory; frozen before model outcomes from this runner**.

The preceding scalar test showed that the L8 H3/H4 payload is almost completely the
shared $\lambda_8v_1$ token code, but one unconditional scalar per head is not a
faithful gate.  This experiment asks whether either of two very small input-side
signals explains the missing variation.

Fit documents are cached rows 1--32.  Evaluation documents are cached rows 33--128.
Both belong to an exposed selection role, so passing results remain exploratory.

The four unchanged baseline arms are reused from
`copy_edge_constant_scalar_results.json`, SHA-256
`3da06d79c0d28bbb6f4d13082aa8c0dcc1bd3315a5ef9ec485e347136774603f`, rather
than spending GPU time recomputing identical forwards.  The new runner verifies the
hash, split, checkpoint, and per-document schema before using them.  Only the three
new gate arms receive new model forwards.

## Exact source policy and target scalar

At destination $p$, let $j(p)$ be the nearest earlier equal-token position within
128 tokens and $k(p)=j(p)+1$.  The target to approximate is the native L8 pattern
pair

$$
y(p)=\left(a_{8,3}(p,k(p)),a_{8,4}(p,k(p))\right).
$$

The replacement always writes the shared broadcast payload from $k(p)$ through the
fixed L8 H3/H4 projection slices.  Only the two pattern scalars vary.

## Gate A: reused weights-computed matcher score

The older copy stand-in used L2 H5 and L3 H8 as matchers.  For each matcher, run its
owned query/key projection and rotary pipeline on the normalized token embedding
$e(x_p)$, without any contextual residual stream.  At the exact equal-token pair
$(p,j(p))$, compute its bilinear attention-pattern product.  The one-dimensional
match score is

$$
s(p)=-\left(a^{\mathrm{static}}_{2,5}(p,j(p))
             +a^{\mathrm{static}}_{3,8}(p,j(p))\right).
$$

The minus sign follows the older finding that same-token matcher products are
negative.  Fit two affine maps by ordinary least squares on every input-eligible fit
position:

$$
\widehat y_h(p)=\alpha_hs(p)+\beta_h.
$$

This stores four fitted scalars and reuses the embedding table plus the two matcher
heads' fixed projection/rotary programs.  It does not use evaluation targets or live
L2/L3 activations.

## Gate B: distance-binned constants

Let $d(p)=p-j(p)$.  Fit one two-scalar mean for each frozen bin

$$
[1,8],\ [9,32],\ [33,64],\ [65,128].
$$

This stores eight scalars plus four interval boundaries and uses only token equality
and positions.  It is cheaper than Gate A but less expressive.

## Frozen arms

All replacements apply at every input-eligible destination, including before the
scoring window.  Scoring remains positions 64--255.

1. `native`.
2. `edge_removed`: exact mixed-value successor edge deleted.
3. `native_pattern_broadcast`: native scalar, shared payload; the value-side ceiling.
4. `eligible_constant_broadcast`: two all-repeat constants from the prior experiment.
5. `static_match_affine_broadcast`: Gate A.
6. `static_match_shift_control`: Gate A coefficients applied to a deterministic
   one-position circular shift of the score within each document.
7. `distance_bin_broadcast`: Gate B.

## Frozen metrics and gates

Copy recovery relative to edge deletion is

$$
R_r=1-\frac{\Delta\mathrm{CE}_r}{
\Delta\mathrm{CE}_{\mathrm{edge\ removed}}}.
$$

Report CE, native-to-arm KL, top-1 accuracy, document mean/SE, fit/evaluation scalar
$R^2$, all coefficients, and an executable-price inventory.

- G1, matcher gate useful: Gate A copy recovery $\ge0.70$.
- G2, improvement over unconditional constants: Gate A recovery exceeds the constant
  arm by at least 0.20.
- G3, matched association: Gate A recovery exceeds shifted control by at least 0.20.
- G4, selectivity: Gate A repeat-negative and nonrepeat absolute $\Delta$CE are each
  at most `0.02` nat and at most 25% of edge-deletion copy damage.
- G5, cheap alternative: distance-bin recovery $\ge0.60$ and lies within 0.10 of Gate
  A recovery.

G5 is independent of G1--G4; failure of distance bins does not reject the matcher.
If Gate A fails G1 or G2, stop treating the historical static matcher score as a
direct explanation of the L8 gate.  If it passes G1--G4, the next experiment composes
it with the upstream matcher replacement on fresh rows.  No new feature search is
allowed after evaluation outcomes are seen in this runner.
