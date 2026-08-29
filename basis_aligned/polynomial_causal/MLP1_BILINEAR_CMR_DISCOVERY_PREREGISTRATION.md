# MLP1 bilinear causal-mechanism-reduction discovery preregistration

Frozen before inspecting any outcome from this experiment on 2026-08-29 UTC.

## Question

On the already validated `C512` MLP0 trajectory, can MLP1 be made into an
actually smaller bilinear program by deleting product channels and folding each
deleted channel's fitted mean write into the MLP1 bias?  At equal executable
price, does a gauge-invariant causal-mechanism score select deletions better than
raw activation variance, an invariant weight-only norm, or random selection?

This is a discovery experiment on already exposed cached natural-text rows.  It
cannot promote a circuit, move the strict whole-model ledger, or establish OOD
authority.

## Frozen inputs and split

- Checkpoint revision: `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`.
- Checkpoint weights SHA-256:
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
- Cached-row file SHA-256:
  `cc8e1c3e468b7bc249e0cf8fc00640955ae17251c7f0c7640350f65a86202cac`.
- Cached-row tensor SHA-256:
  `625258ae1128823194fd27c94c241bd197dfd8daba77cfa2d1a0156ae1daaf8a`.
- `C512` binary SHA-256:
  `3ecf43b485d343bc5413e817dbd4236e5ce6cdaa7a3e0e653214e812b84ce470`.
- Shared-HOSVD result SHA-256:
  `8e2e27a7231472bce1389167414898c343ebf8ac2e1bcac6b78f220fe7b5801e`;
  its already selected canonical rank is 256.
- Rows 0--31 fit MLP1 product means and variances.  Rows 32--127 evaluate.
- Positions 64 onward are scored.  Copy cells use the frozen nearest-repeat
  policy from `discover_copy_source_edges.py`.
- Random control seed: 20260829.

The baseline (`BASE`) contains C512 at MLP0, native MLP1, and the selected
rank-256 shared-HOSVD replacement of the layer-8 H3/H4 copy-source gate.  Every
candidate changes only MLP1 relative to that baseline.  `ZERO` replaces the
whole MLP1 write with zero and supplies the local causal-effect floor.

## Programs and four selection rules

For MLP1 product channel $j$,

$$
a_j(x)=(L_jx)(R_jx),\qquad
y(x)=\sum_j D_{:j}a_j(x)+b.
$$

For each removal fraction $q\in\{0.25,0.50,0.75\}$, delete exactly the lowest
scoring $q\times4608$ channels under each rule:

1. **VAR**: $\operatorname{Var}(a_j)$.  This is deliberately a gauge-invalid
   control.
2. **MASS**: $\|L_j\|_2^2\|R_j\|_2^2\|D_{:j}\|_2^2$.  This is a
   gauge-invariant, weight-only tensor-mass control.
3. **CMR**: $\operatorname{Var}(a_j)\|D_{:j}\|_2^2$.  This is the diagonal
   local logit/write-distortion score.
4. **RAND**: a fixed random ordering, evaluated at the primary 50% budget.

If $S$ is deleted, compile its mean effect exactly into the bias:

$$
b'=b+D_{:S}\mathbb E[a_S].
$$

The candidate computes only retained rows of `Left` and `Right` and retained
columns of `Down`; it is not a full-width mask.  Its parameter price is the
number of retained values in those three matrices plus the unchanged bias.

## Measurements

For every arm and each of `all_scored`, `copy_positive`, `repeat_negative`, and
`nonrepeat`, record paired document-level CE change, standard error, teacher KL,
and top-1 accuracy.  Also record at the layer-8 gate:

- $R^2$ and cosine similarity of the rank-256 HOSVD latent relative to `BASE`;
- fraction of the `ZERO`-MLP1 latent error removed;
- copy-gate pattern $R^2$ where available.

At the 50% budget, measure the true joint local MLP1 write distortion

$$
J_S=\mathbb E\left\|\sum_{j\in S}D_{:j}(a_j-\bar a_j)\right\|_2^2
$$

and the diagonal estimate $A_S=\sum_{j\in S}s_j$.  Report
$(J_S-A_S)/J_S$; a large absolute value means channelwise ranking ignores
important covariance and should be replaced by a block/group method.

Finally apply independent exact product gauges with log-scales in $[-3,3]$.
Recompute rankings algebraically.  VAR should change; MASS and CMR must have
top-set Jaccard 1 up to numerical ties.

## Frozen primary gates

The primary comparison is 50% channel removal.

- **G1, useful causal simplification:** CMR50 recovers at least 90% of the
  `ZERO` MLP1 all-scored CE effect and has all-scored $\Delta\mathrm{CE}\leq0.02$.
- **G2, downstream composition:** CMR50 has layer-8 latent $R^2\geq0.95$ and
  copy-positive top-1 accuracy falls by at most 0.01 from `BASE`.
- **G3, selection value:** CMR50 has lower all-scored CE damage than VAR50,
  MASS50, and RAND50; its advantage over the best matched control must be at
  least 0.005 nat or two paired standard errors of their documentwise
  difference, whichever is smaller.
- **G4, diagonal adequacy:** CMR50 has
  $|(J_S-A_S)/J_S|\leq0.20$.
- **G5, gauge audit:** MASS50 and CMR50 Jaccard are at least 0.999 after the
  exact gauge, while VAR50 Jaccard is below 0.50.

The 25% and 75% points are descriptive curve points, not substitutes for the
primary gate.

## Interpretation fixed in advance

- If G1--G5 pass, the next experiment is a fresh-row replay plus interchange
  and margin-certificate evaluation of the frozen CMR50 program.
- If G1--G3 pass but G4 fails, diagonal CMR is only a screening rule; the next
  move is covariance-aware block selection or affine replacement.
- If G1/G2 pass but G3 fails, mean-folding is useful but CMR is not a validated
  simplicity definition; retain the best invariant selector and do not claim
  a CMR advantage.
- If G1 or G2 fails, prune native MLP1 product-channel deletion at 50% and move
  to response-conditioned multi-view factors or a lower removal budget.
- If G5 fails for MASS/CMR, the implementation is invalid and no model result
  is interpreted.

