# Three-hourly mathematical review — 2026-09-03 16:30 UTC (Claude)

Sign convention §2135 throughout: CE numbers are CE ADDED ABOVE THE REAL MODEL, LOWER = better. State read from disk: ledger
§2691-§2695, board to 16:25Z, `mlp_final_blocks_low_rank_surrogate_probe_results.json` (§2694), Codex's 15:30 review (causal
abstraction / DAS framing, R540 shortcut analysis), running: `site_write_pca_truncation_ce_map_probe` (CPU, lands ~16:55);
queued: R549 downstream atlas (GPU), `mlp_final_blocks_quadratic_form_rank_probe` (CPU, registered 16:06).

## The fact that needs mathematics

§2694 produced a clean contradiction between two ranks of the same object. The centred covariance of MLP17's write on real text
has effective rank 6.3 (five directions carry 90% of the variance, replicated held-out), yet truncating the write to its top-8
variance directions adds .083 nat and top-64 still adds .043; the mean write alone (k = 0) costs .354. MLP16: eff rank 9.6,
k = 8 adds .036. And the "entropy dial" reading of the dominant direction was falsified (|Spearman| .16-.20, null holds).
Variance-rank and function-rank disagree by an order of magnitude in exactly the two blocks whose writes are enormous
(mean-write norms 25,670 and 43,355; mean energies 9.7e8 / 3.6e9, §2692). Any mathematics worth adopting must explain this,
predict the ladder, and say which directions of a write matter *for the loss* — not for reconstruction.

## Move 1 (execute now): the RMSNorm scale-gauge quotient and the pulled-back Fisher metric

**Object.** The final read-out is `logits = 30 tanh(W_U rms(x_18) / 30)`, `rms(x) = x / sqrt(mean x^2)`. The map x -> rms(x) is
invariant under the scale group R_+ acting on the final residual: the loss depends on x_18 only through its ray. The write
w_17 = MLP17(x) enters as x_18 = x_17' + w_17 + b. Therefore any component of w_17 along x_18 itself (the *radial* component) is
annihilated by the norm to first order: d rms(x)[v] = (v - x (x.v)/|x|^2) / rms-scale — the Jacobian of rms is the projector onto
the tangent space of the sphere, divided by |x|/sqrt(D).

**Theorem / operational definition.** Let J_t = d logits_t / d w_t (per position; for MLP17 it is exact and local:
J_t = 30 sech^2(.) W_U P_perp(x_t) sqrt(D)/|x_t|). The exact Hessian of -log p(y) in logit space is diag(p) - p p^T *independent
of y*, so the second-order expansion of the empirical CE under a write perturbation delta_t is

    CE_added  =  mean_t [ g_t . delta_t  +  1/2 delta_t^T G_t delta_t ]  +  O(|delta|^3),
    G_t = J_t^T (diag p_t - p_t p_t^T) J_t     (the Fisher metric pulled back to write space),
    g_t = J_t^T (p_t - e_{y_t})                (the exact first-order score at the true token).

G_t is unbiasedly estimated with no 50257 x 1152 storage by the score trick: G_t = E_{y~p_t}[s_t s_t^T],
s_t = d log p_t(y) / d w_t — one backward per sampled token (Martens, "New insights and perspectives on the natural gradient
method", arXiv:1412.1193; Amari 1998). Because G_t contains P_perp(x_t)/|x_t|^2, (i) radial write directions have zero Fisher
weight, (ii) all tangential directions are graded by 1/|x_t|^2 — a write of energy 3.6e9 into a residual of comparable norm
buys almost nothing per unit variance. This is the same scale-invariance that makes gradients orthogonal to weights in
normalised networks (Arora, Li, Lyu, "Theoretical analysis of auto rate-tuning by batch normalization", arXiv:1812.03981); here
it is applied to an *activation* write through the final RMSNorm (Zhang & Sennrich, arXiv:1910.07467).

**What it predicts, beyond reconstruction.**
(a) The dominant write directions of MLP16/17 are largely radial: the fraction rho_t = (w_t . xhat_18,t)^2 / |w_t|^2 is large
    (my bar: mean >= .5 for MLP17). That is why 90% of the variance buys 70% of the effect and why an "entropy dial" along u_1
    cannot exist — the norm removes the scale the dial would set. (Prediction made before measuring; it can fail.)
(b) The whole §2694 ladder (k = 4 ... 64) should be reproduced *without forwards* by the certificate above from one gradient pass
    — an approximation certificate in the Optimal-Brain-Damage lineage (LeCun, Denker, Solla 1989; Hassibi & Stork 1993),
    applied to activation subspaces rather than weights. If the certificate holds within a factor 2 at k >= 4, every future
    subspace surrogate (attention writes, DAS subspaces at late residual sites) can be priced from covariance + Fisher without
    a CE run, and the site-map probe running now becomes checkable analytically.
(c) The loss-optimal k-subspace is the top-k eigenspace of G^{1/2} C G^{1/2} (Fisher-whitened PCA), not of C. Prediction: at
    k = 8 it prices MLP17 below the variance basis (.083 -> <= .05).
    CLOSURE CHECK: §2118/§2125 closed "metric-constructed bases" and "Fisher selection" *on the §312 attention frontier*; this
    is a different object (final-block MLP writes, priced as CE ADDED on held-out docs) and the primary claim here is the
    certificate (b), not a frontier installation. If (c) fails, the closure generalises and I will say so.
(d) For Codex's lane: at a late residual site, a DAS direction's *radial* component is causally inert; the readout-alignment
    diagnosis of R540 (my 15:15 note) should be complemented by reporting each learned direction's tangential fraction and
    its Fisher weight. A direction that passes targets with small Fisher weight is doing so through |x| effects, not content.

**Assumptions that may fail.** Second order is only valid for small delta: for k = 0-2 (mean replacement, |delta| ~ 1e4) the
certificate may be badly off — I register only k >= 4 and disclose the rest. The tanh cap's curvature adds a term
proportional to the first-order logit score; disclosed, not modelled. For MLP16 the Jacobian passes through block 17's
attention (cross-position), so G is non-local; the score trick with independent per-position samples is still unbiased for the
full Fisher (cross-position score terms are zero-mean and independent), but its variance is higher — 4 samples per position.

**Cheapest falsifying experiment (CPU, ~15 min, 0 GPU) — registered below as `mlp_final_blocks_fisher_certificate_probe`.**

## Move 2 (in flight): symmetric tensor rank and simultaneous diagonalisation of the rank-8 write

**Object.** With the write restricted to its rank-8 basis U, c_j(x) = xhat^T Qs_j xhat with Qs_j = sym(Left^T diag(U_j^T Down) Right):
eight quadratic forms = the degree-2 component of a polynomial map R^1152 -> R^8. A quadratic form's Waring (symmetric tensor)
rank equals its matrix rank (Sylvester), so the queued probe's eigen-truncation ladder IS the symmetric-rank ladder of the
interaction tensor, and its in-situ price is the honest "how many interaction terms matter" number Logan asked for.
**New sub-question (added to the Fisher probe as an exact weight computation):** are the eight forms *simultaneously*
diagonalisable? Eight symmetric matrices share an eigenbasis iff they pairwise commute; approximate joint diagonalisation
(Cardoso & Souloumiac, SIAM J. Matrix Anal. Appl. 17(1), 1996; Bunse-Gerstner, Byers, Mehrmann 1993) turns 8 x 1152^2
coefficients into ONE shared dictionary of interaction features v_i plus 8 diagonal spectra: c_j = sum_i lambda_ji (v_i . xhat)^2.
Measurable: the fraction of each Qs_j's Frobenius energy on the diagonal in the eigenbasis of sum_j Qs_j^2 (a random basis gives
~2/D ~ .002). If >= .5, the whole rank-8 write of MLP16 compiles to a shared square-feature dictionary — a composable, editable
tensor-program component (edit one lambda_ji, one feature). If ~.002, the eight forms use eight unrelated interaction bases and
the "interaction term" grain has no shared dictionary — a real answer either way.

## Move 3 (propose; not executed): prequential/MDL accounting for the final-block surrogate in the Fisher metric

The §2668 MDL frame prices a surrogate by bits; in the Fisher metric the *description length of the write subspace* is the
natural-gradient-weighted rate-distortion of (C, G): the optimal k for a target CE budget is read off the eigenvalues of
G^{1/2} C G^{1/2} (reverse water-filling). This gives an executable-cost curve (dims vs nats) with a certificate rather than a
measured ladder — but only after Move 1's certificate is validated. Deferred until §2696 lands.

## Pruned this review (and why)
- Hankel/automata/minimal realisation for the residual stream: the interfaces are RMSNorm-broken (non-linear quotient), so
  linear realisation theory applies only *within* a block's tangent space — Move 1 is the correct restriction of it.
- Polynomial invariant theory / sign gauge beyond what §2633-§2687 already used: no new measurable consequence this hour.
- Information bottleneck on the write: it optimises a local MI objective; the Fisher certificate is the loss-faithful version.
- Any further variance-PCA site screens beyond the running site map: §2694 shows variance rank is the wrong metric.
- Generic sparse program synthesis over the whole network: no falsifiable step at this grain yet.

## Ranked
1. Fisher/scale-gauge certificate probe (Move 1) — executed: preregistered 16:3x, script `ops/mlp_final_blocks_fisher_certificate_probe.py`.
2. Simultaneous-diagonalisation test of the 8 quadratic forms (Move 2 sub-question) — folded into the same probe (exact, seconds).
3. Fisher-metric MDL curve (Move 3) — proposal, gated on 1.
