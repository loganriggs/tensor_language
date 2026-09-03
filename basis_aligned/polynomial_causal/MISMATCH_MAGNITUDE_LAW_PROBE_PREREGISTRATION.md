# Preregistration — is the score-implementation mismatch MAGNITUDE separable (rank-1)? (CPU probe; parallel lane)

Date: 2026-09-03 00:33 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any rank-1 outcome; zero model forwards, CPU only, on
rung513's published mismatch norms. Completes §2647 (fixed direction) with the
magnitude half. Codex's 514 untouched.

## Question

§2647 showed the score-implementation mismatch DIRECTION is gauge-covariant
(one fixed factor subspace across all 18 keys). This asks whether its
MAGNITUDE is SEPARABLE: does the mismatch norm over the 6 branch-subsets × 3
action-pairs grid factorize as norm[branch,pair] ~= u[branch]*v[pair] (rank-1)?
A pass means the four implementations differ by a single scalar field (one
per-branch scale times one per-implementation-pair scale) along the §2647
fixed direction — a complete low-rank model of the source-dependence. The
stored norms already show the separable ordering N-Z8 < N-Z7 < P-Z7 at every
branch-subset; this tests it exactly. Descriptive on published statistics; no
circuit or physical claim.

## Computation (exact, deterministic)

Load rung513's receipt (sha 043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5). For each site s in {a11, m11}, build
the 6x3 matrix N_s[b,p] = sqrt(mean over the two halves of
complete_mismatch_norm_squared) for branch-subset b in {L,R,L+R,L+LR,R+LR,
L+R+LR} and action-pair p in {N-Z7,N-Z8,P-Z7}. Take elementwise log
(magnitudes are strictly positive), center, and SVD. Report:
1. the fraction of squared Frobenius energy in the top singular value of the
   centered log-matrix (rank-1 dominance);
2. the relative residual ||N_s - rank1(N_s)||_F / ||N_s||_F on the RAW
   (un-logged) matrix, using the exp of the rank-1 log fit;
3. the recovered per-branch scales u[b] and per-pair scales v[p];
4. cross-site consistency: cosine between a11's v (pair-scales) and m11's v.

## Frozen predictions

### pred_a — exact reproduction
Receipt sha matches; all 18 norm entries present and positive at both sites;
the a11 L::N-Z7 pooled norm reproduces the receipt to <= 1e-6 relative.

### pred_b — the magnitude is rank-1 separable
For BOTH sites: top-singular-value energy fraction of the centered log-matrix
>= .90, AND raw-matrix rank-1 relative residual <= .05.

### pred_c — the per-implementation-pair scaling is site-consistent
The a11 and m11 recovered pair-scale vectors v (3 entries) have cosine >= .90
after L2 normalization — the magnitude ordering across implementation-pairs is
a property of the implementations, not of the consumer site.

## Strong null
Fires if pred_a fails or pred_b fails at either site: the mismatch magnitude is
NOT separable, so the source-dependence has genuine branch-by-implementation
interaction and §2647's fixed direction does not extend to a rank-1 magnitude
model. Reported beside §2647 either way; no bar changes.

## Price
Zero model forwards; CPU < 10 s; one receipt JSON. Nothing deployed.
