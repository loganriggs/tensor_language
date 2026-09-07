# Three-hour mathematical tensor-network review — 2026-09-07 23:26 UTC

## Current mathematical object

Bilin18 has residual width $d=1152$, $L=18$ decoder blocks, $H=9$ attention heads of width
$p=128$, bilinear-MLP width $m=4608$, and vocabulary size $V=50304$.  For example $i$, layer
$\ell$, query position $q$, source position $s$, head $h$, and head coordinate $a$, attention forms

\[
Q^{r}_{i\ell qha}=X_{i\ell qd}W^{Qr}_{\ell hda},\qquad
K^{r}_{i\ell sha}=X_{i\ell sd}W^{Kr}_{\ell hda},\quad r\in\{1,2\},
\]

an input-dependent causal pattern $P_{i\ell hqs}$ from softmaxes or squared score products, values
$V_{i\ell sha}$, head responses

\[
H_{i\ell qha}=\sum_sP_{i\ell hqs}V_{i\ell sha},
\]

and a residual write $W^O_{\ell,da}H_{i\ell qha}$.  Each local MLP is

\[
M_\ell(x)=D_\ell[(L_\ell x)\odot(R_\ell x)],
\quad L_\ell,R_\ell\in\mathbb R^{4608\times1152},
\quad D_\ell\in\mathbb R^{1152\times4608}.
\]

The MLP is degree two in its normalized local input.  The whole contraction is not a polynomial
because RMS normalization and attention softmax are input-dependent nonlinearities.  Token
embedding and output weights are tied.  The current intervention has four independently stored
rank-one head axes $u_h\in\mathbb R^{128}$ at
$h\in\{L8H1,L9H1,L9H4,L11H3\}$ and acts in causal layer order by

\[
H^{\mathrm{live}}_{ihq:}\leftarrow H^{\mathrm{base}}_{ihq:}
 +(H^{\mathrm{donor}}_{ihq:}-H^{\mathrm{base}}_{ihq:})u_hu_h^\top,
\]

followed by the fixed complete attention-15 donor clamp.  This is an absolute intervention, not an
additive delta on the already changed live stream.

For construction $e\in\{A1,A2\}$, group parity $f$, and projector tuple $U=(u_h)_h$, the finite
causal response operator reports target signed projection $r_{ef}(U)$, direction fraction
$d_{ef}(U)$, final-residual response in $\mathbb R^{1152}$, and control vocabulary divergences
$K_{cf,i}(U)=D_{KL}(p^{base}_{cfi}\|p^U_{cfi})$ for $c\in\{P,C\}$.  The fit is constrained by

\[
\min_e r_{ef}(U)\ge .75,\qquad \min_e d_{ef}(U)\ge .875,
\]

then minimizes the worse control median plus $.25$ times the worse control mean.  It trains on one
parity and licenses a checkpoint only on the other.  V16 A1/A2/P remains unavailable to selection
and is opened once after both v15 fold projectors and their initialization family are frozen.

Allowed patched inputs have rowwise equal base/donor token lengths and equal terminal semantic
positions.  V16 C violates this contract and is excluded.  Approximation is measured separately by
signed projection, direction fraction, relative squared error, full-vocabulary KL, and top-one
flips; no scalar average is a certificate.  Gauge freedoms are $u_h\sim -u_h$ and residual basis
changes paired with inverse adjacent-weight changes.  The projector $u_hu_h^\top$ and exact
weight contractions are sign-gauge invariant.

Literal projector storage is 512 scalars, or $4(128-1)=508$ projective/Grassmann degrees of freedom,
plus four declared intervention sites, causal order, unit dose, and the attention-15 branch.  The
registered empirical cap is 20 native captures, 200 differentiable transformer forwards, 48
backward forwards/updates, 8,000 example evaluations, and 512 fit parameters.  This is an
identification price, not yet an adopted executable-program price.

## Exact neighboring objects and assumption audit

### Common principal components and joint diagonalization

Flury's common-principal-component model asks whether covariance matrices $\Sigma_e$ from multiple
populations share one orthogonal eigenbasis, $\Sigma_e=B\Lambda_eB^\top$, and derives maximum
likelihood estimates and likelihood-ratio tests
([JASA 1984](https://doi.org/10.1080/01621459.1984.10477108)).  Afsari analyzes uniqueness and
noise sensitivity for exact/nonorthogonal joint diagonalization, showing dependence on diagonalizer
conditioning and a modulus of uniqueness
([SIAM J. Matrix Anal. Appl.](https://doi.org/10.1137/060655997)).  Cai and Li give sufficient
identification conditions and an algorithm for exact joint block diagonalization
([AISTATS 2021](https://proceedings.mlr.press/v130/cai21a.html)).

An exact mapping can be defined diagnostically.  For head $h$ and construction $e$, stack aligned
donor-minus-base response rows into $D_{eh}\in\mathbb R^{N_e\times128}$ and choose a preregistered
positive task metric $G_e$ on observed downstream tests.  Then

\[
C_{eh}=D_{eh}^\top G_eD_{eh}\in\mathbb R^{128\times128}
\]

is a construction-indexed symmetric causal-response Gram matrix.  A shared one-dimensional axis
would be a common leading block of all $C_{eh}$.  But none of the cited theorems solves the present
object: our $G_e$ is induced by a nonlinear suffix and finite interventions rather than Gaussian
population covariance; the four heads interact in causal order; target and control constraints are
not simultaneous-diagonalization loss; and exact common eigenvectors need not exist.  Joint
diagonalization is therefore an identifiability diagnostic after causal response measurement, not a
replacement for the queued intervention.

### Minimum enclosing subspace and projective minimax geometry

Marrinan, Absil, and Gillis formulate the minimax center of several subspaces as a minimum enclosing
ball on a Grassmann manifold and solve a rank-penalized version with a dual subgradient method
([arXiv:2003.12455](https://arxiv.org/abs/2003.12455)).  Their object maps exactly to finding a
geometric center of independently identified construction-specific projector subspaces.  It does
not establish that the nonlinear transformer behavior at that center equals the behaviors at the
endpoints, and its rank-selection objective is not needed here because rank is fixed prospectively.

For two rank-one axes there is a simpler exact certificate requiring no optimizer.  Sign-align unit
vectors $u,v$ so $u^\top v\ge0$, let $\theta=\arccos(u^\top v)$, and define the projective bisector

\[
w={u+v\over\|u+v\|}.
\]

By symmetry and the spherical triangle inequality,

\[
\max_{\|w\|=1}\min\{|u^\top w|^2,|v^\top w|^2\}
=\cos^2(\theta/2).
\]

Thus the pairwise projective angle supplies an exact best-possible *geometric* common-overlap bound.
It does not supply a behavioral bound through the nonlinear suffix, which is precisely why the
bisector must itself be causally evaluated.

### Tensor decompositions and minimal realization

The environment/head/test response inventory can be stored as a tensor
$\mathcal R[e,i,o,h,a]$.  CANDECOMP/PARAFAC or tensor-train decompositions could compress it, and
joint diagonalization is related to uniqueness of some tensor decompositions, but low tensor rank
does not choose an intervention, preserve controls, or resolve absolute causal order.  Ho--Kalman
minimal realization would require a linear time-invariant Markov-parameter Hankel operator; layers,
inputs, RMS normalization, and softmax make Theseus neither time invariant nor linear.  These routes
remain post-identification pricing tools or restricted diagnostics.

## Executable consequence and opposing predictions

If the queued fixed-projector test fails, do not immediately add rank or regularization.  Fit two
additional rank-one *diagnostic oracle* axes per head using the identical parity split and absolute
patcher: one can observe only v15 A1 and the other only v15 A2, with P/C still used solely for
checkpoint selection.  Then:

1. report within-construction fold cosines and between-construction projective angles per head;
2. construct each deterministic sign-aligned bisector, with no further gradient updates;
3. report the exact $\cos^2(\theta_h/2)$ common-overlap ceiling;
4. causally evaluate the four-head bisector tuple on held v15 parity and already sealed/opened v16;
5. compare it with the failed joint optimizer under identical controls and dose.

Opposing outcomes are discriminating:

- **One fixed invariant exists, optimizer failure:** A1 and A2 oracle axes are individually stable,
  their bisector has high geometric overlap, and the deterministic bisector passes both observed and
  OOD causal bars even though the joint learned checkpoint failed.
- **Construction-conditioned coordinate:** each oracle is stable and effective on its own
  construction, between-construction angles are large, cross-use fails, and the bisector fails one
  target or controls.  This licenses an input-conditioned projector/finite routed mixture.
- **No stable rank-one construction axis:** within-construction folds themselves are unstable or
  ineffective.  This rejects the rank-one head-local object rather than diagnosing routing.
- **Nonlinear cross-head interaction:** per-head overlap ceilings are high but the complete bisector
  tuple fails causally.  This localizes the obstruction to ordered multi-head composition rather
  than individual subspace geometry.

This consequence changes stable identification, within-module splitting, held-out prediction, and
the computational specification.  It is not a rank sweep: rank stays one, and the result decides
whether the next circuit object is fixed, construction-routed, or interaction-conditioned.  Its
implementation should reuse the current fit core and add only per-panel masking plus the analytic
bisector helper.  The queued fixed test remains cheaper and more decisive first; the geometry audit
is a preregistered contingent falsifier, not a reason to delay it.
