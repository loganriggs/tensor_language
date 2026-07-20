# Anatomy of a gain: how mlp16 computes its fast structural state

Logan's pick (2026-07-21): dig into the top-MLP fast-structure — what computes it,
from where. The tensor-network structure gives an exact opening: because the MLP is
pure bilinear, the gain coefficient of any output direction $d$ is a **closed-form
quadratic in the layer input**:
$$c_d(x) = \hat x^\top M_d\, \hat x + d\cdot b_D,\qquad
M_d = \sum_j (W_D^\top d)_j\; W_L[j]\otimes W_R[j],$$
weight-exact, no estimation. **Gate: the form reproduces live coefficients to
rel-max 8.3×10⁻⁷.**

## MA-1: the weight-space form is dense

Eigen-anatomy of $\mathrm{sym}(M_d)$ for the four named gain directions: effective
rank **580–640 of 1152**. The rank-4–16 behavioral gain is *not* a weight-space
property — the weights implement a broad quadratic surface.

## MA-2: the data metric collapses it to boundary-feature quadratics

Whitening by the layer-input covariance ($\Sigma^{1/2} M_d\, \Sigma^{1/2}$):
effective rank falls to **25–52** (dir0: 25.6, dir3: 24.3; top-8 features ≈ ⅓ of
mass). And the top data-space features decode uniformly as **punctuation/boundary
structure** — dir0's leading feature is the `.` `).` `:` `,` direction; dirs 1–3
lead with newline/dash/quote directions. The fast structural state is quadratic
detection over boundary features: *where am I relative to structure boundaries,
squared.* The low rank everyone measures downstream is data concentration of a
dense form — a clean instance of weights-vs-function requiring the data measure
(the same lesson as EH-4, constructively this time).

## MA-3: causally sufficient, and the feeders named

Replace all four coefficients in the live forward by their rank-$k$ whitened-form
approximations:

| k (features per dir) | ΔCE | dir0 coefficient R² |
|---|---|---|
| 64 | **+0.028** | 0.954 |
| 16 | +0.033 | 0.865 |
| 4 | +0.066 | 0.907* |

The gains *run* on named quadratic features (\*R² non-monotonicity at small k is an
off-manifold sampling quirk; ΔCE is the binding number). The exact stream-pair split
of dir0's coefficient variance names the feeders: **mlp15 ⊗ mlp15** dominates, with
**attn5 ⊗ mlp15** next — the penultimate MLP's output interacting with the model's
one global attention stream, at coefficient resolution what SI-1/TM-1 showed at
energy resolution.

## The complete mechanism chain

exact weight form (gated) → ~25–50 boundary-feature quadratics (data-concentrated) →
fed by mlp15 self-interaction + the attn5 hub → expressed as rank-4–16 output gains →
behaviorally an intra-register distribution shaper (card 3, ~100× controls).

**Next rung (queued):** recursion — mlp15 is itself a pure bilinear MLP, so its
boundary-feature outputs have their own exact quadratic forms; the same anatomy can
walk upstream until it grounds in token-static structure (which the windowed-D
results say must happen within a few layers).

Files: `../mlp16_anatomy.py/.json`, `../mlp16_anatomy2.py/.json`,
`../mlp16_anatomy3.py/.json`.
