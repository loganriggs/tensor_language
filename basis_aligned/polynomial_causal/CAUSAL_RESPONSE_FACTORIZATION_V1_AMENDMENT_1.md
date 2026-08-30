# Causal-response factorization v1 — prospective amendment 1

Status: frozen before the causal-response FIT authority or bundle exists. This
amendment controls the calibration design and price. The original preregistration is
preserved as evidence of the defect described below.

## Why the cell-level anchor price was wrong

The original document selected 384 individual $(p,s,t)$ response cells. That is a
valid statistical mask but not the physical cost of the experiment. One intervention
forward fixes a phase $p$ and source $s$, and returns all 49 target responses $t$.
Thus the expensive sensor is the entire target block

$$
\mathcal B_{ps}=\{(p,s,t):t=1,\ldots,49\},
$$

not one scalar cell. A scattered set of 384 cells could touch nearly all 98 expensive
phase/source arms and falsely appear cheap.

## Controlling block-sensor designs

Every calibration mask must be a union of complete 49-target blocks. The compared
budgets are $m\in\{2,4,8,16\}$ phase/source arms. The primary executable price is
$m$ intervention forwards per new document batch; the associated scalar observations
$49m$ and code-solve multiply-adds are reported separately.

Two designs are frozen:

1. **Outcome-blind block baseline.** Order the 98 $(p,s)$ arms by SHA-256 of
   `causal-response-factorization-v1-arm|p|s`; take the first $m$ arms.
2. **Training-only block D-optimal design.** For a fitted candidate, form its
   observation-by-code basis $B\in\mathbb R^{4802\times K}$ on the 229 library-training
   documents and orthonormalize its column span to $Q$. For each arm $a=(p,s)$ let
   $Q_a\in\mathbb R^{49\times r}$ contain its target rows. Starting with
   $G_0=10^{-8}I_r$, greedily choose

   $$
   a_j=\arg\max_{a\notin S_{j-1}}
   \log\det\left(G_{j-1}+Q_a^\top Q_a\right),
   \qquad
   G_j=G_{j-1}+Q_{a_j}^\top Q_{a_j}.
   $$

   Ties within $10^{-12}$ select the smaller row-major arm index. Selection sees no
   internal-validation or EVAL response.

The log-determinant is the D-optimal information-volume objective. For additive PSD
information blocks with a positive prior, it is monotone submodular, so greedy
selection has the standard $(1-1/e)$ objective guarantee. This guarantee is about the
log-determinant objective, not response MSE. The block design is also invariant to an
arbitrary invertible change of candidate code coordinates: orthonormal bases of the
same column span differ only by an orthogonal transform, which conjugates every
$Q_a^\top Q_a$ without changing determinants. Drmač and Gugercin's Q-DEIM analysis
provides the closely related basis-invariant pivoted-QR interpolation construction and
error conditioning result: <https://doi.org/10.1137/15M1019271>. The log-determinant
sensor formulation and submodular greedy guarantee are given directly by Shamaiah,
Banerjee, and Vikalo: <https://sidbanerjee.orie.cornell.edu/docs/CDC_sensorsel.pdf>.

## Acceptance and claim boundary

At each budget and for every candidate, report validation support, non-anchor signed
MSE/correlation, worst owner-pair NRMSE, smallest singular value of the valid selected
design, and the same scores for the outcome-blind block baseline. At least 90% of the
114 validation documents must retain at least twice the code dimension valid selected
cells. A D-optimal panel earns a computational-simplicity gain only if it strictly
improves calibrated validation error or support at the same number of physical arms.

The selected panel may be frozen with a FIT candidate before EVAL. It is response
tomography, not zero-shot OOD prediction, semantic extraction, or terminal-circuit
evidence. If the fitted response subspace is wrong on new documents, D-optimal design
can confidently choose the wrong sensors; held-out prediction remains authoritative.

