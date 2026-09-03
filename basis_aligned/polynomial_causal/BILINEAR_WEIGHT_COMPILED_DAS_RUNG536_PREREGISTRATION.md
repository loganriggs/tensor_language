# Rung 536 preregistration: product-space DAS compiled into quadratic weights

**Registered:** 2026-09-03 13:27 UTC

**Owner:** Codex

**Status:** Stage A CPU algebra and planted recovery authorized; no real-model optimization or GPU run authorized

## Question

Can a circuit be represented by a small rotated subspace of an MLP's 4,608 bilinear product activations, selected by
interchange behavior, and then translated exactly into explicit quadratic forms and residual-stream output vectors?

This is a circuit grouping/splitting question. The dimensionality is a constrained minimality criterion only after
held-out circuit prediction, selective intervention, composition, and stable identification pass. Reconstruction or
low rank alone cannot pass.

## Exact object

For one bilinear MLP,

$$
g(x)=(W_Lx)\odot(W_Rx),\qquad F(x)=W_Dg(x)+b.
$$

For an orthonormal product-space basis $U\in\mathbb{R}^{4608\times k}$, $P=UU^\top$. The selected component is

$$
F_P(x)=W_DPg(x).
$$

For each column $u_\ell$ of $U$, define

$$
Q_\ell=\frac12\left[
W_L^\top\operatorname{diag}(u_\ell)W_R+
W_R^\top\operatorname{diag}(u_\ell)W_L
\right],
\qquad d_\ell=W_Du_\ell.
$$

Then the exact compiled implementation is

$$
F_P(x)=\sum_{\ell=1}^k d_\ell\,x^\top Q_\ell x.
$$

An interchange from donor to base is

$$
\Delta F_P=W_DP\left[g(x_{\rm donor})-g(x_{\rm base})\right],
$$

which must equal the difference of the compiled quadratic programs on donor and base.

## Duplicate-work boundary

This is not the completed residual-stream DAS family, which searched subspaces of 1,152-dimensional module writes.
It is not the completed MLP0 49-term minimum-norm probe (§2655), which fit a linear combination of an already chosen
effect table and localized 0/32 circuits out of sample. It is not an MLP Down-rank or tensor-rank sweep. The new object
is the full native 4,608-dimensional product activation, with a causal interchange objective and exact weight
compilation.

The §2655 result is a live warning: its per-term circuit pattern correlated only 0.106 between document halves.
Therefore no real-model DAS is authorized until a pre-optimization stability/power gate passes at a substantially
larger document count.

## Stage A: CPU algebra and planted recovery

Use deterministic float64 toy MLPs. Plant an orthonormal product-space projector $P_\star$ and donor/base product
differences. Fit an orthonormal $U$ by gradient descent to reproduce the planted projected interchange, with no
access to $U_\star$ in the objective.

Registered checks:

- **A — exact weight compilation:** direct $W_DUU^\top g(x)$ and the $k$ compiled quadratic forms agree to maximum
  absolute error at most $10^{-10}$.
- **B — exact interchange compilation:** direct projected donor/base interchange and compiled quadratic difference
  agree to maximum absolute error at most $10^{-10}$.
- **C — basis gauge:** replacing $U$ with $UO$ for a deterministic orthogonal $O$ changes the projector output by at
  most $10^{-10}$.
- **D — planted recovery:** after the frozen optimizer budget, normalized projector overlap
  $\operatorname{tr}(P_\star P)/k\ge0.99$ and held-out projected-interchange relative error is at most 0.05.

Failure of A--C rejects the proposed weight translation. Failure of D means the proposed optimizer is not a valid
instrument and no real-model fit may be interpreted.

## Stage B gate before a real-model run

Stage B must be separately preregistered after Stage A and a complete dossier audit. It must, at minimum:

1. use more documents than the unstable §2655 term-effect screen;
2. show that the target 32-circuit training signal has adequate split-half reliability before fitting DAS;
3. reserve the 30 held-out circuits and code OOD for frozen evaluation;
4. compare multiple fit seeds, dimension-matched random subspaces, shuffled circuit identities, and native-unit
   subsets;
5. test physical interchange, selective removal, and compiled-weight equivalence;
6. select the smallest preregistered $k$ satisfying circuit gates, not the smallest reconstruction rank.

No GPU work is authorized by this Stage A registration.
