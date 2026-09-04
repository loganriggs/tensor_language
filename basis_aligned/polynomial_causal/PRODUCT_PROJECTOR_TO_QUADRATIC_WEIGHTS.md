# Product-space DAS projector to exact quadratic weights

**Written:** 2026-09-04 UTC
**Scope:** CPU algebra only; no model call, fitting, or scientific result

## What is compiled

For a bias-free bilinear MLP path,

$$
g(x)=(W_Lx)\odot(W_Rx),
\qquad
y_P(x)=W_DUU^\top g(x),
$$

where the columns of $U\in\mathbb R^{h\times k}$ are orthonormal. The compiler turns the already chosen product-space
subspace into ordinary quadratic weights. For column $u_\ell$ of $U$, it emits

$$
Q_\ell=\frac12\left[
W_L^\top\operatorname{diag}(u_\ell)W_R+
W_R^\top\operatorname{diag}(u_\ell)W_L
\right]
$$

and the output direction

$$
d_\ell=W_Du_\ell.
$$

The exact compiled computation is

$$
y_P(x)=\sum_{\ell=1}^k d_\ell\left(x^\top Q_\ell x\right).
$$

The symmetrization is exact because $x^\top A x=x^\top(A+A^\top)x/2$. If a dense output-indexed tensor is useful,
the code also forms

$$
\mathcal Q_{oij}=\sum_\ell d_{o\ell}(Q_\ell)_{ij},
\qquad
(y_P(x))_o=x^\top\mathcal Q_o x.
$$

For a recipient $x_b$ and donor $x_d$, the selected interchange contribution is therefore exactly

$$
W_DUU^\top[g(x_d)-g(x_b)]=y_P(x_d)-y_P(x_b).
$$

This is a translation into weights, not an approximation or a new compression claim. The factorized
$(Q_\ell,d_\ell)$ representation usually avoids the much larger dense $\mathcal Q$ tensor.

## Basis gauge and input checks

Changing coordinates inside the same subspace, $U\mapsto UR$ for orthogonal $R$, rotates the individual
$(Q_\ell,d_\ell)$ factors but leaves $UU^\top$, $\mathcal Q$, and every output unchanged. The subspace is the invariant
object; its displayed basis vectors are not.

By default the compiler rejects a matrix whose columns are not orthonormal. Otherwise writing $UU^\top$ would not be
an orthogonal projector and scaling a column would silently change the intervention. With
`normalize_basis=True`, it instead uses an SVD to compile the orthogonal projector onto the supplied column span and
drops linearly dependent columns. That option changes the meaning from the raw matrix $UU^\top$ to the projector onto
`col(U)`, so it must be requested explicitly.

## Audit of the earlier DAS work

- R536 already derived this factorization and passed toy float64 compilation, interchange, and within-subspace rotation
  checks. Its reusable helper, however, assumes orthonormal input and exposes only the factorized computation. The new
  standalone compiler preserves that algebra, validates shapes/dtypes/finiteness/orthonormality, handles an explicitly
  requested normalization, and can materialize the dense output quadratic for inspection.
- R540 learned residual-stream projectors for pending-opener swaps. Those projectors often moved the desired closer
  logit, but also moved answer-preserving controls; the rank-one direction behaved like a general answer-steering
  direction. It was not an identified pending-opener variable.
- R556 moved to the 128-dimensional output of layer 13 head 8 and trained with answer-preserving penalties. Ranks 2--16
  still failed at least one control, so no rank was selected and FINAL/OOD remained closed. These attention/residual
  projectors cannot themselves be compiled by this bilinear-MLP formula: one first needs a projector in a specific
  bilinear product space, or an exact map from the learned site into that product space.

The tests cover random and planted weights, proper low-rank and zero-rank projectors, a dependent supplied basis under
explicit normalization, donor-minus-recipient interchange, orthogonal basis rotations, the symmetric $Q_\ell$ formula,
and malformed inputs.

## What this does not solve

The compiler does **not**:

1. find $U$ or show that any small linear subspace exists;
2. decide which recipient/donor pairs are meaningful counterfactuals, or handle the fact that several different
   counterfactual constructions may represent the same high-level change;
3. show that the projected swap changes the intended internal variable rather than steering the final answer through a
   shortcut;
4. establish held-out or OOD prediction, sufficiency, selective removal, interaction isolation, composition/reuse, or
   stability across data splits and fitting restarts;
5. remove the gauge between different physical subspaces that downstream computation cannot distinguish. It only
   removes the trivial choice of basis inside one fixed subspace;
6. include MLP biases, normalization, residual paths, attention, or later nonlinear computation. Biases would add
   linear/constant terms, and a projector learned at another activation site requires its own exact translation.

Accordingly, this code is the last algebraic step after causal identification, not evidence that identification has
occurred. A real circuit still needs several valid counterfactual families, live answer-preserving controls, held-out
causal response, and selective interventions before its compiled weights are interpretable.
