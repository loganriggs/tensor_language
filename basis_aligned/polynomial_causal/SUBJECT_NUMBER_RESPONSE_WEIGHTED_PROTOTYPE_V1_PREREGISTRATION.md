# Subject-number response-weighted prototype V1

Registered before model execution on the already-open 32-row subject-number
authority. This is a discovery assay, not a fresh OOD confirmation.

The donor-free mean grouped-MLP6/7 displacement predicts the exact native-axis
L11H3 response with relative L2 error `.2880428105400366`, but its complete
coefficient program has error `.5158500295544283`. A rank-2 map selected by
recipient/displacement activation variance changes those errors only to
`.2872163287046907` and `.5107208113072614`. The DCT briefing predicts that the
next coordinates should be chosen in the causal-response metric rather than by
activation energy.

For each leave-one-construction-out fold and answer direction, let

$$
s_i=u^T[H_i(x_i+\delta_i)-H_i(x_i)]
$$

be the donor-dependent discovery response, and let $p_0$ be the existing fixed
mean displacement from the training construction. Across all 16 E/A/U/W
backgrounds, form the centered training displacement matrix and retain its top
16 right singular vectors $V$. This candidate width is fixed in advance; it is
not swept.

At each training background, compute the exact input gradient of the scalar
head response at the donor-free point $x_i+p_0$,

$$
g_i=\nabla_x\,u^T H_i(x)\big|_{x=x_i+p_0}.
$$

The response design is $A_i=g_i^TV$. Fit one ridge-regularized correction

$$
c=(A^TA+\lambda I)^{-1}A^T(s-s_0),\qquad
\lambda=10^{-2}\operatorname{tr}(A^TA)/16,
$$

where $s_0=u^T[H_i(x_i+p_0)-H_i(x_i)]$. The executable prototype is the one
fixed 1,152-vector $p=p_0+Vc$ for that fold and direction. At held-out execution
it reads only the native recipient/background state and frozen model weights:

$$
\hat s_i=u^T[H_i(x_i+p)-H_i(x_i)].
$$

The already-frozen leave-one-construction-out interaction coefficients then
produce $\hat\alpha_i=[1,z_i,\hat s_i,z_i\hat s_i]\beta$. They are not refit.

Eight deterministic permutation controls repeat the identical response fit
after permuting the training residual-response rows. They share the same mean,
rank-16 displacement span, gradients, ridge, parameter count, and evaluation;
only the association between state and response is destroyed. This controls a
fortuitous high-dimensional correction. The candidate must beat the median
permutation program error by at least `.02`.

Instrumentation passes only if:

- exact-response and mean-prototype metrics replay their frozen parent values
  within `1e-8`;
- 512 held-out examples, four fold/direction fits, rank 16, and eight controls
  are present;
- role-state decomposition closure remains at most `5e-5`;
- all gradients, fits, and predictions are finite; and
- every prototype norm is at most twice its mean-prototype norm.

The scientific response gate requires relative L2 improvement of at least
`.03` over `.2880428105400366` and cosine at least `.95`. The complete program
gate requires relative L2 at most `.45`, improvement of at least `.10` over the
native baseline `.6093072967652493`, and degradation of at most `.10` from the
exact-response oracle `.37689362716534275`. The permutation gate above must
also pass.

A failure is a valid null for this fixed rank-16, one-step response-Jacobian
prototype on the opened authority. It does not close nonlinear optimization,
higher-rank response bases, token-level native generation, or fresh-text OOD.
A pass freezes an all-row prototype and requires fresh-authority causal
substitution, selective removal against equal-norm directions, and two-site
composition before upgrading the sparse circuit.
