# A shared arithmetic parent after folding through MLP16

12 September2026. Prepared while the direct input-block fit runs; not submitted
until the local and prepared spectral block outcomes have been interpreted.

Independent blocks are only one structural assumption. A shared parent can
intentionally couple many partners; the exact star counterexample proves that
block irreducibility does not exclude cheap reuse. Use the existing
`shared_input_factor_v1.optimize`, `value_gradient` and `native_partner` directly.
The prior single-layer result is in `COMPOSED_SHARED_PARENT_V1_PRIOR_AUDIT.json`:
all four centered starts converged to1.04876% coefficient capture, but rank16
partners retained only55.26% of that component's energy. Do not repeat it as new.

## New object, unchanged solver

Let $p(x)$ be the homogeneous MLP16 producer including the MLP17 residual
coefficient, and $H$ its exact quadratic coefficient Gram. Set

$$
z=H^{-1/2}p(x),\qquad L=L_{17}H^{1/2},\qquad R=R_{17}H^{1/2}.
$$

Search a unit shared parent $a$ with **unrestricted** output-linear partners.
With $P_a=aa^T$ and the full centered-U forms $S_v$, the extracted star is

$$
S_{v,a}=P_aS_v+S_vP_a-P_aS_vP_a.
$$

It includes all interactions touching that parent, not an independent block.
The old exact objective and gradient apply unchanged after transforming $L,R$:

$$
E(a)=2a^TKa-h(a)^TGh(a),\quad
h(a)=(La)\odot(Ra),\quad G=(U_cD_{17})^T(U_cD_{17}).
$$

The residual partner $M_a$ comes from `native_partner`. The executable extracted
numerator is $(a^Tz)M_az$. Save physical maps $a_p=H^{-1/2}a$ and
$M_p=M_aH^{-1/2}$, so it runs directly as $(a_p^Tp)M_pp$ without retaining the
dense coordinate transform. This is a quartic computation of the upstream input.
Native RMS denominators, other source paths, bias, head and background remain
external. Deleting this numerator component is not the same as renormalizing
or deleting an upstream activation globally.

Four fixed starts: top eigenvector of $K$, Gaussian seeds1103/1109/1117.
Existing sphere/Armijo algorithm,4000steps, normalized tangent gradient<=1e-8.
No convergence inferred from reaching the step or900second job limit.
The bound $E(a)/\operatorname{tr}K\le2\lambda_{\max}(K)/\operatorname{tr}K$
is reported to distinguish intrinsic one-parent capacity from optimization loss.

- A: paired total energy replays the completed full-U producer spectrum<=1e-8;
  star projection/weighted partner energy and physical producer-coordinate
  evaluation agree<=1e-8 on64fixed formal inputs, finite outputs.
- B: all four fits converge at normalized tangent gradient<=1e-8.
- C: all four absolute parent cosines with the best>=0.99 and best fractional
  capture>=1.25times the prior centered single-layer capture. The norms differ
  by composition; this is a structural-concentration comparison, not behavioral gain.
- D: best fixed-parent rank16 partner retains>=90% of its star energy.

Compute the partner spectrum using $J=(I+aa^T)^{1/2}$ and
$\tfrac12\|\operatorname{chol}(U_c^TU_c)^TM_aJ\|_F^2$.
Report rank90 and literal implementation costs. Full parent+partner costs
1,328,256floats plus the15,925,248native producer matrix values; no whole-model
compression claim. A rank16 partner costs38,016fitted values in whitened
coordinates; physical maps can absorb the coordinate transform. Its approximation
error and retained background must be charged separately.

Null: this folded one-parent model does not achieve the registered concentration,
stability or compact-partner bars. Even convergence is local. A passing structure
needs component-dossier alias checks and frozen native extraction, selective
removal/interchange, composition and held-out/OOD prediction. No corpus fitting
or preemptive semantic label is part of this experiment.
