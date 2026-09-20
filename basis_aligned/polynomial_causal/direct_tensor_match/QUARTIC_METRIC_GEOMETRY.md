# Covariance weighting for the folded quartic

Instrument note,20 September2026,18:40UTC. The user-requested data-informed metric needs a precise definition. These are three different objectives, even with the same covariance matrix. Derivations below are independently checked in `check_quartic_covariance_metric.py`; they do not presume an empirical Gaussian input distribution.

Let the symmetric coefficient error tensor be $\Delta_{vijkl}$ and $e_v(x)=\Delta_v[x,x,x,x]$. Let $M=LL^\top$ be positive semidefinite. Write $\Delta_L$ for the tensor obtained by applying $L$ to all four input slots.

**Four-slot coefficient metric:**

$$
\mathcal L_{\mathrm{coeff},M}=\|\Delta_L\|_F^2
=\sum_{v,i,j,k,l,a,b,c,d}\Delta_{vijkl}\Delta_{vabcd}
M_{ia}M_{jb}M_{kc}M_{ld}.
$$

It can also be interpreted as expected squared error of the four-linear map on four independent zero-mean inputs with covariance$M$. Their higher moments are irrelevant because each independent slot enters linearly. For a bilinear-network quartic, changing its two first-layer reader matrices $A,B$ to $AL,BL$ implements this transform exactly. Transform student readers inside the objective too; keep original reader parameters if controlling for optimization parameterization.

**Gaussian function metric:** for one repeated input $x\sim\mathcal N(0,M)$,

$$
\mathcal L_{\mathrm{Gauss},M}
=24\|\Delta_L\|_F^2
+72\|\operatorname{Tr}_{2}(\Delta_L)\|_F^2
+9\|\operatorname{Tr}_{4}(\Delta_L)\|_2^2,
$$

where $(\operatorname{Tr}_{2}H)_{vij}=\sum_k H_{vijkk}$ and $(\operatorname{Tr}_{4}H)_v=\sum_{i,k}H_{viikk}$. The105Gaussian pairings split into24fully cross pairings,72pairings with one internal pair in each tensor, and9pairings with two internal pairs in each tensor. The extra trace terms explain why weighted coefficient matching is not repeated-input Gaussian matching.

**Empirical function metric:** $N^{-1}\sum_n\|e(x_n)\|^2$ depends on actual eighth moments. Covariance alone cannot determine it. For example, $e(x)=x_1^4-x_2^4$ has coefficient norm squared2, Gaussian function loss192 under$M=I$, and loss0 for independent symmetric Rademacher coordinates, also with$M=I$. A nonzero mean adds further terms; replacing covariance with the uncentered second moment does not make a zero-mean Gaussian objective equal to a noncentral or empirical objective.

**Floors change the science, not just numerical conditioning.** Suppose a pure quartic lies in one covariance eigendirection with eigenvalue$\epsilon$. Its squared weighted coefficient norm scales as$\epsilon^4$. A floor0.01 therefore assigns that direction a weight10⁻⁸ relative to a unit eigenvalue. A singular metric can hide nonzero errors completely. Always retain isotropic diagnostics and specify trace normalization, floor convention, coordinate system, and mean treatment.

The new native comparison uses original MLP16 input coordinates, calibration-only covariance, trace-normalization before floors, centered floors0.01/0.1, uncentered second-moment floor0.01, and an isotropic control. Original student parameter coordinates remain fixed across objectives. Gaussian and actual-row losses are diagnostics, not silently substituted training targets. These comparisons may reveal useful directions but do not identify circuits without predictive/intervention evidence.

Validation receipt `QUARTIC_COVARIANCE_METRIC_CHECK_V1.json`: dense weighted coefficient loss agrees3.63×10⁻¹⁶ relative, gradients4.15×10⁻¹⁶, and the Gaussian trace formula agrees with exact fifth-order-per-axis Gauss–Hermite quadrature1.37×10⁻¹⁶. The covariance-equal counterexample and floor scaling are also executable checks.
