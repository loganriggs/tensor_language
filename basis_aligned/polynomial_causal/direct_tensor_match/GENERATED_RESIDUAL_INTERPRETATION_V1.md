**Consistent source generation changes cancellation, but still fails transfer**

The original graph approximates six source reads while receiving the original MLP17 residual state $h$. That is a declared partial-computation interface, not a complete replacement of the preceding MLP contribution. With true reads $q_a,q_b$, write the relevant projected residual as $h^\top A=c+q_a$. Then the target scalar is

$$
\phi=\left(\frac{c+\tfrac12q_a}{s_{17}}-\alpha\right)
\left(\frac{q_b}{s_{17}}-\beta\right).
$$

Replacing source reads everywhere in this selected numerator gives

$$
\hat\phi_{\rm generated}=\left(\frac{c+\tfrac12\hat q_a}{s_{17}}-\alpha\right)
\left(\frac{\hat q_b}{s_{17}}-\beta\right).
$$

By contrast, the old interface gives $(h^\top A-\tfrac12\hat q_a)/s_{17}-\alpha$ as the first factor. Both are exact at the true reads, but their approximation errors differ. Neither interface alone regenerates the RMS17 denominator or attention17.

The tested carry projection is

$$
c=\lambda_{17,0}s_{16}(z^\top A)+\gamma,
$$

where $z$ is the normalized MLP16 input, $s_{16}$ is its RMS scale and $\gamma$ is the projected embedding skip, attention17 and MLP16 output bias. These remain explicitly supplied native quantities. In donor tests, the recipient carry stays fixed while the source reads use the donor input. This is not token-to-output extraction and does not close the attention dependence on the preceding state.

**Native and independent checks**

The managed run used both already-opened panels,96document captures in5.46seconds. Residual16 reconstruction error is6.31e-8 and exact-read component replay9.06e-8, below1e-5. An independently computed prediction from the previous signed error Gram matrices agrees with the new code-panel energies within4.07e-7 relative error. The three old error terms $(t_1,t_2,t_3)$ become $(-t_1,t_2,-t_3)$, explaining the changed cancellation without fitting any parameter.

Across144 centered-scalar comparisons (two panels, two domains, three intervention types, three cohorts and four component selections):

| Interface | Absolute failures | Covariance-baseline relative failures | Isotropic-baseline relative failures |
|---|---:|---:|---:|
| Original supplied residual | 0 | 6 | 4 |
| Consistently generated projection | 1 | 0 | 7 |

The absolute failure is panel-two FineWeb/component-three/natural/continuation:16.94% versus15%. Relative failures require graph error no more than1.10times each baseline. Baselines receive the same consistent generation. This is a scalar screen, not the earlier vocabulary-effect metric, so these counts should not be compared directly to the earlier logit tables.

**Actual successor: original-fitting-state assessment**

On the original448states, graph generated-interface errors are3.10%,2.83%,8.87%, versus3.93%,2.95%,12.05% for the covariance baseline and3.93%,3.53%,11.85% for the isotropic baseline. All three are within1.10times both baselines and below15%. The expanded-interface transfer failure therefore is not revealed by those fitting-state comparisons. Simply claiming that the old objective omitted this interface does not explain the observed distribution gap sufficiently.

The useful progress is a more explicit residual/MLP cubic-and-quartic computation, with a checked executable boundary and predictable change in error cancellation. It has not passed adoption. A subsequent optimization must preserve the actual generated interface and price its supplied scales/context, while testing transfer on data that have not been reused for diagnosis. The current panels are opened data; no positive result on them can restore fresh status. Blindly adding another small fitting-state penalty is not yet a justified repair.

[Native screen](GENERATED_RESIDUAL_V1.json) · [All failures and independent replay](GENERATED_RESIDUAL_AUDIT_V1.json) · [Fitting-state comparison](TWO_INTERFACE_FIT_STATES_V1.json) · [Gram prediction](GENERATED_ERROR_GRAM_PREDICTION_V1.json).
