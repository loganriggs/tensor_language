# Executable extraction of fixed scalar features

2026-09-20 22:20 UTC. Freeze all four previously tested output directions and
primaryalpha.5. Folding their readouts has already produced four scalar
programs with a shared13,916coefficient/10product dictionary; each standalone
scalar costs13,883coefficients/10products. No native-output approximation is
changed except FP32 contraction rounding (<6e-8 on cached rows).

These four scalars are projections of the primary output, not a complete
replacement for that vector output. The quadratic skip can write outside the
four-dimensional canonical span. Do not count discarded directions as savings.

For residual intervention, export writer V=teacher_scale*R_u^{-1}U, using the
actual unembedding QR from the fitting frame. The executable edit is

$$
x_{17}^{\rm out}\mapsto x_{17}^{\rm out}
-\frac{V\operatorname{diag}(g)s(x_{16}^{\rm norm})}{s(h_{17})^2}.
$$

Here g is an explicit four-component intervention strength vector, h17 is the
actual pre-normalization input to MLP17, and all other native paths stay fixed.
Final RMSNorm and softcap remain downstream. This is extraction with a stated
native background/interface, not an autonomous language model or semantic claim.
The writer adds4608 coefficients for all four features (1152 per single mode).
Joint feature computation is shared once; separate copies duplicate work.

Managed validation on the already used code confirmation panel(context256):
- pred_a_frame: exported FP32 residual-writer reprojection relative error<1e-5.
- pred_b_scalar: extracted scalar activations agree with the previous
  1152-output projection within1e-5 relative on native captured inputs.
- pred_c_intervention: all removal logit-effect energies and native targets
  agree with the archived primary confirmation within1e-5 relative.
Preserve prior mode1/KL failures; this validates executable extraction only.
