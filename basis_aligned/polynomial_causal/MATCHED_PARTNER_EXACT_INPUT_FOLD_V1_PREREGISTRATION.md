# Exact conditional producer-folded branch program

Use fixed producer reader columns $C_0,C_3,C_8$ from the current program.

$$A_j=\operatorname{sym}(L_{16}^T\operatorname{diag}(C_j)R_{16}),\quad q_j=x_{16}^TA_jx_{16},\quad w_j=q_0q_jd_j/\rho_{17}^2.$$

This is the existing dense quadratic folding identity applied to the current candidate. C already contains the producer scale, Down map and whitened reader. The pure producer path excludes MLP16 bias and the other MLP17 input paths; those remain external. It does not reconstruct the native normalized x16 input or denominator.

Store upper triangle coefficients, doubling off-diagonal values, so each scalar is the dot product with $x_ix_j$ for i<=j. Three forms share the parent across branches3/8; two output writers finish the local program. FP64 construction is algebraically exact; actual artifact is FP32, so its roundoff is measured separately.

A: FP64 packed/dense/native-factor scalar and write replay<=1e-8. B: FP32artifact branch write relative errors<=1e-5, shared native scorer reference replay<=1e-5 and all swap/removal fidelity bars on the128 fresh endpoints. C: literal packed parameter count1,994,688joint/1,329,408single and at least4x fewer FP32bytes than native-factor joint implementation (10,632,960floats). Report actual disk bytes, including metadata. Spectra/signature/radial fraction for allthreequadratics are descriptive; no rank selection or refit. Null: packing/rounding breaks local execution, or price omits necessary maps. Full U/background and input-state production are explicitly outside this conditional price.
