# Native pure-quartic branch substitution

2026-09-20 21:29 UTC. Implementation begun in `native_quartic_branch.py`.

Question: does the extracted ten-product program preserve the causal effect of
one explicit cross-layer polynomial term when installed in the native model?
This is a branch replacement, not a whole-block replacement or semantic circuit.

Let x be the normalized input of MLP16 and define its bias-free output
m16 = D16[(A16 x) * (B16 x)]. Write the actual pre-normalization input of
MLP17 as h = r + m, where m = lambda17[0] m16. The remaining r includes
MLP16's bias, all other residual sources and actual attention17 output.
With B17(t) = D17[(L17 t) * (R17 t)], the native MLP17 output is

$$
\frac{B_{17}(r)+D_{17}[(L_{17}r)\odot(R_{17}m)
 +(L_{17}m)\odot(R_{17}r)]+B_{17}(m)}{s(h)^2}+b_{17}.
$$

Replace only B17(m). Keep s(h)^2 = mean(h^2)+epsilon, the native attention,
all cross terms, biases, final RMSNorm and logit softcap explicit and unchanged.
An ablation sets this branch to zero with the same denominator. It is an
intervention on the algebraic branch, not removal of MLP16's upstream state.

The fitted output is R_u B17(m)/teacher_scale, where U=Q_u R_u is the
native unembedding QR used for fitting. Solve R_u for the student's output
writer and constant once; do not invert per token. Audit conditioning and
reprojection error. The target is pre-final-normalization residual space.

## Registered native screen

Freeze ten-product candidate; include original 26-product approximation,
exact branch replay and branch ablation controls. Use FineWeb cached documents
64:80 from fineweb_n192_skip7000.pt at context128 (unseen by prior panel tests).
Report document-level CE added above native (lower is better), post-softcap
logit RMS error, KL(native || replacement), argmax agreement and effects of
ablation. Do not tune on this panel.

- pred_a_instrument: exact branch replay relative residual and logit error <1e-5;
  output-frame solve/reprojection error <1e-5. If failed, invalidate native claims.
- pred_b_preservation: ten-product CE damage <0.02 nats/token and KL <0.02.
- pred_c_effect: ten-product logit-change norm relative to native is less than
  half the branch-ablation change norm. A zero ablation effect makes this bar
  undefined, not a success. Report signed CE even when ablation improves CE.

Null: good isolated error does not preserve the installed branch's effect.
No selective semantic manipulation or OOD claim follows from passing this screen.
Price: candidate 19,632 coefficients and ten products plus shared vocabulary
frame; instrumentation retains the native branch to subtract it and therefore
has added compute. A compiled full replacement requires separate accounting.
All GPU execution must use the managed queue. The CPU oracle already checks
bias, residual scaling and normalization with deliberately wrong controls.
