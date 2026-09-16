# MLP17 one-layer causal-Hessian identity certificate

Before using derivative-based component discovery, verify the DCT bridge on the
actual ungated bilinear MLP17 with its input normalizer held outside the slice.
For

$$f(x)=D[(Lx)\odot(Rx)]+b,$$

the exact Hessian action is

$$H(v,w)=D[(Lv)\odot(Rw)+(Lw)\odot(Rv)].$$

Freeze seed `202609160400`, a 16-dimensional orthonormal input probe and an
8-dimensional orthonormal output probe. Materialize the resulting
`8×16×16` tensor once from weights and once from FP64 autodiff at two independent
backgrounds. Also compare it with finite bilinear cross-differences. No prompts,
behavioral readers, activations, fits to model outputs, or normalizer derivatives
enter this identity check.

Run the same deterministic rank-4 symmetric orthogonalized alternating solver on
the weight and autodiff tensors. Match recovered components up to permutation
and input sign. Separately run it on a planted rank-4 tensor with orthogonal
input/output factors and distinct amplitudes.

Frozen predictions:

1. `pred_a_weight_autodiff_identity`: relative tensor error is at most `1e-10`
   and maximum absolute error is at most `1e-9`.
2. `pred_b_background_invariance`: autodiff tensors at two random backgrounds
   agree within `1e-10` relative error.
3. `pred_c_cross_difference_identity`: eight deterministic random input pairs
   have maximum relative cross-difference error at most `1e-10`.
4. `pred_d_factor_agreement`: after optimal permutation, every analytic/autodiff
   input and output factor cosine is at least `.999999`, amplitude relative error
   is at most `1e-6`, and reconstruction-error difference is at most `1e-8`.
5. `pred_e_planted_solver_control`: every planted input/output factor cosine is
   at least `.999`, and amplitude relative error is at most `1e-3`.
6. `pred_f_tensor_symmetry`: both native tensors are symmetric in their two input
   modes within maximum absolute error `1e-10`.

Price: one checkpoint load, no language/model forward executions, two native
subspace Hessians, eight finite cross-differences, three rank-4 solver calls,
zero training, parameter updates, or quantization. Passing validates only the
one-layer fixed-normalizer bridge and factor implementation, not deep-slice
moment contraction, identifiability of native components, or any behavioral
circuit.
