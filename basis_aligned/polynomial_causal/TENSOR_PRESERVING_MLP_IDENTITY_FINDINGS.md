# Exact bilinear MLP tensor-program identity

Date: 2026-08-28

The owned 18-site MLP bank exactly executes

$$
\operatorname{Down}(\operatorname{Left}(x)\odot\operatorname{Right}(x))
+\operatorname{DownBias}
$$

without retaining or calling native MLP objects. On the role-free deterministic
production-shaped fixture:

- every offline same-input write has maximum absolute error 0;
- every write on the full sequential program trajectory has error 0;
- native and program logits are bitwise identical with SHA256
  `ecd96381a8d062a09b7d6387224fb6bf1c9dde9924952d4dec05ffbca50d09c9`;
- both synthetic CEs equal 12.686808586120605;
- all 18 program MLPs and native attention modules execute once, while literal native
  MLP calls in the program arm are zero;
- block/order closure, replacement restoration, and storage disjointness pass.

The complete dense MLP denominator is 286,675,200 float32 stored values, including
Left, Right, Down, and all 18 Down biases. It has total input support, no token tables,
and no fallback. Runtime was 12.7 seconds.

This is an executable identity point, not compression. Its purpose is to replace the
post-forward-hook boundary with an object whose polynomial products can be selected,
factored, or compiled before execution and whose native calls can be mechanically
forbidden.

The current table/correction class cannot yet populate this bank: applying its learned
outputs to uncovered tokens makes all-position recovery negative. The next useful MLP
program must generalize over the full token/state support, for example by combining an
embedding-conditioned lexical map with a small causally selected subset of exact
bilinear products. Any candidate must be installed through this bank, score all
positions, and preserve the Down bias explicitly.
