# C512 MLP0 x shared-HOSVD copy-gate composition

Status: **exploratory; frozen before outcomes from this runner**.

This is the first use of the new downstream-defined copy state to test an existing
upstream MLP simplification.  It crosses the authoritative C512 approximation of
MLP0 `Down` with the selected canonical shared-rank-256 L8 copy gate.

Evaluation uses exposed cached documents 33--128.  Frozen inputs:

- C512 binary SHA-256:
  `3ecf43b485d343bc5413e817dbd4236e5ce6cdaa7a3e0e653214e812b84ce470`;
- C512 fit receipt SHA-256:
  `79d0069864e9df521a99fc36531dd86c7ed31106f58f029d681fb1788a269f82`;
- shared-HOSVD result SHA-256:
  `8e2e27a7231472bce1389167414898c343ebf8ac2e1bcac6b78f220fe7b5801e`.

## Programs and arms

C512 keeps exact MLP0 RMSNorm, Left, Right, coordinatewise products, and Down bias.
It replaces only `Down` by its frozen rank-512 program.

The HOSVD gate maps the live normalized L8 attention input $x^{(8)}$ to

$$
z=V_{256}^\top x^{(8)},
$$

reuses $z$ across the eight H3/H4 Q/K cores, and writes the shared
$\lambda_8v_1$ successor payload.

Frozen arms:

1. `NN`: native MLP0, native L8 copy edge;
2. `CN`: C512 MLP0, native L8 copy edge;
3. `NH`: native MLP0, HOSVD copy edge;
4. `CH`: C512 MLP0, HOSVD copy edge;
5. `ZN`: complete MLP0 write set to zero, native L8.  This is a causal scale control,
   not a proposed program.

Every other MLP and attention component remains native.  HOSVD replacement affects
only the exact H3/H4 successor edge at input-eligible repeat destinations.

## Measurements

Report CE, native-to-arm KL, top-1, document mean/SE in the existing copy-positive,
repeat-negative, nonrepeat, and all-scored cells.  Also report:

- $R^2$, cosine, and relative RMS error of C512 versus native $z$;
- fraction of the zero-MLP0-to-native $z$ squared-error gap removed by C512;
- native and HOSVD L8 edge-scalar errors under native and C512 upstream states;
- CE composition interaction

$$
I=\Delta\mathrm{CE}_{CH}-\Delta\mathrm{CE}_{CN}-\Delta\mathrm{CE}_{NH}.
$$

## Frozen gates

- J1, C512 observational: all-scored $\Delta$CE `CN` $\le0.0075`.
- J2, C512 preserves copy: copy-positive $\Delta$CE `CN` $\le0.02` and top-1 loss
  at most 1 percentage point.
- J3, downstream state: C512 $z$ has all-scored $R^2\ge0.90$ and removes at least
  75% of the zero-MLP0 squared-error gap.
- J4, joint composition: all-scored $\Delta$CE `CH` $\le0.0075`, copy-positive
  $\Delta$CE $\le0.03`, and copy top-1 loss at most 1 percentage point.
- J5, bounded interaction: absolute $I\le0.002` all-scored and $\le0.01` on
  copy-positive positions.
- J6, HOSVD stability under C512: its edge-scalar $R^2$ under the C512 state is at
  least 0.90 for both H3/H4.

Failure of J2/J3 means C512 discards state needed by this copy consumer despite good
ordinary FineWeb output metrics.  Passing J1--J6 licenses C512 plus HOSVD as an
exploratory composed local program and moves the telescope to MLP1/MLP2.  It does not
override prior fresh-data authority boundaries or certify whole-model equivalence.

