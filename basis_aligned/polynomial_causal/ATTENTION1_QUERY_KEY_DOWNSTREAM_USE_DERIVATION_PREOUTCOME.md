# Pre-outcome derivation: split attention1 score branches into query-side and key-side downstream use

**Written:** 2026-09-02 16:52 UTC

**Status:** CPU mathematical continuation while rung495 runs. This is not a preregistration and does not open or
change rung495. It specifies the registered null-route object precisely enough that a rung496 preregistration can be
written without falling back to rung418's weight-space overlap test.

## Decision this resolves

Rung495 asks whether any complete finite QK1/QK2/OV interaction pieces from different attention1 heads are treated
as the same variable by 62 downstream circuits. If no complete pieces match, a shared computation could still exist
on only one input side of a score branch: several heads may read the same query-side feature but combine it with
different key-side features, or conversely.

The next object must therefore answer:

> Do different attention1 heads read the same residual-stream directions on the query or key side, for the same
> downstream purposes, even when their other score side and output differ?

This advances computational specification, cross-head grouping, within-head splitting, held-out prediction, and
stable identification. It is not a rank-reduction question.

## Computation in ordinary attention language

For one attention head `h`, one of its two score branches, query position `i`, and key position `j`, write

`s_h(i,j) = q_h(x_i)^T R_(i-j) k_h(x_j) / 128`,

where `x_i,x_j in R^1152` are the real normalized attention1 inputs, `q_h` and `k_h` are the learned linear maps to
128 numbers, and `R` is rotary position encoding. The model multiplies the two score branches, applies the causal
mask, weights the value vectors, and maps the head output back to the 1152-dimensional residual stream.

For downstream circuit loss `L_c`, the query-side contribution to the input gradient is

`g^Q_(h,c,i) = sum_j (dL_c/ds_h(i,j)) W_Q,h^T R_(i-j) W_K,h x_j / 128`.

The key-side contribution is

`g^K_(h,c,j) = sum_i (dL_c/ds_h(i,j)) W_K,h^T R_(i-j)^T W_Q,h x_i / 128`.

Both are 1152-dimensional vectors in the model's shared residual-stream coordinates. They say which change to the
query-position or key-position input would most change circuit `c` through this one score side while the other score
side, value/output path, normalization, and suffix remain at their actual values.

In implementation these should be obtained by retaining the real computation graph and differentiating through only
the chosen query or key path. The formulas above are definitions and cross-checks, not a replacement model.

## Why this removes the private Q/K coordinate ambiguity

The 128 internal score coordinates have a gauge freedom. For any invertible matrix `G`, replace

`q by G q` and `k by G^(-T) k`.

Every score is unchanged. Raw Q/K coordinates, sparse atoms, and coordinate cosines can therefore change without the
model changing. But the query-side residual gradient is invariant:

`(G W_Q)^T (G^(-T) k) = W_Q^T k`.

The key-side residual gradient is invariant by the same calculation. Comparing these gradients in the common
1152-dimensional input space therefore measures the function, not a private head basis.

## Candidate signature and controls

For each of the 36 score sides (nine heads times two score branches times query/key), collect its residual-gradient
field over frozen documents, positions, four MLP0 branches, and the 62 downstream circuit masks. A useful signature
must retain position and circuit structure; averaging everything to one vector would allow shared token difficulty
or a generic loss direction to masquerade as a shared read.

A later preregistration should:

- select a cross-head side pair on one document half and freeze it before confirmation;
- compare raw fields and fields with the per-branch common gradient removed;
- use one-sided circuit-label permutations, position rolls, and matched random residual subspaces;
- require mutual matching rather than allowing one generic side to match many targets;
- confirm on held-out documents and held-out circuit tags without reselection;
- retain the opposite score side and the value/output path as explicit controls, so a whole-head duplicate is not
  mislabeled as a shared query or key read;
- require the matched query/key-side response to transport when recomposed with each head's own opposite side; and
- route any survivor to a finite physical input-side swap or removal with unrelated-circuit preservation.

The primary similarity should compare the complete gradient fields after fitting at most one scalar magnitude. A
rank or low-dimensional subspace may be reported as a price or matched control, but cannot make the claim pass.

## What rung418 did and why this is different

Rung418 compared the query and key column subspaces of attention0's exact all-token folded score functions. It found
large overlap relative to random subspaces but no eligible multi-head relation: zero edges passed the full shared-half,
unseen-token transport, and different-companion conditions. Its result was a gauge-invariant weight/function-space
screen, not a downstream-use decomposition.

Repeating that test at attention1 would only change layer number. The present object additionally conditions each
side on:

1. the actual document state and positions;
2. the other score branch and value/output computation;
3. the real normalized downstream suffix; and
4. which of the 62 circuits supplies the loss.

Thus two sides group only when they read the same input directions for the same later computations. Conversely, one
native score branch can split if its query and key paths serve different circuit families.

## Opposing outcomes

- **Shared-side outcome:** a cross-head query or key pair selected on one half remains the mutual match on held-out
  documents/circuits, beats all controls, keeps its match after common-gradient removal, and transports through each
  head's distinct opposite side. This licenses a finite input-side interchange test.
- **Whole-head-duplicate outcome:** query, key, opposite side, and output all match together. This is redundancy at a
  larger grain, not the proposed shared-vocabulary-with-different-partners structure.
- **Common-loss outcome:** raw similarity disappears after branch-mean removal or circuit permutation. The apparent
  sharing was generic sensitivity, not a circuit variable.
- **No-shared-side outcome:** no frozen pair confirms. Complete below-head pieces and individual Q/K input paths are
  both distributed at this resolution; move to the broader predictive-state causal quotient rather than tune rank or
  sparsity.

## Literal cost to price before registration

The exact backward-call count depends on whether all 36 side fields can be recovered from one retained graph per
circuit mask or require isolated vector-Jacobian products. The preregistration must derive that count from the actual
autograd implementation, include document-prefix/full-forward counts, and demonstrate a nonzero path-isolation
tripwire. No GPU run should be queued until those counts and the conditional validation cost are frozen.
