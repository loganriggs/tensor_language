# Cross-task causal-mode weight pullback preregistration

## Object

The validated finite response matrix factors as `H = C R^T`, where each row of `C` is a
reachable shared-Q8 write in an orthonormal residual-state basis and each row of `R` is the
actual final RMS-normalization/unembedding reader in that same basis. After the frozen
task-diagonal scaling, compute `H_b = C_b R_b^T = U Sigma V^T`.

For each of the two identified response modes, solve

`C_b t_k = U_k sqrt(Sigma_k)` and `R_b o_k = V_k sqrt(Sigma_k)`

by the minimum-norm pseudoinverse. `t_k` and `o_k` are respectively a source-scoring covector
and reader-context covector in Q8 coordinates. Mapping them through the orthonormal physical
Q8 basis gives gauge-invariant residual-space vectors under the allowed orthogonal basis gauge.

## Test

Recompute the frozen 32-command/32-reader factors using exactly the parent instruments. Confirm
that the two pulled-back mode scores reproduce the rank-two canonical response matrix. Then:

1. contract each source covector backward through H3 value rows and rank all upstream attention
   heads and MLP output matrices;
2. contract each reader covector into downstream attention Q/K/Q2/K2/V and MLP Left/Right maps;
3. compare the two modes' rankings rather than pooling their norms;
4. causally test the top novel component for each mode against matched low-ranked components on
   both temporal and is/was commands.

## Predictions and null

- Pullback score replay relative RMSE is at most 1% for both leading modes.
- Orthogonal Q8 gauge rotation changes physical pullbacks and weight scores by at most `1e-5`.
- Known temporal writer L8H1 and readers L15H5/H1 are enriched in at least one mode.
- At least one additional top-decile component has a mode-selective causal effect that transfers
  across tasks and exceeds the matched low-ranked controls.

If algebra closes but the weight rankings do not predict causal effects, weights provide an
incidence prior rather than circuit identification; retain the two-mode response program and use
finite downstream sensitivity for edge selection. No return to scalar DAS or penalty tuning is
licensed.
