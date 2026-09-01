# MLP0 rank-448 causal-output-interface oracle — preregistration

Date: 2026-09-01 16:49 UTC  
Rung: 409  
Claim level: held-out output-repair ceiling and path-decomposition screen; not an executable repair

## Question

Does the already frozen rank-64 MLP0 causal output interface contain the error left by the covariance rank-448
program, and is that recoverable error better represented by separate token and token×context output bases than by
one global output basis?

This changes the represented object from the failed rank-448 input-metric family. It does not rotate or reweight the
rank-448 input subspace. It adds an oracle correction in MLP0 **output space**.

## Immutable authorities

- Native checkpoint, exact rank-448 construction, exact T/C/I/S/A grammar, source rows, and scoring convention are
  those of rungs 401 and 404–407.
- The frozen large-document branch receipt is
  `mlp0_rank448_branch_large_confirmation_results.json`, file SHA-256
  `545941a446338d4470d712feccd1a88e986eea33454046a3ed2cdd830224d76f`.
- The historical causal-output bundle is
  `joint_early_mlp_pca_composition_authoritative_v3_bases.pt`, file SHA-256
  `0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9`.
- Its site-0 basis is shape `[1152,64]` with raw tensor SHA-256
  `cb57c81a5b5ecbe8a1ad0f13f0f8b9e9df20d01ea237399db40659da75ab4b52`.
- Use source documents 0:192 only to fit new PCA controls. Score only documents 192:384, with waves 192:288 and
  288:384. Positions are `[64,256)`. `FINAL` remains unopened.

All CE numbers are nats added above the native model; lower is better.

## Exact objects

For each scored position let `y_native` and `y_448` be the complete native and covariance-p448 MLP0 writes. Let

`e = y_native - y_448`.

Using the exact rung-401 grammar reconstructed for both programs, define branch errors

`e_T = T_native - T_448`, `e_C`, `e_I`, `e_S`, and arithmetic closure `e_A`,

with the pointwise identity `e = e_T+e_C+e_I+e_S+e_A` checked before scoring.

For an orthonormal output basis `U`, the oracle correction for branch set `J` is

`repair_U,J = (sum_{j in J} e_j) U U^T`,

and the intervened write is `y_448 + repair_U,J`. This reads the native error at evaluation and is therefore an
impossible oracle ceiling, not an executable program.

## Frozen bases and arms

1. Native, covariance-p448, covariance-p640, and covariance-p768 physical baselines.
2. Frozen historical `B0_64` basis, correcting `T`, `I`, `T+I`, or the complete error.
3. `JOINT_TI_64`: top 64 output PCA directions of `e_T+e_I` on training documents only; correct held-out `e_T+e_I`.
4. `SPLIT_T32_I32`: top 32 directions of training `e_T` and top 32 of training `e_I`; correct each held-out branch
   in its own basis and add the two corrections. It stores the same 64 output directions as `JOINT_TI_64`.
5. `TOTAL_ERROR_64`: top 64 directions of complete training error `e`; correct complete held-out error.
6. `RANDOM_64`: seed-409 Haar output basis; correct complete held-out error. This is the negative control.

No rank, split, allocation, basis, or branch set may change after any evaluation result is viewed.

## Literal price

- covariance-p448 layer program: 9,954,432 values;
- historical complete rank-64 affine output interface: 153,920 values;
- their optimistic executable sum: 10,108,352 values;
- covariance-p640: 11,945,088 values;
- covariance-p768: 13,272,192 values.

The oracle itself has no executable coefficient producer and earns no adoption credit. The equal-direction
`JOINT_TI_64` and `SPLIT_T32_I32` comparison prices only their 64 stored output directions for this screen; any
successor must price its producer and operations before execution.

## Registered predictions

### Prediction A — authority and instrument

All checkpoint, source-row, parent-program, retained-energy, branch-identity, frozen-basis hash, fit/evaluation split,
orthonormality, endpoint, call-count, and finite-value checks hold. Physical p448/p640/p768 damage on the two
evaluation waves reproduces rung 407 within `1e-6`. Random correction moves the output and `FINAL_opened=0`.

### Prediction B — the frozen causal interface contains current p448 error

On evaluation documents, the complete-error `B0_64` oracle:

- removes at least 30% of p448 CE damage;
- improves p448 by at least `0.001` nat in each evaluation wave; and
- beats `RANDOM_64` by at least `0.001` nat pooled.

### Prediction C — the useful frozen-interface correction is token-grammar-led

Within `B0_64`, the `T+I` correction recovers at least 70% of the complete-error oracle's CE gain. The `T`-only and
`I`-only corrections each improve p448 by at least `0.0002` nat pooled, and neither loses more than `0.0005` in an
evaluation wave.

### Prediction D — different paths prefer different output descriptions

At the same 64 stored output directions, `SPLIT_T32_I32`:

- lowers the geometric mean of held-out T/I relative squared error by at least 10% versus `JOINT_TI_64`;
- improves physical pooled CE by at least `0.0002` nat versus `JOINT_TI_64`; and
- is no worse in either evaluation wave.

## Strong null

The strong null fires if A fails, if complete-error `B0_64` improves p448 by less than `0.0002`, if `RANDOM_64` is
within `0.0002` of it, or if none of `B0_64`, `TOTAL_ERROR_64`, `JOINT_TI_64`, and `SPLIT_T32_I32` improves p448 by
at least `0.0002` on held-out CE.

## Frozen decision

- If A/B/C hold, test the already serialized `B_l5_r64` causal-interface predictor as a physical correction on the
  p448 program. It must be repriced at 10,108,352 values plus verified runtime operations and compared with p640 and
  p768 on fresh and out-of-distribution text.
- If A holds, B fails, but `TOTAL_ERROR_64` or `JOINT_TI_64` succeeds, the historical interface is the wrong output
  basis; design one held-out producer for the successful new basis.
- If D holds, design separate token and interaction coefficient producers; otherwise prefer one shared output basis.
- If the strong null fires, close rank-64 output repair of p448 and move to direct nonlinear CE fitting or the
  already successful late-layer quadratic-surrogate family. Do not tune rank 64 or revisit input metrics.

No outcome from this rung alone is an executable compression, composition result, intervention certificate, or
adoption point.
