# Hourly strategic review — 2026-08-29 08:55 UTC

## Bottom line

The newest scientific result remains E3.2's negative: a pointwise rank-64 state cannot
carry the finite L8 $\rightarrow$ L11 $\rightarrow$ L14 response language. The action
completed in this review is an engineering prerequisite for the next highest-return
branch, E4 copy/continuation: the source-owned per-head attention adapter now reproduces
the live checkpoint exactly before head separation at all five relevant layers.

This is real progress on a missing interface, not progress on explained behavior. E4
still has no scored model outcome.

## How much of the model is actually explained?

There is no honest single percentage because the project tracks different claims:

| Claim | Current strict fraction | Meaning |
|---|---:|---|
| structural interception | 36/36 sites | every attention/MLP write has an executable interception surface; this is plumbing, not semantic explanation |
| removal-certified stored model | 5.3481% | this fraction of literal stored model state has passed the project's strict removal/certification ledger |
| named causal CE | 10.923% | named mechanisms account for this fraction under the causal CE ledger; 4.72714 nat remains unnamed |
| terminal executable actions | 0/68 | no final extraction/removal/transplant cell has passed its complete contract |
| E1--E4 entry points | E1 negative; E2 compression-only/negative semantics; E3 rank-64 negative; E4 unmeasured | none yet supplies a whole-model semantic program |

The main positive whole-program fact is still the exact covered-token table behavior:
on covered current tokens, live and table MLP writes agree to about $3.6\times10^{-7}$
relative error. The remaining late-MLP prediction changes are almost entirely the
uncovered-token fallback (`99.24% / 99.41% / 100%` of MLP16 changes). Attention remains
genuinely contextual: attention 5 changes about 96% of covered and uncovered positions.

## Largest remaining gaps

1. **No behavior-level executable circuit.** We have component candidates and causal
   effects, but no held-out program that predicts, extracts, selectively removes and
   transports one behavior with controlled collateral damage.
2. **Uncovered-token MLP fallback.** Covered tables are essentially exact. The learned
   fallback differs from the live late-MLP write by roughly `0.328--0.336` relative
   error and causes nearly all MLP restoration changes.
3. **No composable intermediate state language.** The tested pointwise rank-64 L8/L11/
   L14 basis is insufficient even at the destination (`E_out=0.2709`) and predicts
   unseen composition poorly (`E_out=0.4520`, coordinate $R^2=0.4024`).
4. **Native-Down's causal promise is unvalidated.** The MLP3 Family-F K512 support has
   better suffix KL with native Down (`0.05772`) than with the locally refitted Down
   (`0.08476`), despite worse local write NRMSE. Its fresh finite-edit port is still
   blocked by independent row audit and a complete measurement/semantic validator.
5. **Compression coordinates are not yet semantic APIs.** Shared rank-64/128 output
   structure saves prediction at matched storage, but global/typed/hierarchical large-
   budget dictionaries do not beat independent maps strongly enough to license edits.

## What ran in this review

The new adapter decomposes an attention write after all nonlinear QK computations:

$$
w_{\mathrm{attn}}=\sum_{h=0}^{8} z_h\left(W_O^{(h)}\right)^\top.
$$

Here $z_h$ is head $h$'s context-dependent value result and $W_O^{(h)}$ is the matching
128-column block of the output projection. This is a physical additive residual-write
interface, so removing a head write has an exact real-number meaning.

Two fail-closed attempts found implementation details before behavioral rows were
spent. V1 exposed bfloat16 recomposition error. V2 plus a read-only localization found
that `Rotary.inv_freq` is a plain float32 attribute: a blanket adapter conversion had
silently cast it to bfloat16, perturbing rotated queries by 0.623% and the full write
by 0.810%. V3 preserves source dtypes and completed receipt-last in 9.72 seconds:

- named layers 5, 7, 8, 13 and 14: unpartitioned adapter write bit-identical to native;
- shared block-0 value bus: bit-identical on every layer;
- all-nine-head recomposition relative error: `0.002627--0.002667`;
- zero native calls after adapter construction;
- literal price per site: 7,962,689 stored values, including all six matrices, the
  scalar and rotary frequencies.

The nonzero head-sum error is finite-precision accumulation order, not missing model
structure. It remains reported as a numerical residual and must be small relative to
actual causal head effects in the future behavior run. Receipt SHA256:
`c5ef51670b6e23bb3cddbbef6c5cd451dff55eea8b8f7ddfdf20aca7374bb324`.

## Pruned actions

- Do not edit a rank-64 E3 state: its representation failed sufficiency first.
- Do not enlarge normalized-energy uneven rank allocation: it lost uniform by
  `0.0194--0.0231` nat. The raw-energy near-hit requires a new causal-weighted design.
- Do not refit Family-F Down by local MSE: it improved NRMSE while worsening suffix KL.
- Do not revive direct-sum/HOSVD with the frozen projectors: real quadratic closure was
  worse than matched Haar controls.
- Do not treat attention write norm as a causal law. The all-layer follow-up and
  amplification falsified both ordering and threshold causality; norm is at most a
  locator for the special attention-5/6 failure regime.
- Do not rescue stream maps with more closed-input fitting: recursive closure and
  self-consistent refit failed by large margins.

## Top five next actions

1. **Complete E4 copy/continuation rows, scorer and behavioral screen.** Highest expected
   information gain per GPU minute and directly tests prediction, extraction, removal,
   OOD transport and price. The riskiest tensor-formula adapter is now checkpoint-exact.
2. **Run the prospective MLP3 native-Down finite behavioral port.** It tests the strongest
   new causal signal and whether low suffix KL represents a downstream null or merely
   one-sided nonlinear compensation. It remains blocked by its declared independent
   row audit and missing full measurement/result validator.
3. **Replace the failed E3 pointwise state with a behavior-conditioned temporal state.**
   Fit a small state only after E4 identifies a behavior and allow short position kernels
   or two-edit diamonds. This exploits tensor composition without repeating the failed
   universal rank-64 grammar.
4. **Fit a nonlinear uncovered-token fallback jointly against downstream CE.** Keep the
   exact covered table and target only the actual gap. Candidate grammars are low-rank
   quadratic or sparse mixture maps conditioned on attention writes, with a native-free
   recursive deployment test.
5. **Test tight-budget shared-plus-private maps or causal-weighted rank allocation.** The
   only surviving E2 signal is at rank-64/128 storage. This is cheap and falsifiable but
   lower priority because it primarily offers compression, not a selective edit API.

Ranking uses expected information gain, causal relevance, whole-model composability,
falsifiability, GPU cost and redundancy with completed failures. The current critical
path is E4 row/scorer lifecycle; the adapter result removes one of its six launch
bindings but does not justify opening behavioral data before the others are frozen.

