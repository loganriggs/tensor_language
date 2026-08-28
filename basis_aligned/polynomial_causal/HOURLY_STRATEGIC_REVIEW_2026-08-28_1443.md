# Hourly strategic review — 2026-08-28 14:43 UTC

## Strategic change

The MLP1 physical-gate branch has resolved negatively. The selected native gates
were stable, but no registered 32/128/512-gate program passed the predictive and
harm controls. This is useful discrimination: the next attempt should change the
quadratic basis rather than keep pruning the checkpoint basis.

The attention-5/6 anomaly is also now localized. Replacing only the suffix after
the live attention write does not repair the 12-point failure, while restoring a
compatible prefix recovers 2.39 percentage points. In the compiled input stream,
live attention 5 emits a write with mean norm 1,202,366 versus 7,878 for its
compiled row, a ratio of 152.6. This is an incoming-state scale/interface mismatch,
not evidence that attention 5 is intrinsically harmful.

## Honest explanation balance sheet

| Ledger | Current value | Meaning |
|---|---:|---|
| Structural execution | 36/36 sites | All operations are executable; this is not semantic understanding |
| Certified whole-program storage reduction | 5.35% | Rank-640 attention passed predictive/causal tests, but not exact top-token identity |
| Older named-behavior ledger | $32.1\%\pm6.4\%$ | Human-labeled behaviors under its own denominator |
| Strict named causal recovery | 10.923% | Named intervention effects under a different denominator |
| Strict current-ship recovery against the $+0.8976$ CE gap | 0% | No composed replacement has yet earned this credit |

The dense MLP banks remain the largest unexplained parameter mass. More
importantly, their producer/consumer contracts remain unexplained: MLP0 can be
locally compressed while changing what MLP1 computes, and MLP2 can hide much of
that mismatch at the final output.

## Largest remaining gaps

1. **No low-product-rank MLP program.** Down-only compression reduces the final
   mixing matrix but still executes all 4,608 products. Native-gate pruning has now
   failed its registered MLP1 test. A new tensor basis is the live alternative.
2. **No composable early-MLP interface.** We do not yet know whether independent
   MLP0/1/2 reductions compose, require a shared basis, or require conditional
   refitting.
3. **No certified semantic meaning for compressed coordinates.** MLP0 has shared
   lexical structure and continuous refinement, but those descriptions have not
   beaten matched continuous programs under downstream tests.
4. **Distributed contextual computation.** No single live attention layer recovers
   more than 1.7% of the context-free program's missing accuracy. Context is spread
   across compatible interfaces.
5. **Insufficient final-scale validation for new candidates.** Small 16/32-document
   assays are branch discriminators. Any candidate promoted on CE, extraction,
   removal, or OOD transport still needs larger independent replication and a
   data-doubling check.

## Candidate actions and pruning

- **Implicit folded-tensor HOSVD for MLP1/2:** retained. It is CPU-feasible,
  gauge-invariant at the function level, and directly tests whether a different
  basis has small input/output multilinear rank.
- **Executable Tucker/CP or block-term fit:** retained conditionally. It becomes
  worthwhile only if the spectrum/core curve is cheaper than native or Down-only
  execution at a useful error threshold.
- **Eight-cell MLP0/1/2 composition cube:** retained. It is the cheapest direct test
  of independent composition and pair/triple interactions.
- **Joint sparse lexical dictionary with sparse downstream readers:** retained but
  deferred. It operationalizes overlapping features, but should not run before the
  weight tensor tells us whether a shared polynomial basis is plausible.
- **Attention-5/6 typed-interface calibration:** retained as a separate interface
  lesson, but below the dense-MLP path because it does not attack the largest storage
  or strict recovery gap.
- More native-gate subset sizes, arbitrary PCA bases, token-only coverage, local MSE
  optimization, and full dense tensor materialization are pruned for now.

## Ranked top five

1. **Source-close, freeze, and run the MLP1 implicit folded-tensor diagnostic.**
   Highest information gain after the native-gate negative; CPU-only; directly
   separates Down rank from function-level multilinear rank and sparse-core price.
2. **Run the already implemented MLP2 tensor diagnostic under the same cost
   contract.** This tests whether the compensating stage has a different intrinsic
   tensor structure and enables matched MLP1/MLP2 comparison.
3. **If either spectrum has a real knee, build an executable Tucker/CP candidate and
   measure both cost and local error.** HOSVD alone is not a product-count result.
4. **Evaluate the matched-cost eight-cell MLP0/1/2 composition cube.** Compute pair
   and triple Möbius interactions so compensation is measured rather than inferred.
5. **Fit overlapping MLP0 lexical features jointly with sparse MLP1/2 readers and a
   centered continuous context residual.** Admit it only if it improves prediction,
   extraction/removal, OOD transport, or executable price beyond reconstruction.

## Highest-priority action executed

The completed MLP1 result has been consolidated into
`MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md`, including the exact distinction between
stable selection and failed executable utility. In parallel, an independent agent is
now building and testing a create-only authority/collector for the committed MLP1
implicit folded-tensor source. That authority will bind the exact checkpoint tensor
hashes and source closure before any numerical spectrum is opened.

There is no GPU, FineWeb, cache, or `rspd` blocker for this branch. No scientific job
is currently consuming a GPU. The remaining temporary gate is procedural: the new
MLP1 tensor runner must pass source-closure tests and freeze its no-outcome authority
before it may read and publish checkpoint-derived spectra.

## Late-arriving result: partial compilation is directional

The bottom-up mirror curve finished after the initial review and materially changes
the interface story. The earlier top-down curve remains numerically valid, but its
per-layer increments cannot be read as independent layer importance.

Compiling only layer 0 while leaving layers 1--17 live loses
$62.6\%/60.4\%/61.9\%$ of the full live-to-compiled gap. Compiling only layer 17
loses only $4.8\%/4.4\%/3.7\%$. Worse, compiling layers 0--3 beneath live upper
layers performs below the fully compiled program by roughly 43--47% of the gap.
Keeping only the apparently expensive layers 6--9 live also fails catastrophically.

The correct interpretation is typed and directional:

$$
\text{compiled lower state}\longrightarrow\text{native upper module}
$$

is an out-of-distribution interface, whereas a compiled upper module can consume a
native lower state much more safely. This agrees with the attention-5 norm explosion.
Individual layer effects therefore do not compose by addition, and subset selection
from singleton or ordered-prefix marginals is invalid.

This raises the composition action in priority. The early-MLP cube must compare both
independently fitted replacements and **top-down conditional refits**. That separates
a fixable calibration/type error from a structural need for a joint program. The
folded-tensor diagnostic remains first because its source closure is almost complete
and it is a cheap branch selector, but no favorable local spectrum may be promoted
without this directional interface test.

The contemporaneous mathematical red-team also fixes the tensor decision boundary.
The existing MLP1 output-rank screen gives a numerical exact-product lower bound of
1,152, not incompressibility. A full-output dense Tucker program can beat native on
both storage and products only in the certificate-compatible input-rank window
$48\le r_i\le95$. Ranks 96--157 can win storage only; above 157, dense Tucker loses
even on storage. CP can still be useful between the lower bound of 1,152 products
and the native 4,608, but HOSVD cannot certify that rank.
