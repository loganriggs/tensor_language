# Block-3 native-Down behavioral port v1 — execution addendum

**Status:** prospective and outcome-blind. This addendum resolves numerical ambiguities
in the parent design but does not authorize rows, model loading, or GPU execution.

## Fresh role and immutable parents

Use 192 source documents, one 257-token chunk per document, from the pinned ordered
FineWeb-v2 parquet. Starting at dataset index 60,000, select the first documents that
survive the recursive registry census used by
`prepare_mlp0_c512_mlp2_compensation_v1_rows.py`. Exclude prior document IDs, dataset
indices, full rows, and 32-token prefixes. The future row freezer must use a create-only
`block3_native_down_behavioral_port_v1_rows_*` namespace and publish exact document
provenance, row-to-document identity, source/registry hashes, and receipt last. With one
row per document, the required row-to-document map is exactly `range(192)`.

Bind exact Family-F parents: v2 recovery authority `ca759beb...13dbd`, result
`18b03ccf...97c5`, receipt `e8167309...65a5a`, and the sealed v1 program tensor
`d4af5bfb...a038`. The selected program key is exactly
`real_F_binary_native_down_k512`. Its outcome may motivate this experiment but cannot
change any rule below.

## Programs and deterministic controls

The candidate support is the artifact's ordered 512-index tensor. The random support
is `torch.randperm(4608, generator=manual_seed(2026082907))[:512].sort().values`.
Both use balanced native Left/Right rows, exact native Down columns at their support,
and the native bias once. The decoder-shift arm keeps candidate feature rows in artifact
order and uses native Down columns indexed by `torch.roll(candidate_support, -1)`.
This preserves decoder-column norm multiset while breaking each feature/column pair.

Ordinary arms are native, zero write, candidate, random, shifted decoder, and a second
autonomous exact-native replay. The candidate/random/shift callbacks execute 512
products and zero native MLP3 calls. An ordinary full-model forward supplies the exact
replay known answer, never an extra scientific arm.

## Frozen fit-error directions

On all positions 64--255 of the old n480 fit role, give every physical error vector
equal weight. Let `E` be float64 CPU candidate-minus-native writes. Center by its global
row mean and form `E.T @ E / len(E)`. Use `torch.linalg.eigh`, descending. Require every
adjacent gap among eigenvalues 1--5 to exceed `1e-10*max(lambda_1,1)`; otherwise fail
before fresh rows. Orient each top-four vector so its largest-absolute coordinate
(smallest coordinate index on ties) is positive. Scale each to RMS one:
`u *= sqrt(1152)/||u||_2`. The scalar fit-error RMS is `sqrt(mean(E**2))` before
centering. Publish float64 directions, mean, eigenvalues, gaps, error RMS, tensor hashes,
candidate/program hashes, and a receipt before the fresh row tensor is deserialized.

## Exact edit and score semantics

All suffix logits are sliced to positions 64--255 and softcapped exactly once. KL means
`KL(native || arm)` over all 50,304 logits. CE uses ground-truth next tokens. Centered
logits subtract their vocabulary mean independently at each row/position.

Error secants use response `center(G(write))-center(G(N))`. For each amplitude 0.5 and
1.0 and sign, compare response for `N + sign*amplitude*(C-N)` with the corresponding
linear target `sign*amplitude*(center(G(C))-center(G(N)))`; cosine is between those two
responses. Candidate `N+e` is reused byte-for-byte from the ordinary arm, not rerun.

For edit-port cell `(direction,sign,epsilon)`, compare candidate response
`center(G(C+delta))-center(G(C))` against native response
`center(G(N+delta))-center(G(N))`, where
`delta=sign*epsilon*fit_error_rms*u`. NRMSE denominator is native-response energy;
cosine and norm ratio use those same centered responses. Native materiality is
`KL(G(N)||G(N+delta))/KL(G(N)||G(0)) >= .05` for that exact cell. Every registered
cell must be material; nonmaterial cells fail the edit-port gate rather than disappear.
“No ordinary regression” means candidate ordinary CE q95 remains at most .01 and is
conjoined into the edit-port pass.

## Bootstrap and gates

Use batch size four, 2,000 source-document bootstrap draws, seed 2026082911, shared
multinomial document multiplicities, and PyTorch/type-7 linear quantiles. Point values
are document-summed numerators divided by document-summed denominators. For a metric
family, simultaneous upper bounds are `point_i + q95(max_i(draw_i-point_i))`; lower
bounds are `point_i - q95(max_i(point_i-draw_i))`. Norm-ratio two-sided bounds apply
this construction separately to upper and lower deviations.

The parent thresholds and interpretation table are unchanged. Raw predicates are named
`*_qualifies`; every `*_pass` is conjoined with exact replay, positive finite stake,
source/row/program/call integrity, and ordinary-baseline controls. No averaging or
dropping cells is allowed.

## Exact call plan and lifecycle

Fit direction capture: 120 four-row prefixes and no suffix. Fresh scoring: 48 prefixes,
48 ordinary full-model replays, 48 teacher/native suffixes, and 1,920 student suffixes
(40 per batch: five ordinary, three non-reused error-secants, and 32 edit-port suffixes).
Physical attention/MLP calls are 216 at sites 0--3 and 2,016 at sites 4--17. Native
MLP3 calls are exactly 216; compiled student arms call it zero times.

Required transaction: committed/pushed complete sources and row freezer -> independent
audit -> pristine namespace/lock -> exact parent, row receipt, registry, source, and
checkpoint bindings -> create-only authority -> load old fit rows/program/model ->
publish fit-direction artifact and receipt -> final-attempt token -> load the fresh role
once -> execute exact ledger -> restore hooks/model equality -> semantic result and
manifest -> result -> receipt last. A failure spends v1 and preserves every prior write.
