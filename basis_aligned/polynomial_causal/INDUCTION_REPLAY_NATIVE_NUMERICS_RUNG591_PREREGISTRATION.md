# R591 prospective replay/native numerical diagnostic

Date frozen: 2026-09-03 UTC

Lane: diagnostic GPU work only after independent implementation review. This
document and its present work are CPU-only.

## Purpose and non-scientific status

The R585 managed retry passed the input-shape repair and then stopped at the
hard integrity check

```text
replay/native full-logit comparison failed before publishable evidence
```

No R585 result, receipt, or evidence namespace exists. This is not an R585
scientific null. R591 is a minimal diagnostic to locate the numerical source of
that failed identity while preserving the frozen absolute threshold `1e-5`.

R591 must write no result, receipt, evidence directory, registry record, or
scientific terminal. Its only output is one strict-finite JSON object on stdout,
captured by the managed run log. It must not call any R585 scoring, selection,
terminal, or publication function.

## Frozen source

The diagnostic is derived from R585 managed-repair commit
`c4288dbe8ee6213dfc4dcb538024dc119fbb642e`:

| Input | SHA-256 |
|---|---|
| R585 producer | `fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b` |
| R585 owner test | `fcaba664269de12a41a5adb8ff089fc9963eeec91577ef94993ff032c02fc885` |
| facade | `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c` |
| induction contraction helper | `b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a` |
| R585 manifest builder | `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962` |
| R585 dependency lock | `908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7` |

Checkpoint loading remains local, production-validated, float32, eval mode,
inference-only, and hash-checked against
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.

## Exact difference between the two R585 paths

Let $x$ be an endpoint and $\ell(x;D,B,P)$ its full 50,304-dimensional final
logit vector under dispatcher $D$, batch membership/order $B$, and padded
sequence length $P$.

### Replay capture

`collect_capture_replay` sorts endpoints by endpoint ID, chunks them into
batch-32 mixed-length batches, and pads each batch to its maximum length. Its
dispatcher calls native attention at every layer. At L5H5, L7H3, L8H3, and
L8H4 it additionally reconstructs the equality-supported term and adds, at the
saved final query,

$$
\delta_{h,x}
=\sum_{r\in\{A,C\}}e_{h,x,r}\,u_{h,x,r}
-W_{O,h}\!\left(\sum_{k}p_{h,x,k}M_{x,k}v_{h,x,k}\right).
$$

The first expression projects each selected value and then sums two
1,152-vectors. The second sums selected values in 128-dimensional head space
using `torch.bmm` and then applies $W_{O,h}$. They are algebraically equal when
the two registered roles exhaust equality support, but floating-point
associativity, GEMV/BMM contraction order, and kernel tiling differ. The small
`term - canonical` delta is explicitly written into the residual stream and can
be amplified by all downstream layers.

The path otherwise returns the native attention write and native first-value
state, and uses the native MLP at every layer.

### Native comparator

`collect_native_comparator` uses native attention and native MLP without the
four reconstructed writes. It sorts endpoints by `(length, endpoint_id)` and
chunks them into batch-32 batches. Thus an endpoint can have different batch
neighbors, row index, matrix dimensions, and padded length from its replay
forward.

### What is algebraically invariant but not necessarily bitwise invariant

- There is no operation coupling different batch examples. Changing batch
  neighbors or order should not change the mathematical function of an
  endpoint.
- Padding is strictly after the saved final query. Causal attention prevents
  valid queries from reading padding. Rotary positions on the valid prefix and
  tokenwise MLP/RMS normalization are unchanged.
- Nevertheless, changing batch size/shape, flattened GEMM dimensions, row
  placement, or sequence length can select different GPU kernels and reduction
  tilings. Float32 results can therefore differ even when the real-arithmetic
  function is identical.
- The model is in eval mode with inference mode and no updates, so dropout,
  optimizer state, and stochastic training are not candidate causes.
- The frozen check is an absolute maximum over all 50,304 logits. A local error
  below `1e-5` does not imply the final-logit maximum stays below `1e-5` after
  downstream composition.

These leave three candidate sources: the explicit reconstructed hook, native
batch/padding numerics, or an unintended mutation caused merely by running the
factor observer.

## Frozen diagnostic conditions

Use three dispatchers:

- **N — native:** native attention and MLP only.
- **F — factor observer, no write:** execute exactly the R585 factor
  reconstruction and collect its local errors, but return the untouched native
  attention write. This tests whether observation itself mutates an alias or
  state.
- **R — current replay:** execute the factor reconstruction and add the exact
  four `term - canonical` deltas as R585 does.

### Full-FIT reproduction and first split

Run all 1,728 frozen FIT endpoints in three conditions:

1. `R_M`: R under the exact endpoint-ID mixed schedule and its natural padding;
2. `N_M`: N on the identical mixed batches and identical padded token tensors;
3. `N_L`: N under the exact current length-sorted comparator schedule.

This costs `3 * 54 = 162` forwards and gives, endpoint by endpoint,

$$
\Delta_{\rm total}=\ell(R_M)-\ell(N_L),
$$

$$
\Delta_{\rm hook}=\ell(R_M)-\ell(N_M),
\qquad
\Delta_{\rm batch+pad}=\ell(N_M)-\ell(N_L).
$$

Log the maximum absolute residual of the vector identity

$$
\Delta_{\rm total}
=\Delta_{\rm hook}+\Delta_{\rm batch+pad}.
$$

It is only an arithmetic consistency check, not an attribution score.

### Controlled 256-endpoint panel

For each frozen FIT length in `{19,20,21,22,27,28,29,30}`, choose the first 32
endpoint IDs in lexical order. This produces 256 endpoints before any model
output is seen.

Construct three eight-batch schedules, all containing exactly these endpoints:

- `L_native`: eight length-homogeneous batch-32 batches at their native lengths;
- `L_30`: the same eight memberships and row orders, each padded to length 30;
- `M_30`: for batch index $j=0,\ldots,7$, take lexical ranks
  `4*j : 4*j+4` from each of the eight length classes, preserving length then
  lexical order within the batch. Every batch has 32 rows and padded length 30.

Run N, F, and R on all three schedules: `3 * 3 * 8 = 72` forwards.

The paired differences isolate:

$$
\Delta_{\rm padding}^{D}=\ell(D,L_{30})-\ell(D,L_{native}),
$$

holding endpoint membership and row order fixed, and

$$
\Delta_{\rm membership}^{D}=\ell(D,M_{30})-\ell(D,L_{30}),
$$

holding batch size and padded tensor shape fixed. For each schedule also compute

$$
\Delta_{\rm observer}=\ell(F)-\ell(N),
\qquad
\Delta_{\rm hook}=\ell(R)-\ell(N).
$$

Total price is exactly 234 forwards, zero backwards, zero updates, FIT only.
SELECT, FINAL, OOD, R585 interventions, scores, and terminals remain unopened.

## Frozen run-log fields

For every named difference above, emit:

- exact maximum absolute full-vocabulary error;
- endpoint ID, token ID, length, batch index, and row index attaining the
  maximum;
- number and fraction of endpoints whose maximum exceeds exactly `1e-5`;
- maximum full-vocabulary RMS, for scale description only; and
- for R/F cells, maximum `term - canonical`, `canonical + remainder - head`,
  and reconstructed-attention-versus-native-write errors by site.

Emit the maximum decomposition-residual error and exact realized forward count.
All scalar fields must be finite JSON numbers with `allow_nan=False`. Do not
round the recorded maxima. Do not evaluate alternative tolerances or choose a
tolerance from these data.

The script must fail nonzero if source/checkpoint hashes, panel membership,
schedule shapes, call count, finiteness, or vector-decomposition identity differ
from this registration. Such a failure is a diagnostic implementation failure,
not a scientific terminal.

## Registered interpretation and repair implications

The `1e-5` boundary is unchanged in every clause.

1. **Observer failure:** if any `max |F-N| > 1e-5` within an identical token
   tensor, the factor observer mutates state or aliases unexpectedly. Kill the
   current instrument and repair the observer before considering any replay
   change.
2. **Hook-dominated:** if same-batch `max |R-N| > 1e-5` while `F-N <= 1e-5`
   and native batch/padding comparisons pass, the explicit
   `term - canonical` write is the cause. Do not relax the tolerance. Re-express
   the frozen term with the same contraction order as the live canonical term,
   most directly by combining selected values in 128-dimensional head space and
   applying $W_O$ once, then prospectively re-audit R585.
3. **Padding-dominated:** if `max |N(L_30)-N(L_native)| > 1e-5`, the frozen
   padded/unpadded native identity itself fails. R585 remains an invalid
   instrument at its registered threshold; merely comparing replay and native
   in one batch would hide rather than repair clause 5.
4. **Membership/GEMM-dominated:** if
   `max |N(M_30)-N(L_30)| > 1e-5`, identical-shape batch membership or row
   placement changes float32 output beyond the frozen threshold. R585's
   cross-batch identity is invalid as written. A repair must use a prospectively
   fixed paired execution geometry or a demonstrably invariant numerical
   implementation, not a post-hoc larger tolerance.
5. **Mixed:** if more than one component exceeds `1e-5`, repair each active
   source and rerun the same diagnostic unchanged. No component may be dismissed
   because another is larger.
6. **All components pass but original total fails:** treat this as a failed
   diagnostic or missing factor. The exact vector decomposition and endpoint
   attaining the original maximum must localize the discrepancy before any
   R585 retry.

Passing R591 only restores instrument plausibility. It cannot establish R585's
selector/payload causal claim and cannot license a result by itself.

## Prospective panel amendment — 2026-09-03T23:47Z UTC

This amendment was frozen after CPU-only implementation work found an
impossibility in the original panel definition and before any R591 model
execution or outcome. The exact R585 authority has these endpoint-length
histograms:

- FIT: `{19: 960, 20: 480, 27: 192, 28: 96}`;
- SELECT: `{21: 480, 22: 240, 29: 96, 30: 48}`.

Thus the original request for 32 **FIT** endpoints at each of all eight lengths
cannot be materialized without opening SELECT, which is forbidden. Replace only
the controlled-panel membership and schedules with the following exact
definition:

- select the first 64 FIT endpoint IDs in lexical order at each of lengths
  `{19,20,27,28}`, still totaling 256 endpoints;
- `L_native` and `L_30` each contain two consecutive lexical batch-32 groups per
  length, ordered by length and then within-length group;
- for `M_30` batch index $j=0,\ldots,7$, take lexical ranks
  `8*j : 8*j+8` from each of the four FIT length classes, preserving length then
  lexical order inside the batch.

All other conditions remain unchanged. There are still eight batches per panel
schedule, nine dispatcher-by-schedule cells, 72 panel forwards, 162 full-FIT
forwards, and 234 forwards total. The revised panel strengthens the per-length
sample size while preserving the padding and fixed-shape membership contrasts.
The implementation dry run must record both split-length histograms and reject
the impossible original eight-length FIT panel.

## Prospective contract amendment — 2026-09-04 UTC

This amendment follows the independent exact-byte review of commit `1396747c0`
and precedes any R591 model execution. It changes no rows, schedules, numerical
measurements, threshold, or price.

- The registered **padding** and **membership/GEMM** cause booleans use only the
  N-dispatcher contrasts stated above. F/R padding and membership contrasts are
  still emitted descriptively but cannot assign a native numerical cause.
- The dry-run panel receipt emits the exact ordered 256 FIT endpoint IDs, their
  ordered hash, and the per-length census, satisfying the shared v5 support
  contract without changing panel membership.
- Endpoint authority is rebuilt from directly hash-pinned R578 rows and the R585
  semantic manifest. Dry-run authority construction must not open or parse R586
  or R587 outcome artifacts.
- Managed execution must run the exact producer bytes that passed the adapter's
  hash check, not look up the mutable producer pathname again after preflight.

These are prospective implementation-boundary corrections. A different agent
must review the repaired exact bytes before execution.
