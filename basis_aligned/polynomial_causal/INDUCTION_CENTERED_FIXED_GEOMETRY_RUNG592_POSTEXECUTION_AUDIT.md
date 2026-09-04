# Independent post-execution audit: R592

**Audited:** 2026-09-04 UTC

**Executed candidate:** `7c6be867fcca7a64b3e6dffbff4540e645a32c4e`

**Pre-execution approval:** `4149ae2644defa9786b2046911d860afdc95805b`

**Classification:** **INVALID INSTRUMENT — no scientific null or hold may be inferred**

The managed run stopped after its first FIT endpoint call. It did not run any directed native/replay/interchange arm,
score FIT, open SELECT, construct a bootstrap, test an active control, or evaluate counterfactual selectivity. The
result therefore says nothing about whether coefficient or projected-content factors causally transfer induction
behavior.

## Bound terminal bytes

| artifact | SHA-256 |
|---|---|
| invalid diagnostic | `e2d858f8e830d25defab60a38bd4ff7a245d2e1ae2460cdbbba64119ec21f8ae` |
| invalid receipt | `069f3f65b119d8d0a5884aef7a4c7e4b9a518f8bb801d619b272f5787ad0ca24` |
| one-call prefix | `ca61f9475f42a84d870685fcd78f22ef26a36e19797d813f026eb62ab79a5261` |
| managed run log | `f06050f4ed6ec59ffe39f89a0c6c6185d28249b2a5e3888a68cc418c7d8a7e5b` |

The independent auditor streamed and verified every byte length and SHA-256 in the receipt: 36 files containing
5,210,059,038 bytes. The receipt binds the diagnostic, call prefix, complete preallocated FIT files, and all raw arrays
from `FIT:endpoint:0000`. The prefix binds the first 32 endpoint IDs, their order, token bytes, query positions, every
array's dtype/shape/length/hash, and labels the call as uncommitted `raw_current_chunk`. The canonical slice ledger is
empty, correctly indicating that none of the large preallocated canonical array rows is valid evidence.

All JSON is finite and strictly parseable. All first-call floating arrays are finite. The tokens exactly equal the
first 32 FIT endpoints independently reconstructed from the R578/R585 authority, padded to physical width 30. The
logits have the full registered shape `[32, 50,304]`. Normal result, receipt, and evidence namespaces are absent; only
the atomic invalid evidence-diagnostic-receipt terminal exists.

The currently installed producer, runtime, adapter, R585 authority, R585 manifest, and R578 rows have the exact approved
hashes. The managed adapter verifies those bytes before its immutable producer dispatch. One post-execution limitation
is worth preserving: the invalid receipt itself does not include the source or checkpoint hash. Thus exact execution
lineage is supported by the approved managed path and contemporaneous bytes, but is not self-contained in the invalid
receipt as it would be in a normal result. No scientific claim depends on that limitation because this terminal is
invalid.

## Primitive computations

The diagnostic reports the first failure in the frozen predicate order, but the same completed call triggers two
invalidity predicates.

### 1. Saved native decomposition exceeds the frozen tolerance

For every endpoint $i$, site $s$, and residual coordinate $d$, the audit independently computed

$$
\epsilon_{i,s,d}
=
\left|
C_{i,s,d}+R_{i,s,d}-H_{i,s,d}
\right|,
$$

where $C$ is the saved native equality contribution, $R$ is the saved non-equality remainder, and $H$ is the saved
native head output. The result is

$$
\max_{i,s,d}\epsilon_{i,s,d}
=5.340576171875\times10^{-5}>10^{-5}.
$$

There are 5,246 saved coordinates above $10^{-5}$. This exactly reproduces the diagnostic's primary
`native_equality_remainder_reconstruction_failed` predicate. In contrast, the independently reconstructed complete
nine-head attention write equals the captured full native attention write exactly: maximum absolute error `0.0`.

The separately saved product-factor expression also differs from the native equality contribution by
`5.340576171875e-05` at maximum, with 4,540 coordinates above tolerance. Under the frozen R592 contract this latter
comparison blocks a literal “remove the native term and insert the factorized term” claim but is not itself an R592
validity predicate. It remains important negative evidence for that stronger interpretation.

### 2. The support predicate contradicts the registered dataset

The raw `support.npy` is not corrupted. The audit recomputed each bit directly from the registered token and semantic
coordinates:

$$
S_{i,s,r}
=
\mathbf{1}\{x_i[p_{i,r}-1]=x_i[q_i]\},
$$

where $p_{i,r}$ is the payload position for semantic role $r\in\{A,C\}$ and $q_i$ is the query position. The saved
mask equals this recomputation exactly.

In the first batch, 28 endpoints have exactly one supported role and four have none. Repeated across four sites, that
is 112 true and 144 false support entries. The runtime nevertheless treats `not support.all()` as
`factor_transport_failed`. This does not test whether the observed support equals the registered support pattern; it
demands that both roles be equality-supported at every endpoint and site.

That demand is impossible for this frozen dataset. Across the entire authority:

| split | endpoints with 0 supported roles | endpoints with 1 supported role | true site-role entries | false site-role entries |
|---|---:|---:|---:|---:|
| FIT | 288 | 1,440 | 5,760 | 8,064 |
| SELECT | 144 | 720 | 2,880 | 4,032 |

No endpoint has both roles supported. Therefore the exact implementation was guaranteed to invalidate on its first
nonempty endpoint batch even if every model observation was correct. The pre-execution fake executors filled
`support.npy` with all `True`, so they tested the implementation's assumption instead of the registered counterfactual
data. This is the main reusable review failure.

## Call accounting and unopened analyses

| quantity | audited value |
|---|---:|
| model forwards | 1 |
| completed FIT calls | 1 endpoint call |
| observed endpoint rows | 32 |
| observed directed rows | 0 |
| SELECT calls | 0 |
| backwards / weight updates | 0 / 0 |

The single forward count follows independently from the one executed call ID and matches the managed run log. FIT was
not scored. SELECT, FINAL, and OOD were not opened. Consequently the following registered objects have no realized
evidence and must not be described as passing or failing: directed recipient/donor joins, native-versus-zero replay,
coefficient/projected-content/joint additions, hook-delta identities, active controls, FIT-frozen scales, target/control
thresholds, 124 bootstrap cells, 2,000 replicates, FIT-to-SELECT replication, and the scientific terminal.

There are no nonfinite masks because the failing arrays are finite. The frozen invalid-predicate ordering correctly
selects native decomposition mismatch before support/factor transport. Atomic publication and the receipt bindings are
valid.

## Repair boundary

Do not rerun this exact candidate. A successor needs a prospective, independently reviewed instrument amendment that:

1. compares the observed support mask with the authority-derived expected mask and exact split census, instead of
   requiring every semantic role to be supported; and
2. makes the native equality-plus-remainder identity numerically well-defined at the unchanged $10^{-5}$ boundary.
   The repair should preserve enough primitive precision or use a compensated residual representation; simply
   loosening the tolerance would discard the registered falsifier.

The successor should plant real authority examples with zero and one supported roles, plus values near the actual
float32 scale, rather than all-true/small synthetic fixtures. Only after those instrument gates hold may R592's causal
counterfactual, active-control, bootstrap, and held-out predictions be evaluated.
