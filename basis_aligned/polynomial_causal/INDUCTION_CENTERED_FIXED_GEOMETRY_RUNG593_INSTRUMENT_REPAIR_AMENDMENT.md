# Rung 593 prospective instrument-repair amendment

**Frozen:** 2026-09-04 UTC, after the R592 invalid terminal and before any R593 implementation, model call, or outcome

**Status:** prospective instrument repair; no execution authority until a different agent approves exact implementation bytes

## Scope and immutable authorities

R593 repairs only two instrument predicates that made R592 invalid after its first FIT endpoint call. It does not reuse
an R592 namespace and it does not reinterpret that invalid terminal. These post-outcome authorities are bound:

- exact executed R592 candidate commit: `7c6be867fcca7a64b3e6dffbff4540e645a32c4e`;
- R592 invalid diagnostic SHA-256:
  `e2d858f8e830d25defab60a38bd4ff7a245d2e1ae2460cdbbba64119ec21f8ae`;
- independent post-execution audit commit: `658c9db0e`;
- independent audit script SHA-256:
  `cc36365e6dc95d6975b181ff96ad6c1f1bc44980d05c1afb25e84ff1252ddace`;
- independent audit report SHA-256:
  `1398d4907d868ff3053c3e0690861a8c0be48f19b1b1286cbbb2534d56622b46`;
- R585 execution authority SHA-256 values remain exactly those pinned by R592, including R578 rows
  `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6` and R585 manifest
  `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962`.

All R592 scientific hypotheses, target/control cells, recipient/donor directions, centered interventions, scoring
formulas, numerical science thresholds, FIT-frozen scales, 2,000 group-bootstrap replicates, FIT-first SELECT opening,
FINAL/OOD closure, and partial-output-factor claim boundary remain unchanged. Physical width remains 30; vocabulary
width remains 50,304; the call price remains 639 FIT plus conditional 322 SELECT, or 961 maximum forwards, with zero
backwards and zero weight updates.

R593 uses new result, receipt, evidence, invalid-diagnostic, invalid-receipt, and invalid-evidence namespaces containing
`rung593`. Every R592 outcome byte remains read-only.

## Repair 1: exact authority-derived support, not all-true support

For endpoint $i$, semantic role $r\in\{A,C\}$, and each registered site $s$, define

$$
S_{i,s,r}
=
\mathbf 1\left\{x_i[p_{i,r}-1]=x_i[q_i]\right\},
$$

where $p_{i,r}$ is the authority's payload position and $q_i$ is its final query position. The site index repeats the
same token-equality fact because all four sites read the same semantic token coordinates. `support.npy` remains a bool
array of shape `[batch,4,2]`, but its validity condition is now exact array equality with this model-free authority mask.
`support.all()` is forbidden.

Each endpoint call manifest must bind the expected mask's C-order raw-byte SHA-256, true count, false count, endpoint
IDs, and order. Runtime evidence must equal those bytes exactly. Complete phase evidence must independently reconstruct
the expected mask from the authority, compare every bit, and bind an ordered support-record hash and the exact counts.

The frozen ordered support record is

```json
{"endpoint_id": "...", "expected_support": true, "role": "A", "site": "L5H5", "split": "FIT"}
```

sorted by `(split, endpoint_id, site, role)`. Its exact census is:

| scope | records | true | false | canonical-record SHA-256 |
|---|---:|---:|---:|---|
| FIT | 13,824 | 5,760 | 8,064 | `ad2e827af9d7fada09327aa27c9465173aa283ee918599bfd5cb5ee107f79d6a` |
| SELECT | 6,912 | 2,880 | 4,032 | `b33ebe9b6d971dd1d09cd3ab797b888703c718457e548ed8f2b244a6698397c9` |
| combined | 20,736 | 8,640 | 12,096 | `25a8b2e9c4cf2175c37f8aa08e3fd5b127397b441ab2e30d609b125bf03dcceb` |

At endpoint level the FIT histogram is exactly 288 endpoints with zero supported roles and 1,440 with one; SELECT is
144 with zero and 720 with one. No endpoint has two. The 54 ordered FIT endpoint-call mask descriptors have canonical
hash `8ad3c99504273cb41873700bf240bca42b62cae7ae42942a063f7fe853ea0f5c`; the 27 SELECT descriptors have hash
`904d2eb8327a7092b40b0ad6f80c242a4a90150a03b79fd83e48452dc806ede5`. A descriptor contains call index, true
count, false count, and raw mask SHA-256.

A mismatch remains `factor_transport_failed` and remains instrument invalidity. A planted test must use real frozen
authority examples with both zero and one supported roles, flip one true bit and one false bit separately, and show
that both mutations fail. Filling the mask with all true must fail.

## Repair 2: independent float64 decomposition primitive at the unchanged tolerance

The R592 failure was an unscaled comparison after three separately rounded float32 contractions. R593 does not loosen
the $10^{-5}$ threshold and does not define the remainder as `head - equality`, which would make the test tautological.
Instead, it evaluates a separate audit primitive in adequate precision.

Let $p_{qk}$ be the already-computed native float32 attention pattern, $v_k$ the already-computed native float32 value
vector, $W_O$ the native projection slice for the head, and $m_{qk}$ the exact equality-successor mask. Cast $p$, $v$,
and $W_O$ individually to IEEE float64 before any of the following three contractions. Independently compute

$$
C^{64}=W_O^{64}\left(\sum_k m_{qk}p_{qk}^{64}v_k^{64}\right),
$$

$$
R^{64}=W_O^{64}\left(\sum_k (1-m_{qk})p_{qk}^{64}v_k^{64}\right),
$$

and

$$
H^{64}=W_O^{64}\left(\sum_k p_{qk}^{64}v_k^{64}\right).
$$

The equality and complement contractions must use separate masked reductions; $R^{64}$ may not be defined from
$H^{64}-C^{64}$, and $H^{64}$ may not be defined from $C^{64}+R^{64}$. The retained falsifier is exactly

$$
\max_{i,s,d}\left|C^{64}_{i,s,d}+R^{64}_{i,s,d}-H^{64}_{i,s,d}\right|\le 10^{-5}.
$$

The three arrays `native_equality_term.npy`, `native_non_equality_remainder.npy`, and `native_head_write.npy` become
little-endian float64 in endpoint and directed-native raw/canonical evidence. Their shapes and names do not change.
`factorized_equality_term.npy`, factors, hook deltas, logits, full native write, and independently reconstructed full
native write remain float32. The factorized-versus-native equality comparison still blocks only a literal
remove-and-insert claim; it is not promoted to a validity gate.

The original complete-nine-head float32 reconstruction gate remains independently required at $10^{-5}$. Thus the new
primitive tests the equality/complement partition in high precision while the existing gate tests fidelity to the
actual native model write. Neither can substitute for the other.

Model-free tests must use values at the observed fp32 scale (head-output RMS approximately 28, including values with
ULPs above $10^{-5}$), demonstrate that independent float64 masked contractions pass at $10^{-5}$, and show that a
planted $2\times10^{-5}$ structural error fails. They must also show that a subtraction-defined remainder is rejected
by the implementation contract.

## Evidence bytes and capacity

Changing exactly three endpoint and three directed-native arrays from float32 to float64 adds 453,869,568 bytes to
complete FIT+SELECT evidence and 1,769,472 bytes to the largest live 32-row five-call chunk. The exact registered data
sizes become:

| quantity | bytes |
|---|---:|
| complete FIT canonical arrays | 5,501,463,552 |
| complete SELECT canonical arrays | 2,750,731,776 |
| complete FIT + SELECT canonical arrays | 8,252,195,328 |
| largest live five-call chunk | 43,440,640 |
| maximum streaming data peak | 8,295,635,968 |

The prior safety margin remains exactly 1,160,003,072 bytes. Therefore the pre-model threshold becomes
9,455,639,040 bytes and the phase-relative pre-SELECT threshold becomes

$$
2{,}750{,}731{,}776+43{,}440{,}640+1{,}160{,}003{,}072
=3{,}954{,}175{,}488.
$$

Equivalently, `9,455,639,040 - 5,501,463,552 = 3,954,175,488`. Equality passes and one byte below either boundary
fails before crossing the protected model-call boundary. These are storage guards only; no scientific row, model call,
score, bootstrap, or decision threshold changes.

## Terminal and publication rules

All R592 instrument predicates, ordering, finite/mask behavior, strict JSON, streaming state machine, incomplete-call
hard abort, and evidence-result-receipt publication order carry forward under R593 names. The repaired support mismatch
and float64 equality/remainder mismatch are still instrument-invalid terminals, never scientific nulls. FIT must clear
every instrument and scientific gate before SELECT. FINAL and OOD remain closed.

The invalid receipt must additionally bind the exact producer/runtime/adapter source hashes and checkpoint weights hash
when model construction succeeded, closing the lineage limitation recorded in the R592 post-execution audit. No R593
execution may be queued until a different agent verifies the amendment, exact implementation dependency closure,
capacity arithmetic, real authority fixtures, and all inherited R592 state-machine attacks.
