# R591 replay/native numerical diagnostic findings

Date executed: 2026-09-04 UTC  
Evidence level: diagnostic only; no R585 scientific terminal  
Managed run-log SHA-256: `85403a5ed99e76734229d3063fa9cd666d005eb5d30507f74d4ad6b7a4257002`  
Managed run-log bytes: `32272`  
Reviewed producer commit: `a5e1dd022729c28dad99c1782f557b3162cdf45e`  
Independent approval commit: `d80ce3f39`

## Result

R591 completed all 234 registered FIT-only forwards with zero backwards and zero
weight updates. It emitted no result, receipt, evidence directory, score,
selection, or scientific terminal. The frozen absolute threshold remained
$10^{-5}$.

The registered classification is **mixed**:

- the current replay hook changes final logits beyond the threshold; and
- changing padded sequence length changes native final logits beyond the
  threshold.

Batch membership at a fixed tensor shape is exactly invariant in the controlled
panel, and merely running the factor observer without writing is also exactly
invariant.

## Exact measurements

| comparison | maximum absolute logit difference | endpoints above $10^{-5}$ |
|---|---:|---:|
| full FIT: current replay vs length-sorted native | $4.9591064453125\times10^{-5}$ | 1599 / 1728 (92.53%) |
| full FIT: replay vs native on the same mixed batches | $1.811981201171875\times10^{-5}$ | 946 / 1728 (54.75%) |
| full FIT: native mixed batches vs length-sorted native | $4.9114227294921875\times10^{-5}$ | 1394 / 1728 (80.67%) |
| controlled panel: native padding to length 30 vs native length | $2.8848648071289062\times10^{-5}$ | 256 / 256 (100%) |
| controlled panel: native mixed vs length-grouped membership, both length 30 | $0$ | 0 / 256 |
| controlled panel: factor observer vs native, every schedule | $0$ | 0 / 256 |
| controlled panel: current replay vs native, native-length schedule | $1.811981201171875\times10^{-5}$ | 149 / 256 (58.20%) |
| controlled panel: current replay vs native, length-30 schedules | $1.9073486328125\times10^{-5}$ | 183 / 256 (71.48%) |

The vector identity

$$
\Delta_{\mathrm{total}}
=\Delta_{\mathrm{hook}}+\Delta_{\mathrm{batch+padding}}
$$

closed to a maximum residual of
$4.547473508864641\times10^{-13}$. This confirms that the two measured sources
account for the original discrepancy at numerical precision.

## Local computation

The independently reconstructed complete attention write matched the native
attention write exactly at all four registered sites. The local expression that
projects each role value and then sums did not match the canonical expression
that sums in 128-dimensional head space before the output matrix:

| site | max $|\text{role factor}-\text{canonical}|$ |
|---|---:|
| L5H5 | $3.814697265625\times10^{-5}$ |
| L7H3 | $1.1444091796875\times10^{-5}$ |
| L8H3 | $9.1552734375\times10^{-5}$ |
| L8H4 | $5.340576171875\times10^{-5}$ |

Thus the hook failure has the predicted contraction-order mechanism: R585 writes
the floating-point difference between two algebraically equal implementations
into the residual stream. Later layers amplify enough of that difference to
cross the final-logit threshold.

The equality-support audit remains exact on all 2,592 endpoints: 432 endpoints
have no registered equality successor, 2,160 have one, and there are zero extra
or missing positions.

## Consequences

R585 is not licensed to produce science in its present form. The result rules
out three tempting but incorrect responses:

1. Do not loosen the $10^{-5}$ threshold.
2. Do not blame batch neighbors: fixed-shape membership was exactly invariant.
3. Do not treat factor observation as a mutation: every observer comparison was
   exactly zero.

A prospective successor must repair both active sources:

- use one fixed padded tensor geometry, and compare every intervention to native
  execution with the same membership, order, and padding; and
- use a centered change in one fixed factor expression so self-replay is a
  literal zero tensor.

The centered operation is only a registered equality-factor intervention in
output space. It is not automatically a realizable query/key swap, a full
attention-pattern swap, literal native remove-and-insert, or a sufficiency test.
Those remain separately named gates under handoff v7.

Because the original R585 result namespaces remain absent, this diagnostic does
not retract or reinterpret a scientific result. It explains why the instrument
correctly refused to publish one.
