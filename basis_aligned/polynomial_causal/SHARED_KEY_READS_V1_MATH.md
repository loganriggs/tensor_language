# One shared key-subspace read for both QK factors

13 September 2026. This is exact computational sharing inside the existing folded head8-to-head9 response program. It reduces stored tensor scalars by 19.96%, without changing its function or resolving its conditional-input limitations.

Let $B\in\mathbb R^{1152\times64}$ be the frozen key subspace, $K_1,K_2\in\mathbb R^{128\times1152}$ the two key maps, and $x$ a residual vector. Both reflected-key numerators need

$$
K_iBB^\top x=(K_iB)t,\qquad t=B^\top x\in\mathbb R^{64}.
$$

The existing compiled response used separate dense readers $K_1BB^\top$ and $K_2BB^\top$, totaling 256 intermediate coordinates. The new representation computes the same 64 coordinates $t$ once and sends them through two $128\times64$ adapters. The four full QK reads and value read remain. Total reader width is therefore $512+64+1=577$, versus 769.

This sharing also applies when the input is produced by the folded directional response: project each generated component into $B$ once. Both QK factors consume that result. It does not separate the two QK factors into independent semantic tasks.

[The implementation](shared_key_reads_v1.py) reuses the existing response-feature and scalar executors. It changes the reader matrix, its folded response map and writer read, while charging both adapters. Program tensor counts fall from **1,088,643 to 871,363 scalars**. The saved 217,280 scalars are

$$
192\cdot1152+192\cdot64+192-2\cdot128\cdot64=217280.
$$

The terms respectively account for reader weights, folded rank64 response weights, direction reads, and the new adapters. Counts include all tensor fields in these compiled dictionaries; they do not include native context generators, producer-amplitude generation or suffix weights.

[The actual-weight control](SHARED_KEY_READS_V1_CONTROL.json) evaluates 72 existing cached prompts. Expanded feature error is at most $1.03\times10^{-15}$; scalar error against the previous executor is at most $5.32\times10^{-15}$. The underlying rank64 approximation is unchanged. The reported comparison to the frozen runtime uses that same approximate response, so it is not a new native causal validation.

[The CPU timing](SHARED_KEY_READS_V1_PRICE.json) measures folded feature generation plus the complete scalar calculation, including temporary expansion for the existing consumer. Seven interleaved trials of twenty evaluations on one 16-token prefix give 1.274 ms versus 1.154 ms, or **1.104×**. This narrowly passes the declared 1.1× bar in that setting; it is not broad throughput evidence. No peak-memory reduction is claimed because the consumer currently expands the compact reads temporarily.

The original regional runtime already expresses its inside reads through the shared basis, albeit with the projection repeated in its loop. This change removes redundancy introduced by the earlier dense folded-reader compilation. It is a concrete shared computation and local implementation improvement, not a newly discovered semantic circuit or a full-model speedup.

## Can the shared read be smaller than 64?

Stack the two consumer maps into $A=[(K_1B)^\top,(K_2B)^\top]^\top\in\mathbb R^{256\times64}$. The rank-$r$ truncated SVD gives the least-Frobenius-error common linear read subspace: replace $B^\top x$ by $V_r^\top B^\top x$, with adapters $AV_r$. This optimizes the stated matrix objective directly, without iterative solver or initialization uncertainty. The checked basis is orthonormal within $1.58\times10^{-14}$, so its input folding preserves that coefficient norm.

[The weight-only fit and paired-response screen](SHARED_KEY_CONSUMER_RANK_V1_CONTROL.json) fails the 10% weight and response bars at all tested ranks:

| Shared width | Weight error | Scalar-response error across three families | Additional program storage saved |
|---|---:|---:|---:|
| 16 | 71.8% | 31.3–38.1% | 8.11% |
| 32 | 53.6% | 28.4–32.8% | 5.41% |
| 48 | 35.1% | 20.1–23.2% | 2.70% |

Response error compares the difference between edited and pristine outputs for each candidate against that same difference in the uncompressed shared-read executor. It therefore does not hide error behind a large pristine output. The existing rank64 transport approximation remains common to both sides; these are cached examples, not fresh native validation.

[An equal-consumer countercheck](SHARED_KEY_CONSUMER_BALANCED_V1_CONTROL.json) normalizes each consumer by its Frobenius norm before fitting the shared basis, then restores actual amplitudes in the adapters. It also fails: at width48 response errors remain 19.3–22.7%. Consumer weighting does not explain away this miss.

Retain the exact 64-coordinate shared read. This null concerns a common linear subspace chosen by two coefficient objectives. It does not rule out structure in the joint QK product, position-dependent interactions, or sparse/block representations with a different approximation target.
