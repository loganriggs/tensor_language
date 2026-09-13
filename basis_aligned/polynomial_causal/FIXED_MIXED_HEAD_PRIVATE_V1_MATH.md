# Shared factors with head-specific corrections for the mixed operator

13 September 2026. Registered weights-only fit; consult the result receipt for execution status.

The preceding rank-128 SVD preserved the regional target contribution but missed controls and FineWeb. This experiment changes the assumed structure, rather than fitting that same rank objective longer. The object remains the exact fixed residual/attention mixed matrix $M=J_{\lambda w}O_{10}$; see [its derivation and native evidence](FIXED_WRITER_ATTENTION_OPERATOR_V1_MATH.md).

Partition the input columns into nine attention heads, each 128 wide. Seek

$$
M\approx AB+[C_1D_1\;C_2D_2\;\cdots\;C_9D_9],
$$

with $A\in\mathbb R^{1152\times128}$, $B\in\mathbb R^{128\times1152}$, $C_h\in\mathbb R^{1152\times16}$, and $D_h\in\mathbb R^{16\times128}$. Shared factors can read every head; a private factor can read only its own head. All may write into the same residual output space. Native heads provide candidate input blocks, not assumed semantic circuit boundaries.

The parameter count is exactly

$$
128(1152+1152)+9\cdot16(1152+128)=479232.
$$

An ordinary global rank-208 factorization has precisely the same count, $208(1152+1152)$. The structured model can use up to 272 intermediate directions, but private input connectivity is sparse. Thus it tests whether additional output directions with restricted input connections are more useful than fewer globally connected directions. Dense output adapters are fully counted.

Each alternating step solves two conditional problems exactly: truncated rank-128 SVD of M minus the private matrix, followed by separate rank-16 SVDs of each head block after subtracting the shared matrix. These steps decrease squared Frobenius loss, but the joint problem is nonconvex. Ten initializations receive six cycles each; the three strongest continue for up to 80 more cycles under a 420-second total cap. Initializations include shared-first, private-first and seeded perturbations at two scales. Three consecutive relative objective gains at most $10^{-7}$ constitute an objective plateau, not a global-optimality or stable-identification certificate.

Predictions: exact serialized replay and monotonicity; at least two promoted fits reach the plateau criterion; and at least 5% smaller squared weight error than the exactly solved rank-208 baseline. All outcomes remain weight-only. Neither successful fitting nor equal scalar count establishes runtime savings or native circuit preservation.

[The near-planted CPU control](HEAD_PRIVATE_MATRIX_V1_CONTROL.json) converges monotonically from a perturbed true private component to relative squared error $5.02\times10^{-30}$. That is a conditional algebra/control check, not proof of independent recovery. [The executable control](HEAD_PRIVATE_MATRIX_EXECUTOR_V1_CONTROL.json) compares factor execution with an independently reconstructed dense matrix on arbitrary factors and batched token inputs, agreeing within $4.11\times10^{-16}$.

The [executor](head_private_matrix_executor_v1.py) computes shared features once, adds the nine private contributions, and never materializes the dense sum. Native evaluation will need to compare its error against the mixed term's own effect, as in the rank-128 test, rather than letting the full background hide errors. The head-write producer, normalization, other residual modes and suffix remain outside this component's parameter count.

The initial enqueue was rejected before GPU execution because the static gate did not recognize keyword-style `dict(pred_a=...)` construction. Explicit predicate keys repaired that mechanical issue; the reviewed runner hash and binding were updated before enqueue. No failed scientific run was restarted.
