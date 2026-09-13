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

## Initial fit result and executed counterchecks

[The managed fit](FIXED_MIXED_HEAD_PRIVATE_V1_RESULT.json) finished in 381.37 seconds. Monotonicity and serialization replay pass; convergence and the matched-capacity improvement criterion fail. The best squared relative weight error is 0.253271, versus 0.222835 for exact global rank208: about 13.7% worse squared error. Two promoted starts completed 80 extension cycles without plateau; the third stopped at the time limit after eight cycles. Their failure is not evidence that the structure is absent or globally optimized.

[The postfit audit](FIXED_MIXED_HEAD_PRIVATE_V1_AUDIT.json) finds shared and private norms 0.823 and 0.425 times the target norm, with cosine -0.159. There is no large-cancellation pathology in these aggregate numbers. One additional exact conditional cycle improves squared error by $1.28\times10^{-5}$ relatively, consistent with the failed convergence criterion. The factor executor replays the dense reconstructed fit within $1.77\times10^{-15}$.

Private rank allocation was also tested rather than assuming every head needs 16 factors. Holding the shared matrix fixed, globally selecting the largest 144 private residual singular values gives ranks [16,16,15,23,18,16,12,12,16]. This reduces squared error by only 0.2334%, leaving most of the matched-baseline gap. It is an exact conditional allocation result, not a converged joint fit with variable topology.

The structured executor is faster than the exact dense matrix on the measured CPU batches, but slower than the equally priced rank208 executor: about 0.084/0.228/1.163 ms versus 0.063/0.185/1.098 ms at batches1/16/128. These comparisons have different approximation errors. Better speed than the larger exact matrix does not establish an adopted fidelity/runtime tradeoff.

An [independent planted control](HEAD_PRIVATE_MATRIX_V1_INDEPENDENT_CONTROL.json) recovers one exact 48-by-72 structured matrix from all ten initializations to squared relative error below $10^{-8}$. This strengthens the implementation check beyond the earlier near-planted control, without guaranteeing recovery on native weights or unique shared/private factors.

The [CPU kernel measurement](HEAD_PRIVATE_MATRIX_V1_CPU_PRICE.json) takes 0.267–0.294 seconds per exact alternating cycle. The managed GPU run performed 228 cycles in roughly381 seconds including setup and bookkeeping; this is not a precisely matched device-kernel benchmark, but it supports using CPU for the bounded convergence extension. That extension starts from the best serialized private component, retains the same objective and ranks, and allows up to800 cycles/240seconds with the unchanged three-gain plateau criterion. Its result belongs in a separate receipt; it cannot retroactively make the original multi-start convergence predicate pass.

## CPU convergence extension completed

[The continuation](FIXED_MIXED_HEAD_PRIVATE_CONTINUE_V1_RESULT.json) reaches the declared objective plateau in112.14seconds. Squared relative error falls from0.2532711714 to0.2530416660, still13.56%worse than globalrank208 at equal parameter count. Its monotonicity/replay and plateau criteria pass; its improvement criterion fails. This is convergence of the continued best fit under the declared objective-change test, not proof of global optimality or convergence of every original start.

The initial remaining optimization gap explains little of this matched-baseline deficit. Together with the independent planted controls, moderate component norms and weak budget-reallocation gain, this makes another fixed native-head private-factor fit lower priority. It does not rule out different partitions or interaction-specific arithmetic structure. The next registered action validates the stronger three-term rational representation on fresh template prompts with live-generated upstream fields.
