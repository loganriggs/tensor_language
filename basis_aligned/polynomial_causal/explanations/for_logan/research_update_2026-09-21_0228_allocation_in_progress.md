# Allocation experiment in progress; linear-term cost is not free

21 September 2026, 02:28 UTC.

**Completed:** see the [uniform and adaptive allocation results](research_update_2026-09-21_0235_adaptive_allocation.md). The following preserves the earlier in-progress record.

The managed centered-decomposition experiment is still running. At the latest process check it had completed256 of512 matrix decompositions. No final native comparison is available yet. The experiment compares256output directions with two products each against512directions with one each, plus corresponding1,024-product variants.

The candidates retain exact native constant and single-input terms around calibration means. These terms cost2,654,208 weight coefficients; they are explicitly included rather than treated as free background.

While the GPU run continued, two independent CPU steps were completed:

1. Prepared an adaptive512-product allocation using the run's first two singular terms for every output direction. The existing toy oracle checks this allocation against exhaustive enumeration. Export and native evaluation will wait for the parent results to be interpreted; no shared dependency of the live job was changed.
2. Measured how well shared linear features compress the exact first-order terms. This can reduce cost later, but introduces an additional error source.

The linear calculation represents

$$
F_{\mathrm{linear}}(n,m)=F(\bar n,\bar m)
+(n-\bar n)^\top J_n+(m-\bar m)^\top J_m,
$$

with row/column conventions chosen consistently in code. It uses exact original-weight maps and selects output directions from their joint paired-calibration variation. A rank-$r$ implementation reads $r$ shared linear features from both inputs and uses one output writer, requiring $3\times1152\times r$ weight coefficients.

| Shared linear features | Weight coefficients | Calibration linear-variation error |
|---|---:|---:|
| 32 | 110,592 | 25.43% |
| 64 | 221,184 | 21.03% |
| 128 | 442,368 | 16.07% |
| 256 | 884,736 | 10.36% |

The constant is retained and counted as state separately. These errors concern the linear variation alone, not total model error or native intervention fidelity. They do not justify replacing the exact linear terms in the current controlled comparison.

This preparation exposes a second allocation decision: products are not the only cost. Sharing linear reads and writes can save substantial storage, but those losses must be scored separately before combining approximations. The broader circuit goal still requires stable identities, selective behavior, extraction, composition and OOD evidence.

Evidence: live managed log `run_direct_midpoint_centered_allocation_v1.log`; CPU results `MIDPOINT_FIRST_ORDER_SHARED_V1.json` and `ADAPTIVE_OUTPUT_RANK_ORACLE_V1.json` under `direct_tensor_match`. This report records an in-progress state, not a completed model result.
