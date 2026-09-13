# Native shared/private fit: first sweep and execution cost

13 September 2026. [Registration](FULLU_SHARED_LOCAL_TIMING_V1_PREREGISTRATION.md), [result](FULLU_SHARED_LOCAL_TIMING_V1_RESULT.json), [representation and prices](FULLU_SHARED_LOCAL_FEASIBILITY_V1_MATH.md).

All three registered bars held. Full-vocabulary metric construction plus two native initializations and sweeps took 14.95 seconds, with peak allocated GPU memory 3.17 GB. No text or model body forwards were used. Native coefficient energy replays the earlier CPU calculation to 5.20e-14 relative error; GPU smaller-Gram/SVD controls pass.

| Global / groups / private | Initialization | One sweep | First-sweep full coefficient error | Optimal global error at matched bytes |
|---|---:|---:|---:|---:|
| 64 / 32 / 8 | 0.72 s | 2.60 s | 85.08% | 85.85% |
| 128 / 64 / 16 | 2.31 s | 8.32 s | 76.49% | 77.74% |

Both initial fits slightly improve on a matched-storage single global subspace. These are one-sweep results, not evidence of convergence, identified token groups, or behavioral preservation. All bilinear products remain; only the output map is being approximated. The native sweep cost permits meaningful multistart optimization.

## Sharing the computation during optimization

The original group assignment projects each token onto the joined span of global bank P and private bank Q_k. With row-orthonormal P, write

$$
H_k=Q_k-(Q_kP^\top)P=U_k\Sigma_kV_k^\top.
$$

The joined projection decomposes into a common part and an orthogonal private part:

$$
\|\operatorname{proj}_{[P;Q_k]}X_v\|^2
=\|X_vP^\top\|^2+\|X_vV_k\|^2.
$$

The global term is identical for every candidate group. Compute it once. For a chosen group with independent joined rows, recover coefficients by

$$
b_v=(X_vV_k)\Sigma_k^{-1}U_k^\top,\qquad
a_v=X_vP^\top-b_vQ_kP^\top.
$$

The new [encoder](shared_local_reused_encode_v1.py) falls back to the original joined SVD if the private complement is nearly deficient, preserving its numerical-rank and coefficient-gauge conventions.

The main assignment projection widths drop from k(g+r) to g+kr: 2304 to 320 and 9216 to 1152. This counts the principal projection multiplications, not every operation or an end-to-end speedup. On a synthetic 1024-by-1152 CPU fixture, the controlled encoder took 0.031 versus 0.148 seconds (4.81 times faster), with identical assignments, coefficient error 3.33e-15 and prediction error 7.73e-16. [Control](SHARED_LOCAL_REUSED_ENCODE_V1_CONTROL.json). Native GPU equivalence and full-fit speed remain to be measured separately. The completed bound timing run used the original encoder throughout.

## Token-wise baseline audit while the multistart fit runs

The full fit started at 05:19:11 under the [new registration](FULLU_SHARED_LOCAL_FIT_V1_PREREGISTRATION.md). Its results remain separate from the completed timing experiment above.

The [read-only token audit](audit_shared_local_tokens_v1.py) computes each token's coefficient error relative to its own full quadratic-function norm. It retains the exact vocabulary mean, so this is the same function target as the aggregate fit. The optimal global baselines were executed on CPU in 6.39 seconds, with exact aggregate spectral replay:

| Matched global rank | Aggregate error | Median token error | 90th percentile token error | Tokens with error at most 50% |
|---|---:|---:|---:|---:|
| 78 | 85.85% | 86.29% | 89.79% | 0.93% |
| 167 | 77.74% | 78.00% | 81.84% | 1.05% |

[Receipt](SHARED_LOCAL_GLOBAL_TOKEN_BASELINES_V1_RESULT.json). These large baseline errors are widespread across token functions; a few unusually large token rows do not explain them. This narrows one possible explanation, without proving poor behavior on actual model inputs. Coefficient error still weights arbitrary polynomial input directions, rather than the model's producer-constrained input distribution.

Once each frozen compiled program and its configuration receipt exist, the same audit measures per-token improvement/worsening against the matched baseline, group-wise errors, literal nonzero code counts, and global/private component cancellation. It also independently replays the saved FP32 program's full coefficient error. No token labels, text fitting or post-hoc regrouping enter the audit.
