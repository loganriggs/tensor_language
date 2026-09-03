# R591 token-support exhaustion audit

**Time:** 2026-09-03T23:36Z  
**Scope:** CPU-only analysis of the frozen R585 endpoint authority  
**Model, CUDA, queue, or outcome access:** none

## Question

R585 reconstructs its proposed equality-supported attention term from the two semantic source roles $A$ and $C$.
The independent canonical contraction instead includes every successor key $k$ satisfying

$$
1\le k\le q,\qquad \operatorname{token}(k-1)=\operatorname{token}(q),
$$

where $q$ is the final query position. If another prompt position also satisfied this equality, the proposed
$\sum_{r\in\{A,C\}}e(r)u(r)$ term would omit real canonical support. That would make R585's replay/native mismatch
structural rather than numerical.

## Computation

For each of the 2,592 frozen FIT+SELECT endpoint token sequences, independently enumerate all successor keys satisfying
the equality above. Separately enumerate the registered $A,C$ payload positions whose predecessor equals the query.
Compare the two integer sets exactly; no attention scores, activations, weights, or model outputs are used.

## Result

- endpoints with an extra canonical equality position: **0 / 2,592**;
- endpoints with a registered equality position absent from the canonical mask: **0 / 2,592**;
- canonical equality-support count per endpoint: **432 have zero positions; 2,160 have one position**.

Therefore the two semantic roles exhaust the canonical equality mask on every frozen endpoint. R591 need not test a
missing-support hypothesis. The remaining live explanations are contraction-order rounding and downstream
amplification, batch/padding numerical geometry, or unintended mutation by the factor observer.

This does not show that replay is numerically exact and does not change R585's frozen $10^{-5}$ full-logit threshold.

