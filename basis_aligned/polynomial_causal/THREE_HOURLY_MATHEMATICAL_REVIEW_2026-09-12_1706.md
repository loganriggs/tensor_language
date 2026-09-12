# Three-hour mathematical review — 12 September,17:06 cycle

Recorded17:12UTC after source inspection and an executed CPU consequence. Goal: a simpler executable model with fresh/OOD prediction, extraction, selective manipulation and composition/reuse. The fresh shared producer pair is conditional; its native current-state inputs remain unresolved. This review distinguishes local read compression from a closed upstream computation.

## Actual object and price

For each of head8.2 and head9.8, current inputs x(t) are1152-dimensional normalized residual states; Q1,K1,Q2,K2 are128x1152; the current value reader v is1152-dimensional; first-value readings are a50304-entry table per producer. With positions explicit, the scalar output is

$$
a(t)=\sum_{s\le t}\gamma(t,s)\,[v^Tx(s)+b(\operatorname{token}_s)],
$$

where gamma is the product of the two native normalized, rotated QK scores. Two such outputs, on different layer states, feed one four-dimensional writer. With a formal independent first source, the numerator has query degree2/source degree3. With token lookup fixed, source degree is at most3. Actual RMS gates remain nonpolynomial. Later bilinear MLPs add polynomial interactions with normalization; final tanh is not rational.

The validated pair stores1,282,566scalars/5,541,936tensorbytes;1,179,648scalars are full QK maps. Allowed controls below use arbitrary real input sequences, not language examples. Error norm is relative Frobenius on outputs/projections, with native FP32 replay separately checked. Orthogonal read-coordinate changes and compensated reader/writer scaling are gauge freedoms; no new semantic interpretation follows from choosing a basis.

## Literature mapping and limits

[CLUE](https://arxiv.org/html/2004.11961) computes the smallest exact linear lumping containing specified linear observables for polynomial dynamics. Its Jacobian coefficient matrices define the required common invariant rowspace; the supplement explicitly includes discrete-time polynomial maps. For constant matrices with total T nonzeros, state dimension n and resulting dimension r, its invariant-space routine has expected arithmetic cost O(r n(T+r)); this does not include unlimited coefficient expansion or bit growth. Mapping here: residual coordinates are states, chosen downstream readers are observables, and a normalization-free bilinear block is a quadratic transition. The guarantee is minimality within constrained linear lumpings, not minimum arithmetic DAG size or semantic identification. A fixed point of rowspace closure would establish sufficiency for that restricted transition. Layer-varying maps require stage-specific interfaces. Native normalization and final tanh violate the polynomial assumption, so we cannot claim the theorem solves the real model. Dense learned Jacobian coefficients can also erase practical sparsity.

The [rational extension](https://arxiv.org/html/2201.13373) offers symbolic and sampled-Jacobian algorithms; the sampled version has a user-specified correctness probability and post-verification. It avoids materializing an entire symbolic Jacobian. Native square-root normalization is algebraic rather than rational in raw residuals, and tanh remains outside this class. Adding a norm variable with a polynomial constraint does not automatically satisfy the theorem's unconstrained-state assumptions. Thus sampled floating-point rank alone would be an exploratory test here, not its exact guarantee.

## Executable consequence1: a precise stagewise closure test

For a proposed linear interface L_in r and next-stage observable L_out f(r), exact dependence only on that interface requires

$$
L_{\rm out}J_f(r)\delta=0
\quad\text{for every }r\text{ and every }\delta\in\ker L_{\rm in}.
$$

This is the fiber-constancy criterion specialized to a layer transition. For a scalar quadratic output h(r)=r^T S r, symmetric S, it reduces to

$$
S\delta=0\quad\forall\delta\in\ker L_{\rm in}.
$$

For a bilinear MLP and output reader c, S=sym(L^T diag(cD)R). This gives a factored coefficient test for whether an upstream reader set is closed under that output, instead of guessing from activation reconstruction. A residual linear term adds its corresponding row constraint. This is an exact restricted test; no large native closure run was claimed or launched during this review.

Norm propagation is the crucial additional observable. If r'=r+F(r),

$$
\|r'\|^2=\|r\|^2+2r^TF(r)+\|F(r)\|^2.
$$

For quadratic F the terms have degrees2,3,4. A compact update therefore needs shared norm/cross observables or an explicitly retained norm-producing background. Repeated naive polynomial lifting can grow rapidly; a finite closed arithmetic state has to be demonstrated, not assumed.

## Executable consequence2: read coordinates plus an explicit norm

Stack the four128-row QK maps and the current value reader in E (513x1152). On the **already normalized** input x, an orthogonal projection onto row(E) preserves every quantity used by the scalar producer. This is a sufficient interface, not a proof that the scalar output's minimum representation needs513 coordinates.

For raw residual r, let rho²=||r||²/1152+epsilon and x=r/rho. A head-normalized Q reading is exactly

$$
\frac{Qr}{\sqrt{\operatorname{mean}((Qr)^2)+\epsilon\rho^2}}.
$$

The value is v^Tr/rho plus the token-table term. Hence E r **together with rho²** suffices. E r alone generally does not. This remains true with native nonzero epsilon; approximate scale cancellation must not silently remove it.

The [CPU control](SCALAR_PRODUCER_INPUT_QUOTIENT_V1_RESULT.json) actually ran:

- Both stacks have numerical rank513, condition numbers54.4and80.1.
- Normalized-input projection preserves scalar outputs within5.13e-15.
- Constructed raw perturbations preserve all E r coordinates within9.56e-16 but change outputs by15.4%and24.2% through the missing RMS information.
- Adding the actual rho² observable restores outputs within5.72e-15; it agrees with the native FP32 calculation within4.20e-7.

These are random mathematical interface controls, not native-language effect tests. They establish a concrete failure mode for a proposed upstream interface and an exact corrected specification.

The price prevents a misleading compression claim. Per head, the original stacked maps store590,976scalars. The reduced maps would store263,169, but their dense raw-state encoder adds590,976more:854,145total before norm generation. The coordinate form is worthwhile only if an upstream program produces those coordinates directly or shares their computation. We do not adopt the larger adapter as progress toward a simpler model.

## Decision

The literature supplies a useful closure criterion for restricted polynomial transitions, not a ready-made whole-model solution. The executed norm counterexample changes the extraction specification: next upstream candidates must expose both read coordinates and the required norm/cross-observable generator. Do not restart a rank-only projection sweep or call missing normalizers harmless.

Continue the already registered newline capability/wholehead-positive-control experiment now; it addresses selective manipulation of the identified local pair. Then use the explicit coordinate-plus-norm interface when choosing the next backward fold. Broad weights-first composed-path discovery remains active; local circuit work is evidence for which interfaces to preserve, not a replacement objective.

Next math review20:12UTC; hourly17:43UTC unchanged.
