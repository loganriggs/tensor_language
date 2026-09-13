# Compressing the upstream response map: preserve the intervention direction

13 September2026. This tests a shared computation feeding the successful three-term interaction representation. Both tested approximations fail local live-input preservation, so the exact map remains in use.

The fixed-writer MLP9 response contains

$$
\Delta(a)=-aw+\frac{a(2\beta-a\gamma)}{\rho(a)}m_0
-\frac{a}{\rho(a)}Jz+\frac{a^2}{2\rho(a)}Jw.
$$

J is the1152-by-1152 mixed bilinear map for the fixed writer w. It is reused across contexts. Compressing J could reduce response-context preparation; it would not remove the native computation producing z or the pristine MLP output $m_0$.

## A protected low-rank approximation

Let $v=w/\|w\|$. Preserve $Jv$ exactly and truncate only the orthogonal-input map:

$$
\widehat J=(Jv)v^\top+[J(I-vv^\top)]_{r-1}.
$$

The bracket is the best rank-$r-1$ matrix approximation from SVD. This uses total factor rank r, matching an ordinary rank-r factorization, and ensures $\widehat Jw=Jw$. It protects the quadratic response coefficient and the map's action on inputs parallel to w. For $E=\widehat J-J$, the response error is exactly

$$
\widehat\Delta(a)-\Delta(a)=-\frac{a}{\rho(a)}Ez,
$$

up to numerical error in the protected equality. Thus preservation of one physically relevant direction is explicit, rather than hoped for from the Frobenius fit.

[The weight screen](UPSTREAM_RESPONSE_MAP_V1_CONTROL.json) finds constrained errors88.25%,81.38%,70.12%,52.62%,28.01%at total ranks32/64/128/256/512. Protecting w costs little additional Frobenius error versus unconstrained SVD, while reducing writer-action error to numerical precision. But strong global low-rank compression is not supported at these tolerances.

[Actual-input validation](UPSTREAM_MAP_LIVE_V1_RESULT.json) uses48live template contexts and child/remainder/parent fields at strengths-1,1,2. It compares local response vectors, not just their contribution to a larger background. Protected-writer and response-error identities pass within approximately $1.5\times10^{-15}$. Rank256still gives23.05–24.21%aggregate response error across branch/strength groups; rank128gives30.63–32.52%, and rank32gives38.50–40.62%. None meets the1%criterion. No suffix test or replacement is justified by these results.

## A different assumption: sparse connections plus an exact correction

At the same nominal FP32 storage as rank256, fit

$$
\widehat J=S+d v^\top,\qquad d=(J-S)v.
$$

S is sparse; the rank-one correction again protects Jv. Charge546048sparse values, a165888-byte support bitmap, and2304rank-one factor scalars:2359296bytes total. This proposes edge sparsity, not node sparsity, and does not yet price resident indices or execution time.

For a fixed support in row i, write $R=J(I-vv^\top)$ and let $v_i$ denote v restricted to that support. The exact least-squares coefficients are

$$
S_{i,\mathrm{support}}=R_{i,\mathrm{support}}
+\frac{R_{i,\mathrm{support}}v_i}{1-\|v_i\|^2}v_i^\top.
$$

This follows by inverting the rank-one Gram correction $I-v_iv_i^\top$. The support is proposed by largest absolute entries of R; it is not globally optimized. [The CPU check](UPSTREAM_PROTECTED_SPARSE_V1_CONTROL.json) verifies a supported gradient below $9\times10^{-17}$ and exact protected action. Weight error improves to34.52%, versus52.62%for the equally priced protected low-rank map. Refitting coefficients only slightly improves on raw thresholding at34.53%.

Nevertheless, [live-input validation](UPSTREAM_SPARSE_LIVE_V1_RESULT.json) gives41.42–42.99%response error—worse than low rank. Its algebraic identities pass and both preservation criteria fail. The fidelity screen materializes the proposed map; no efficient sparse executor or achieved storage saving is claimed.

[The matched audit](UPSTREAM_MAP_ERROR_ALIGNMENT_V1_AUDIT.json) verifies identical reference norms for all432response cases. Sparse residual Frobenius error is0.656times the low-rank residual, but actual response error is1.768times larger. After accounting for residual magnitude, the squared amplification ratio is7.26. This quantifies different alignment of the residual errors with the actual weighted inputs; it is not a fitted covariance model or causal attribution.

These failures rule out the tested low-rank and fixed-support sparse approximations as accurate response generators at this budget. They do not establish absence of other structure. The current three-term result retains exact J: the successful compression concerns the composed polynomial interaction, and should not be weakened by substituting an inaccurate upstream map merely to reduce its parameter count.
