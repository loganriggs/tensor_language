# Complete even-key replacement error without a dense fourth-order tensor

13 September 2026. The [mathematical review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-13_1129.md) derived the complete score error, including the cross terms omitted by the earlier fit. This note implements its weight-only coefficient objective. No new fit or behavioral improvement is claimed yet.

For one position pair, fold query/key maps into $M_i=Q_i^\top R_s^\top R_tK_i$. With orthonormal source basis $B$, write $N_i=M_iB$ and $O_i=M_i-N_iB^\top$. These are inside and outside maps, respectively. Define $G_i=N_i^\top N_i$, $C=N_1^\top N_2$, and $Z_{ij}=O_i^\top N_j$. All optimization-dependent contractions can use 64×64 matrices. Set

$$
H=\tfrac14\left[
\|O_1\|_F^2G_2+\|O_2\|_F^2G_1
+Z_{12}^\top Z_{12}+Z_{21}^\top Z_{21}
+\langle O_1,O_2\rangle_F(C+C^\top)
+Z_{22}^\top Z_{11}+Z_{11}^\top Z_{22}\right].
$$

For retained projector $P$ and discarded projector $D=I-P$, the complete squared error of the double-symmetric numerator coefficient tensor is

$$
\begin{aligned}
L(P)={}&\operatorname{tr}(DH)\\
&+\tfrac14\bigl[
\operatorname{tr}(PG_1)\operatorname{tr}(DG_2)
+\operatorname{tr}(PG_2)\operatorname{tr}(DG_1)\\
&\qquad+\operatorname{tr}(PCDC^\top)
+\operatorname{tr}(PC^\top DC)
+2\operatorname{tr}(PC)\operatorname{tr}(DC^\top)
+2\operatorname{tr}(PG_1DG_2)\bigr].
\end{aligned}
$$

The first term covers outside/discarded interactions. The bracket covers retained/discarded interactions. They occupy orthogonal key-coordinate blocks, so their squared coefficient errors add. Products involving two discarded inside directions cancel from the complete replacement error. The equality requires an orthogonal projector; its polynomial extension supplies derivatives, with optimization restricted to the projector manifold.

[The explicit-tensor control](COMPLETE_EVEN_KEY_OBJECTIVE_V1_CONTROL.json) checks four independent random models and ranks 0, 1, 3 and full rank. Maximum normalized coefficient error is $4.30\times10^{-16}$; finite-difference gradient error is at most $5.30\times10^{-12}$. These include the empty/full projector edge cases.

[Actual-weight preparation](COMPLETE_EVEN_KEY_V1_PREPARATION.json) uses the same four position pairs as before. The four sets of $(G_1,G_2,C,H)$ total **65,536 scalars**. Setup took 0.217 CPU seconds; median objective-plus-autograd evaluation took 1.85 ms on two threads. Setup forms 1152×1152 folded matrices temporarily, but no fourth-order tensor; only the small Gram matrices are retained for fitting.

At rank48 the original key-map SVD basis has complete squared coefficient error $3.11674\times10^{10}$; the previous inside-product query-folded basis has $2.70235\times10^{10}$, a **13.30% reduction** under this fuller objective. A basis from the outside-only quadratic term gives $2.77136\times10^{10}$. These are starting-point comparisons, not completed fits. Absolute coefficient scales are not logit or behavioral units.

The saved [Gram artifact](COMPLETE_EVEN_KEY_V1_GRAMS.pt) includes these starting bases and normalization for the next fit. Keep rank48, price and frozen validation unchanged. This objective now includes all even-key numerator replacement terms, but it still does not weight errors by the input-dependent denominators or value/suffix effects. Those functions remain exact in execution; behavioral preservation must still be measured.

## Executed multistart fit and signed response check

[The ten-start fit](COMPLETE_EVEN_KEY_FIT_V1_RESULT.json) uses the original SVD, prior query-folded basis, outside-only basis, two perturbed prior bases and five random bases. QR-retracted gradient descent minimizes the complete objective, normalized by the original SVD loss. All ten runs are monotone and converge below the $10^{-7}$ tangent-gradient threshold; their final normalized losses agree within $1.4\times10^{-13}$. Total CPU time is 79.3 seconds. This is local convergence and objective agreement, not proof of a global optimum.

Loss falls from 0.86704584 for the prior query-folded basis to 0.86570599, a **0.1545% improvement**, missing the declared 1% meaningful-gain bar. Rank48 and stored program size remain unchanged.

[The frozen signed-response screen](SHARED_KEY_COMPLETE_STRENGTH_V1_CONTROL.json) still fails. Maximum family errors at strengths $-2,-1,0.5,1,2$ are respectively **10.97%, 15.55%, 8.73%, 7.90%, 7.74%**. The complete-objective fit is slightly worse in this maximum-error comparison than the prior fit at every tested strength. These are conditional responses on the same 72 cached contexts; no new native suffix or fresh-data result is implied.

Thus the incomplete numerator objective was a real mathematical mismatch, but correcting it did not fix the signed behavioral failure. Keep exact shared64 for general edits. Denominator/value weighting, available rank and other representation families remain possible explanations; more iterations of these already-converged runs are not justified by an optimizer-stopping failure.

### Lower-bound countercheck

The retained/discarded contribution is nonnegative, so a global rank48 lower bound is the sum of the sixteen smallest eigenvalues of $\sum_s H_s$. [The executed bound](COMPLETE_EVEN_KEY_V1_LOWER_BOUND.json) is $7.38183\times10^9$, versus best loss $2.69818\times10^{10}$. This bound is loose: it leaves up to 72.6% possible coefficient-error reduction and cannot certify near-global optimality. It neither proves that another optimizer can achieve that gain nor that behavioral error has the same bound. Preserve this uncertainty rather than treating agreement among ten starts as a global theorem.
