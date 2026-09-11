# Exact reduced-Hessian trust regions for the same overlapping block fit

Source: ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_RESULT.json and its hash-bound cache.
Library CG stopped on a tiny step after35.52fit seconds; loss.9145133300887056,
capture.08628536781479246, relative stationarity.0004750817. The original custom
manifold run and the earlier raw fit remain charged. No structural negative.

Keep16blocks x16readers x4quadratic outputs, all50304unembedding rows, full
coefficient metric, lambda=.01 and the same377344number price. Only the local
optimizer changes: existing Pymanopt2.2.1 trust regions with exact reduced
Euclidean Hessian-vector products and its built-in manifold connection.
Initial radius.1, maximum radius1, maximum50inner truncated-CG iterations,
120seconds/1000outer iterations. Limits are not convergence. Save final state.

If K is target-feature cross matrix, G feature Gram and M=U^TU, the exact
conditional writer is W=K R^-1, R=G+lambda diag(G). The minimized loss is

$$
\ell(\theta)=1-\frac{\langle MK,KR^{-1}\rangle_F}{E_{\rm native}}.
$$

Autodifferentiate this solve for Hessian-vector products. Holding W fixed is
valid for the first envelope gradient, but misses its response at second order.
CPU controls verify objective/first gradient, Hessian-vector finite differences,
symmetry, and the built-in Riemannian Hessian. A deliberately frozen-writer
Hessian fails the finite-difference control by1.398relative; true error1.36e-10.
The manifold check error is1.41e-10. The standard solver's toy fit converged in
1.04seconds; that is not a native result or matched performance benchmark.

- pred_a: latest native loss/residual replay, writer residual and constraints
 <=1e-8; native Euclidean Hessian-vector finite-difference relative error<=1e-5
 before optimization. No accepted objective increase above1e-8.
- pred_b: absolute manifold gradient<=1e-7, original scaled relative
 stationarity<=1e-4, original maximum tangent entry<=1e-7, and relative loss
 change across five diagnostics<=1e-5. Diagnostics occur every five accepted
 new points and at the final point. Fewer than five diagnostics cannot pass.
- pred_c: full coefficient capture>=1.05*.08623021231576344, unchanged from the
 previous registered library-CG comparison. Penalty and capacity are unchanged.

Null: exact local curvature still does not converge or gives negligible capture
gain. Report stopping condition, Hessian-vector cost and local conditioning;
do not interpret iteration limits as absent block structure. An improved fit
still requires stable blocks/output patterns and frozen FineWeb intervention
tests before any circuit claim. No text used, no native data fitting.

V1 CPU trust-region adapter failed only when reading its diagnostic history:
Pymanopt's trust-region implementation does not call the CG logging hook. It
was never queued. V2 records scalar diagnostics at accepted-point gradient
callbacks; source inspection verifies proposed-point gradients are requested
only after acceptance. Repeated Hessian gradient requests at the same point
are deduplicated. No copied/forked optimizer implementation or iterate-array log.

Primary context: [variable projection](https://www.cs.umd.edu/~oleary/software/varpro/varpro.pdf),
[Pymanopt trust regions](https://pymanopt.org/docs/stable/optimizers.html),
[manifold Hessian conversion](https://pymanopt.org/docs/stable/manifolds.html).
These methods supply local optimization tools, not a global tensor recovery guarantee.
