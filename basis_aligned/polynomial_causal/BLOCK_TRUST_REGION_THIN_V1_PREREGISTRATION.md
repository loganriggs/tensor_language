# Equivalent thin Stiefel contraction and longer native continuation

Resume BLOCK_TRUST_REGION_V1_RESULT.json's latest hash-bound state: loss
.9145115632005731, residual.9137125138193017. The first exact-curvature run
completed only24outer iterations and1134Hessian products in122.98fit seconds.
It did not converge; the negative result remains an optimization failure.

The two-thread hypothesis alone failed its wall-time screen (ratio1.102), though
CPU consumption fell. Inspecting the installed Pymanopt2.2.1 Stiefel Weingarten
map found a different bottleneck: left association constructs V X^T of size
1152square per block. Compute V(X^T N) instead. This is the identical Hessian
geometry in real arithmetic, using a16square intermediate. The installed
library and old experiment sources stay unchanged; a local subclass overrides
only the Weingarten contraction order.

CPU same-point/same-shape full Hessian-conversion checks give2.27e-17relative
error. Reverse-order timing shows14.68xhost speedup at16threads and44.26xat2.
The native run uses2BLASthreads with the thin contraction. This is not a claim
of44xwhole-solver speedup. Native state, iteration count and budget differ from
the preceding run; report actual throughput without presenting an unmatched
end-to-end comparison as a controlled benchmark.

Algorithm/objective/family unchanged: exact reduced-Hessian Pymanopt trust
regions,16overlapping16-reader/4-output blocks, all50304outputs, lambda=.01,
377344stored coefficients, radius.1/max1, max50inner iterations. Extend the
allowed native fit to600seconds (1000outer iterations remains the limit).
This provides a substantial continuation after the diagnosed implementation
bottleneck, not another short chunk of the identical slow code. Time limits
are never convergence. Save final state and all scalar diagnostics.

- pred_a: latest loss/residual replay, constraints/writer solve and monotonicity
 <=1e-8; native reduced-Hessian finite-difference relative error<=1e-5; CPU
 reference-Hessian equivalence held before queueing. Record live BLAS pools.
- pred_b: canonical gradient<=1e-7, original scaled relative gradient<=1e-4,
 original maxentry<=1e-7 and five-diagnostic relative loss change<=1e-5; at
 least five diagnostics. All must hold.
- pred_c: capture>=1.05*.08623021231576344 on the same full coefficient metric.

Null: implementation repair improves throughput but does not establish
convergence or better structure. A failed result cannot rule out other local
optima/blocks. If converged, next inspect block functions and restart stability;
no circuit/behavioral adoption from coefficient capture alone. No text used.
