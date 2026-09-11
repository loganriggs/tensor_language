# Independent mixed-precision congruence spectrum

V1 full FP64 search completed with3of4requested modes converged. Core operator
checks passed, but the complete registered convergence bar did not. Preserve its
result as partial convergence; do not infer a structural negative from its flags.

Independentseed997, ncv40,maxiter150,tol1e-5 FP32search matvecs, with all returned
eigenpairs verified against the original full FP64operator. Full50304outputs,
all1152²coordinates, zero text. No change to the final native acceptance bars.
The search tolerance is not the final eigenpair tolerance.

A: search returns4converged modes; independentFP64residual/operatorbound<=1e-7;
identity overlaps and orthogonality<=1e-8; no negative eigenvalue below-1e-8bound;
native total matches prior<=1e-8; fixed-matrix FP32/FP64action discrepancy<=1e-5.
B: both smallest nativeFP64Rayleighvalues<=.01*2/(1152+1).
C: lambda2/lambda3<=.10 withlambda3>1e-10operatorbound. B/CrequireA.
D performance only: median synchronized matvec FP64/FP32 speed ratio>=3.
No compression or interpretability credit for lower search precision.

Compare the returned values and subspaces with V1's3converged modes. A negative
structural screen must preserve metric/conditioning and distinguish the global
noninteracting block assumption from overlapping or task-specific computation.
Partial convergence still means unresolved optimization. Extra output-metric or
block-count searches are not silently folded into this result.

Price:0forwards0tokens; fixed5synchronized calls per precision for timing, about
425MBCPU Krylov vectors, ~42.5MBephemeral /dev/shm cache, no large disk artifact.
900s hard process limit, not a convergence declaration. CPU mixed-precision
control passed at independently checked residuals<=3.1e-8of operatorbound.
