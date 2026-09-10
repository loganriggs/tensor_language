# Weight-only symmetric-product ALS, 10 September 21:57 UTC

Compare a tensor-aware optimizer with the earlier generic Adam/L-BFGS result.
Use the same128products, explicitlambda.01 separate-component-energy penalty,
all-U coefficient metric, and frozen stalled initial readers in unit gauge as
PENALIZED_WEIGHT_PRODUCT_V1. No corpus inputs, validation scores or new capacity.
Historical reference reached objective0.9140457259720446 in57.04additionalseconds;
its prior540s warm-start source remains charged to both methods.

Alternate A-reader, B-reader and exact conditional writer solves. With fixed
other readers/writers, the energy penalty adds lambda times the diagonal blocks
of the reader normal matrix, equivalently replace outputGram K by
K+lambda*diag(K). Solve implicitly with block-Jacobi PCG, relative true residual
target1e-9,max300iterations. Normalize reader rows and refit writers, preserving
the represented candidate family and explicit penalty. No hidden ridge.

Initial120second fit budget, saves current/best states. Prediction A: existing
dense/gradient/CG controls held; native initial penalized objective agrees with
the fixed-reader baseline within1e-8; every inner solve has true residual<=1e-9;
no half-step increases objective by>1e-9; final checkpoint replay<=1e-10.
B: five diagnostics meet relative objective plateau<=1e-5 of captured energy,
relative stationarity<=1e-4 and max gradient<=1e-7. C: best objective is within
1e-4 of the historical L-BFGS result and cancellation<=10. C is comparable fit
quality at this budget, not a speedup claim or global optimum.

Charge actual CG iterations and outer sweeps, not maximum allowed steps.
Timer check occurs between complete sweeps; report any overrun. No corpus or
native body forwards. Checkpoint~5MB,900s hard alarm, managed lane1. Preserve
nonconvergence/conditional-solve failures; they do not reject structure. The
one-factor update applies to independent product operands, not tied squares or
general blocks; those require separate solvers.
