# Full-U nonorthogonal congruence spectrum

Target: all50304token forms Q_v from entire U folded through MLP17. Native1152²
matrix coordinates Z, no input/output projection and no text. Seek additional
solutions of Q_vZ=Z^TQ_v, a necessary algebraic structure for congruence blocks.
Use frozen congruence_block_operator_v1.py; operator controls include indefinite
nonorthogonal blocks, free bilinear products and dense null. Previous orthogonal
consumer commutant toy is a different target and is not repeated.

Normal operator L(Z)=2sum_v Q_v(Q_vZ-Z^TQ_v), normalized bysum_v||Q_v||F².
Remove scalar identity by orthogonal trace projection and shift only that trivial
direction above the spectrum. Bound ||L||<=4||sum_vQ_v²||/total. The exact mean of
the remaining d²-1eigenvalues is2/(d+1). This avoids using a dimension-dependent
absolute threshold without calibration.

SciPy1.18.1 eigsh, smallest algebraic4modes, ncv20,maxiter100,tol1e-8,seed961.
GPU matvec; CPU Krylov basis. Independently replay every eigenpair, identity
overlap and orthogonality; catch and report partial-convergence exceptions.

A numerical/convergence prediction: native total replay<=1e-8, operator adjoint
and identity replay<=1e-10, all4modes converged with residual/operator-bound<=1e-7,
identity overlaps and mutual orthogonality<=1e-8. All values finite, no significant
negative eigenvalues (<-1e-8*operatorbound). A miss is instrument/optimization
unresolved, never evidence against native block structure.
B near-null prediction: the two smallest nontrivial eigenvalues are each<=.01
times the exact mean2/(d+1). This would motivate several-block recovery, not
identify circuits by itself.
C gap prediction: lambda2/lambda3<=.10 (indices1/2 in sorted fourvalues), provided
lambda3>1e-10*operatorbound; otherwise preserve a larger unresolved near-null space.
Both B/C require A. A near-null witness can reflect weak input directions; follow
with original-coordinate block error and conditioning rather than labeling it a circuit.

Price: 0forwards0tokens. ~170MBFP64native Gram; normal operator never materialized.
CPU Krylov storage roughly212MBfor20vectors;4retained FP64witnesses~42.5MBstored
only in a uniquely named /dev/shm cache, with checksum and reproducible source
bindings in durable JSON. Cache is ephemeral, not an off-box artifact. Runtime
unmeasured; hard process limit1200s. A limit is not convergence. No new disk-heavy
checkpoint. Existing11MBfree disk does not limit shared-memory capacity.

Null: converged smallest modes are broad and/or not near-null. This only weakens
the tested full-U globally separated block assumption in this coefficient metric.
Red-team negative with independent-start confirmation, congruence conditioning,
and common/centered metric distinctions before a stronger claim. Overlapping
shared computations and task-specific blocks remain possible.

Algorithm reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
Structural mapping: https://arxiv.org/html/1607.00716v2
