# Head9.8 fixed-value weighted quadratic SVD

Registered before native execution,12September2026. Uses the frozen current-value
reader from the known scalar producer and original fulljointkey or selected
reflection-even key numerator. Does not refit/select value readers or use data.
Prior exact metric, adjoint, cubic execution and coefficient-energy controls
hold. Fixedvalue metric powers now use rank-one contractions; dense controls rerun.

Exact source support: row(K1,K2,v), at most257; query support row(Q1,Q2),atmost256.
Fit relativequery8/source7. Each target receives two seeded rank8 matrix-free
SciPy ARPACK SVDs,120seconds each/500iterations; total600seconds. Products run
FP64 on managedGPU; CPU drives iterative solver. No bodyforwards. Storedsource
quadratics cost8×257² scalars perprogram plusbases/value and retainednativeQK.
This is a different, larger-block family than rank8CP; no equal-rank priceclaim.

A: all four arms return, leading singular value>1e-12 and both singular-pair
residual norms divided by leading singular value<=1e-6.
B: A and each arm captures>=50% of its exact symmetric quadratic-query/cubic-source
coefficient norm. The norm is evaluated independently of retainedsingularvalues.
C: A and the two seeds' singular-value-vector relative differences<=1e-6 for
both targets. This checks numerics, not uniqueness of repeated singularvectors.

Frozen sourcequadratics are rescored atquery8/source0; heldcapture descriptive.
Original fullQKnorms, nativecontexts andsuffix remain required by laterphysical
validation. A/B/C cannot promote a circuit. Timelimits/convergencefailure are
unconverged, not absentstructure. Residuals do not certify that no larger mode
was missed; two seeds mitigate that concern without proving it impossible.
