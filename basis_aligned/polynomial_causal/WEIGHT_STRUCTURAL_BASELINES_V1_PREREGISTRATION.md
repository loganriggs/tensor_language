# Weight-only structural baselines, 10 September 2026

Two native-weight candidates, no corpus access or model body forwards:

* Signed squares:256 independent input readers with256 signed output vectors.
  589824 parameters. A general128product model embeds in this family by
  polarization, but the paired-output constraint is removed and capacity rises.
* Multi-output blocks:16 overlapping, nonorthogonal16-dimensional input bases;
  each basis supports4 different symmetric quadratic forms and4output writers.
  377344 parameters:294912 input,8704 core,73728 writers. Outputrank<=64.
  There are256 input projections and2176 distinct within-block pair monomials,
  reused by the four cores. This tests shared input subspaces with richer outputs.

Optimize exact all-token folded coefficient error through U-transpose-U, with
explicitlambda.01 sum-of-feature-component-energy penalty. Conditional writers
are solved exactly; Adam2000 steps then controlled L-BFGS. Seed0 independently
initializes each representation. Initial540second chunks preserve optimizer
state; a time limit is not convergence. No claim these first baseline optimizers
are advanced block-term solvers. Structural Gauss-Newton and restart stability
remain pending. No final cross-family ranking from unequal parameter budgets.

A: preexisting dense/gradient controls pass, objective finite and nonnegative
within1e-8, regularized Gram condition<=1e12, saved-model objective replay<=1e-10.
B: five L-BFGS checks plateau<=1e-5 of captured energy, relative stationarity
<=1e-4 and maxgradient<=1e-7. C: raw coefficient capture exceeds the previous
penalized128product value0.08698912596 by at least0.01 absolute. C is a first
screen, not a complexity-adjusted winner or circuit identification.

Report reconstruction separately from penalty. Penalizing each of64block
quadratics versus256squares imposes different priors; within-feature cancellation
is not charged. Failures require solver/capacity/init red teaming before any
claim about absence of structure. Squares allow signs through output vectors;
no positivity, disjoint-token support or orthogonality assumption.

Managed lane1 only; FP64; checkpoint at most~100MB per arm; require120MB free
before starting each, never delete research artifacts to make space. Full
native quadratic bias is unchanged and is outside this coefficient objective.
