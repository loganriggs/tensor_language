# Quantify the remaining MLP8 normalization dependence

Fixed reducedcross midpoint,40now-opened freshcross sequences. Two variants:
(1) omit_correction: remove D[(Lg)*(Rg)]*(1/s1-1/s0), keep s1 for other terms;
(2) freeze_denominator: also replace s1 bys0 for cross/quadratic numerator.
s0=mean(g²)+eps; s1=mean((g+delta)²)+eps. No other changed terms or sources.
Both still need full attention8 context for norm; saving two reader projections
is not state-port closure or parameter compression.

CPU preflight: exact-mode matches fused executor<=1e-10relative and allmodes
zero atstrength0; report localvector approximation errors without causal claims.
Planned native arms: native,fullnative8,reducedreference,omit_correction,
freeze_denominator =200bodyforwards/300s. Direct block9 input edits; fullsuffix.
pred_a: native/full/reduced anchors replay storedfresh[0,1,3]<=1e-4maxabs AND
<=1e-5relativeL2,finite,outsidezero,200forwards. pred_b: omit_correction predicts
reducedtarget effect<=.35relativeL2 andtargetRMS>=1e-5 EVERYfamily.
pred_c: freeze_denominator samegate. pred_d: eachvariantfourcollateralRMS<=.5
own targetRMS per family. No random-null or fresh selectivity upgrade. Preserve
all gates/failures. Passing is a candidate forfreshconfirmation; no denominator
freeze is incorporated into the verified exactpackage by this screen.
