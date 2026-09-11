# Same-objective ADMM repair for fixed product writers

V1completed362.61s, numericalAheld, bothwriterarmsunconverged. Highpenalty
feasibility.0021222/stationarity8.50e-7; lowpenaltyfeasibility1.4791e-5/
stationarity9.16e-8. Originalbar1e-5 stays unchanged. No structuralnegative.

Same source512products, fullU target, centeredL1 penalties.05739531and.005739531.
Primal warm start from V1 savedZ; initializeA=FZ andscaleddual0 becauseV1didnot
retain its multiplier. Explicitly not an exactV1optimizerresume. V2retains full
Z/A/dual/rho/global-step state; CPU40step versus20+20resume is bitexact.

Every10steps compare canonical ADMM residuals r=FZ-A and s=rho F^T(Anew-Aold).
If one norm exceeds10times the other, double orhalve rho withbounds1e-6/1e6;
rescale the scaleddual so its unscaled value stays fixed. This is the standard
residual-balancing heuristic described in
[Boyd et al.](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf), not a
change to the fitting objective or a convergence certificate on its own.

A: priornativeinstrumentheld androot/nativeU plusinitialmetricreplay<=1e-8.
B: botharms originalfeasibility/stationarity/L1subgradientresiduals<=1e-5.
Skip any arm alreadyconverged; eachunfinishedarm gets540seconds/20000steps.
Preserveunfinishedstatus andfullstate if this expires. Originalfunctionbudget,
penalties,inputs andqualitybars stay frozen.
C: original lowerpenalty concentration/capturebar unchanged: median top16contrast
loadingenergy>=1.25times unpenalizedbaseline and fullcapture>=.8timesbaseline,
afterA andlowerpenaltyconvergence. Do not treatloadingenergyasadditivefunctionenergy.

Priorunpenalizedfullcapture6.9925%,top16concentration41.5869%. LowerpenaltyV1
provisional6.0660%/58.6725% meetsqualityvalues but convergence is stillmissing.
No joint input-reader optimization, no physicaladoption or four-propertyclaim.
Fullstatecachesinsharedmemory, smallresults/checksumsdurable.
