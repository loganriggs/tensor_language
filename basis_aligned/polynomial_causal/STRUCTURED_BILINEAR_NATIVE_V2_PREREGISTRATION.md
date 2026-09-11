# V2 execution repair: finite-difference truncation, same native experiment

V1 completedseed0factorinitialization but failed its nativegradientpreflight
beforejointoptimization. Its initialstate andfailedreceipt are preserved.
STRUCTURED_GRADIENT_STEP_AUDIT_V1_RESULT demonstrates exactgradientreplay,
100.07xFDerrorreduction for10xsmallerstep, and6.09e-10Richardsonrelativeerror.

Keep the V1 objective, two seeds, representation, prices, optimizer andthree
predictions. Replace only the numericalgradientcheck with Richardson
extrapolation fromcentral differences atrelativeparametersteps1e-5and5e-6.
The relativeerrorbar remains1e-5. Save the two rawderivatives andextrapolation.
Seed0loads the exacthash-bound V1initialcheckpoint; do not rerun its56.4s
factorinitialization orrepeat itsbranchamplitudesolve. Seed937 retains the
registeredindependentinitialization andsamecorrectednativecheck.

The V1failure remains a failedinstrumentreceipt, not a failednativefit or
negative result about structure. V2sources/binding/results are separate.
All convergence andcaptureclaims require the actual V2finalreceipt.
