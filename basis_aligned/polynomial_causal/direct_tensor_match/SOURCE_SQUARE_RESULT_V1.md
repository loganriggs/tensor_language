# Targeted second-read corrections: execution passes, fidelity fails

21 September2026. All four fixed spectral-correction programs pass dense/profile/actual-execution checks and physical cost accounting. At most14 new squares retain the original20%source multiplication saving.

Covariance coefficient selection allocates13/1/0 corrections, reaches7.8169% coefficient error and leaves component3 at10.6723%. The explicitly exploratory opened-state fidelity selector chooses2/0/12, gives7.8422% coefficient error and10.0769% component3 error; its worst normalized fidelity ratio is1.04039, still a failure. Isotropic candidates remain far from the covariance requirement. No graph is accepted or promoted to fresh validation.

Allocation-only reanalysis permits up to42 total corrections, still at most14per pair, using the saved profiles. Covariance's best worst-case fidelity remains unchanged: the available spectral corrections do not fix it even when this limited total-budget restriction is loosened. This is not a bound on larger per-pair banks, refitting, alternative source objectives or other graphs. The original20%budget verdict remains unchanged. Isotropic's worst ratio improves only1.7131 to1.6956.

The useful distinction is between identifying a consequential source error and finding a cheap correction for it. Exact-second-read substitution helped, but leading coefficient-residual squares do not recover enough of that benefit. More accurate coefficients do not monotonically improve the composed response because errors can cancel. This argues for comparing functional residual directions against spectral residual directions, while retaining a global coefficient constraint and independent behavioral validation.

[Registered plan](SOURCE_SQUARE_PLAN_V1.json), [executable results](SOURCE_SQUARE_V1.json), [exploratory frontier](SOURCE_SQUARE_OPENED_FRONTIER_V1.json), [implementation](fit_source_squares.py).
