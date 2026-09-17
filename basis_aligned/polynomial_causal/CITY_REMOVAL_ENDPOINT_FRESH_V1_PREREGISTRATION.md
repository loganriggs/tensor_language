# Donor-free removal on fresh constructions, cities and endpoints

Freeze before outcomes.20cells/40sequences/240endpoint rows in
CITY_REMOVAL_ENDPOINT_FRESH_V1_ROWS.json: ten new vocabulary-focused templates,
York/Portland and Oxford/Seattle (held out from donor-free selection, not globally
new cities), six endpoints disjoint from selection: colour/color,centre/center,
favourite/favorite,flavour/flavor,travelling/traveling,cancelled/canceled.
No exact prior prompt overlap. No outcome-based row replacement or deletion.

Fixed formula city_inherited_removal_v1.py: native city inherited-value half-removal
and its single-head frozen-norm MLP8 approximation. One recipient residual7 input;
no donor. Token tables are algebraic extensions from checkpoint, no fitting.
800bodyforwards/300s: native, native8 removal, exact local response reentry,
one-input candidate,16per-position norm-matched block9 Gaussian nulls.
Seeds17092700+1000*k+context_id; paired signs. Same four unrelated readers.

pred_a: exact reentry agrees with native removal<=1e-4absolute AND1e-5relative
allreadouts; exact local delta<=1e-4relative; finite/outsidezero; null normerror
<=1e-5;800forwards.
pred_b: EVERY family>=18of24capable UK-US margins>=.1, candidate targetRMS>=1e-5,
and candidate/native effect relativeL2error<=.35.
pred_c: EVERY family's each4reader RMS<=.5candidate targetRMS.
pred_d: EVERY family>=.90capable pairs attenuate and mean attenuation>=.02.
pred_e: EVERY family candidate targetRMS>=2random median and beats16of16nulls.
pred_f: EVERY family effecterror<=.8times EACH frozen cueconstant and textOLS
error. Four text features (intercept,cue,length,city-position) frozen on old removal
panel, no endpoint identities or new outcomes. Candidate has native-state access
and more compute than baselines; disclose asymmetry.

These stricter positive-fraction/null gates follow2355 review, fixed before new
outcomes; old screen still uses its original.75/15gates. If capability fails,
report failed generalization on this panel; do not filter examples or soften gates.
If only approximation fails, retain the native removal result separately.
Freshness ends when outcomes are inspected. Corpus/token-only/composition remain
untested. One native input belongs to this removal counterfactual, not originalswap.
