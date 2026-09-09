# Can any proper query block split survive all readers?

QUERY_COMMON_SQUARED_CHANNELS_V1 rejected exact three-channel separation in
the fixed first context; its first two centered I output forms already fail
commutation. This successor uses only that opened coefficient artifact,
SHA256 3f195fe72cc86a62b96ba08733c89e4768a6462608b97a0cbf4483fe1b4e3151.
No new native forwards, contexts, selected output coordinates, or fit.

Prior art: toy_consumer_commutant_blocks.py and equality reader rung479
already use consumer commutants. Here the object is three explicit query-root
amplitudes with a positive native norm Gram, not a fitted 32D MLP subspace or
gradient-averaged task form. Reuse the commutator-operator column convention
and exact rational matrix operations. Do not rerun the closed equality study.

For G>0 and symmetric forms A_k, put B_k=adj(G) A_k. A common 1+2 (or finer)
congruence block split of G and every A_k implies a nontrivial projector in
the common commutant of the B_k. This follows by whitening G, taking the
orthogonal block projector, then conjugating back. Therefore a scalar-only
commutant proves there is no proper common split. Solve XB_k-B_kX=0 exactly
over the stored dyadic coefficients: nine variables, lexicographic forms and
row-major equations; stop once rank8 is certified. Identity is an exact
null vector, so rank cannot exceed8. Record pivot source equations and exact
rank. A larger nullspace is inconclusive here; do not claim a split without
constructing its projector. Circuit consequence: keep the three-input
computation together under this norm-preserving linear-interface grammar.

Also quantify the already opened first witness pair without a numerical
eigenspace search. For its whitened forms U,V define
c=||UV-VU||F/(2||U||F||V||F). If commuting approximants U0,V0 each have
relative Frobenius error at most e, then expansion of their zero commutator
and ||XY-YX||F<=2||X||F||Y||F give c<=2e+e². Hence
max(relative errors)>=sqrt(1+c)-1. This is a whitened coefficient metric,
not full-logit KL, average task accuracy, or the earlier route-error bar.

Compute c² exactly without Cholesky rounding. For B=adj(G)A_i and
C=adj(G)A_j, similarity to symmetric whitened forms gives
c²=-tr((BC-CB)²)/(4 tr(B²) tr(C²)); determinant scaling cancels.
Compare this rational value with (2/100+1/10000)² to decide whether a
1% per-form approximation by common squared channels is impossible.
Report sqrt(1+sqrt(c²))-1 numerically only as a readable bound.
This is an exact certificate for the compiled artifact, not an interval
certificate for ideal-real native weights. It does not bound approximate
1+2 block models or general bilinear circuits.

A: source hash and parent's instrument pass; exact identity commutation;
planted coupled 1+2 family retains commutant dimension2; a connected dense
pair has dimension1; a diagonal pair has c²=0; both decisions survive a
fixed invertible nonorthogonal congruence transformation. B: exact scalar
commutant (no proper block split). C: exact rational c² exceeds the1% bar.
Preserve either negative result with no family/metric rescue.

Bounded saved-artifact CPU, threads2, alarm180s, tensors<256MiB. All387968
native constants remain charged; no extracted circuit or structural saving.
