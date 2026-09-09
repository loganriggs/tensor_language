# How the final reader uses a join write

The contextual matcher's paired gate cannot be freely transferred between
matched middle-label cases. Its signed values change substantially, and the
native reader is sensitive to those changes. Rather than fit a sign correction,
we now ask which algebraic operations the final reader performs on the write.

The scope is the existing four-layer attention model, with a selected L2 write
to binding positions and one final attention layer remaining. The final query
is not changed by this intervention. A curve for this fixed query is an exact
partial response program; it is not an independently reduced whole model.

## Exact token-derived response

Let d_s be the selected join write at source position s, and let z_s be that
source's post-L2 state with the write removed. Scaling the native write gives
`x_s(alpha) = z_s + alpha d_s`. All other source states and the final query
stay fixed. For each final attention head, define linear functions

    a(x) = q1 · K1(x) /32,
    b(x) = q2 · K2(x) /32,
    v(x) = V(x),

with actual absolute-position RoPE inside K1/K2 and the fixed normalized native
query inside q1/q2. These key/value projections act on the unnormalized source;
its three RMS factors will be supplied together. The folded output map and
final .5 interpolation factor are included when decoding the following vectors.

Write a(x)=a0+alpha a1, b(x)=b0+alpha b1, v(x)=v0+alpha v1. Multiplication gives

    C0 = a0 b0 v0
    C1 = a1 b0 v0 + a0 b1 v0 + a0 b0 v1
    C2 = a1 b1 v0 + a1 b0 v1 + a0 b1 v1
    C3 = a1 b1 v1.

After summing heads and decoding, each Ck is a29-dimensional source contribution.
The exact final-query logits are a fixed background plus

    sum_s g_s(alpha)^3 * (C0_s + alpha C1_s + alpha² C2_s + alpha³ C3_s),

where g_s(alpha) is the original RMS gain at that source. There is no degree
truncation in this identity. Separate source normalizers prevent merging their
coefficient vectors before normalization.

For stable scalar evaluation, let t=mean(z*d)/mean(d²), with t=0 when d=0.
Then the squared RMS denominator is

    mean(d²)*(alpha+t)² + mean((z-t*d)²) + epsilon.

This completed-square form avoids subtracting large terms in an expanded
quadratic. The identity is over real arithmetic; the implementation must still
pass numerical correspondence. Epsilon follows the actual affine-free RMSNorm.

The query-removal effect at alpha1 versus alpha0 splits exactly into
`(g1³-g0³)C0`, `g1³ C1`, `g1³ C2`, and `g1³ C3`, summed over sources. The first
term accounts for normalization changing the unmodified-source contribution.
These are degree attributions, not separately editable physical circuit states
or independent variance fractions.

## The operation hypotheses and their tests

Earlier native port restoration found predominantly value use for the forward
write and joint key-pair use for the backward write. That motivates two fixed
tests: retain C0, the actual normalizer and C1 forward; retain C0, the normalizer
and C2 backward. It does not establish either truncation in advance.

[JOIN_WRITE_READ_DEGREE_V1](../JOIN_WRITE_READ_DEGREE_V1_PREREGISTRATION.md)
uses all opened32 worlds, six orders and four query hops. The full curve must
match physical source edits and native score scaling at alpha=-1,0,.5,1,2.
Each fixed truncation must predict the complete centered query effect relative
to alpha0 within1%, separately for every population/orientation/hop/scale group.
No degree or scale range may be selected after seeing the result.

The [primitive](../join_write_read_degree_reference.py) passes12 CPU controls:
five physical source-edit scales, exact removal partition, live curve, nonnegative
norm representation, planted linear/quadratic cases, a live omitted-degree
negative, and rejection of writes that change the final query. The managed
trained-model audit is now complete: exact response and partition pass, both
fixed degree hypotheses fail. The derivation is invalid for a changed query or
an additional contextual layer after the edited states without further work.

## What is and is not saved

Compilation explicitly runs the token-derived native prefix and readers. It
retains all387968 export constants. For two changed sources and29 outputs, a
curve stores232 coefficient values, six norm values and29 background logits:
267 FP64 values,2136 bytes, plus source indices and the RMS epsilon. The weights
and compilation cost remain charged. Evaluating several scales can reuse these
values without rerunning the prefix or projecting changed source states.

The background is computed by the explicit native executor, not an unexplained
external activation cache. Nevertheless, this is a query/context-specific
response representation. No whole-model storage reduction, novel semantic
operation or OOD causal sufficiency follows from compiling it exactly. The
degree hypotheses, if supported, would identify a simpler operation in the
specified reader; their broader extraction and structural cost remain separate.

## Trained result and constraint

The [managed receipt](../JOIN_WRITE_READ_DEGREE_V1_RESULT.json) covers768 requests
from32 opened worlds in4.384 seconds. Five-scale native/source-edit correspondence
is1.28e-13 absolute and1.34e-15 relative RMS; removal partition closes1.33e-14.
At the native scale, hop3 forward-C1 relative errors are .06999 IID/.06772 OOD;
backward-C2 errors are .65392/.53191. Both registered hypotheses are rejected.
No degree, scale range, head or task subset is selected to repair the null.

For backward hop3 removal, the linear term's projection onto the full removal
is .498 IID/.422 OOD; the quadratic term's is .488/.575. Both contribute
substantially. These correlated terms are neither variance fractions nor
independently editable causal variables. Forward removal has a large linear
term but the omitted terms still violate the1% prediction bar. Port necessity
and polynomial degree therefore describe different properties of the reader.

Curve compilation excluding native context takes .252 seconds; saved cohort
curve tensors total1,643,520 bytes. All387968 opaque export coefficients remain.
The exact response compiler is useful intervention machinery, not the requested
structural discovery.

[Metadata correction](../QUERY_HOP_METADATA_CORRECTION_V1.json) supplies576
answer/expected-answer fixes for the degree rows: their hop-fork loop retained
hop3 labels. These labels were not consumed by the registered exactness, KL,
centered-effect or degree/partition scores. Saved logits, curves and results
remain unchanged. Apply the overlay before future answer-based analysis.

The next completed screen groups the existing direct forward endpoint writer
with all final readers; see [matcher dossier](join_matcher_factors.md). It finds
sign compensation at hop3 but fails both grouped-vector portability and agreement
with physical removal. Neither exact response compilation nor conditional
path attribution has yet supplied a structurally simpler sufficient circuit.


## Two joins writing into the same binding

For a->b->c->d with b->c serialized after its two neighbors, the existing native
L2H1 forward join and L2H2 backward join write into the same two positions.
Queries a and F^-1(a) use different two-edge tails there. This tests composition
of context-dependent computations, without assuming invariant gate strengths.

The [overlap screen](../OVERLAPPING_JOIN_NORMALIZER_V1_RESULT.json) uses512
opened requests from32 worlds, two overlapping orders, two origins and four
hops. Native execution and the complete single-write response curves agree to
1.42e-13/1.03e-13; source writes are identical across the eight query forks.
The CPU run takes5.224 seconds. Native hop3 accuracy is .969–1 and each
intended join removal has a large effect, but the writes are not independently
selective: backward removal changes the other, forward query's gold probability
by mean absolute .17580 IID/.35091 OOD. Forward removal's OOD cross-change .05944
also exceeds the .05 bar. No subset of eligible groups is adopted.

A fixed composition hypothesis retained both complete univariate cubic
numerators and the exact joint RMS denominator, omitting only mixed numerator
terms around native state. For source state x and writes u,v it uses

    g(x+(a-1)u+(b-1)v)^3 * [P(x+(a-1)u)+P(x+(b-1)v)-P(x)].

The axes a=1 or b=1 are exact. The joint hypothesis fails every registered
panel: at both-cut, interaction-relative errors are4.784/4.693 forward and
2.322/2.178 backward IID/OOD. Shared normalization alone is insufficient;
mixed key/value numerator products are required. This is a different overlap
from the earlier raw/endpoint field test, which also found that freezing RMS
did not restore independent field semantics. Neither null is rescued here.

## The backward writer affects both readers through keys and values

The [native port factorial](../OVERLAPPING_JOIN_PORTS_V1_RESULT.json) then uses
all eight K1/K2/V subsets, with K1+K2 fixed as the address hypothesis. Native
projection-hook correspondence1.28e-13, saved full-cut replay1.42e-13 and query
reuse0 validate the instrument on the same512 requests; CPU2.622 seconds.
All opaque export coefficients and routing are still retained.

For the backward query, key-pair effects recover1.054 IID/1.043 OOD of the full
cut's signed gold-probability effect; V-only is .045/.035 in absolute ratio.
For the forward query, the key-pair cut overshoots the full cut: ratios1.351/
1.176, and the IID value effect is .169 of the full effect. The registered
shared-address nomination therefore fails; no backward-query-only success
replaces it. Value-only cuts improve gold probability on both origins, so
keys and values have partially opposing effects under these interventions.
Probability effects are nonlinear; their factorial interaction is retained.

Key-pair-only full-vector errors are .187/.187 on the forward query and
.092/.111 on the backward query. The sufficient-port hypothesis fails too.
The next [key gain protocol](../OVERLAPPING_JOIN_KEY_GAIN_V1_PREREGISTRATION.md)
separates direction K(x-v) from RMS gain g(x-v), before interpreting a changed
key as address information. It is a confound diagnostic, not a replacement for
the failed composition or full-source sufficiency tests. No structural
simplification has yet been established by these response decompositions.


## Directional key changes and normalization are coupled

The [direction/gain audit](../OVERLAPPING_JOIN_KEY_GAIN_V1_RESULT.json) passes
14 controls and native correspondence1.14e-13, with saved-query replay and
source reuse exactly0 on512 opened requests. CPU1.647 seconds. Direction-only
key cuts overshoot full-key hop3 gold effects by ratios1.690/1.410 forward,
versus1.059/1.059 backward. Gain-only changes partly offset the damage. The
fixed direction-dominance nomination fails across the two consumers; it cannot
be rescued by retaining only the backward query. Direction-only full-vector
errors are .220/.234 forward and .125/.136 backward. This is explicit coupling,
not evidence that address information is absent.

The [producer split](../OVERLAPPING_JOIN_ADDRESS_PRODUCERS_V1_RESULT.json) then
isolates E, the direct earlier source-value embedding; O, the existing L1
previous-key path; and exact remainder R within the L2H2 write. With final
source gain and values fixed, their key response is quadratic. Native/oracle
error1.56e-13, saved full-direction replay1.42e-14, reuse0, and vanishing
three-way interaction2.84e-13 verify the512-request CPU screen (3.540 seconds).
The fixed E-forward/O-backward address split fails: E/full forward task ratios
are -.00554/-.00021, while O/full is1.183/1.160. O also accounts for the backward
query, with ratios1.014/1.010. The E+O sufficient interface fails, with hop3
full-vector errors .119–.165. No path expansion or new origin-field interchange
claim follows. This points to a shared computation involving O, rather than
the proposed two independent address payloads.

## An explicit origin-key × endpoint-value interaction

The [interaction screen](../ORIGIN_ENDPOINT_INTERACTION_V1_RESULT.json) groups
O through the final keys with the direct forward endpoint field F through the
final values, at the same two source positions. It is an additive interaction
node in the expanded native read, with original source gains and queries:

    I = native - O_key_cut - F_value_cut + both_cuts.

Equivalently, it multiplies the change in key-pair score caused by O by the
projected endpoint payload F, summed over all four final heads. This gives
explicit producers and consumers across L1, L2H1/H2 and L3. O's input remains
contextual; the formula alone does not prove a pure three-edge lookup.

Seven controls and full-position native projection-hook correspondence1.14e-13
validate the node knockout native-I on512 opened requests, CPU1.859 seconds.
It is not a physical L2 source removal. The registered circuit nomination
**fails**: IID forward-hop3 gold loss .48058 is below .5; OOD loss .66782 and
all other query/hop selectivity panels pass. The largest control mean absolute
gold change is .01444. No rescaling, sample selection or threshold relaxation
is allowed. All387968 native export coefficients and the remainder remain.

The [saved-output route census](../ENDPOINT_ROUTE_PARTITION_V1_RESULT.json)
partitions total endpoint read T into I and J, where J is the remaining read
with O removed from keys. CPU .0246 seconds; exact knockout/partition replay
1.14e-13. On32 IID target cases,13 fail only the I single knockout,7 only the
J single knockout,1 either single,10 only the joint, and1 is a native error.
On32 OOD cases,21 fail only I,4 only J,6 only the joint, and1 survives all
knockouts. No nonmonotone joint rescue occurs in these target cases; all other
query/hop groups and both order strata remain in the receipt.

Thus16/63 initially correct target cases exhibit backup behavior: both single
knockouts preserve correctness but their joint removal fails. Total endpoint
read removal lowers target goldP by .93313/.95114. This explains why a partial
route need not be individually necessary on every example, but does not turn
its failed nomination into a pass. J is not yet an identified advance-then-read
algorithm. The next [query-lineage protocol](../ENDPOINT_ROUTE_QUERY_LINEAGE_V1_PREREGISTRATION.md)
tests which query payloads feed I and J, keeping mixed query terms explicit.
No reduced structural description has yet been established.


## Query roots must remain coupled in this interface

The [query-lineage screen](../ENDPOINT_ROUTE_QUERY_LINEAGE_V1_RESULT.json)
transports the original query-entity payload L through the frozen native prefix;
C is the exact remainder of the final query state. Native queries are then
replaced only at Q1/Q2, using the original query gain. The512-request CPU run
(4.847 seconds) passes10 controls, native correspondence1.42e-13, saved replay
9.24e-14 and prefix-lineage closure2.13e-14. Zero query gives zero route.

The proposed I-local/J-complement assignment fails: forward-hop3 task-effect
ratios are .0284/.00161 for I-local and .0963/.0357 for J-complement IID/OOD.
Selected-term full-vector errors are .922–.945 for I and .700–.734 for J.
Mixed query terms have RMS .667–.724 of the complete routes; they cannot be
silently dropped. Payload lineage is not an input Jacobian or an entity code:
its context-dependent gates can transform information from the query token.

The [mixed-input screen](../ENDPOINT_MIXED_QUERY_INPUTS_V1_RESULT.json) splits
C into task-token roots H (delimiter/instruction) and binding-document roots D.
The exact entity/complement interaction is LH+LD. On512 opened cases, CPU5.048
seconds, native correspondence7.11e-14, local replay0, root closure5.11e-15 and
mixed partition4.62e-14 all pass. Both fixed dominance hypotheses fail. Task
term effect ratios .195–.361 and document ratios .021–.184 do not explain the
full mixed-node removal. Probability effects of additive logits need not add.
Neither a selected route nor a different root subset is adopted.

The [additivity bound](../QUERY_ROOT_ADDITIVITY_BOUND_V1_RESULT.json) changes
from testing a chosen approximation to constraining a whole function class.
For a program additive across two query-root blocks, the mixed four-corner
contrast is zero. The native contrast is therefore a signed sum of four
approximation errors; at least one corner has RMS error >=contrast_RMS/4.
This remains true for arbitrary nonlinear functions inside either block.

On forward-hop3 I/J groups, each of L|HD, H|LD and D|LH has a lower bound
of7.8–9.4% relative to the fixed native route RMS. Across all query/hop groups,
84/96 partition tests exceed1%; no reference was floored. Stored-table replay
closes4.62e-14, CPU .0257 seconds. This is a bound on the specified binary
Q-root intervention cube with a fixed native-route denominator. It is not the
previous per-intervention error bar or a whole-model impossibility result.
The inequality is exact; its RHS is FP64 evaluated, not interval-certified.
The separate4e-9 contrast-slack check leaves the forward conclusions unchanged.

Further root-dominance tests are now demoted. The next
[common-channel protocol](../QUERY_COMMON_SQUARED_CHANNELS_V1_PREREGISTRATION.md)
asks whether changing coordinates can remove the coupling while preserving
both the query norm and all route readers. For a positive root Gram G, this
is simultaneous diagonalization by congruence. The whitening/commutation
criterion in [Jiang and Li, Theorems3.2/3.3](https://arxiv.org/pdf/1507.05703)
gives a finite exact test of the compiled3x3 forms. The implemented rational
primitive passes nonorthogonal positive, noncommuting negative, singular and
indefinite controls. The trained-form audit and its consequence are now below;
all387968 native coefficients are still required by the current executor.

## Changing coordinates does not separate this query interface

The [fixed-context audit](../QUERY_COMMON_SQUARED_CHANNELS_V1_RESULT.json)
compiles the query norm Gram and all58 centered I/J output quadratic forms
from six root/pair evaluations. Four additional amplitude vectors, including
zero and signed inputs, match the independent native Q-projection oracle
within8.88e-15; coefficient replay1.78e-15, norm5.55e-17, lineage8.88e-16.
Execution took .087 seconds on CPU. The source is the first registered IID
world/order, forward query/hop3; no context or output subset was selected.

Exact rational principal minors certify the stored Gram positive definite.
The very first two forms, centered output coordinates0/1 of I, have a nonzero
congruence commutator. Thus no invertible linear coordinate change turns
the norm and all readers into three independent squared channels, even in
this single context. This is an exact statement about the stored FP64
coefficient artifact. Its relationship to ideal-real native execution is
numerically checked, not interval-certified.

The [block and approximation bound](../QUERY_COUPLED_BLOCK_BOUND_V1_RESULT.json)
strengthens the result in .011 seconds of saved-artifact CPU analysis. The
exact equations XB_k=B_kX, B_k=adj(G)A_k, have rank8 in nine unknowns after
the first11 equations. Identity spans the nullspace. Any proper common block
split would provide a nonscalar commuting projector, so even a1+2 split is
impossible for this norm-and-reader interface. Six planted and coordinate
change controls pass. This rules out an exact block split, not every kind
of approximate block model.

For the original witness pair, let U,V be the Gram-whitened forms and
c=||UV-VU||F/(2||U||F||V||F)=.0308595. Commuting approximations with at most
e relative Frobenius error in each form require c<=2e+e². Consequently at
least one error is >=sqrt(1+c)-1=.0153125. A rational trace calculation
certifies that the bound exceeds1% without relying on a rounded Cholesky
factor. The metric is per-form whitened coefficient error; it is not the
previous route-logit error or a KL bound.

These results close independent query-channel proposals at this interface.
They do not mean that coupling precludes reuse. The elementary circuit
(z0²,z0*z1,z0*z2) shares the gate z0 while its complete reader family has
only a scalar commutant. The next [shared-gate protocol](../QUERY_SHARED_LINEAR_GATE_V1_PREREGISTRATION.md)
therefore asks whether the contracted trained route functions share a
linear multiplicative gate. Its planted coupled-but-shared control and
exact determinant falsifier are implemented; the trained determinant scan
is pending. Singular forms alone will be reported as inconclusive, never
as proof of a shared factor. No extracted circuit or structural saving is
claimed from either the certificates or the toy.


## Single shared linear gate also rejected

The [trained determinant screen](../QUERY_SHARED_LINEAR_GATE_V1_RESULT.json)
now completes in .00436 seconds of saved-artifact CPU. The very first form
has an exact nonzero determinant, so it cannot be a product of two linear
forms; a single common linear gate across all58 outputs is therefore excluded
in this fixed interface. Eleven planted, coordinate-change and root-edit
controls pass. The certificate concerns the stored dyadic artifact, not an
interval proof of ideal-real native arithmetic. This does not license adding
factors or selecting outputs to rescue the nomination.

The next research branch examines the actual18-layer model's
[shared first-value payload](shared_first_value_payload.md), with live
producer deletions and full-token output scoring. It preserves these local
query-factor nulls and the distinction between coupling and lack of reuse.
