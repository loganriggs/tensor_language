# Separating position from content in a shared attention computation

**Completed result:** the query-phase proposal fails. Its long-context transfer
falls from68.8% to37.8% of the original complete head-set effect. The native
instrument and unchanged short-context controls pass. The final section
explains the result and the separate answer-reader/complement calculation.

The new lead concerns a verb–preposition circuit rather than the closed is/was
local-write branch. Claude's v473 result found that a fitted about/for direction
loses transfer on a longer intervening phrase, while its selected attention
heads remain effective. This provides a concrete opportunity to investigate
how a consumer changes with context instead of merely decomposing a weak effect.

The result does not yet identify a distance-reading algorithm. Adding words
changes both position and contextual representations. We have implemented and
tested an intervention that separates them: keep the computed query, keys and
values fixed, and move only the query's rotary position. The native test is
registered; its result belongs below when available.

## The mathematical separation

An attention query describes what a position is looking for; keys describe
candidate sources; values carry the information to read. Rotary position
embedding, or RoPE, transforms each query/key coordinate pair using

    R(t) = [[c_t, s_t], [-s_t, c_t]].

For an unrotated query pair `(a,b)` and key pair `(u,v)`, expanding the dot
product gives

    (R(t)q)ᵀ(R(s)k) = (a*u+b*v) C(t,s) + (a*v-b*u) S(t,s)
    C(t,s) = c_t*c_s + s_t*s_s
    S(t,s) = c_t*s_s - s_t*c_s.

Sum these expressions over the 64 coordinate pairs in a width128 head. The
two content products are independent of position; `C,S` describe the geometry.
In ideal trigonometric arithmetic they depend only on the relative position.
This is the relative-position structure introduced in
[RoFormer](https://arxiv.org/abs/2104.09864).

Bilin18 multiplies two such query–key scores and divides by 128². The resulting
routing coefficient multiplies a value vector before summation over causal
sources. Thus both score factors and the value must be accounted for when
asking whether two circuits reuse the same matcher. Sharing RoPE tables is
known architectural sharing, not a newly discovered semantic computation.

This formula also supports weight folding. If `a,b` and `u,v` are linear
projections of inputs `x,y`, each content coefficient is a weight-defined
bilinear form in `x,y`. Query/key RMS denominators stay explicit. An output
reader can fold into the value/output maps, but intervening nonlinear layers
cannot simply be omitted. Comparing complete consumer functions is still
required; identical payloads or identical positional tables alone are too weak.

## The actual rounded rotations require care

Native code rounds cosine and sine to BF16. Consequently `c_t²+s_t²` is not
exactly one. The inverse is

    R(t)^-1 = R(t)ᵀ / (c_t²+s_t²).

To keep content fixed and change its phase from `t` to `t'`, apply
`R(t') R(t)^-1` to the captured rotated vector. A transpose without the
denominator subtly changes content magnitude. It is also unsafe to replace
the two absolute rounded tables by an ideal table for `t-s` while claiming
an exact native intervention.

Executed controls use the actual native table code, head width128 and all512
positions. In FP64, inverse/transport errors are at most1.34e-15; the folded
dot-product error is1.43e-14. FP32 transport error is9.54e-7. The incorrect
transpose shortcut has max error.01738 on the same random-vector fixture.
Squared rotary scales range from.994644 to1.005508; equal-displacement
coefficient pairs at different absolute positions differ by as much as.006028.
Those last numbers describe numerical geometry, not measured behavioral damage.

A separate two-layer native-model control verifies the read-only capture:
original outputs remain unchanged, methods restore, identity reads agree
exactly, and a direct pre-rotation oracle matches the alternate read within
5.56e-17. The phase edit is live. No trained checkpoint forward was needed for
these controls.

## The fixed native question

We reproduce v473's five-frame native fit, freeze its chosen heads and per-layer
directions, and evaluate the same eight previously opened held-row panels.
No new direction is fitted using the alternate phases.

The token audit found actual query-minus-cue index differences of3 or4 on the
fit frames,4 on the new short phrase, and9 on the long phrase. Every A1 pair
has equal lengths and a single changed token. The descriptions' “five” and
“ten” count the cue inclusively. The registered operation caps the query phase
at cue_index+4; only the long phrase changes, by minus5 positions.

Both query factors move. Keys, values, masks and earlier states stay native
during capture. For each chosen head we form

    alternate_donor = native_base + virtual_donor - virtual_base.

The original fitted direction then interchanges toward that endpoint, with
the actual downstream model recomputed. This is a native-prefix-conditioned
counterfactual program, not an executor that independently reads raw text.

The long-frame direction must recover at least80% of the complete selected
head-set effect and gain at least10 percentage points over its unchanged
counterpart. Separately, the candidate must reproduce the complete-set causal
effect within10% error in both the full-vocabulary and answer-margin frames
on all seven target panels. The different-mapping control and every unchanged
short phase must retain their registered behavior. A margin-only gain cannot
satisfy the stronger fidelity criterion.

A failure closes this direct query-phase proposal. We will not search another
shift, individual head, or key-phase variant to rescue it. A success would
identify a useful conditional positional adapter; fresh OOD, selective removal,
independent extraction and shared use across circuits would still need testing.
All native weights and prefix/suffix computations remain charged.

- [Frozen native protocol](../../ABOUT_FOR_QUERY_PHASE_V1_PREREGISTRATION.md)
- [Native-phase controls](../../ROTARY_PHASE_TRANSPORT_V1_CONTROLS.json)
- [Frozen rows and token-gap audit](../../ABOUT_FOR_QUERY_PHASE_V1_ROWS.json)
- [Transport implementation](../../rotary_phase_transport.py)
- [Read-only phase capture](../../rotary_phase_capture.py)
- [Claude's v473 result](../../../bilinear_quotient/circuits/followups/unit_broad_circuit_v473_result.json)

## Native outcome and what the remaining error means

The managed run completed in58.72 seconds, with1405 forwards,107792 sequence
evaluations and120 fitting backward steps. It reproduces the original24-head
fit across13 layers; maximum recovery drift from the rounded v473 receipt is
.000460. Every native cache and unchanged short-phase bridge is exactly zero.
Instrument and specificity/identity predictions pass; phase portability and
complete-effect fidelity predictions fail.

For the long phrase, original complete-set recovery is.9371 and original
fitted-axis recovery is.6449. The alternate phase reduces those to.5266 and
.3542 respectively. Dividing the latter axis recovery by the ORIGINAL
complete-set recovery gives.3780, versus.6881 before transport. These ratios
measure registered signed answer-margin recovery, not percentages of all
behavior explained. The proposed phase adapter makes transfer worse.

We close this direct query-phase proposal without choosing another shift,
key phase, head subset or rank. The experiment does not rule out position
dependence elsewhere; it rules out this specific content-held explanation.

The full-vector failure requires a second distinction. Let e be the error
between the candidate and complete head-set vocabulary effects. For the
answer-minus-foil vector w, `||w||²=2`, so

    squared error along the answer axis = (wᵀe)² / 2
    squared error orthogonal to it = ||e||² - (wᵀe)² / 2.

The saved norms permit this exact calculation without another model run.
On all seven about/for panels, at least99.9912% of squared error lies outside
the answer axis; on unchanged short panels the minimum is99.9990%. These are
Euclidean squared-error shares, not probabilities or semantic attributions.

This means the fitted direction is not a reconstruction of the entire head
set. It does not establish that a narrower task-specific circuit is wrong:
the complete heads may serve other consumers. Equally, success on the one
answer contrast cannot explain those other computations or satisfy the
whole-output goal. The next decomposition must account for multiple actual
consumer functions instead of treating a task direction as a complete module.

With several reader rows W, the observed squared error is
`(We)ᵀ(WWᵀ)^+(We)`, where + denotes the pseudoinverse. The Gram matrix `WWᵀ`
records overlap between readers. For instance, about-minus-for and
from-minus-for have normalized reader cosine.5 merely because they share
the word “for”; that is not evidence of a shared hidden operation.

Executed controls verify the joint metric under duplicated readers, rescaled
coordinates and a dependent three-pair cycle, with max error3.56e-15 against
an exact fixture. Inconsistent cycle values are rejected. This supplies a
tool for comparing consumers; dynamic/intervention closure and meaningful
shared computation remain unproved. The mathematical review records these
limits and preserves the full four-property goal.

- [Native phase result](../../ABOUT_FOR_QUERY_PHASE_V1_RESULT.json)
- [Executed answer/complement partition](../../OUTPUT_READER_ERROR_PARTITION_V1_RESULT.json)
- [Joint-reader metric controls](../../CONSUMER_QUOTIENT_GRAM_V1_CONTROLS.json)
- [Three-hour mathematical review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0449.md)
