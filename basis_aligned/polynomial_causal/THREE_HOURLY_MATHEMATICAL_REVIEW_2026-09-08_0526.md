# Three-hour mathematical tensor-network review — 2026-09-08 05:26 UTC

## Exact model and current circuit object

Bilin18 has residual width `d=1152`, `L=18` blocks, `H=9` attention heads of width `p=128`,
bilinear-MLP hidden width `m=4608`, and vocabulary size `V=50304`. For batch row `i`, token `t`,
residual coordinate `a`, head `h`, source token `s`, and hidden coordinate `j`, one block has the
schematic form

\[
\bar x_{ita}=\operatorname{RMS}(x_{it:})_a,
\quad q^{(1)}_{ith:}=W^{Q1}_{\ell h}\bar x_{it:},
\quad q^{(2)}_{ith:}=W^{Q2}_{\ell h}\bar x_{it:},
\]

with analogous keys, per-head RMS normalization and rotary maps, and unnormalized causal attention

\[
A_{ihts}=1_{s\le t}\,
\frac{\langle \widehat q^{(1)}_{ith},\widehat k^{(1)}_{ish}\rangle}{p}
\frac{\langle \widehat q^{(2)}_{ith},\widehat k^{(2)}_{ish}\rangle}{p},
\quad z_{ithc}=\sum_s A_{ihts}v_{ishc}.
\]

There is no softmax. Before normalization, the attention score is degree four in residual inputs
and its contraction with values is degree five. RMS normalization makes the deployed block a
rational/algebraic rather than globally polynomial map. The MLP core is

\[
M_{ita}=\sum_j W^D_{aj}\big[(W^L\operatorname{RMS}(x))_{itj}
(W^R\operatorname{RMS}(x))_{itj}\big],
\]

implemented exactly as `Down(Left(RMS(x)) * Right(RMS(x)))`; it is degree two after its normalized
input. Learned recurrent coefficients mix the live residual and the layer-zero value stream. The
final decoder is exact RMS normalization, tied unembedding, and elementwise soft cap
`30 tanh(logit/30)`.

The new physical state objects are

\[
S\in\operatorname{St}(1152,8),\qquad U_0,U_1\in\operatorname{St}(1152,2),
\]

where `S` is temporal Q8 and `U_f` is the opposite-fold is-was construction union used when fold
`f` is held out. The measured singular values of `U_0^T U_1` are `.9050,.8087`, so the is-was
plane is stable but not identical across folds. The singular values of `S^T U_f` are
`.6752,.2465` and `.6520,.1941`. Thus neither is-was plane is contained in temporal Q8. Numerically,

\[
\operatorname{rank}[S\ U_0\ U_1]=12,
\]

while a single deployed fold uses a ten-dimensional sum `span[S,U_f]`. “Rank-12 task-typed direct
sum” refers to storage of both cross-fit realizations, not to twelve simultaneously fitted degrees
of freedom in one held fold.

Allowed inputs for the next causal test are exactly 32 frozen rows, each with four same-sequence
cells `00,10,01,11`. The first bit toggles a temporal will/had command and the second an is/was
command. Every earlier temporal prefix in the successor is token-identical to a capability-qualified
standalone prefix. Outputs are the two canonical signed margins

\[
m_T(x)=\ell_{will}(x)-\ell_{had}(x),\qquad
m_I(x)=\ell_{is}(x)-\ell_{was}(x),
\]

never “correct minus foil,” because that coordinate would reverse sign when the cell bit changes.
The capability gate requires at least `6/8` correct independently in all 32
phase/template/cell/role cells before any causal outcome opens.

## Gauge symmetries and exact weight translation

Each stored basis has a right orthogonal gauge: `S ~ S R`, `U_f ~ U_f Q` for `R in O(8)` and
`Q in O(2)`. For an orthonormal residual basis `Q` of rank `r`, an output writer
`W_w in R^{1152 x k}` and input reader `W_r in R^{k x 1152}` have exact projected weight fractions

\[
\phi_w(Q)=\frac{\|Q^T W_w\|_F^2}{\|W_w\|_F^2},\qquad
\phi_r(Q)=\frac{\|W_r Q\|_F^2}{\|W_r\|_F^2}.
\]

Both are invariant under the basis gauges. Dividing by the isotropic expectation `r/1152` gives
rank-comparable enrichment. The complete inventory contains 180 writers—nine `W_O` head slices
and one MLP Down tensor per layer—and 846 readers—five Q/K/Q2/K2/V head maps and two MLP input maps
per layer. Three bases times 1,026 interfaces gives exactly 3,078 contractions.

This atlas found six interfaces above fourfold enrichment for `S`, `U_0`, and `U_1`: output writers
L9H1, L11H3, L15H5; value reader L11H3:v; and query readers L15H5:q/q2. The most extreme scores are
`24.38x` for temporal writing through L11H3, `13.80x` for its temporal value read, `12.51/12.54x`
for is-was writing through L15H5, and `7.79/7.46x` for its is-was q2 read. Is-was fold top-20
Jaccards are `.8182` for readers and `.7391` for writers.

The atlas is invariant geometry but not causal identification. RMS/QK normalization means a large
`||WQ||` need not give a large finite response; downstream nonlinearities can cancel a head write;
and a matrix can geometrically touch both task spaces without being exercised by either population.

## Causal head/module tensor and interaction law

Let `c in {00,10,01,11}` and let `tau_r(c)` toggle only role `r`. At layer `ell` and the query token
immediately preceding role `r`'s answer, capture the concatenated native pre-output-projection head
tensor

\[
Z_{i c \ell}\in\mathbb R^{9\times128}.
\]

A singleton intervention `do(h <- donor)` substitutes only
`Z_{ic\ell h}:=Z_{i,tau_r(c),\ell h}`. A complete-module intervention substitutes all nine slices.
Because `W_O` is linear, this is exactly a full head-output or full attention-output patch before
the common projection, while leaving every unselected slice unchanged. It does not claim which
Q/K/V factor generated the head; factor identification is a later split conditional on a causal
head result.

For canonical margin `m_r`, define the native paired-cell effect and intervention effect

\[
g_r(c)=m_r(x_{\tau_r(c)})-m_r(x_c),\qquad
e_{r,H}(c)=m_r(\operatorname{do}_H(x_c))-m_r(x_c).
\]

The primary projection is `<e,g>/<g,g>`, accompanied by cosine, direction agreement, and
`||e-g||/||g||`. The non-target margin is evaluated from the same patched forward. Since the
attention mask is lower triangular and the is-was query occurs after the temporal query,

\[
\operatorname{do}(Z_{q_I})\quad\Longrightarrow\quad
\Delta m_T=0
\]

exactly in real arithmetic. A violation above `1e-5` is therefore an instrumentation failure, not
a scientific interaction.

For the three nominated heads `H={L9H1,L11H3,L15H5}`, define the finite-intervention interaction

\[
\eta_r=e_{r,H}-\sum_{h\in H}e_{r,\{h\}},\qquad
\rho_r=\frac{\|\eta_r\|_2}{\|e_{r,H}\|_2}.
\]

This is a Möbius first-order remainder around the native `000` head arm. It includes cross-layer
and downstream nonlinear interaction. Only `rho_r <= .25` in both phases, plus union recovery no
worse than the best singleton by more than `.05`, licenses greedy composition. Otherwise the exact
seven nonempty subsets are the correct next finite factorial; summing singleton effects would be
mathematically unjustified.

## Neighboring theorems and assumption audit

1. **Grassmann principal-angle geometry.** Singular values of `S^T U` are invariant under separate
   right-orthogonal gauges and equal cosines of principal angles. Containment of a rank-two `U` in
   `S` would require both singular values one (within tolerance). Our second values near `.2-.25`
   decisively violate that assumption, so a shared-Q8 interpretation is false even though some
   physical weights touch both spaces.
2. **Orthogonal projection/Pythagorean energy.** For orthonormal `Q`, the projected Frobenius
   fraction lies in `[0,1]`; rank normalization compares against an isotropic subspace. This theorem
   assumes only a Euclidean residual gauge. It says reachability through weights, not activation
   occupancy or causal use; those missing assumptions are exactly why intervention follows.
3. **Causal triangularity.** A strictly causal decoder guarantees later-token edits cannot change
   earlier-token states or logits. The deployed mask satisfies this. It does not guarantee the
   converse: the earlier temporal patch may alter the later is-was margin, so that collateral must
   be measured rather than assumed absent.
4. **Möbius/ANOVA decomposition on a finite Boolean cube.** Every three-component intervention
   response has an exact expansion into singleton and higher-order interaction terms. Greedy
   addition assumes the higher-order remainder is small. The model's normalized bilinear attention,
   quadratic MLPs, and cross-layer placement violate exact additivity a priori; the union arm is the
   required empirical test.
5. **Weight-sharing versus representation-sharing.** A linear map can have large projection onto
   two subspaces without those subspaces intersecting. The observed direct-sum geometry plus shared
   weight enrichment is a concrete example. Therefore “same downstream component” and “same
   subspace” are distinct hypotheses and must not be conflated.

## Literal prices and executable consequence

Physical storage is 8, 2, and 2 columns of length 1,152 for `S,U_0,U_1`: 13,824 float32 values plus
layout and hashes. One held fold operationally needs `S` and one `U_f`, or 11,520 values before any
coefficient tensors. The complete weight atlas costs one checkpoint load, zero model forwards,
1,026 interfaces, and 3,078 basis-weight contractions; it changes no model parameters.

The full causal head/module test costs exactly one native capture plus, for each of two roles, three
singleton heads, three parent modules, and one union: 15 forwards, 1,920 sequence evaluations,
3,840 scored token positions, zero backwards, zero updates, and zero fitted parameters. Its stored
intervention table has `2 roles x 7 arms x 128 rows = 1,792` records.

The strongest executable consequence is now fixed: if the pending prefix-preserved capability
issues all 32 licenses, hash-bind that immutable result into the staged runner and execute the
full-head/module factorial. Material module recovery with failed head recovery localizes only the
parent; head recovery plus low collateral identifies a shared physical interface carrying distinct
task-typed states; a failed union additivity gate routes to the exact seven-subset factorial. If the
native license fails, no causal arm opens. The runner, accounting contract, canonical-margin test,
and nonfinite guard are already implemented; the immediate action after this review is to continue
the managed capability queue and finalize the result hash only on a valid license.
