# Exact effect-game reading of the P7 plus identity interface

**Time:** 2026-09-08 13:18 UTC  
**Scope:** CPU analysis of immutable result
`temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json`; no new model
execution, fitting, rank objective, or outcome-dependent intervention.

## Question and prior-work gate

The transferred `shap_tensor.md` note proposes an effect game

\[
E_a(S)=F(1,S)-F(0,S)
\]

to allocate how downstream components change a selected source component's causal
effect. Repository search confirms that exact Mobius/Shapley factorial accounting is
already used in several Bilin18 circuits and implemented in `ops/mobius.py`. The new
part worth importing is therefore not another generic Shapley implementation. It is
the **source-conditioned effect game**, together with the warning that a deletion
null can hide redundant routes and that an interaction index is not itself a directed
edge.

The current two-stream result already supplies a complete two-player intervention
game. Let `I` be the exact block-10 residual identity replacement and `P` be the fixed
seven-module response replacement

`P7 = A11 + M11 + M12 + M15 + M13 + M16 + M10`.

For signed recovery relative to the native-to-writer effect, define

\[
F(0,0)=0,\quad F(1,0)=I,\quad F(0,1)=P,\quad F(1,1)=1.
\]

Then the identity effect with no P7 response is `E_I(empty)=I`, its effect with P7
present is `E_I(P)=1-P`, and the exact two-player dividend is

\[
d_{IP}=F(1,1)-F(1,0)-F(0,1)+F(0,0)=1-I-P.
\]

The Shapley credits are

\[
\phi_I=I+d_{IP}/2,\qquad \phi_P=P+d_{IP}/2.
\]

## Exact immutable-result computation

| OOD phase | `I` alone | `P7` alone | identity effect with P7 | `d_IP` | Shapley identity | Shapley P7 | vector nonadditivity relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIT | 0.458161 | 0.568457 | 0.431543 | -0.026619 | 0.444852 | 0.555148 | 0.029631 |
| HOLDOUT | 0.480321 | 0.547874 | 0.452126 | -0.028195 | 0.466224 | 0.533776 | 0.032576 |

The credits sum to exactly one up to floating-point rounding. The independently
reported `identity_joint_contribution` equals `1-P7` in both phases, providing an
internal check on the effect-game mapping.

The negative 2.66--2.82 point dividend means mild redundancy, suppression, or
cancellation between the two installed streams under signed recovery. It agrees in
scale and sign with the small finite-vector nonadditivity. It does **not** weaken the
physical two-stream result: both singleton and full-model marginal contributions are
large and stable, and the joint intervention remains exact.

## What this does and does not identify

This is an exact decomposition of the measured **replacement game**. It is not yet a
directed identity-to-P7 edge decomposition. In the existing experiment the P7 module
outputs are captured from the selected writer run and clamped. They are not recomputed
under each value of the identity intervention. Consequently, `d_IP` measures nonlinear
readout/composition of two physical replacement streams, not how identity changes the
native computation performed by each P7 module.

A genuine downstream effect game must, for every identity state `b` and subset `S`,
recompute enabled downstream modules after installing `b`, while using one fixed
disable/replacement rule for modules outside `S`. Its values would answer which
downstream routes transmit, redundantly reconstruct, complement, or suppress the
identity effect. Large Shapley or interaction values would still nominate dependence;
edge-specific interchange/removal is required before calling them direct edges.

## Executable consequence for the live branch

Do not replace the queued P7 leave-one-module-out necessity audit. It is the cheaper
first discriminator and directly licenses the already-frozen A11 head and M11 product
splits.

After that receipt:

1. If A11 and/or M11 are stably necessary, execute the already-bound within-module
   splits first. They more directly improve within-module grouping, extraction, and
   manipulation than a broad coalition average.
2. If several P7 modules fail single-deletion necessity while the full P7 remains
   sufficient, that is the registered signature where redundancy may fool leave-one-out.
   Freeze an exact seven-player source-conditioned effect game over the unchanged P7
   set. Seven modules require `2 * 2^7 = 256` identity-by-subset arms per population,
   batchable with the existing clamp machinery.
3. Report singleton effects, full-model marginals, Mobius dividends, Shapley credits,
   pair interactions, and exact efficiency/closure. Use negative pair interactions as
   redundancy/cancellation evidence and positive values as complementarity evidence,
   but inspect the four corresponding intervention outcomes and do not relabel the
   interaction index as an edge coefficient.
4. Only fit a tensor-network surrogate after exact enumeration demonstrates a useful
   game and only with a held-out sup-norm error bound. Weight tensor rank or held-out
   regression R-squared alone does not license Shapley accuracy.

This conditional use adds a redundancy-sensitive circuit diagnostic without changing
the current frozen experiment, opening an 8,192-arm twelve-player sweep, or treating
generic attribution as causal identification.

## Independent effect-game receipt after this analysis

Claude's independently preregistered nine-player head-patch game
`unit_shapley_effect_game_v271_result.json` subsequently landed valid in 19.3 seconds.
Its exact efficiency residual is `-2.08e-17`; Shapley attribution reorders the singleton
ranking, but its redundancy prediction fails. The largest negative reported pair value
is `-0.027131`. The direct/unlisted route is large:

\[
E_a(\varnothing)/E_a(D)=0.203774/0.252417=0.8073.
\]

Thus only about 19.3% of that source effect is allocated to the enumerated set. This is
an informative prior for the P7 game: a broad downstream list may still leave most of a
source effect in the residual or unlisted background, and pair effects near three points
are plausible. It does not license changing the P7 game's frozen `.03` bars.

The comparison also sharpens two semantic distinctions. V271's game patches the source
and coalition members to donor values; the P7 successor instead installs one source and
lets enabled modules **recompute live**, clamping the complement to native outputs. In
addition, all P7 players are causally after the block-10 source, whereas v271 deliberately
included one pre-source negative control. Finally, v271 reports an unweighted coalition
mean of pair second differences, while the P7 runner reports the Grabisch--Roubens index
from Harsanyi dividends; their magnitudes must not be treated as the same estimator.
