# Three-hour mathematical review — 2026-08-29 22:48 UTC

## Update: the composition experiment is now decisive

The highest-priority experiment from the preceding review is complete on 192 fresh,
registry-disjoint FineWeb documents.  It compared three equal-price rank-512 MLP2
programs, each alone and after MLP0-C512:

- `FULL512`: the original rank-512 fit;
- `CONTINUE512`: the same bytes followed by 1,200 native-only training steps;
- `ROBUST512`: the same bytes and compute, but trained equally on native and
  MLP0-C512-produced inputs.

For a program $P$, define its composition interaction with C512 by

$$
I_P = \Delta CE(C512+P)-\Delta CE(C512)-\Delta CE(P).
$$

This is the extra error that cannot be predicted by adding the two standalone errors.
The old interaction replicated:

$$
I_{FULL}=0.008569
\quad\text{with simultaneous 95\% CI}\quad
[0.006623,0.010655].
$$

The trajectory-trained program had

$$
I_{ROBUST}=0.007442
\quad\text{with CI}\quad
[0.005668,0.009337].
$$

Thus the point reduction is only

$$
1-\frac{|I_{ROBUST}|}{|I_{FULL}|}=13.2\%,
$$

not the preregistered 50%, and its upper confidence bound is not below 0.005 nat.
The robust program improves the composed arm over old FULL512 by 0.010202 nat
([0.007997, 0.012438]), but it improves over the equal-compute CONTINUE512 control by
only 0.000311 nat ([-0.000293, 0.000916]).  That is statistically compatible with no
CE benefit.  Its small KL advantage over CONTINUE512 is positive but was explicitly
diagnostic, not a pass gate.

**Ruling:** `trajectory_exposure_rejected`.  Almost all improvement came from giving
the same rank-512 program more optimization, not from exposing it to the C512 state
distribution.  The interaction is real and stable, but ordinary local complete-write
MSE on two backgrounds is not the correct objective for removing it.

The run took 43.27 seconds.  Result SHA-256:
`73feb593114cf133d449ecd1970aea9937a00a65fbffe997c57a1b1efa2d98d7`.
Receipt SHA-256:
`22026cd77420e8cf739796e2283782bbe971be1852eaa1996902aaf7e0bab30e`.

## Mathematical consequence

The earlier polynomial certificate showed that every global rank-512 replacement is
necessarily wrong: a fixed polarization slice of each native MLP0--2 has rank 1,152.
The new experiment shows that merely sampling both observed input distributions does
not identify which part of that unavoidable error matters downstream.  Therefore the
next representation must be **consumer-weighted** or **interaction-targeted**.  More
unweighted CP/Tucker/HOSVD, local PCA/RRR, or ordinary two-environment MSE is pruned.

The attention-lane interpretation was also corrected during this review.  Attention 5
is close to a presence/control interface, but attention 6's *content above a constant
row is high-dimensional*: rank 16 recovers only about 17--22%, rank 64 about 54--58%,
and rank 128 about 79--81%.  Earlier claims that one direction carried most content
used deletion, rather than the constant row, as the denominator and measured presence.
Attention 5/6 remains a useful verified downstream response, but attention 6 should
not be described as a one-dimensional semantic code.

## Ranked move 1 — consumer-adjoint weighted polarization

### Exact object and operation

Let $A_y$ be the $1152\times1152$ polarization slice of an early MLP.  For a suffix
consumer vector $g$ (centered final logits, a copy score, capitalization score, or an
attention-interface response), let $J_g$ be its Jacobian with respect to the MLP write.
Let $P$ be a covariance or controllability metric for reachable MLP inputs.  Analyze
and fit the weighted slices

$$
\widetilde A_{y,g}=W_g^{1/2}J_g A_y P^{1/2},
$$

or a randomized joint sketch over several $g$ and $y$.  This asks for product rank in
directions the suffix can observe, rather than in the raw 1,152-dimensional output.

### Mathematics, assumptions, and prediction

For fixed linear weights, Eckart--Young again gives the optimal rank-$r$ approximation
and a tail-energy lower bound.  Empirical controllability/observability balancing is
the nonlinear local analogue; classical nonlinear empirical balancing is described by
[Condon and Ivanov](https://arxiv.org/abs/math/0606430) and
[Kawano and Scherpen](https://arxiv.org/abs/1902.09836).

The suffix is nonlinear and RMSNorm makes the metric state-dependent, so this is a
local/finite-distribution certificate, not a global transformer theorem.  It also
fails if the consumer bank is incomplete.

It predicts something beyond reconstruction: at the same 512-product price, a
consumer-weighted fit should beat **CONTINUE512**, not merely old FULL512, on composed
CE and reduce $I_P$ while preserving standalone CE.  The cheapest falsifier is a
64-document randomized JVP/VJP sketch on native and C512 backgrounds.  Reject it if
the weighted spectrum has no stable knee or if a held-out finite intervention is not
predicted better than equal-price local MSE.

## Ranked move 2 — directly factor the mixed interaction functional

### Exact object and operation

The failure is a mixed finite difference, not a large standalone error.  Treat

$$
\mathcal I(C,P)=CE(C+P)-CE(C)-CE(P)+CE(N)
$$

as the quantity to model.  Differentiate or finite-difference it with respect to the
MLP2 program coefficients and factor the resulting bilinear sensitivity tensor.  A
shared part preserves native behavior; a small private correction is selected only
by directions that contribute to the mixed term.  Constant bias and scalar calibration
remain effectively free, but are fitted inside every arm rather than post hoc.

This is an ANOVA/Möbius decomposition on the intervention lattice: the mixed term is
the second-order interaction after removing both main effects.  Its attraction is
falsifiability—the target is exactly the independently replicated 0.0086 nat.  Its
assumption is that the interaction has a stable low-rank tangent or sparse support;
large finite interventions may make first-order sensitivities misleading.

The cheapest falsifier is to compute directional derivatives for 32--64 documents,
fit a rank-16/32 correction on half, and test whether it predicts the sign and document
ordering of held-out interaction contributions.  If not, do not spend a full refit.

## Ranked move 3 — intervention-complete causal quotient, then finite Hankel rank

Define early states to be equivalent only when a bank of verified consumers agrees
under a registered family of suffix interventions:

$$
x\sim_\varepsilon x'
\Longleftrightarrow
\sup_{a\in\mathcal A}d(G_a(x),G_a(x'))\le\varepsilon.
$$

This is the operational answer to whether MLP0 tokens form clusters: raw vectors may
all be distinct while downstream computation identifies many of them.  Approximate
causal abstraction supplies the commutation criterion
([Beckers, Eberhardt, and Halpern](https://proceedings.mlr.press/v115/beckers20a.html)).
Once at least three independent late consumers exist, a prefix-by-intervention/
continuation Hankel block can test whether those response classes have a smaller
minimal realization.

The main assumption is consumer completeness.  Copy plus attention 5/6 is not enough;
capitalization, numeric, syntax, and entity consumers must be added and independently
validated.  The cheapest falsifier is leave-one-consumer-out and leave-one-intervention-
composition-out prediction.  If quotient distance does not predict the withheld
response, it is not a useful definition of simplicity.

## Pruning and project impact

- **Pruned:** more raw rank-512 tensor factorization as a route to global identity;
  exact slice rank already rules it out.
- **Pruned:** native+C512 local-write MSE as the explanation of composition error;
  ROBUST512 did not beat its matched continuation control in CE.
- **Not pruned:** rank-512 as a distributional executable family.  Continued training
  improved standalone dCE from 0.051274 to 0.042253 and composed dCE from 0.061994 to
  0.052102 at no added storage or products.
- **Not yet established:** semantic extraction, selective removal, OOD transport, a
  complete causal quotient, or any terminal action.

The strict project ledger therefore does not move: **5.348245316%** certified storage,
**10.923302467%** named causal CE, **4.72714 nat / 89.077%** unexplained, and **0/68**
terminal actions.
