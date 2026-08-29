# Terminal copy four-head interaction cube v1 — preregistration scaffold

**Status:** source scaffold only; launch **NO-GO**. This document is written after the
E4 selection outcome and before creating a new role, authority, checkpoint outcome, or
interaction-cube result. It does not authorize a model forward.

## Scientific question and historical evidence

The completed E4 selection role fixed the four-head set

$$
F=\{\mathrm{L5H5},\mathrm{L7H3},\mathrm{L8H3},\mathrm{L8H4}\}.
$$

Its full position-mean replacement had copy CE effect `0.44870` nat and specificity
`0.46352` nat, but off-target CE effect `0.02441` nat exceeded the frozen `0.01`
budget. A post-hoc receipt-bound analysis found full-minus-singleton-sum copy excess
`0.34165` nat (simultaneous lower bound `0.20810`). Output-distribution KL was also
non-additive: joint/singleton-sum ratio `6.487` on copy positives and `1.683` off
target.

These observations select this experiment and may not be tested confirmatorily on the
exposed E4 selection role. They do not establish which pair/triple interaction is
responsible, nor distinguish interaction from finite-intervention geometry.

## New natural role

Create one new role of 192 unique source documents, one 257-token row per document,
using the existing recursive row registry to exclude every prior document ID, source
index, exact row, and registered prefix. The role must be frozen create-only and
receipt-last before model access. It must not deserialize or reuse E4 final/OOD rows;
the E4 negative receipt forbids those roles from opening.

The existing E4 fit-only head-position means may be reused because they were frozen
before E4 selection and contain no selection outcome. Their complete parent chain and
bytes must be rebound by the new authority. All copy-positive, matched-negative, and
off-target masks must be reconstructed from the new rows using the already frozen E4
label definition. Require at least 48 positions and 24 documents in each positive and
matched-negative cell or fail before checkpoint loading.

## Exact candidate bank

For every subset $S\subseteq F$, define

$$
w_S'=w_{\mathrm{native}}-\sum_{h\in S}w_h+\sum_{h\in S}\mu_h(p).
$$

The empty subset is the shared native arm. The complete bank has 16 subset arms. E4
already measured the native arm, four singletons, and full set on an exposed role; this
new transaction nevertheless reruns all 16 arms on shared new support. It may not copy
old point estimates into the new cube. The ten historically missing arms are six pairs
and four triples.

Add three finite-amplitude controls for the full set:

$$
w_{\alpha F}'=w_{\mathrm{native}}+
\alpha(w_F'-w_{\mathrm{native}}),
\qquad \alpha\in\{0.25,0.5,0.75\}.
$$

Together with full amplitude one, these form the frozen scaled-full-set curve. They are
diagnostic controls and cannot themselves select a subset.

All unselected heads, all MLPs, and all other layers remain native. Multi-layer subsets
execute sequentially on their live counterfactual states. Candidate order, batching,
call counts, and all-head recomposition checks must be frozen in the later execution
authority. No early stopping or adaptive candidate removal is permitted.

## Behavioral estimands

For cell $c$ and subset $S$, compute pooled-token effects

$$
v_c(S)=\frac{\sum_d
(\mathrm{NLL}_{d,c,S}-\mathrm{NLL}_{d,c,\mathrm{native}})}
{\sum_d n_{d,c}}.
$$

Compute native-to-candidate KL with the same pooled denominator, plus CE, correct-token
log probability, and top-1 accuracy for positive, matched-negative, and off-target
cells. Every arm must share exactly the same support hash, counts, and native sufficient
statistics within a cell.

The unique interaction coefficients are

$$
m_c(T)=\sum_{S\subseteq T}(-1)^{|T|-|S|}v_c(S),
\qquad
v_c(S)=\sum_{T\subseteq S}m_c(T).
$$

Report signed coefficients without clipping, separately by cell and metric. Also
report:

- the sum of coefficients at orders 1, 2, 3, and 4;
- prediction of the full-set effect using terms through each order;
- Shapley allocation as a secondary attribution, never as an executable component;
- full-minus-singleton-sum excess, to replicate the historical descriptive result;
- copy specificity for every coefficient, defined as positive minus matched-negative;
- off-target coefficient and total effect separately, not hidden inside specificity.

Möbius coefficients are intervention contrasts. They do not by themselves supply a
physical interaction tensor or a removable program.

## Displacement-geometry measurements

For every arm, document, and cell, stream and discard activations after accumulating:

$$
\sum\|\Delta_S\|^2,\quad
\sum\|w_S\|^2,\quad
\sum\|\mu_S\|^2,\quad
\sum\langle w_S,\mu_S\rangle,\quad
\sum\|r_{\mathrm{entry}}\|^2,
$$

where $\Delta_S=\mu_S-w_S$ is the projected-head replacement displacement and
$r_{\mathrm{entry}}$ is the residual stream entering the affected attention site.
For multi-layer subsets, report these quantities separately at each affected layer;
do not add vectors from different residual interfaces.

Derived descriptive metrics are live/mean norm ratio, live/mean cosine, displacement
RMS, and displacement-to-entry-stream RMS ratio. The scaled-full-set curve reports
deviation from the linear secant $\alpha v_c(F)$ for CE and KL.

These measurements test whether the joint effect coincides with a much larger,
highly aligned displacement and whether behavior varies smoothly with amplitude. They
cannot prove an internal mechanism: RMSNorm, later attention/MLPs, and softmax can all
create nonlinear response curves.

## Frozen inference family

Use 10,000 shared source-document bootstrap draws with a seed and exact quantile rule
to be fixed in the execution authority. Every bootstrap replicate must recompute pooled
numerators and denominators, the complete Möbius transform, specificity contrasts,
full-minus-singleton excess, and scaled-curve residuals.

Use a single simultaneous maximum-deviation family over all promotive CE coordinates:

- positive and specificity Möbius coefficients for all non-singleton terms;
- the positive and specificity historical excess replication;
- off-target total effect and scaled-full-set CE coordinates.

KL, geometry, Shapley values, and order-truncated reconstructions are explanatory and
cannot promote a subset in v1.

## Prospective decisions and claim boundaries

The historical non-additivity **replicates** only if simultaneous lower bounds for the
new role's positive and specificity full-minus-singleton excess are both above zero.

Call the interaction **low-order localized** only if at least one pair or triple has
simultaneous positive lower bounds for both its positive coefficient and specificity,
and its point positive coefficient accounts for at least 50% of the new full-set
positive effect in absolute units. The 50% threshold is a deliberately demanding
screen, not a theorem about true circuitry. Ties select lower order, then lower
literal intervention price, then canonical lexical order.

Even a localized coefficient licenses only a new conditional-component proposal. It
does not license final/OOD opening, extraction, selective removal, or an explained-CE
ledger change, because a Möbius contrast is not an independently executable tensor.

If non-additivity replicates but no term localizes, the result favors either diffuse
head interactions or an unsuitable head coordinate system. The next allowed route is
the downstream-Fisher basis. If non-additivity fails to replicate, the historical
post-hoc result remains descriptive and the subset route stops.

If the scaled curve and displacement geometry account for the apparent interaction,
report that bounded explanation without relabeling it a circuit. No post-outcome
threshold or candidate-bank change is permitted.

## Launch blockers

Launch remains NO-GO until all of the following exist and pass independent review:

1. a new 192-document row freezer, authority, and receipt with registry exclusion;
2. an exact 16-subset plus three-scaled-arm physical dispatcher;
3. streamed displacement-geometry sufficient statistics with known-answer tests;
4. a source-closed scorer that replays every Möbius/bootstrap decision;
5. a create-only three-terminal-state lifecycle: passer, scientific negative, or
   integrity failure;
6. a committed/pushed execution authority fixing rows, parents, sources, checkpoint,
   candidate order, call counts, seed, thresholds, and empty output paths.

The pure CPU contract is `terminal_copy_interaction_cube_v1.py`. Its tests establish
the exact 16-arm bank, ten missing pair/triple arms, signed Möbius recovery, order
truncation, and scaled-curve arithmetic. Passing those tests is infrastructure, not a
model outcome.
