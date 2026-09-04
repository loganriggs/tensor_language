# Task 14 MLP15/17 full-rank conditional-response panel

**Frozen:** 2026-09-04 18:03 UTC

**Finalized after independent audit:** 2026-09-04 18:16 UTC

**Status:** Phase-0 design and CPU metadata checks only; no model, GPU, queue, optimization, or subspace fit is authorized

## Why this panel comes before a learned subspace

The existing screens show a direction-dependent, mostly additive effect through MLP15 and MLP17 after the Task-14
head-11.3 interchange. That is not enough evidence that the same response exists for cross-noun donors, literal
cross-syntax donors, or complete-subject donors. Learning a small subspace now could merely fit the one direction
already known to work.

Phase 0 therefore makes no compression claim and fits nothing. It asks whether the **entire** bilinear output of
MLP15, MLP17, and their depth-ordered combination has a causal effect across the full frozen inner Task-14 panel.
Only a later, separately authorized phase may ask for the smallest subspace that reproduces this full-rank effect.

## Exact data and relation roles

The panel reuses the existing DISCOVERY inner split:

- FIT: 153 relations, 64 endpoints, ordinal hash
  `5c24f97e98de6ff351514e19586d6ec4e72b5d1af6a3d5971d3d0b7d1b2267db`;
- SELECT: 145 relations, 64 disjoint endpoints, ordinal hash
  `4b7de6802d6f6fd23c669cde5276e57987ba552dd2b1ac67068c21fea9c0f823`.

The outer 544 VALIDATION relations remain unopened. The exact metadata census is frozen in
`task14_mlp15_17_full_rank_panel_relation_audit_v1.json`.

Target versus control is determined only by `expected_relation`:

$$
\text{target}\iff\texttt{expected\_relation=opposite\_subject\_toward\_donor},
$$

$$
\text{control}\iff\texttt{expected\_relation=same\_subject\_zero\_projected\_effect}.
$$

The family letters are not roles. In particular, P has true `P_positive_transfer` targets as well as zero-effect
controls, while C has true complete-subject transfer targets as well as controls. FIT contains 116 targets and 37
controls; SELECT contains 106 targets and 39 controls.

Targets are reported separately as paired, cross-noun, literal cross-syntax, and complete-subject transfer. Every
prediction is evaluated separately for recipient state -1 (singular) and +1 (plural). No result may pool away arm,
family, matching rule, recipient subject state, module, or intervention direction.

## Conditional product response and signs

For a relation, $B$ is the native recipient trajectory. $H$ is the same recipient with only attention head 11.3's
final-token pre-output value replaced by the natural donor value. At MLP layer $\ell\in\{15,17\}$,

$$
z_\ell=(W_{L,\ell}x_\ell)\odot(W_{R,\ell}x_\ell)\in\mathbb R^{4608},
$$

and the response being tested is

$$
\delta z_\ell=z_\ell(H)-z_\ell(B).
$$

This is a conditional response to the actual head-swap trajectory. It is not the product activation from an unrelated
natural donor prompt.

Let $f_\ell(z)$ mean: run the $H$ trajectory to layer $\ell$, override only that layer's product vector with $z$,
and then recompute the rest of the model normally. The equations apply both to the donor-oriented `is`/`are` margin
and to the full logit vector. With $P=I_{4608}$, define

$$
E_{\mathrm{full}}=f_\ell(z_B)-f_\ell(z_H).
$$

Removal must satisfy

$$
E_{\mathrm{remove}}
=f_\ell(z_H-P\delta z)-f_\ell(z_H)
=E_{\mathrm{full}},
$$

and sufficiency must satisfy

$$
E_{\mathrm{suff}}
=f_\ell(z_B+P\delta z)-f_\ell(z_B)
=-E_{\mathrm{full}}.
$$

The sign is important. Negative effects are retained and described as compensatory when appropriate; they are never
clipped or silently converted to magnitudes. The product endpoint equalities and margin/logit equalities must hold to
maximum absolute error $10^{-4}$, otherwise the intervention is invalid.

## Exact bilinear expansion and Phase 0B

Let the change in normalized MLP input be

$$
\delta x_\ell=x_\ell(H)-x_\ell(B).
$$

The total product response has the exact three-term expansion

$$
\begin{aligned}
\delta z_\ell
={}&(W_Lx_B)\odot(W_R\delta x)
 +(W_L\delta x)\odot(W_Rx_B)\\
 &+(W_L\delta x)\odot(W_R\delta x).
\end{aligned}
$$

These are the base-left/changed-right interaction, changed-left/base-right interaction, and changed-left/changed-right
interaction. Phase 0 must numerically verify this identity to maximum absolute error $10^{-5}$ on every relation and
module.

Testing which term actually causes the downstream effect is separately priced as **Phase 0B**, not silently added to
the cheap panel. In the same fixed H upstream background, Phase 0B installs all eight subsets of the three terms above
the base product, for both MLP15 and MLP17. The eight-cell Boolean factorial reports each term's main effect, all three
pair interactions, and the three-way interaction by exact Boolean-lattice Möbius differences. The empty and full
subsets reuse Phase 0's reset and H endpoints, so only six new states per module are needed. At batch size 32 this
costs at most 120 additional model calls and 3,576 sequence examples, with no backward pass or optimizer. A
64-condition cross-module term factorial is deferred and is not part of this price. Term vectors are interpreted only
after applying $W_D$, or modulo its null space.

## Depth-ordered MLP15 plus MLP17 intervention

The earlier MLP15-by-MLP17 factorial combined cached module outputs. It remains useful evidence about additivity, but
it does not prove that MLP17 was reached after recomputing the network under an MLP15 intervention.

The new joint reset is executable in model order:

1. Run $H$ to MLP15 and replace $z_{15,H}$ by $z_{15,B}$.
2. Continue through the modified MLP15 output and block 16 normally.
3. At MLP17, record the newly recomputed product $z_{17\mid15^-}$.
4. Compute its live response to the native base reference,
   $\delta z_{17\mid15^-}=z_{17\mid15^-}-z_{17,B}$, and set the product to $z_{17,B}$.
5. Recompute the remaining model normally.

The rescue traverses the same hooks but installs $z_B+\delta z$, returning the live current product at each site.
The joint reset-to-rescue difference must be the negative of the reset effect. For every relation, the runner must
record exactly one ordered sequence of `MLP15 product entered`, `MLP15 reset applied`, `MLP17 entered after MLP15`,
`live MLP17 product captured`, and `MLP17 reset applied`. It retains hashes of both $z_{17\mid15^-}$ and
$z_{17\mid15^-}-z_{17,H}$. The MLP17 value must be copied at the live product hook after the MLP15 intervention;
a cached $z_{17,H}$ or precomputed substitute is forbidden. A difference norm and hash by themselves are not accepted
as evidence of recomputation.

## What is measured

For each relation, let

$$
h_i=s(H_i)-s(B_i),\qquad e_i=s(\mathrm{reset}_i)-s(H_i),
$$

where $s$ is the donor-oriented answer margin. Within each registered target class and recipient-state cell $G$,
report

$$
\beta_G=-\frac{e_G^\mathsf{T}h_G}{h_G^\mathsf{T}h_G},
\qquad
q_G=\frac{\operatorname{RMS}(e_G)}{\max(\operatorname{RMS}(h_G),10^{-12})},
$$

and the absolute cosine between $e_G$ and $h_G$, with the signed dot product alongside it. Positive $\beta_G$
means resetting the MLP tends to undo the head effect; negative $\beta_G$ means the MLP tends to compensate for it.
Full-vocabulary RMS and maximum logit changes are computed online for every relation, but full vocabulary arrays are
not retained.

## Opposing predictions

The **broad-response** prediction requires the depth-ordered joint response to have $q_G\ge0.10$ and absolute cosine
at least 0.50 separately for singular and plural recipients in all four target classes. At least one of MLP15 or
MLP17 must also have $q_G\ge0.05$ in each of those eight directional cells. This allows restorative and compensatory
signs, but requires a coherent causal effect outside the original paired direction.

The **direction-specific** prediction gates specifically on the depth-ordered joint MLP15-plus-MLP17 reset effect. It
requires paired joint $q_G\ge0.10$ for recipient state +1, the plural-recipient to singular-donor direction that was
previously live, and paired joint $q_G<0.05$ for recipient state -1, its opposite. It also requires the joint effect
to have at least one separately reported recipient-state cell with $q_G<0.05$ in at least two of cross-noun,
literal cross-syntax, and complete-subject transfer. Single-module values are reported but do not select this terminal.
If this occurs, the strongest justified label remains
"direction-specific compensatory response," not "subject-number feature." Results between these patterns are reported
as mixed, without upgrading the claim.

Controls use the FIT median absolute target head effect as a fixed scale. If any registered control arm has RMS MLP
effect above 0.20 of that scale, the screen ends in control failure.

## The product-space gauge

The MLP output is

$$
y_\ell=W_{D,\ell}z_\ell+b_{D,\ell}.
$$

Because $W_D:\mathbb R^{4608}\rightarrow\mathbb R^{1152}$, at least 3,456 product-space directions are invisible at
the MLP output. For every $n\in\ker(W_D)$, $z$ and $z+n$ produce the same MLP output. Phase 0 therefore identifies
the dense output response $W_D\delta z$ and its downstream causal function, not a unique 4,608-dimensional hidden
basis. Any later product-space feature claim must be stated modulo this null space or compiled into its dense quadratic
output tensor.

## Deferred unrelated-task control

The numbered-list behavior is currently localized to L8H7/L8H3 with candidate writes in MLP8--14, not MLP15/17.
Using it now could create a trivial collateral-effect ratio because the MLP15/17 full-module denominator may be nearly
zero. It is explicitly deferred from Phase 0. A later unrelated-task control must first pass a preregistered MLP15/17
full-module liveness floor.

## Exact price and execution boundary

At batch size 32, the price ceiling is:

- 128 native endpoint examples in at most 4 model calls;
- 298 relations under 7 conditions (`H`, two MLP15 conditions, two MLP17 conditions, and two joint conditions) in at
  most 70 calls;
- 74 total forward calls and 2,214 sequence examples;
- zero backward calls and zero optimizer updates.

This is a screen, not a subspace fit, learned circuit, OOD test, or adoption result. No execution is authorized by this
document. A create-only implementation amendment must hash the runner and sources, freeze exact retained bytes and a
measured runtime cap, and use the managed GPU queue.
