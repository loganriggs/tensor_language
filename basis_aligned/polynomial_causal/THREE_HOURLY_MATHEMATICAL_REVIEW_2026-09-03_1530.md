# Three-hour mathematical review — 2026-09-03 15:30 UTC

## The object we actually want

Let $z\in\mathbb{R}^{1152}$ be the activation at a candidate site, and let $F_c(z)$ denote all downstream
measurements in context $c$: not only one answer-logit difference, but the relevant answer distribution, registered
reader states, unrelated-task outputs, and any later variables that the proposed circuit should or should not change.

For a base/donor pair with activation difference $d=z_{\mathrm{donor}}-z_{\mathrm{base}}$, a linear interchange uses

$$
z' = z_{\mathrm{base}} + P d,
$$

where $P$ is an orthogonal projector onto a learned subspace. If the semantic variable changes from $v_b$ to $v_d$,
the required target is a *vector of causal consequences*

$$
F_c(z')-F_c(z_{\mathrm{base}}) \approx
F_c\!\left(\operatorname{do}(v=v_d)\right)-F_c\!\left(\operatorname{do}(v=v_b)\right).
$$

For an answer-preserving change, the required target is zero on the behavior and on every registered downstream
reader of that variable, even when the complete activation changes substantially. This is causal abstraction by
interchange intervention, not ordinary supervised dimension reduction; the formal framing follows
[Geiger et al.](https://arxiv.org/abs/2106.02997) and the distributed-basis search follows
[Geiger et al.](https://arxiv.org/abs/2303.02536).

## Why the R540 transfer result was not identification

Suppose the fitted and evaluated measurement is the scalar closer margin $m(z)$. Locally,

$$
m(z+Pd)-m(z) \approx \nabla m(z)^\top P d.
$$

A rank-one projector aligned with an average $\nabla m$ can move that margin for many unrelated donor pairs. At a
late residual site, this gradient includes all downstream computation leading to the closer logits. The raw
unembedding contrast is only its cheapest approximation; the more faithful shortcut span is

$$
S_{\mathrm{endpoint}}=
\operatorname{span}\left\{\nabla_z(m_i-m_j)(z_c): c\text{ in FIT},\ i,j\text{ closer values}\right\}.
$$

R540's target transfer and control leakage are exactly the signature predicted by this shortcut. Deflating
$S_{\mathrm{endpoint}}$ is a useful diagnostic, but it cannot by itself identify the variable: at first order it also
removes the route by which the variable changes that endpoint. The real repair is to add independent downstream
consequences and semantically distinct interventions, then ask for one alignment that satisfies all of them.

This is consistent with causal-representation identifiability results: interventions can resolve otherwise
indistinguishable latent rotations, but only under explicit coverage assumptions. For example,
[Squires et al.](https://proceedings.mlr.press/v202/squires23a.html) prove identifiability for a linear latent causal
model when every latent variable is intervened on; their theorem does not directly apply to this nonlinear fixed
network, but it explains why one scalar outcome and one edit family cannot fix our gauge. Recent work on
[identifiability of causal abstractions](https://proceedings.mlr.press/v258/li25g.html) likewise treats paired
interventions as the information that selects among observationally equivalent representations.

## Interactions are part of the circuit, not noise

For two proposed variables $A$ and $B$, run the four factorial interventions and measure

$$
I_{A,B}=Y(A_1,B_1)-Y(A_1,B_0)-Y(A_0,B_1)+Y(A_0,B_0).
$$

$I_{A,B}$ is the finite interaction: the part of the joint effect that cannot be assigned to either intervention
alone. Higher-order terms follow the same inclusion--exclusion rule. This is the right way to test whether, for
example, an induction selector composes with a payload value, or whether two MLP decoder-vector/encoder-composition
paths only matter jointly. It also avoids interpreting an individual activation-patching effect as purely belonging
to that component. The multiple-mediators analysis of
[Vaidyanathan et al.](https://arxiv.org/abs/2606.27510) derives precisely why individual indirect effects absorb
prompt-dependent interactions and why group interventions are necessary.

## A basis-independent grouping rule

Native heads, neurons, and MLP product coordinates are not privileged semantic units. For a physical component or
subspace $u$, define its causal response fingerprint as the vector of finite changes it induces across all registered
counterfactual families, downstream readers, unrelated controls, and joint interventions:

$$
\Phi(u)=\left(\Delta Y_1(u),\ldots,\Delta Y_K(u), I_{u,v_1},\ldots,I_{u,v_J}\right).
$$

Components from different modules should be grouped when they have the same role in the high-level computation and
interchange as substitutes; a single module should be split when parts have different stable fingerprints. The
fingerprint is invariant to rotations inside any physical subspace that leaves all tested causal responses
unchanged. It therefore addresses the gauge problem more directly than clustering weights or reducing rank.

This is related to causal sufficient dimension reduction, whose aim is to preserve a treatment--outcome relationship
under a lower-dimensional representation ([Nabi et al.](https://proceedings.mlr.press/v180/nabi22a.html)). Our
requirement is stricter: the representation must preserve several interventions and intended interactions while
being selectively inert on unrelated outcomes.

## Executable consequence

1. Let R544 decide whether the four-valued pending-opener variable has valid complete-state targets and controls.
2. If it passes, fit against a multi-output causal response: closer distribution plus at least one independently
   computed downstream reader, while enforcing all three invariant families.
3. Report overlap with the raw closer-unembedding span and the pooled downstream endpoint-gradient span. Compare the
   ordinary fit with a shortcut-deflated diagnostic, but do not call deflation a circuit by itself.
4. Add pairwise factorial interventions between the pending-opener candidate and its downstream reader. Acceptance
   requires predicting the joint effect, not merely both marginal effects.
5. Apply the same response-fingerprint schema to induction selector × payload and to the equality-score shared
   subroutine. These are better tests of grouping across heads and splitting within heads than further rank screens.

The hypothesis is falsified if no single alignment survives multiple answer-changing constructions, all variable
values, answer-preserving edits, and independent downstream consequences. In that case the pending opener is not one
linearly represented variable at the tested site; it must be split, modeled nonlinearly, or moved to an earlier
physical site.
