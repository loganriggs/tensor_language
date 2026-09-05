# Three-hour mathematical circuit review — 2026-09-05 04:58 UTC

## Object and symmetries

Theseus has residual states $x_{\ell,t}\in\mathbb{R}^{1152}$. Attention block $\ell$ has nine heads of width 128. For
head $h$ at the prediction position $t$,

$$
o_{\ell,h,t}=\sum_{s\le t}
\left(\frac{q_{\ell,h,t}^{\mathsf T}k_{\ell,h,s}}{128}\right)
\left(\frac{{q'}_{\ell,h,t}^{\mathsf T}{k'}_{\ell,h,s}}{128}\right)
v_{\ell,h,s}\in\mathbb{R}^{128}.
$$

Its residual write is $W_{O,h}o_{\ell,h,t}\in\mathbb{R}^{1152}$. Each MLP is quadratic in the normalized residual:

$$
M_\ell(x)=W_{D,\ell}\big[(W_{L,\ell}\hat x)\odot(W_{R,\ell}\hat x)\big]+b_{D,\ell},
\qquad W_{L,\ell},W_{R,\ell}\in\mathbb{R}^{4608\times1152}.
$$

The current low-level variable is $o_{11,3,t}\in\mathbb{R}^{128}$ and the proposed high-level variable is grammatical
subject number $z\in\{-1,+1\}$. A head-coordinate change $G\in GL(128)$ can be absorbed by the output projection, so
native coordinates are not semantic. The operational object is the equivalence class of states that produce the same
registered downstream counterfactual responses.

## Mathematical issue exposed by the SELECT result

An interchange score is a property of a representation **and an intervention family**, not of the representation alone.
Let $D$ be a rule that maps each target prompt $i$ to a donor $D(i)$. For fixed site $S$, define the response vector

$$
F_{S,D}(i)=m\!\left(\operatorname{do}left[S_i\leftarrow S_{D(i)}\right]\right)-m(i),
$$

where $m$ is the target-oriented `is`/`are` logit margin. Matched-noun and cross-noun donor rules are two different
probes of the same proposed quotient. If $F_{S,D}$ is stable across both rules after conditioning on subject-number
direction and attractor plurality, lexical identity is less likely to be part of the carried variable. A discrepancy is
positive evidence that the proposed one-variable quotient is too coarse.

This is the concrete mapping to causal abstraction: the high-level intervention replaces $z_i$ by $z_{D(i)}$ and the
low-level intervention replaces the fixed head output. Interchange-intervention work establishes faithfulness only for
the tested input and intervention family; it does not grant uniqueness outside that family
([Geiger et al., 2022](https://proceedings.mlr.press/v162/geiger22a.html)). Distributed alignment search can locate a
rotated subspace, but without multiple meaningful counterfactual families an expressive alignment can still encode the
wrong variable ([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html)).

The multiple-mediators result also matters: an individual patch includes interactions with the unchanged state of other
paths, so the response cannot yet be called the isolated effect of head 11.3
([Vaidyanathan et al., 2026](https://arxiv.org/abs/2606.27510)). The present complete-head transfer is therefore a
screen for a usable carrier. A later common-equation factorial or selective path removal is required to separate its
main effect from interactions and redundancy.

## Executable consequence and opposing predictions

Partition SELECT groups by $(\text{target subject number},\text{attractor plurality})$. Within each stratum, map each
group to the next group in a frozen cyclic order, guaranteeing a different head noun while preserving the two
conditioners. Use the same A1-to-A2 and A2-to-A1 syntax swaps and the same two sites. This costs the same eight forward
calls and 256 example evaluations as the matched-noun run.

- **Syntax-general subject-number carrier:** head 11.3 retains donor-direction fraction at least 0.75 and mean recovery
  at least 0.40 in every direction cell under both donor rules.
- **Lexically conditional carrier or interaction:** cross-noun recovery falls below the frozen bar or becomes strongly
  asymmetric even though native capability remains high.

The comparison should also report the paired change $F_{S,D_{cross}}-F_{S,D_{matched}}$ rather than using two unrelated
means. This is a direct stability test for the causal-response basis. It is not rank reduction, and no subspace fit is
licensed by it.

No tensor-rank or Tucker theorem solves this semantic identification problem: they minimize algebraic factor count or
reconstruction error and remain invariant under gauges that the downstream intervention family must resolve. The
empirical counterfactual route therefore remains the higher-information next step. Once the carrier survives donor and
selectivity tests, exact contractions into later $Q/K/Q'/K'$ and bilinear MLP weights can test which downstream tensor
factors read it.
