# Three-hour mathematical circuit review — 2026-09-05 07:58 UTC

## Current tensor-network object

Theseus has residual states $x_{\ell,t}\in\mathbb{R}^{1152}$. Each attention block has nine heads of width 128. For the current bracket circuit, head 8 in layer 13 contributes the exact residual write

$$
t_{g,c}=p_{g,c}(q,o)\,W_{O,13,8}v_{13,8}(o)\in\mathbb{R}^{1152},
$$

where $g$ is a bracket triplet group, $c$ is one of its three closer-token constructions, $p(q,o)$ is the product of the two bilinear attention scores from the final query position $q$ to the opener position $o$, and $v(o)$ is that opener's value vector. This term is an exact part of the model's computation, not a fitted activation direction.

For each group, the three writes have the exact decomposition

$$
t_{g,c}=\mu_g+\delta_{g,c},
\qquad
\mu_g=\frac{1}{3}\sum_{c=1}^{3}t_{g,c},
\qquad
\sum_{c=1}^{3}\delta_{g,c}=0.
$$

$\mu_g$ is the part shared by the three possible closer constructions and $\delta_{g,c}$ is the construction-specific difference. The zero-sum condition fixes the otherwise arbitrary shift between these two terms. Removing either term changes the task output, but restoring MLP15, three response-localized attention heads, their joint write, or any single complete downstream module write recovers essentially none of that causal effect. The remaining question is therefore about **paths**, not about finding another high-response component.

## Exact path factorial

Let $A\in\{N,R\}$ denote whether the L13H8 factor is native or removed, and let $B\in\{N,R\}$ denote whether the complete bank of later module writes is the bank cached under native or removed-opener execution. The bank contains

$$
\{\operatorname{MLP}_{13},\operatorname{Attn}_{14},\operatorname{MLP}_{14},
\operatorname{Attn}_{15},\operatorname{MLP}_{15},\operatorname{Attn}_{16},
\operatorname{MLP}_{16},\operatorname{Attn}_{17},\operatorname{MLP}_{17}\}.
$$

Write $Y_{AB}\in\mathbb{R}^{1152}$ for the final centered residual response under one of the four combinations. Then

$$
D_{\mathrm{total}}=Y_{NN}-Y_{RR},
$$

$$
D_{\mathrm{residual}}=Y_{NN}-Y_{RN},
\qquad
D_{\mathrm{writes}}=Y_{NN}-Y_{NR},
$$

and the exact second-order interaction is

$$
I=Y_{NN}-Y_{RN}-Y_{NR}+Y_{RR}.
$$

These obey

$$
D_{\mathrm{total}}=D_{\mathrm{residual}}+D_{\mathrm{writes}}-I.
$$

This is the Möbius decomposition of a function on the two-bit Boolean lattice. In ordinary terms, it asks how much of the opener effect remains when later module responses are held native, how much is produced by changing those later responses while the opener stays native, and how much exists only because both changes occur together. It is exact for the four counterfactual model executions and requires no orthogonal basis, rank cutoff, or activation reconstruction.

The same factorial is applied to cross-entropy loss. If $L_{AB}$ is loss, then

$$
\Delta L_{\mathrm{total}}=L_{RR}-L_{NN},
$$

$$
\Delta L_{\mathrm{residual}}=L_{RN}-L_{NN},
\qquad
\Delta L_{\mathrm{writes}}=L_{NR}-L_{NN},
$$

$$
\Delta L_{\mathrm{interaction}}=L_{RR}-L_{RN}-L_{NR}+L_{NN},
$$

with

$$
\Delta L_{\mathrm{total}}=
\Delta L_{\mathrm{residual}}+
\Delta L_{\mathrm{writes}}+
\Delta L_{\mathrm{interaction}}.
$$

## Relation to existing mathematics

This is directly motivated by the multiple-mediators result: a usual natural-indirect-effect patch of one module contains interactions with every mediator left in its natural state. Vaidyanathan et al. show that this interaction can be decomposed into pairwise and higher-order group terms and vanishes only under a suitable local-affinity condition ([Vaidyanathan et al., 2026](https://arxiv.org/abs/2606.27510)). Our grouped two-factor calculation is the smallest exact decomposition that tests whether the large set of individually null downstream responses matters collectively or through interaction.

Interchange interventions test whether a proposed low-level state realizes a high-level causal variable, but the claim is only as broad as the tested intervention family ([Geiger et al., 2022](https://proceedings.mlr.press/v162/geiger22a.html)). A rotated-subspace search can help locate such a variable, but it would be premature here: the current unknown is whether the opener factor travels through the additive residual route or is rewritten collectively by downstream modules. Distributed alignment search does not answer that path question by itself ([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html)).

No tensor-rank, Tucker, or activation-reconstruction theorem solves the semantic identification problem here. Those objectives can simplify a tensor algebraically while preserving the gauge ambiguity and without preserving CE loss, selective causal removal, or counterfactual interchange. The exact factorial is useful because it identifies causal path terms first. If a path is material, its contractions into later $Q/K/Q'/K'$ and quadratic MLP weights can then be expanded exactly and searched for a reusable shared computation.

## Executable consequence

The frozen next experiment is the eight-forward $2\times2$ factorial above, run separately for $\mu$ and $\delta$ on the existing 24-row authority. The preregistered opposing predictions are:

- **Residual-route account:** for both factors and both target constructions, the projection of $D_{\mathrm{residual}}$ onto $D_{\mathrm{total}}$ is at least 0.75.
- **Collective downstream account:** the projection of $D_{\mathrm{writes}}$, or the absolute projection of $I$, reaches 0.25 for at least one factor and target construction.

Both can partly hold because the decomposition includes interaction. The result is an exact grouped path statement, not evidence that every write in the bank matters or that the bank is a semantic basis. If the bank is material, the next experiment splits it into earlier and later banks with another small factorial. If it is not, the next weight-level computation follows the direct residual contraction of $\mu$ and $\delta$ into the first downstream readers.
