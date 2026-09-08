# Shapley / tensor-network attribution — transferred note

Provided by the user (Logan) 2026-09-08, copied verbatim below. Source paper: https://arxiv.org/pdf/2606.01540
(TN-SHAP-G). Posted for Codex's lane as well as Claude's; see AGENT_BOARD.md for the note and for Claude's
assessment of what does and does not transfer to bilin18.

---

**Yes—the Shapley framework generalizes to your tensor transformer, and it can help trace a layer-3 component through redundant downstream routes.** The paper’s computational speedups require additional structure; they do not follow just from your model being a tensor network.

The most useful extension is to **attribute the effect of your layer-3 component**, rather than attribute the whole prediction.

**What transfers from the paper**

TN-SHAP-G learns a tensor-network surrogate of a *coalition game*:

$$
v(S)=\text{model score when components in }S\text{ are enabled}.
$$

Its players can be internal components or edges, with a specified ablation or replacement rule. The original predictor can have any architecture. Exactness applies to the learned surrogate; agreement with the original model depends on surrogate fidelity. [TN-SHAP-G](https://arxiv.org/abs/2606.01540)

For your model, players could be bilinear channels, attention heads, factorized components, or particular component-to-component connections. Choose a scalar score such as a target-minus-distractor logit.

Here are the mathematical connections I would use.

**1. Shapley divides each interaction equally among its participants**

Every binary-mask game has a unique expansion

$$
v(z)=\sum_{T\subseteq N}a_T\prod_{i\in T}z_i,
\qquad z_i\in\{0,1\},
$$

where

$$
a_T=\sum_{U\subseteq T}(-1)^{|T|-|U|}v(U)
$$

is its *Harsanyi dividend*: the contribution requiring the joint presence of that set, after subtracting all lower-order terms.

Then

$$
\boxed{\phi_i=\sum_{T\ni i}\frac{a_T}{|T|}.}
$$

A short proof: the monomial for \(T\) contributes \(a_T\) only when its last member arrives in a random ordering. Each member is last with probability \(1/|T|\). Add contributions across monomials.

This fits your architecture particularly well: polynomial composition produces explicit interaction terms. **Shapley turns those terms into component scores; retaining the terms themselves gives more circuit structure than the scores alone.** A dividend identifies a jointly contributing set, but does not specify the directed computational path within it.

**2. Your polynomial model is not automatically multilinear in component masks**

Suppose a layer-3 residual is decomposed as

$$
h(z)=r+\sum_i z_i c_i,
$$

where \(r\) is held fixed. For one bilinear layer, define

$$
B(u,v)=D(Lu\odot Rv).
$$

Then

$$
B(h,h)=B(r,r)
+\sum_i z_i[B(c_i,r)+B(r,c_i)]
+\sum_{i,j}z_i z_j B(c_i,c_j).
$$

For binary masks, \(z_i^2=z_i\). After a scalar linear readout \(w^\top\), the game therefore has coefficients

$$
a_i=w^\top[B(c_i,r)+B(r,c_i)+B(c_i,c_i)],
$$

$$
a_{ij}=w^\top[B(c_i,c_j)+B(c_j,c_i)]
\quad(i\ne j).
$$

Consequently,

$$
\boxed{\phi_i=a_i+\frac12\sum_{j\ne i}a_{ij}.}
$$

That is an **exact weight-based Shapley calculation for this one-layer game**, given the component activations and fixed context.

For deeper polynomial networks, the general rule is

$$
P(z)=\sum_\alpha c_\alpha z^\alpha
\quad\Longrightarrow\quad
\phi_i=
\sum_{\alpha:\alpha_i>0}
\frac{c_\alpha}{|\operatorname{supp}\alpha|}.
$$

You divide by the number of **distinct players**, not total polynomial degree.

This matters if you try to integrate gradients through continuously scaled activations. For example,

$$
P(z)=z_1^2z_2
$$

has binary-mask Shapley values \((1/2,1/2)\), whereas integrated gradients along \(z=t\mathbf1\) gives \((2/3,1/3)\). The paper integrates derivatives of the **multilinear extension of the binary game**, which has already replaced repeated mask powers by single occurrences.

Your attached post also mentions RMSNorm and Q/K normalization. With their denominators recomputed under intervention, the ordinary forward map is not polynomial. The binary-game identities still hold exactly; the direct polynomial expansion above requires a polynomial subnetwork or an explicitly fixed-normalization intervention.

**3. To trace your layer-3 component, make its effect the game**

Let \(a\) be the component you found, and \(D\) a collection of downstream components. Define

$$
F(b,S)=\text{final score with }a\text{ set to }b
\text{ and downstream components }S\subseteq D\text{ enabled},
$$

where \(b=1\) retains \(a\), and \(b=0\) replaces it with your chosen baseline. Recompute enabled downstream components after each intervention.

Now define

$$
E_a(S)=F(1,S)-F(0,S).
$$

This asks: **how much does \(a\) matter when these downstream routes are available?**

Compute downstream Shapley values

$$
\psi_{a\to j}
=
\mathbb E_{\pi}\left[
E_a(P^\pi_j\cup\{j\})-E_a(P^\pi_j)
\right],
$$

where \(P^\pi_j\) contains the downstream components preceding \(j\) in a random ordering.

Two exact identities make this useful:

$$
\boxed{
\psi_{a\to j}
=\phi_j(F(1,\cdot))-\phi_j(F(0,\cdot))
}
$$

by Shapley linearity, and

$$
\boxed{
E_a(D)=E_a(\varnothing)+\sum_{j\in D}\psi_{a\to j}
}
$$

by efficiency.

So you get a complete allocation of **how the selected downstream components change the source component’s effect**. The residual \(E_a(\varnothing)\) includes any effect through the unmasked background, such as a direct residual route.

This is my proposed application of the game-theoretic identities to your circuit question. It measures downstream dependence; a large value alone does not establish a direct edge from \(a\) to \(j\). For individual connections, use edge-specific interventions.

**4. Redundancy is where this improves on single ablations**

Consider two downstream routes carrying \(a\)’s effect:

| Effect game \(E_a\)             | Either route suffices | Both routes required |
| ------------------------------- | --------------------: | -------------------: |
| Neither enabled                 |                     0 |                    0 |
| Only \(j\)                      |                     1 |                    0 |
| Only \(k\)                      |                     1 |                    0 |
| Both                            |                     1 |                    1 |
| Single deletion from full model |                     0 |                    1 |
| Shapley credit per route        |               \(1/2\) |              \(1/2\) |
| Pair interaction                |                \(-1\) |               \(+1\) |

**Shapley finds credit for redundant routes even when deleting either one does nothing. But first-order Shapley alone cannot distinguish redundancy from complementarity:** both examples assign half to each route.

For that, inspect

$$
\Delta_{jk}E_a(S)
=
E_a(S\cup\{j,k\})-E_a(S\cup\{j\})
-E_a(S\cup\{k\})+E_a(S).
$$

The Shapley interaction index averages this over appropriately weighted contexts. Negative means diminishing joint contribution; positive means complementarity. In signed neural computations, negative interactions can also reflect cancellation or suppression, so inspect the four intervention outcomes.

In dividend coordinates, the standard interaction index is

$$
\boxed{
I_U=\sum_{T\supseteq U}\frac{a_T}{|T|-|U|+1}.
}
$$

Thus pair interactions also aggregate higher-order effects. They are not isolated pairwise circuit coefficients. [Grabisch–Roubens interaction index](https://ideas.repec.org/a/spr/jogath/v28y1999i4p547-565.html)

**5. What the tensor-network representation actually buys you**

Two separate obstacles remain:

* **Representation:** the coalition tensor must admit small bond dimensions. The paper’s cut-rank bound is

  $$
  \operatorname{rank}_{A|B}(T)\le\prod_{e\in\delta(A,B)}\chi_e.
  $$

  Proof: fix the crossing bond indices; each assignment contributes a rank-one outer product between the two sides.
* **Contraction:** even a compact tensor network can be expensive to contract. Dense interactions and repeated reuse of upstream components can make the required contractions large.

Therefore, a fast ordinary transformer forward pass does not imply fast exact averaging over component coalitions.

The paper also proves that uniform surrogate error \(\|v-\hat v\|_\infty\le\epsilon\) implies

$$
|\phi_i-\hat\phi_i|\le2\epsilon,
\qquad
|I_U-\hat I_U|\le2^{|U|}\epsilon.
$$

Each interaction difference contains \(2^{|U|}\) signed evaluations, which proves the bound by the triangle inequality. High held-out \(R^2\) alone does not establish that uniform guarantee.

**My first experiment would be:** choose one layer-3 component and 10–12 downstream candidates, enumerate \(E_a(S)\) exactly, and calculate its Shapley values and pair interactions. Twelve candidates require \(2\times2^{12}=8192\) intervened evaluations per input, batchable. That directly tests whether you have redundant routes, serial dependencies, or suppressors—and provides ground truth before fitting a TN surrogate.

(from the paper: https://arxiv.org/pdf/2606.01540 )
