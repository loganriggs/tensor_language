# Three-hour mathematical review — 2026-09-04 15:30 UTC

## 1. The current object

Theseus has 18 residual blocks. At sequence position $t$ and block boundary $\ell$, the residual state is

$$
x_{\ell,t}\in\mathbb{R}^{1152}.
$$

Each attention layer has 9 native heads of width 128. For one head $h$, the unprojected value gathered at the final prediction position is

$$
o_{\ell,h,t}=\sum_{s\le t} a_{\ell,h,t,s}v_{\ell,h,s}
\in\mathbb{R}^{128}.
$$

Unlike ordinary softmax attention, Theseus uses the product of two query–key dot products,

$$
a_{\ell,h,t,s}
=
\left(\frac{q_{\ell,h,t}^{\mathsf T}k_{​\ell,h,s}}{128}\right)
\left(\frac{q^{(2)\mathsf T}_{\ell,h,t}k^{(2)}_{​\ell,h,s}}{128}\right),
$$

with a causal mask. The four query/key vectors and the value vector are linear projections of RMS-normalized residual states, followed by head-wise normalization and rotary position maps where applicable. Conditional on those normalization scalars, the attention weight is degree four in its two residual inputs and the value-weighted head output is degree five. Because RMS normalization divides by an input-dependent norm, the complete network is not literally a polynomial in the raw residual coordinates; it is a tensor contraction with simple scalar normalization nodes.

The nine $o_{\ell,h,t}$ vectors are concatenated and multiplied by the attention output projection to write a vector in $\mathbb{R}^{1152}$. Each MLP has 4,608 bilinear hidden products,

$$
M_\ell(x)=W_{D,\ell}
\left[(W_{L,\ell}\hat{x})\odot(W_{R,\ell}\hat{x})\right]+b_{D,\ell},
\qquad \hat{x}=\operatorname{RMSNorm}(x),
$$

so it is exactly quadratic in the normalized residual state.

The present candidate is the final-token output of head 11.3,

$$
o_i=o_{11,3,t_i}\in\mathbb{R}^{128},
$$

on frozen subject–verb agreement prompts. The proposed high-level variable is binary complete-subject number,

$$
z_i\in\{\text{singular},\text{plural}\},
$$

and the measured output is the next-token logit margin

$$
m_i=\operatorname{logit}_i(\text{correct ` is/are`})
-\operatorname{logit}_i(\text{other ` is/are`}).
$$

The allowed inputs for the next decision are the frozen validation PP and relative-clause pairs. No prompt, donor, threshold, or site will be selected from validation outcomes.

## 2. What equivalence and gauge mean here

The 128 native coordinates are not presumed semantic. A change of basis $G\in GL(128)$ inside the head can be absorbed by an inverse change in the corresponding slice of the output projection. Query and key factors also have dot-product-preserving gauge transformations. Therefore “dimension 37” or even “head 11.3” is an implementation coordinate, not the final explanatory unit.

For a candidate $k$-dimensional linear causal subspace with orthonormal basis $U\in\mathbb{R}^{128\times k}$, only the projector

$$
P=UU^{\mathsf T}
$$

is identifiable; $U$ and $UG$ describe the same subspace for every orthogonal $G\in O(k)$. The operational object is the equivalence class of head states that downstream computation cannot distinguish on the registered interventions. This matches the user’s proposed interaction-determined basis: two states count as the same variable when swapping one for the other has the same downstream consequences across the circuit battery.

## 3. Literature mapping, assumptions, and limits

### Causal abstraction and interchange intervention

[Geiger et al. (ICML 2022)](https://proceedings.mlr.press/v162/geiger22a.html) define interchange intervention by replacing a neural representation from a source input into a base input and requiring the resulting behavior to match the counterfactual of a high-level causal model. The exact mapping here is:

- low-level variable: $o_{11,3,t}\in\mathbb{R}^{128}$ or a projector $P o_{11,3,t}$;
- high-level variable: complete-subject number $z$;
- high-level intervention: replace $z_{\mathrm{base}}$ by $z_{\mathrm{source}}$;
- low-level intervention: replace the complete head output, or later only $P(o_{\mathrm{source}}-o_{\mathrm{base}})$;
- agreement criterion: the patched `is`/`are` margin moves toward the source counterfactual on held-out cross-syntax pairs while same-variable controls remain stable.

Zero interchange loss would justify a causal abstraction under the stated alignment and intervention family. It would not prove that the alignment is unique or globally meaningful. [The general causal-abstraction framework](https://arxiv.org/abs/2301.04709) explicitly allows distributed and graded alignments, but an arbitrarily expressive alignment can make the criterion vacuous. We therefore restrict the alignment to an orthogonal linear projector, charge its dimension and parameters, freeze discovery/validation, and demand selective controls.

### Multiple mediators and interactions

[Vaidyanathan et al. (2026)](https://arxiv.org/abs/2606.27510) show that ordinary activation-patching effects include interactions with other mediators and can be unstable when components compensate for one another. This maps directly onto the apparently conflicting Task 14 evidence: mean-replacing heads 11.3 and 15.5 barely changed natural accuracy, while donor-swapping head 11.3 strongly changed the agreement margin. Removal estimates necessity in the native state of every other path; interchange asks whether a source state is causally usable in the recipient. Redundancy makes the first small without making the second zero.

Their combinatorial interaction decomposition applies only when every factorial cell uses a common intervention equation. Our current `resid:11`, `attn:11`, `mlp:11`, and `resid:12` measurements have different recomputation semantics and are not valid factorial corners. A valid $2^3$ replay would cache recipient/source $R,A,M$ components and evaluate every chosen source/recipient combination through the same downstream function. It is useful after identification, but it does not resolve the immediate syntax-generality question as cheaply as literal cross-syntax interchange.

### What no theorem currently gives us

Tensor rank, Tucker decomposition, and weight eigendecomposition do not identify the subject-number variable because their objectives are reconstruction or algebraic factor count, not equivalence under the registered downstream interventions. Minimal weighted-automaton realization would require a finite-state input/output Hankel object closed under prefixes and suffixes; the present transformer state and counterfactual family do not satisfy that setup. No reviewed theorem therefore yields a unique semantic factorization of the full Theseus contraction from weights alone.

The closest principled object is a **restricted causal quotient**: quotient the 128-dimensional head state by indistinguishability under a frozen family of downstream interventions, then seek the smallest linear representative that realizes the binary high-level variable.

## 4. Executable consequence

First run the decisive full-output test. For every frozen validation record with a PP recipient and relative-clause opposite-number source, and vice versa, replace either the full attention-11 output or the complete head-11.3 output at the final position. This tests whether the same low-level state crosses syntax, rather than merely whether one site works separately in two constructions.

If and only if full-output cross-syntax transfer passes, fit projectors on discovery rows. For source $s$ and base $b$, patch

$$
o_b' = o_b + P(o_s-o_b),
\qquad P=UU^{\mathsf T}.
$$

For each $k=1,2,\ldots$, optimize $U$ only on discovery interchange loss plus fixed answer-preserving controls. Select the smallest $k$ that reaches a preregistered fraction of the complete-head effect; then evaluate that frozen $P$ once on validation cross-syntax pairs and unrelated endpoint-matched behavior. Opposing predictions are:

- **one shared variable:** a small common $P$ transfers both PP→relative and relative→PP, both number directions, and leaves controls small;
- **two construction-specific handles or a generic endpoint signal:** transfer is asymmetric/weak across syntax, or an unrelated `is`/`are` task moves similarly.

This is DAS-like subspace search, but the scientific claim is about the held-out interchange behavior of $P$, not its low dimension.

## 5. Exact translation to weights

Let $W_{O,3}\in\mathbb{R}^{1152\times128}$ be head 11.3’s slice of the attention output projection. A one-dimensional causal head coordinate $u\in\mathbb{R}^{128}$ writes the exact residual direction

$$
r=W_{O,3}u\in\mathbb{R}^{1152}.
$$

For any later bilinear MLP and chosen output direction $v$, define

$$
Q_v=\frac12\left[
W_L^{\mathsf T}\operatorname{diag}(W_D^{\mathsf T}v)W_R
+W_R^{\mathsf T}\operatorname{diag}(W_D^{\mathsf T}v)W_L
\right].
$$

Writing a normalized downstream state as $x=\alpha r+c$ gives the exact decomposition

$$
v^{\mathsf T}M(x)
=\alpha^2 r^{\mathsf T}Q_vr
+2\alpha r^{\mathsf T}Q_vc
+c^{\mathsf T}Q_vc.
$$

The first term is self-interaction of the head-written direction; the second is its interaction with the rest of the residual state; the third is computation not using that direction. This contraction provides a weight-level hypothesis about which later MLP output directions use the causal state. An analogous contraction into later $Q,K,Q2,K2$ projections tests which attention pattern factors read it.

These weight contractions are exact algebra, but they become interpreted circuit edges only if their predicted interventions generalize and selective removal of the contracted path changes agreement without damaging unrelated tasks.

## 6. Complexity and price

- The immediate full-output validation has 64 frozen cross-syntax records, two sites, recipient/source native calls, and patched calls. At batch size 32 this is a small number of forward passes and should take seconds on the managed GPU.
- A projector search costs repeated forward/backward passes through the suffix after block 11. Its literal parameter price is $128k$ before quotienting the $O(k)$ basis gauge; the subspace has $128k-k(k+1)/2$ degrees of freedom.
- Exact head-to-residual translation is one matrix–vector multiply. Each later MLP contraction is dominated by the 4,608 hidden factors, not a full dense $1152^3$ tensor.
- Storage, compute, and dimension are prices. Success remains cross-syntax/OOD prediction, extraction, composition, and selective manipulation.

## 7. Decision

The empirical causal route remains higher-information than an unsupervised tensor decomposition because the live uncertainty is semantic: one syntax-general number variable versus construction-specific or output-token state. The mathematical review changes the route in two ways:

1. report only the projector/equivalence class as the eventual feature, never a native head coordinate as intrinsically semantic;
2. require cross-syntax full-state interchange before projector fitting, and require exact weight contraction after it.

The next executable action is already underway as a frozen cross-syntax candidate. The cached-component factorial remains secondary and must use a common replay equation if run.
