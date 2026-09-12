# The strongest current decomposition, derived from the model

**For Logan — 12 September 2026. Evidence includes results completed at 19:55 UTC.**

This explanation stands alone. It follows one regional-spelling computation from a shared downstream feature into two earlier attention heads, derives their interaction through a bilinear layer, and explains the more selective part found inside their joint key operation.

**What we have:** an executable conditional component whose removal and donor swaps change regional-spelling contrasts on fresh prompts, with passing tested newline controls. We also have an exact conditional equation for the interaction of its broader producer components through an intervening MLP.

**What remains:** the native model still generates important input states and executes the remaining layers. The component responds to regional cues in both writer and reader positions; it is not a clean writer-role circuit. It is our strongest current combination of explicit structure and selective intervention evidence, not a completed standalone language circuit or a global unsupervised decomposition.

## 1. Start with what the model actually computes

A token position carries a vector of 1,152 numbers called the **residual state**. An attention head reads earlier positions and writes another 1,152-vector into that state. A bilinear MLP reads two sets of linear features, multiplies them in pairs, and writes the products back.

We number layers from zero. “Head8.2” means head index 2 in block 8; the final block is 17.

| Symbol | Shape | Meaning |
|---|---|---|
| $x_t$ | $1152$ | State at token position $t$; its precise layer is stated when used. |
| $L,R$ | $4608\times1152$ | The two readers in a bilinear MLP. |
| $D$ | $1152\times4608$ | MLP output writers. |
| $Q_1,Q_2,K_1,K_2$ | each $128\times1152$ | One attention head's two query/key systems. |
| $V,V_0$ | each $128\times1152$ | Current-state and first-state value maps. |
| $O$ | $1152\times128$ | One head's output map. |
| $U$ | $50304\times1152$ | Unembedding: one output reader for each vocabulary token. |

Vectors in equations are columns. The code stores token batches as rows, so code matrix products may be transposed relative to the equations.

Ignoring its separately retained bias, the MLP is

$$
B(x)=D[(Lx)\odot(Rx)].
$$

Here $\odot$ multiplies corresponding entries. For example, hidden entry $j$ is the scalar $(L_jx)(R_jx)$, and column $D_{:,j}$ tells us where that product is written.

The MLP receives an RMS-normalized state. Define

$$
\rho(x)^2=\frac{x^Tx}{1152}+\epsilon,
\qquad \operatorname{RMS}(x)=\frac{x}{\rho(x)}.
$$

Because the numerator is quadratic,

$$
B(\operatorname{RMS}(x))=\frac{B(x)}{\rho(x)^2}.
$$

That denominator matters whenever we change $x$. Final output scores also include final RMS normalization and $30\tanh(\cdot/30)$. The complete model is not a fixed polynomial with constant coefficients in its raw token inputs.

## 2. What “folding a reader backward” means

Suppose we only care about one scalar reading of an MLP output, given by $c^T B(x)$ with $c\in\mathbb R^{1152}$. Substitute the MLP equation:

$$
\begin{aligned}
c^TB(x)
&=(D^Tc)^T[(Lx)\odot(Rx)]\\
&=\sum_{j=1}^{4608}(D^Tc)_j(L_jx)(R_jx)\\
&=x^TS_cx,
\end{aligned}
$$

where

$$
S_c=\frac12\left[
L^T\operatorname{diag}(D^Tc)R+
R^T\operatorname{diag}(D^Tc)L
\right]\in\mathbb R^{1152\times1152}.
$$

**Nothing has been fitted here.** We have rewritten the same scalar computation in terms of its inputs. Symmetrizing is exact because $x^TAx=x^T(A+A^T)x/2$.

For an output token, choose $c=U_{v,:}^T$. For a token contrast, choose the difference of two unembedding rows. For the full vocabulary, keep all rows: this yields a tensor of shape $50304\times1152\times1152$. We normally contract its factors rather than allocate that tensor.

This is the starting point you described: each token slice is a polynomial interaction matrix, and a decomposition looks for readings or intermediate products shared across slices. Folding alone does not make those shared factors unique or semantically meaningful.

## 3. The original shared attention component

The regional example was discovered in an attention numerator. For one query/source pair, an attention head combines **both** query–key matches with a value write:

$$
F_h(q,s)=g_h(q,s)
(q^TA_hs)(q^TB_hs)M_hs.
$$

Here $q\in\mathbb R^{1152}$ is the query state and $s\in\mathbb R^{2304}$ concatenates current and first-value source states. The matrices $A_h,B_h$ have shape $1152\times2304$; $M_h$ maps the source into whichever output readings are under study. The scalar $g_h$ keeps the actual Q/K normalization factors explicit. Relative-position rotations are folded into $A_h,B_h$ for the specified position pair.

The displayed numerator has degree two in $q$ and degree three in $s$. We searched for source cubics reused by query-dependent consumers:

$$
h_r(s)=(a_r^Ts)(b_r^Ts)(c_r^Ts),
\qquad
\widehat F_h(q,s)=\sum_r \beta_{hr}(q)h_r(s).
$$

For fixed source readers, the output/query coefficients are a linear least-squares problem and were solved out. The remaining reader search is nonlinear. The broad fit captured little of the complete coefficient function; its small useful component should not be mistaken for a successful full-attention compression.

Two fitted products had nearly coincident readers and large cancelling coefficients. Average/difference coordinates exposed an approximate shared form:

$$
f_j(s)=r_j^Ts,\qquad
p(s)=f_0(s)f_1(s),
$$

$$
\widehat F(q,s)=
\beta_0(q)\,p(s)f_2(s)+
\beta_1(q)\,p(s)f_3(s).
$$

The intermediate $p$ is computed once and reused by two children. The exact average/difference rewrite preserved the original fitted function; retaining the compact shared-parent form was a separately measured approximation. The complete component was then tested with native normalizers and background retained.

This is a small arithmetic DAG: four scalar reads, one shared multiplication, two child multiplications, then their query-dependent writes. It is richer than claiming that two tensor factors have similar vectors. Counterfactual tests found that the child readings carry most of the regional cue; the shared parent is not itself an identified “Britishness” variable.

A dominant consumer is head17.2. Its local package uses explicit readers, query/key maps, writers, and a complete first-token lookup. It still needs the native query/current state. [Original derivation](../../SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md) · [Consumer package](../../extracted_circuits/regional_shared_head2_token_mixed_v1/README.md).

## 4. Fold those downstream readings into earlier heads

The discovered consumer needs four current-state readings. Stack them into

$$
C\in\mathbb R^{4\times1152},\qquad f=Cx\in\mathbb R^4.
$$

An upstream head writes $Ov$ into the residual stream. Its contribution to these readings is $COv$: a $4\times128$ map instead of a full residual writer.

The head's value combines its current source state $x_s$ and a first-state source $e_s$ using a learned scalar $\mu$:

$$
v_s=(1-\mu)Vx_s+\mu V_0e_s.
$$

The learned coefficient need not describe a convex mixture. If $\lambda$ is the actual residual propagation multiplier along the selected direct path, define

$$
M=\lambda CO
\begin{bmatrix}(1-\mu)V&\mu V_0\end{bmatrix}
\in\mathbb R^{4\times2304}.
$$

Then

$$
\lambda COv_s=M
\begin{bmatrix}x_s\\e_s\end{bmatrix}.
$$

Again, this folding is exact for the specified direct value path. It does not include every indirect change to subsequent attention or MLPs; those are tested separately.

To seek a simple value component we used a weight-derived downstream sensitivity metric $H\in\mathbb R^{4\times4}$ and solved

$$
\min_{\operatorname{rank}(\widehat M)\le1}
\left\|H^{1/2}(M-\widehat M)\right\|_F^2.
$$

This is a weighted matrix rank-one problem, solved spectrally. It is not CP optimization with a random local optimum. The metric uses the downstream component's Jacobian and synthetic, specified input moments, not a fit to the language test prompts.

A bank spanning 27 heads in layers8,9,13 produced two behaviorally useful components: heads8.2 and9.8. Their downstream writer cosine in this metric is about **0.99949**. Their source readers differ substantially, especially in the first-state branch. Their full physical writers have cosine about **0.899**, so we retain the actual separate writers rather than pretending they are identical everywhere.

The finding is: **different upstream reads can feed almost the same downstream variable.** The selection of the regional consumer and relevant producer layers was behavior-informed; the folded factors themselves were weight-derived. This is component-conditioned discovery, not a claim of wholly unsupervised model-wide circuit selection.

## 5. Turn each producer into “one scalar times one writer”

Let $d_h\in\mathbb R^{1152}$ be the retained physical writer. Its selected contribution at position $t$ has the form

$$
\delta x_h(t)=a_h(t)d_h.
$$

The scalar still contains a rich attention computation. For $j=1,2$, define

$$
\widehat q_j(t)=R_t\operatorname{RMS}_{128}(Q_jx_t),
\quad
\widehat k_j(s)=R_s\frac{K_jx_s}{\kappa_j(x_s)},
$$

$$
\kappa_j(x)^2=\frac{\|K_jx\|^2}{128}+\epsilon,
\qquad
b_j(t,s)=\frac{\widehat q_j(t)^T\widehat k_j(s)}{128}.
$$

$R_t$ is the actual position rotation, shape $128\times128$. The two scalar matches multiply:

$$
\gamma_h(t,s)=b_1(t,s)b_2(t,s),
$$

$$
a_h(t)=\sum_{s\le t}\gamma_h(t,s)
\left[r_h^Tx_s+\tau_h(\operatorname{token}_s)\right].
$$

The current-value reader $r_h$ has 1,152 entries. The first-value table $\tau_h$ has 50,304 entries and is constructed from the model's actual initial-state computation, not fitted token scores. Causal masking gives $s\le t$. This architecture has **no attention softmax**.

“One scalar output” therefore does not mean “one simple token detector”: both query systems, both key systems, position and normalization remain. The selected writer can be small as an output interface while its generator is complex.

## 6. Why the two producers cannot be evaluated independently

Head8.2 is earlier than head9.8. Removing an amount $ad$ from head8 changes the intervening MLP8 and therefore the state seen by head9. Keeping head9's original scalar after that removal creates a different, deliberately frozen intervention.

We derived this change exactly. Let $z\in\mathbb R^{1152}$ be the state entering MLP8 before its normalization, and let removal change it to

$$
z'=z-ad.
$$

Write the bias-free MLP numerator as $B(z)=D[(Lz)\odot(Rz)]$. Expand both readers:

$$
L(z-ad)=Lz-aLd,\qquad R(z-ad)=Rz-aRd.
$$

Multiplying gives

$$
\begin{aligned}
B(z-ad)=B(z)
&-aD[(Lz)\odot(Rd)+(Ld)\odot(Rz)]\\
&+a^2D[(Ld)\odot(Rd)].
\end{aligned}
$$

There are three pieces: the old quadratic output, a **mixed state/direction term**, and a direction-squared term. The mixed term is precisely what a direct-residual-only picture misses.

Define

$$
J_d=D[\operatorname{diag}(Rd)L+
\operatorname{diag}(Ld)R]\in\mathbb R^{1152\times1152},
$$

$$
h_d=D[(Ld)\odot(Rd)]\in\mathbb R^{1152}.
$$

Then $B(z-ad)=B(z)-aJ_dz+a^2h_d$, and $h_d=J_dd/2$. Let $\rho=\rho(z)$ and $\rho'=\rho(z-ad)$. Including the residual connection, the **exact** change after MLP8 is

$$
\boxed{
\Delta x=-ad+
\left(\frac1{\rho'^2}-\frac1{\rho^2}\right)B(z)
-\frac a{\rho'^2}J_dz+
\frac{a^2}{\rho'^2}h_d.
}
$$

The MLP output bias cancels between the two executions. If $u=B(z)/\rho^2$ is supplied by the native execution, the same equation becomes

$$
\boxed{
\Delta x=-ad+
\left(\frac{\rho^2}{\rho'^2}-1\right)u
-\frac a{\rho'^2}J_d\left(z-\frac a2d\right).
}
$$

The three visible contributions are direct removal, normalization change, and the folded bilinear interaction. Multiply this residual change by block9's actual residual coefficient, add it to the original block9 input, and recompute block9 normalization and attention. This keeps the serial dependency.

The stored $J_d$ and $d$ require 1,328,256 scalars, versus 15,926,400 in the three unfused MLP matrices. This is a smaller **conditional directional bridge**; $z,u,a$ and the background still have to be generated. It is not a twelvefold reduction of the whole model.

On fresh regional prompts, the exact bridge reproduced the signed serial effect to roughly $3.5\times10^{-6}$ relative error. The tested direct-plus-mixed approximation had about 0.87–0.93% error; direct-only failed substantially on earlier panels. Donor tests also supported the bridge, with about 1.43–2.33% error for the approximation. These bridge measurements were made for the earlier full scalar components, not separately re-established for every later key-subspace variant. The identity itself applies to any scalar amplitude along the same $d$.

The signed reference is the difference between dynamic joint removal and a hybrid intervention freezing head9's pristine scalar. It measures their interaction under that specified comparison; it is not the whole spelling effect. [Bridge and native evidence](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md).

## 7. Find a more selective piece inside the joint key product

Removing the broader two-producer component weakened regional spelling but damaged an unrelated newline endpoint too much. Splitting the value sources alone did not fix that. We therefore partitioned what **both keys jointly read**.

For each head, concatenate the key row spaces:

$$
\operatorname{span}(K_1^T,K_2^T)\subseteq\mathbb R^{1152}.
$$

It has 256 directions in this construction. A weight-only coefficient-influence operator, averaged over 32 specified relative positions, orders them spectrally. Four bands of 64 span the complete space. No directions are initially discarded. This operator concerns the numerator's coefficient geometry, not an empirical language activation distribution.

Let $B\in\mathbb R^{1152\times64}$ contain the leading orthonormal band, and

$$
P=BB^T\in\mathbb R^{1152\times1152},
\qquad x=Px+(I-P)x.
$$

Split each key match linearly into the two source pieces, keeping its **original full-state normalizer**:

$$
b_j=b_{j,\mathrm{in}}+b_{j,\mathrm{out}}.
$$

Multiplication gives the exact four terms

$$
\gamma=
\underbrace{b_{1,\mathrm{in}}b_{2,\mathrm{in}}}_{\text{inside}}
+\underbrace{b_{1,\mathrm{in}}b_{2,\mathrm{out}}+
 b_{1,\mathrm{out}}b_{2,\mathrm{in}}}_{\text{mixed}}
+\underbrace{b_{1,\mathrm{out}}b_{2,\mathrm{out}}}_{\text{outside}}.
$$

**Every term uses QK1 and QK2 together.** We are not assigning one task to QK1 and another to QK2. The split concerns whether the two reads use the same source subspace or different subspaces.

The selected candidate keeps inside and outside, omitting mixed:

$$
\gamma_{\mathrm{even}}=
 b_{1,\mathrm{in}}b_{2,\mathrm{in}}+
 b_{1,\mathrm{out}}b_{2,\mathrm{out}}.
$$

Why call it even? Define the reflection $T=I-2P$. It negates the inside part and leaves the outside unchanged. Holding query, value and original key denominators fixed,

$$
\boxed{
\gamma_{\mathrm{even}}(q,x)
=\frac12[\gamma(q,x)+\gamma(q,Tx)].
}
$$

The mixed terms change sign and cancel; the two same-subspace products survive. This is an exact algebraic projection of the specified numerator. It is **not** a discovered symmetry of the complete normalized head. The measured normalizer audit fails that stronger interpretation; recomputing the key norms after reflection changes the function.

Finally, the selected pair uses only the first-value sector of head8.2 and current-value sector of head9.8:

$$
a_8^{\mathrm{sel}}(t)=\sum_{s\le t}
\gamma_{8,\mathrm{even}}(t,s)\tau_8(\operatorname{token}_s),
$$

$$
a_9^{\mathrm{sel}}(t)=\sum_{s\le t}
\gamma_{9,\mathrm{even}}(t,s)r_9^Tx_s.
$$

The physical writes are $a_8^{\mathrm{sel}}d_8$ and $a_9^{\mathrm{sel}}d_9$. Joint removal subtracts the first, executes the intervening model on its changed state, then computes and subtracts the second. It does not reuse a pristine head9 state.

The basis was chosen from weights. Which grades and value sectors to retain was selected by a registered behavioral screen, then frozen for confirmation. The selected program therefore has weights-first structure discovery and behavior-guided selection. The 64-direction boundary is a probe choice, not a uniquely identified semantic boundary.

## 8. What passed, what failed, and how much is extracted

The first fine-path candidate narrowly missed the regional criterion. A later complete inside/mixed/outside screen selected the above same-subspace pair without lowering the thresholds. On fresh confirmation:

| Family | Fraction of native cue contrast removed | Donor transfer relative to native cue contrast |
|---|---:|---:|
| Diary wording | 53.80% | 56.24% |
| School-essay wording | 52.51% | 54.82% |
| Opposing writer/reader cities | 57.30% | 60.26% |

All 36 paired removals and 72 directed donor swaps had the expected sign. Fresh FineWeb newline controls had largest absolute loss changes of 0.05036 and 0.03323 nats in their two halves. The original retained control panel also passed its maximum 0.1 bar, narrowly at 0.09495.

The full value-sector component also passes the new newline panel, so that panel alone does not prove that the finer split is better. The older retained outlier distinguishes them. No unfavorable example was removed to obtain the pass.

The harder writer/reader factorial changes one city at a time and reverses mention order. Writer coverage falls to 48.73% and 44.84%, below 50%. Reader donor effects have 0.677 and 1.200 times the writer-effect norm, far above the specificity ceiling of 0.25. Thus the component has a regional-cue interpretation but no clean role separation. The native model itself also uses both cues.

The portable producer package stores 1,380,864 scalars / 6,328,320 tensor bytes. Its main tensors are:

| Stored object | Shape |
|---|---|
| Each of Q1, Q2, K1, K2 for two heads | $2\times128\times1152$ |
| Two leading key bases | $2\times1152\times64$ |
| Two physical writers | $2\times1152$ |
| Head8 first-token value table | $50304$ |
| Head9 current-value reader | $1152$ |

It accepts token IDs and the two correctly normalized, intervention-consistent native context arrays, each $N\times T\times1152$ for batch size $N$ and sequence length $T$. It returns two physical writes of the same shape. The package replay matches recorded regional and newline effects exactly at output precision, with scalar discrepancy below $4.7\times10^{-15}$.

It still needs the native prefix, background, suffix and full QK maps. Packaging validates an interface; it does not mean the component can predict from token IDs alone. New templates and unused FineWeb cache rows are useful holdouts, but not a broad independent corpus evaluation.

## 9. What this teaches us about continuing the fold

The productive unit was a **composition**: a downstream reader bank, different upstream generators of that bank, and their interaction through an MLP. The path exposed shared writing across heads and a useful split within each head's joint key operation.

Continuing backward means replacing the context inputs or their needed readings with explicit earlier computations. Continuing forward means following the physical writers into their downstream consumers. Both can be done for a token family, token contrasts, or a full-vocabulary coefficient metric. Each extra fold should keep the mixed terms and normalizers needed by the component and then test the resulting combined intervention.

The exact bridge illustrates why this is more than simply deleting everything outside a selected tensor factor: the existing background state participates in the mixed product. A compact DAG can keep that dependency as a declared port while we work backward to generate it. The four-property goal requires eventually closing enough of those ports to demonstrate sufficiency, robust selective manipulation and reuse.

**Primary evidence:** [shared attention derivation](../../SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md), [producer and bridge research note](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md), [weight-derived key bank](../../SCALAR_JOINT_KEY_BANK_V1_RESULT.json), [fresh confirmation](../../SCALAR_JOINT_KEY_GRADES_CONFIRMATION_V1_RESULT.json), [role falsifier](../../SCALAR_EVEN_ROLE_V1_RESULT.json), [normalizer limitation](../../SCALAR_EVEN_NORMALIZER_CLOSURE_V1_RESULT.json), [executable package](../../extracted_circuits/regional_even_key_producers_8_2_9_8_v1/README.md), [native package replay](../../SCALAR_EVEN_PRODUCERS_NATIVE_V1_RESULT.json).
