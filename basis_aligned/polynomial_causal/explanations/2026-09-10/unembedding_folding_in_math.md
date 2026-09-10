**Yes—this gives you a family of explicit quadratic functions, one per vocabulary logit. The interesting structure is in how those functions share factors and subcomputations.** That lets you test concrete hypotheses about hierarchy, DAGs, and linearity.

Let the bilinear layer be

$$
y=D[(Lx)\odot(Rx)]
$$

and the unembedding be \(U\in\mathbb R^{|\mathcal V|\times d}\). Fold it into

$$
C=UD.
$$

The layer’s logit contribution is

$$
\Delta z_v(x)=\sum_k C_{vk}(l_k^\top x)(r_k^\top x)
=x^\top S_vx,
$$

where

$$
\boxed{
S_v=\operatorname{sym}\!\left(L^\top\operatorname{diag}(C_{v,:})R\right).
}
$$

This is the weight-based quadratic representation used in [bilinear MLP interpretability](https://arxiv.org/abs/2410.08417), now anchored to actual output words.

If a final RMSNorm sits between the residual and unembedding, preserve its denominator explicitly. Folding \(U\) into \(D\) then describes the corresponding numerator contribution; the full logits also depend on the residual path and normalization.

**The most revealing object is the whole family \(\{S_v\}\), rather than separate eigendecompositions for individual words.**

Here is an illustrative structure you could discover:

$$
\begin{aligned}
z_{\rm is}   &= up+an,\\
z_{\rm are}  &= up-an,\\
z_{\rm was}  &= ut+an,\\
z_{\rm were} &= ut-an,
\end{aligned}
\qquad u=a+b.
$$

Here \(a,b,p,t,n\) denote linear functions of the layer input—not interpretations we can assume in advance.

This program has:

* a shared intermediate \(u=a+b\);
* two products \(up,ut\);
* an additional product \(an\), reused across both groups;
* four output combinations.

It is naturally a **DAG**: one grouping shares \(up\) or \(ut\), while another crosses those groups through \(an\).

Its structure creates exact signatures in the folded weights:

$$
\boxed{
S_{\rm is}-S_{\rm are}
=
S_{\rm was}-S_{\rm were}.
}
$$

Also,

$$
\frac{S_{\rm is}+S_{\rm are}}2
=\operatorname{sym}(u p^\top),
\qquad
\frac{S_{\rm was}+S_{\rm were}}2
=\operatorname{sym}(u t^\top).
$$

Those last two matrices share the linear factor \(u\). **You can test all of this directly from weights, before assigning meanings to the factors.**

I would look for four kinds of structure.

**1. Shared products and shared linear factors.**

A scalar quadratic consisting of one product,

$$
q(x)=(a^\top x)(b^\top x),
$$

has

$$
S=\operatorname{sym}(ab^\top).
$$

Its rank is at most two. More precisely, a nonzero symmetric quadratic is one real product when it has either rank one, or rank two with one positive and one negative eigenvalue.

For the latter case,

$$
S=\lambda ee^\top-\mu ff^\top
$$

gives the exact factorization

$$
q(x)=
\left[(\sqrt\lambda e+\sqrt\mu f)^\top x\right]
\left[(\sqrt\lambda e-\sqrt\mu f)^\top x\right].
$$

So a concrete search is:

> Find combinations of output logits whose quadratic matrices factor into a few products, then identify factors reused across those combinations.

The output combinations matter. In the example, an individual logit contains two mechanisms, but sums and differences separate them.

This is more targeted than asking whether each word has a low-rank matrix. You are searching for a **joint arithmetic implementation**.

**2. Hierarchical output structure.**

Suppose a group of words has

$$
S_v=S_{\rm group}+E_v.
$$

Then

$$
\Delta z_v(x)
=
x^\top S_{\rm group}x+x^\top E_vx.
$$

The group shares one computation; the corrections distinguish its members. Recursively applying this could give a hierarchy.

But every chosen group admits a mean-plus-residual decomposition. **The evidence for hierarchy is that the shared function and corrections have a substantially simpler joint implementation**—for example, a common product plus sparse corrections using a few additional factors.

A tree may be insufficient. In the is/are/was/were example, agreement is reused across tense groups. Allowing shared nodes gives a DAG without duplicating that operation.

A suitable candidate representation is

$$
\boxed{
\Delta z(x)=A\,p(x),\qquad
p_j(x)=(a_j^\top x)(b_j^\top x),
}
$$

with additional sharing among the linear forms. Count the factors, their coefficients, and the output connections. Merely choosing a different basis for the same dense computation establishes little.

**3. Independent subsystems.**

A stronger hypothesis is that some input subspaces never interact in this final computation.

If one change of coordinates \(x=P\xi\) makes every output quadratic block diagonal,

$$
P^\top S_vP=
\begin{bmatrix}
S_v^{(1)}&0\\
0&S_v^{(2)}
\end{bmatrix},
$$

then

$$
\Delta z_v
=
\xi_1^\top S_v^{(1)}\xi_1+
\xi_2^\top S_v^{(2)}\xi_2.
$$

This certifies two additively separable computational subsystems at this layer. The relevant mathematical problem is [simultaneous block diagonalization by congruence](https://arxiv.org/abs/2503.01166).

This tests a stronger and narrower hypothesis than shared factors. Failure to find blocks does **not** rule out a useful DAG: shared variables can connect otherwise simple operations.

Also, folding in a full-column-rank \(U\) does not create new algebraic structure: it preserves the information in the original output tensor. Its benefit is exposing structure in vocabulary coordinates where hypotheses become meaningful.

**4. Linear versus genuinely quadratic computation.**

For a logit combination \(q(x)=x^\top Sx\),

$$
\nabla^2 q(x)=2S.
$$

Thus its homogeneous quadratic contribution is globally linear only when \(S=0\), in which case that contribution vanishes. Any residual linear path remains separate.

On an affine input domain \(x=x_0+B\xi\),

$$
q(x)=
x_0^\top Sx_0+
2x_0^\top SB\xi+
\xi^\top B^\top SB\xi.
$$

Therefore,

$$
\boxed{B^\top SB=0}
$$

is an exact certificate that this computation is affine on that domain.

Unlabeled activations can suggest such domains or approximate versions. That can reveal “this mechanism acts linearly on the states actually encountered,” even though the global function is quadratic. The claim must still be tested on your intended intervention states.

**You don’t need to construct the vocabulary-sized tensor.**

For any output combination \(\alpha\), define

$$
c=D^\top U^\top\alpha.
$$

Then the corresponding quadratic matrix can be applied to a vector \(v\) as

$$
S_\alpha v
=
\frac12\left[
L^\top(c\odot Rv)+
R^\top(c\odot Lv)
\right].
$$

In PyTorch:

```python
# U: [vocab, d], D: [d, hidden], L/R: [hidden, d]
def quadratic_mv(alpha, v):
    c = D.T @ (U.T @ alpha)
    return 0.5 * (
        L.T @ (c * (R @ v))
        + R.T @ (c * (L @ v))
    )
```

This supports factor searches and spectral screening without allocating \(|\mathcal V|\times d\times d\). Numerical screening nominates structure; exact identities or measured approximation residuals establish what was recovered.

**My first experiment would be to recover a small joint program for a related output family.** Form its logit sums and differences, factor the resulting quadratics, identify shared factors, and reconstruct all selected logits jointly. Preserve the full-vocabulary writes when testing removal.

The final-layer tensor can establish that these shared functions implement the output computation. **To determine whether the model actually constructs those intermediates upstream—and obtain an extractable input-to-output circuit—trace their linear readers backward through earlier MLPs and attention.** That is where the discovered final-layer DAG becomes a hypothesis about the larger circuit.

Logan: How does this carry over to folding back into the attention module at the last layer or the prevouis bilinear layer? We have several computation paths, the direct ones from the unembedding to attn and the unembedding -> bilinear_final -> attn_final, etc

**Yes. The extension is to pull back the entire logit function, retaining its multiple input branches.** A linear readout pulls back to a vector; a quadratic readout pulls back to a computation that includes interactions between paths.

This is the bilinear composition principle behind [weight-based extensions of transformer circuits](https://arxiv.org/abs/2305.03452). Here is the explicit version for your final attention and MLP.

For clarity, first consider additive residuals and linear unembedding, keeping normalization separate.

**1. The final attention contributes through three kinds of downstream computation.**

Let

$$
x=r+A(r),\qquad
z=U[x+B(x)],
$$

where \(r\) is the residual before final attention, \(A\) is final attention, and

$$
B(x)=D[(Lx)\odot(Rx)].
$$

For vocabulary token \(v\), define \(u_v=U^\top e_v\) and

$$
S_v=\operatorname{sym}\!\left(
L^\top\operatorname{diag}(D^\top u_v)R
\right).
$$

Then

$$
z_v=u_v^\top x+x^\top S_vx.
$$

Substitute \(x=r+A\):

$$
\boxed{
z_v=
\underbrace{u_v^\top r+r^\top S_vr}_{\text{computation without attention}}
+
\underbrace{u_v^\top A}_{\text{direct attention write}}
+
\underbrace{2r^\top S_vA}_{\text{attention interacting with residual}}
+
\underbrace{A^\top S_vA}_{\text{attention interacting with attention}}.
}
$$

So your routes include:

| Forward route                                    | Exact contribution |
| ------------------------------------------------ | ------------------ |
| Attention → unembedding                          | \(u_v^\top A\)     |
| Attention and residual → final MLP → unembedding | \(2r^\top S_vA\)   |
| Two attention writes → final MLP → unembedding   | \(A^\top S_vA\)    |

**The MLP-mediated route is not generally one independently scored path. It is a computation with two input branches.** This is where a graph of ordinary paths needs to become a DAG of operations.

**2. Fold the final MLP’s readers into attention’s output maps.**

Write attention as

$$
A=\sum_h O_h a_h,
$$

where

$$
a_h=\sum_{s\le t}
\underbrace{
(q_{1,h,t}^\top k_{1,h,s})
(q_{2,h,t}^\top k_{2,h,s})
}_{g_{h,ts}}
v_{h,s}.
$$

The exact folded weights are:

| Object              | What it specifies                                         |
| ------------------- | --------------------------------------------------------- |
| \(UO_h\)            | Direct vocabulary write from head \(h\)                   |
| \(LO_h,\ RO_h\)     | How head \(h\) supplies the final MLP’s two input factors |
| \(S_vO_h\)          | How its write interacts with the existing residual        |
| \(O_h^\top S_vO_g\) | How heads \(h,g\) interact in producing logit \(v\)       |

For example,

$$
A^\top S_vA
=
\sum_{h,g}
a_h^\top
\underbrace{O_h^\top S_vO_g}_{\text{exact interaction operator}}
a_g.
$$

That operator answers a concrete structural question:

> Which combinations of information written by these two heads can jointly affect this output?

A zero operator certifies that this particular quadratic interaction is absent. A shared factor across several such operators proposes a reusable computation.

Crucially, you need not materialize all these pairwise operators. Compute

$$
p=Lr+\sum_h(LO_h)a_h,\qquad
q=Rr+\sum_h(RO_h)a_h,
$$

and then

$$
\boxed{
z=Ur+\sum_h(UO_h)a_h+(UD)(p\odot q).
}
$$

This preserves every cross term while retaining the original factorization. Expanding the pairwise terms is useful for inspection; the factored form is useful for execution.

**3. Going back into the previous bilinear layer uses the same substitution rule.**

Suppose the residual entering final attention is

$$
r=b+D_0h_0,\qquad
h_0=(L_0b)\odot(R_0b).
$$

Its direct route to logits folds to

$$
UD_0h_0.
$$

Its residual route into the final MLP folds through

$$
LD_0,\qquad RD_0.
$$

For any final quadratic readout,

$$
r^\top S_vr
=
b^\top S_vb
+2b^\top S_vD_0h_0
+h_0^\top D_0^\top S_vD_0h_0.
$$

The last term is quadratic in the previous layer’s products \(h_0\), hence quartic in \(b\). **Keep \(h_0\) as a shared intermediate.** There is no need to expand its quartic coefficients.

But the previous MLP also changes attention’s routing. For either QK factor, let

$$
M=Q^\top K
$$

with positional operators included where necessary. At destination \(t\) and source \(s\),

$$
\begin{aligned}
r_t^\top Mr_s
={}&b_t^\top Mb_s\\
&+h_{0,t}^\top D_0^\top Mb_s\\
&+b_t^\top MD_0h_{0,s}\\
&+h_{0,t}^\top D_0^\top MD_0h_{0,s}.
\end{aligned}
$$

These are respectively:

* the background score;
* previous-MLP influence through the query;
* previous-MLP influence through the key;
* query–key interaction between previous-MLP computations.

Apply this to **both** QK factors, then multiply their scores. Include the value route through \(V_hD_0\) wherever values depend on the current residual.

For the shared \(v_0\) branch, that current-residual value route is absent: the previous MLP changes **where the cached payload is retrieved**, while the payload comes from its separate upstream computation.

**4. This gives a systematic backward-folding procedure.**

Starting at logits:

1. Fold linear maps into the operations that produce their inputs.
2. At an addition, retain the separate sources.
3. At a multiplication, retain both input dependencies.
4. Substitute earlier computations while preserving shared nodes.
5. Search the resulting folded operators for common factors, absent interactions, and repeated subprograms.

For instance, you might discover that:

* several outputs share the same final-MLP product;
* that product reads one factor from an attention head and another from the residual;
* several heads supply the same factor through different source-selection rules;
* those rules share a query predicate constructed by the previous MLP.

That would be a genuine compositional account of multiple routes to logits.

**5. These interactions also tell you how to test removal.**

Let \(w\) be one final attention head’s output and \(x=r+A\). Removing that head while leaving the other final attention heads unchanged gives

$$
\boxed{
z_v(x)-z_v(x-w)
=
u_v^\top w+2x^\top S_vw-w^\top S_vw.
}
$$

This is an exact finite removal effect for the stated final block—not a gradient approximation. It includes interactions with the residual and the other heads.

Removing the **previous MLP** requires recomputing attention scores and values too, because that intervention changes the routing program itself.

Normalization must also be recomputed. For example, if the final MLP reads \(\operatorname{RMSNorm}(x)=\Gamma x/s(x)\), its quadratic logit contribution becomes

$$
\frac{x^\top\widetilde S_vx}{s(x)^2},
$$

with \(\Gamma\) absorbed into the readers. The numerator still has the decomposition above, but removal changes its denominator as well.

**The immediate object I would inspect is therefore the joint family \(UO_h,\ LO_h,\ RO_h,\ LD_0,\ RD_0,\ Q_{jh}D_0,\ K_{jh}D_0\)**, keeping their products and residual connections explicit. It exposes both direct vocabulary effects and the intermediate computations through which those effects are constructed.
