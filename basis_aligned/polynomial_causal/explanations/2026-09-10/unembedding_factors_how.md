Logan: For the original folding in, how do you find the shared factors between the tokens? You fold in, make 3rd order tensor, slice per output, then those are factors(?) and then you use that somehow to see if there's hierarchcial structure ro DAGs or seomthign? Like how many tokens share each factor or having overlapping sets? What's standard methods there?

Yes—the workflow is **fold in → jointly factor across outputs → inspect which outputs use each factor → look for further shared computations inside those factors.** But the output slices themselves are whole quadratic functions, not individual factors.

Here, “tokens” means **output vocabulary entries**. We are initially studying how the final bilinear layer writes to those logits.

**1. What folding gives you**

For a bilinear layer and unembedding,

$$
B(x)=D[(Lx)\odot(Rx)],\qquad C=UD,
$$

its contribution to output token \(v\) is

$$
\Delta z_v(x)=x^\top S_vx,
\qquad
S_v=\sum_k C_{vk}\operatorname{sym}(\ell_kr_k^\top),
$$

where \(\operatorname{sym}(M)=(M+M^\top)/2\).

Stacking the matrices \(S_v\) gives your third-order tensor: **output token × input coordinate × input coordinate**. Each slice describes everything this layer contributes to one logit. Any normalization outside this bilinear map remains explicit.

Notice that **the original neurons already provide an exact shared factorization**:

$$
f_k(x)=(\ell_k^\top x)(r_k^\top x),
\qquad
\Delta z_v(x)=\sum_k C_{vk}f_k(x).
$$

So \(C=UD\) is already a *token-by-factor table*. The research question is whether another exact factorization exposes substantially clearer sharing.

**2. How do you find better shared factors?**

You solve for factors **jointly across the slices**:

$$
\boxed{
S_v=\sum_{j=1}^{K}A_{vj}F_j
\quad\text{for every }v.
}
$$

Here \(F_j\) is a shared quadratic computation and \(A_{vj}\) says how token \(v\) uses it.

Different methods constrain \(F_j\) differently:

| Method                         | What the shared factors are                      | What it establishes                                                               |
| ------------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------- |
| SVD of vectorized slices       | Arbitrary quadratic forms \(F_j\)                | The smallest linear space containing all slices                                   |
| Joint dictionary factorization | Quadratic forms with sparse token loadings \(A\) | A vocabulary of computations that outputs selectively reuse                       |
| CP-style product factorization | \(F_j=\operatorname{sym}(a_jb_j^\top)\)          | Shared products \((a_j^\top x)(b_j^\top x)\), matching your bilinear architecture |
| Tucker decomposition           | Shared input directions plus an interaction core | Which combinations of directions contribute to which outputs                      |

CP and Tucker are standard tensor decompositions. A symmetric-input CP variant uses squares \((b_j^\top x)^2\); your product form allows two different readers. Standard Tucker does **not** automatically produce a sparse core or a hierarchy. [Kolda and Bader’s review](https://www.kolda.net/publication/TensorReview.pdf)

For your objective, I would prioritize **joint product factorization with explicit pressure for reusable factors and sparse output connections**. No SAE is required. A candidate formulation is

$$
\min_{A,\{a_j,b_j\}}
\lambda K+\mu\|A\|_0
\quad\text{subject to}\quad
S_v=\sum_j A_{vj}\operatorname{sym}(a_jb_j^\top)
\quad\forall v.
$$

This is a proposed search objective, not an easy globally solvable algorithm. Practically, fix \(K\), alternate fitting the output coefficients and input factors, and verify the resulting tensor identity. The native factorization supplies an exact baseline; a smaller exact solution is not guaranteed.

**Independently eigendecomposing every slice is usually the wrong primary procedure:** different mixtures of the same shared factors can have different eigenvectors. Joint decomposition is what asks for common computations.

**3. How do you measure token sharing and overlap?**

Suppose a recovered coefficient table looks like this:

| Output | \(f_1\) | \(f_2\) | \(f_3\) |
| ------ | ------: | ------: | ------: |
| is     |       1 |       0 |       1 |
| are    |       1 |       0 |      −1 |
| was    |       0 |       1 |       1 |
| were   |       0 |       1 |      −1 |

Then:

* \(f_1\) is reused by *is* and *are*.
* \(f_2\) is reused by *was* and *were*.
* \(f_3\) affects all four, with a signed distinction crossing the first two groups.

These are hypothetical coefficients; grammatical interpretations would require checking the functions themselves.

Define each factor’s support

$$
\mathcal V_j=\{v:A_{vj}\ne0\}.
$$

You can count \(|\mathcal V_j|\), intersections, and inclusion relationships. Keep the signs: shared upweighting and shared downweighting are both relevant.

In trained weights, exact supports may all be dense. Then normalize the factors before comparing loadings, and distinguish **large loading patterns** from exact zeros. With unlabeled data, a useful effect scale is

$$
|A_{vj}|\sqrt{\mathbb E_x[f_j(x)^2]}.
$$

Thresholding this gives an approximate usage table; it does not make the underlying decomposition exactly sparse.

For analyzing that table:

* **Biclustering** finds groups of tokens sharing groups of factors, allowing overlap.
* **Formal concept analysis** constructs a hierarchy of token sets and their common factors from a binary incidence table. This is particularly close to your “overlapping sets forming a DAG” idea. Its full lattice can become large. [Ganter and Wille](https://link.springer.com/book/10.1007/978-3-642-59830-2)

**4. Token-set hierarchy and computational hierarchy are different discoveries.**

Overlapping supports tell you **who reuses a computation**. They do not establish that one factor computes another.

For actual computational sharing, inspect the factors’ internals. For example,

$$
f_1(x)=(a^\top x)(b^\top x),\qquad
f_2(x)=(a^\top x)(c^\top x)
$$

share the intermediate \(u=a^\top x\). That yields an executable DAG: compute \(u\) once, then reuse it in two products.

There are therefore two complementary graphs:

* **Factor → output:** discovered through the coefficient table \(A\).
* **Intermediate → factor:** discovered through common linear readers, algebraic identities, and eventually folding into earlier modules.

I would start with the native exact factors, inspect their output loadings, then attempt a joint refactorization and search for shared readers. Crucially, **factorization is not generally unique**: an attractive overlap diagram needs algebraic verification and intervention tests before we call its nodes the model’s mechanisms.
