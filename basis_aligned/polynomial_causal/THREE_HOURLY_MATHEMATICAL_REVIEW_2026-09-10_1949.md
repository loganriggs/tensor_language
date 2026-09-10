# Three-hour mathematical review — 10 September 2026, 19:49 UTC

## Decision

**Exact sparse token connections need an interface-feasibility test before a sparse-factor optimizer.** The current residual-writer interface cannot produce a nonzero write supported only on any of the three tested pronoun groups. Even the best possible writer must place most of its squared vocabulary loading outside each group. This limits literal token-support interpretations; it does not forbid selective behavior or shared computations.

The two stable products also failed the registered natural-context reflection and removal screen. Keep them as partial readout components. Do not increase their number to rescue this claim. Continue tracing shared computation inside the factors and the joint QK spaces, with normalization explicit.

## Actual object and price

MLP17 uses $x\in\mathbb R^{1152}$, $L,R\in\mathbb R^{4608\times1152}$ and $D\in\mathbb R^{1152\times4608}$. Its degree-two contribution is $D[(Lx)\odot(Rx)]$; Down bias is separate. The unembedding has 50,304 rows, of which 50,257 are valid tokenizer entries. Flattening the symmetric input coordinates gives

$$
T=UG,\qquad T\in\mathbb R^{V\times664128}.
$$

We do not materialize $T$. A product dictionary has columns $F_j=\operatorname{vec}_{\rm sym}(\operatorname{sym}(a_jb_j^\top))$, with an isometric symmetric vectorization, and represents $T\approx AF^\top$. The native graph contains shared linear readers, multiplication nodes and output sums. Product-reader scaling, reader interchange, component permutation and paired sign changes are gauges; general rotations of products need not preserve product form. No parameter tying between L and R is assumed.

Full-vocabulary Frobenius error is the discovery metric. Natural-state intervention effects are separate measurements. The last-layer input is already normalized; full logits additionally depend on the incoming residual, final RMS denominator and tanh saturation. Thus the full model is not a homogeneous polynomial on unconstrained raw residual inputs. Interventions must specify their site and retain those operations.

The compact two-product executable stores 123,466 bytes, including two fitting restarts and the comparison axis. Its raw product computation requires four length-1152 dot products, two multiplications and two residual writes per restart. It retains all native upstream/background weights. This is not independently extracted text-to-output computation. Native screen: 13 forwards, 104 sequences, 2.238 executor seconds. Support analysis: 1.613 CPU seconds, zero model forwards. These prices do not constitute circuit identification.

## Literature mapping and assumptions

**Cospark / sparse-error decoding.** Zhong and Zhao define cospark as the smallest support of a nonzero vector in a matrix's column space. Set their matrix to our valid-vocabulary $U$ and their unknown vector to a residual writer $w$. The problem is exactly $\min_{w\ne0}\|Uw\|_0$. Exact cospark is generally hard; their polynomial algorithm concerns generic matrices with a prescribed sparsity pattern and independently continuously sampled entries. Those assumptions do not apply to a trained, rounded unembedding. We therefore do not infer its exact cospark from density. [Primary paper](https://arxiv.org/pdf/1701.08925).

**Tensor unfolding and CP.** The output unfolding exposes the column-space constraint directly. A product with distinct readers is the sum of two symmetric partner rank-one terms. Consequently ordinary generic CP uniqueness conditions cannot simply be transferred: the partner terms duplicate output columns. Standard decomposition algorithms remain candidates, not identification guarantees. Tucker changes shared coordinate spaces but does not by itself produce sparse connections or computational semantics. [Kolda and Bader](https://www.kolda.net/publication/TensorReview.pdf).

**Parts-based nonnegative factorization.** Donoho and Stodden's framework depends on nonnegative data/factors and conditions on how parts appear separately. Our quadratics and output coefficients are signed, and no analogous separability condition has been demonstrated. Applying NMF to absolute loadings would alter the algebra being explained, not establish identification of the original products. [Primary paper](https://papers.nips.cc/paper_files/paper/2003/file/1843e35d41ccf6e63273495ba42df3c1-Paper.pdf).

The neighboring concentration formulation suggests a stronger *fixed-group* question that elementary linear algebra solves exactly. A graph-width contraction or tensor train could evaluate the existing program differently, but does not resolve this output-support constraint. Hankel/minimal-state methods would require specified sequential observables and transition closure, absent from the current static final-layer family. They are not substitutes for this support test.

## Derivation: where output factors can live

If the quadratic columns of $F$ are linearly independent and an exact decomposition satisfies $T=AF^\top$, then

$$
A=T F(F^\top F)^{-1}
 =U\underbrace{G F(F^\top F)^{-1}}_W.
$$

Hence every factor's output code lies in $\operatorname{col}(U)$. The same is true of the unconstrained least-squares optimum for fixed independent $F$. Thus using $A=UW$ in the current least-squares optimizer does not arbitrarily exclude a better unrestricted least-squares output fit.

If $F$ is dependent, output components outside $\operatorname{col}(U)$ can cancel between terms. Removing those terms independently can break the cancellation. Such a decomposition requires explicit intervention semantics and pricing; it cannot inherit a native residual-writer interpretation automatically. A sparse approximation with unrestricted $A$ similarly needs a changed output interface if its columns leave $\operatorname{col}(U)$.

For a proposed token support $S$, an exact nonzero supported write exists precisely when

$$
U_{S^c}w=0\quad\text{for some }w\ne0,
$$

assuming $U$ has full column rank. Full column rank of the complement rules it out. If every choice of $d$ rows of $U$ were independent, every nonzero code would have at least $V-d+1=49,106$ nonzeros. This is a conditional generic/full-spark statement, **not** an established property of this checkpoint; verifying three supports does not establish all supports.

## Executable approximate bound

Let $M=U^\top U$ be positive definite and $M=CC^\top$ its Cholesky factorization. The greatest possible fraction of squared loading placed inside a specified group is

$$
\eta_S=\max_{w\ne0}\frac{\|U_Sw\|^2}{\|Uw\|^2}
=\lambda_{\max}\left(U_S M^{-1}U_S^\top\right).
$$

Derivation: substitute $y=C^\top w$, then maximize a Rayleigh quotient. Solve triangular systems; do not invert $M$ explicitly. A maximizing eigenvector gives a writer that attains the bound, providing an independent numerical witness. Every residual-space writer leaves at least $1-\eta_S$ of its squared loading outside the group. This is a sharp bound on this linear numerator metric, not on probabilities or CE.

Cost: form $M$ in $O(Vd^2)$, factor it in $O(d^3)$; each size-$s$ group needs $O(sd^2+s^3)$. The implementation additionally checks the complement spectrum in $O(d^3)$. Storage is $O(Vd+d^2)$. This small eigenproblem is preferable to solving general cospark or constructing the third-order tensor.

The CPU implementation passed the analytic fixture $U=[(1,0);(0,1);(1,1)]$, $S=\{1\}$, whose optimum is $2/3$. Native maximizing witnesses agree with the eigenvalue bound within $1.5\times10^{-14}$.

| Specified output group | Largest possible in-group squared loading | Required outside loading |
|---|---:|---:|
| Six original pronoun targets | 10.45% | 89.55% |
| Twelve largest absolute loadings of stable product 0 | 20.49% | 79.51% |
| Twelve largest absolute loadings of stable product 9 | 22.24% | 77.76% |

Each complement has numerical rank 1152; smallest Gram eigenvalues are 23.21–23.22, with minimum/maximum ratios about $3.33\times10^{-4}$. This is well separated numerically, but no interval-arithmetic exact rank certificate is claimed. The fitted products place 7.33% and 7.88% of their squared loading in their own top-twelve groups, respectively. Sparse exact token support is therefore neither present nor feasible for those groups within this residual interface.

## Circuit consequence and next step

For the natural-context screen, the two stable products leave 85.3–85.6% full-vocabulary response error and 30.6–32.6% target-CE response error. Removal causes mean absolute CE changes of .170–.200 on pronoun targets and .01161–.01204 elsewhere, missing the .01 preservation bar. Paired row intervals also exclude that bar here. This supports partial causal relevance but fails the registered sufficient/selective operation claim.

The support bound does **not** explain away that failure. Small per-token writes across many unlikely alternatives can dominate squared loading while barely changing behavior. We must retain the behavioral test and its failed result. The theorem instead changes the discovery objective: do not demand exact small token supports from a legal residual writer; inspect graded signed consumer profiles and shared linear/product computations, or explicitly change and price the output interface.

Next highest-information step: pull the already fitted pre-position joint QK readers through their actual Q1/Q2 weight maps and compare the two behaviors in the resulting common input-quadratic metric. This tests whether their apparent product-coordinate differences survive folding back into the shared input. It targets within-module splitting and shared input computation, without refitting ranks or adding another reconstruction sweep. Normalization denominators remain separate; raw numerator overlap alone cannot identify normalized attention circuits.

Receipts: [support feasibility and paired intervals](UNEMBEDDING_SUPPORT_FEASIBILITY_V1_AUDIT.json), [causal reflection screen](STABLE_JOINT32_REFLECTION_V1_RESULT.json), [CPU implementation](audit_unembedding_support_feasibility_v1.py). Next math review: 22:49 UTC. Hourly review remains due 20:14 UTC.
