# Shared global functions with local corrections: feasibility, not a fitted result

13 September 2026, 05:01 UTC. User-directed compression setting 3.

The proposal is to let every token use a few shared quadratic functions, plus a small function bank belonging to its learned group. This can have many functions overall while each token reads relatively few. It is an output-edge sparsity hypothesis. It does not yet simplify the multiplication inside those functions.

Prior work matters: `UNEMBEDDING_BACKWARD_VIEWS_V1_RESULT.json` tried fixed group means before folding, and `TOKEN_FUNCTION_DICTIONARY_V1_RESULT.json` fitted an overlapping sparse function dictionary in the folded metric. The new restricted comparison is a shared global parent plus group-specific *subspaces*, learned after folding. It is related to that dictionary work, not an unexplored general concept. Neither earlier result proves this model works.

## Exact object and representation

Write the bias-free last bilinear layer as

$$
B(x)=D[(Lx)\odot(Rx)],\quad
L,R\in\mathbb R^{4608\times1152},\quad
D\in\mathbb R^{1152\times4608}.
$$

The output tensor consists of the quadratic functions \(U_v B(x)\), for all 50,304 rows of \(U\). For their symmetric coefficient Frobenius inner product, define

$$
K_{kl}=\tfrac12[(L_k\cdot L_l)(R_k\cdot R_l)
 +(L_k\cdot R_l)(R_k\cdot L_l)],\qquad
G=DKD^\top=HH^\top.
$$

Here \(H\) is a Cholesky factor, of shape \(1152\times1152\). If \(\mu\) is the mean vocabulary row, the coordinates

$$
X_v=(U_v-\mu)H
$$

preserve exactly the inner products of centered token quadratic functions. No text or giant vocabulary-by-input-by-input tensor is required.

The candidate model is

$$
\widehat X_v=a_v P+b_v Q_{c(v)},
$$

where \(P\) contains \(g\) global rows, each \(Q_k\) contains \(r\) local rows, and \(c(v)\) assigns a token to one of \(k\) groups. Thus each token uses \(g+r\) coefficients, while all groups together can span up to \(g+kr\) dimensions. The global basis need not dictate each local basis. Multiple local parents per token would be a later, more general overlapping model.

A learned coordinate row \(p\) compiles into the native residual reader \(pH^{-1}\). Compute that reader once with a triangular solve and store the resulting row. The inverse is not an additional runtime adapter. Evaluation still needs the existing \(B(x)\), and the exact mean function \(\mu B(x)\) is shared by all tokens.

## Honest price and optimistic bound

With FP32 weights and int32 group IDs, the candidate costs

$$
4\left[d+(g+kr)d+V(g+r)\right]+4V\quad\text{bytes},
\qquad V=50304,\ d=1152.
$$

This replaces the output map only. Keep native L/R/Down, products, bias, residual route, normalization and final saturation in the total executable price. All-output evaluation computes every bank; a selected-token evaluation may need only selected groups. Weight sharing across different inputs is not automatically a computation saving.

Let \(\lambda_i\) be the descending eigenvalues of \(X^\top X\). Any proposed model has centered rank at most \(g+kr\), so its full-tensor relative error is at least

$$
\sqrt{\frac{\sum_{i>g+kr}\lambda_i}
 {\sum_i\lambda_i+V\mu G\mu^\top}}.
$$

This relaxes all group restrictions. A zero bound means only that rank cannot rule the model out. It does not promise exact representation with sparse group usage.

| Global width / groups / local width | Output-map bytes relative to native U | Optimal global fit at the same byte budget | Optimistic grouped error lower bound |
|---|---:|---:|---:|
| 32 / 16 / 4 | 3.40% | 89.94% error | 84.12% |
| 64 / 16 / 8 | 6.72% | 86.14% | 75.60% |
| 64 / 32 / 8 | 6.97% | 85.85% | 65.30% |
| 128 / 32 / 16 | 13.86% | 78.69% | 42.30% |
| 128 / 64 / 16 | 14.88% | 77.74% | 0% |

These are coefficient errors, not language-model damage or native grouped-fit measurements. The mean accounts for 7.20% of coefficient energy and is retained exactly. Native U contains 57,950,208 scalars. The 128/64/16 candidate contains 8,572,032 floats plus 50,304 int32 assignments, and uses 144 coefficients per token. Its combined bank width is 1152: potentially edge-sparse despite full rank. This is the main reason to test the hypothesis rather than repeat only small-rank global fits.

## Executed checks and next decision

[CPU analysis and controls](fullu_shared_local_feasibility_v1.py) ran in 3.54 seconds; [receipt](FULLU_SHARED_LOCAL_FEASIBILITY_V1_RESULT.json). The exact metric agrees with explicitly materialized toy coefficient tensors to 3.88e-16, compiled reader inner products to 3.10e-15, and the native total energy with the earlier independent full-U receipt to 8.88e-16.

A planted shared/private example is represented exactly at a budget where optimal global compression has 62.82% error. Its banks and assignments were supplied: this demonstrates representational advantage, **not recovery by an optimizer**.

The next native comparison should fit the 64/32/8 and 128/64/16 candidates, retain the exact mean, and compare against the exact global solutions at their byte budgets. Use alternating exact conditional projections/subspace solves and group reassignment, with multiple initializations and explicit objective/stationarity checks. Report the shared-versus-private ablation and assignment stability. A plateau is not global convergence. Only after freezing a useful fit should text determine whether its coefficient gain preserves token functions and causal effects. No native grouped fit has run in this receipt.
