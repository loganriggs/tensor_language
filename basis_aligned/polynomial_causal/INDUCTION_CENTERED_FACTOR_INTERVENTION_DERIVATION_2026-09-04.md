# Centered selector/content intervention for the induction reference circuit

Date: 2026-09-04 UTC

Status: CPU-only mathematical derivation. This is not a preregistration, a model
result, or permission to change R585 after observing R591. It records the
prospective repair that becomes appropriate only if the frozen R591 diagnostic
finds that the current replay hook is the source of the numerical mismatch.

## The semantic object

For one registered attention head and one prompt, let

$$
E=(e_A,e_C)
$$

be the two attention weights on the registered equality-matching source roles,
and let

$$
U=(u_A,u_C),\qquad u_r=W_Ov_r\in\mathbb{R}^{1152},
$$

be the corresponding value vectors after the head's output matrix. The proposed
semantic contribution is the bilinear map

$$
B(E,U)=e_Au_A+e_Cu_C.
$$

Here $E$ says which matching source is selected and $U$ says what content those
sources would write. This is smaller than a whole attention head and is shared
across the four registered sites L5H5, L7H3, L8H3, and L8H4.

## Why the present replay is numerically fragile

The current self-replay adds

$$
B(E_x,U_x)-C_x,
$$

where $C_x$ is the same equality-supported contribution reconstructed by a
different contraction order: first sum the 128-dimensional head values, then
apply $W_O$. The two expressions are equal over real arithmetic, but floating
point multiplication and addition are not associative. Therefore self-replay
can write a small nonzero residual into the model even though its intended
causal change is exactly zero. Downstream layers can amplify that residual.

That is a measurement problem, not evidence against the selector/content
hypothesis.

## A centered intervention

For recipient prompt $x$ and donor prompt $y$, define the perturbation relative
to the recipient's factorization, rather than relative to an independently
contracted native term:

$$
\begin{aligned}
\delta_{\mathrm{replay}}(x,y)
  &=B(E_x,U_x)-B(E_x,U_x)=0,\\
\delta_{\mathrm{score}}(x,y)
  &=B(E_y,U_x)-B(E_x,U_x)
    =\sum_r(e_{y,r}-e_{x,r})u_{x,r},\\
\delta_{\mathrm{content}}(x,y)
  &=B(E_x,U_y)-B(E_x,U_x)
    =\sum_re_{x,r}(u_{y,r}-u_{x,r}),\\
\delta_{\mathrm{joint}}(x,y)
  &=B(E_y,U_y)-B(E_x,U_x).
\end{aligned}
$$

The model intervention is simply

$$
h_x' = h_x + \delta,
$$

at the registered query position. Self-replay is now exactly a zero tensor by
construction. It no longer asks two algebraically equivalent but numerically
different contractions to cancel.

The joint change also has the exact semantic decomposition

$$
\delta_{\mathrm{joint}}
=\delta_{\mathrm{score}}+\delta_{\mathrm{content}}
+\sum_r(e_{y,r}-e_{x,r})(u_{y,r}-u_{x,r}).
$$

The last term is the selector-by-content interaction. Saving it explicitly
separates an effect that requires both donor factors from the two single-factor
changes.

## What this would and would not establish

If it passes, the centered intervention would establish that changing the
specified bilinear factor causes the predicted behavior. It would not by itself
show that the factor is unique, sufficient without background computation,
stable under a different corpus, or selectively removable. Those remain
separate held-out, removal, reuse, and gauge tests.

The centered definition also changes the operational claim slightly. It tests a
causal displacement in the factor coordinates $B(E,U)$; it does not literally
delete the native equality term $C_x$ and insert a replacement. That distinction
must be stated in any future preregistration and result.

## Required prospective checks

Before this is allowed to replace R585's current hook, a new frozen experiment
must check all of the following without changing the existing R591 threshold:

1. Self-replay produces an exactly zero inserted tensor for every registered
   endpoint and site.
2. Score-only, content-only, and joint deltas reconstructed from saved $E$ and
   $U$ equal the tensors observed at the hook.
3. The equality-support census remains exact; no unregistered matching source
   is omitted.
4. Native behavior and the full-state transfer ceiling still hold on FIT before
   SELECT is opened.
5. The opposing selector-only and content-only counterfactual predictions are
   unchanged.
6. Answer-preserving controls receive nonzero perturbations but little unrelated
   behavioral damage.
7. Every row-level factor, delta, logit, and decision is saved in one atomic,
   independently auditable evidence package.

## Reusable lesson for later circuit agents

When a proposed circuit coordinate is a multilinear function $F(z_1,\ldots,z_k)$,
define interchange as a centered change

$$
F(z_1',\ldots,z_k')-F(z_1,\ldots,z_k),
$$

using one fixed computational expression for both terms. Do not validate a
no-op by subtracting a second implementation that is only algebraically equal.
Then expose the lower-order changes and mixed finite differences separately.
For bilinear attention, the mixed finite difference is exactly the interaction
term above. This gives later agents a common way to split a head into a selector,
content, and their interaction while keeping self-interchange a literal zero.
