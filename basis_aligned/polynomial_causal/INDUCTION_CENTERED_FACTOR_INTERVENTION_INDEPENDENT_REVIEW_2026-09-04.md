# Independent review: centered induction-factor intervention

Date: 2026-09-04 UTC  
Reviewed derivation SHA-256:
`afb816361603d880dea8dd5daa30b90e841f686d4935da8684ac78c3839a78ca`  
Verdict: **algebra approved; causal claim approved only as an output-space
factor intervention, not as literal attention-score or remove-and-insert
interchange**

This review used only small CPU tensors. It did not load the model, open CUDA,
inspect outcomes, or edit R590/R591.

## Algebra

Let

$$
B(E,U)=\sum_r e_r u_r,
\qquad
\Delta E=E_y-E_x,
\qquad
\Delta U=U_y-U_x.
$$

By bilinearity,

$$
\begin{aligned}
B(E_y,U_y)-B(E_x,U_x)
&=B(\Delta E,U_x)+B(E_x,\Delta U)+B(\Delta E,\Delta U).
\end{aligned}
$$

Therefore the derivation's joint, score-only, content-only, and mixed terms are
correct. The mixed term is the two-factor finite difference, not an unexplained
residual. Self-replay can be made bitwise zero by constructing
`zeros_like(B(E_x,U_x))`; recomputing two nominally identical contractions and
subtracting them is avoidable and should not be the implementation.

## Exact operational difference from literal replacement

Write the native equality contribution as $C_x$ and the factorized computation
as $B_x=B(E_x,U_x)$. Let $B_{new}$ be any donor-factor combination. The centered
arm is

$$
h_{centered}=h_x+B_{new}-B_x,
$$

whereas literal remove-and-insert is

$$
h_{literal}=h_x-C_x+B_{new}.
$$

Ignoring the final floating-point addition roundoff,

$$
h_{centered}-h_{literal}=C_x-B_x.
$$

Thus centered interchange deliberately keeps the recipient's native contraction
error as common background across arms. This is exactly why its self arm is a
true no-op. It also means the arm does not demonstrate that the native term was
deleted and replaced, and it is not by itself a sufficiency or compiled-circuit
test. A later literal-removal test remains necessary.

## What score-only and content-only mean causally

The score-only arm is valid for the narrowly defined question: “What is the
downstream effect of changing the two registered equality-role coefficients
from $E_x$ to $E_y$ while holding their recipient output vectors $U_x$ fixed?”
The content-only arm analogously changes the two registered projected value
vectors while holding their recipient coefficients fixed. These are controlled
interventions on the output-space mediator $B$.

They are not automatically literal internal-attention counterfactuals:

1. Replacing only two attention coefficients while holding every unregistered
   coefficient fixed can violate the attention pattern's fixed total mass. In
   this model the two softmax branches give the full bilinear-attention pattern
   a fixed total determined by their mixing coefficient. A partial equality-role
   score swap need not preserve that total.
2. Even when the two equality coefficients have the same total mass, the new
   partial pattern need not be reachable from any query/key state. Donor
   query/key changes generally alter unregistered positions too.
3. A projected value vector is an output-space object after the head output
   matrix. Swapping it is not the same intervention as swapping a token, the
   pre-output value vector, or the complete value stream.
4. The role labels $A$ and $C$ must be authority-bound. Bilinearity cannot detect
   a donor-role permutation; the tensor calculation remains well formed but
   represents a different semantic counterfactual.

Accordingly, future results should say “registered equality-factor coefficient
swap” and “registered projected-content swap.” They should not shorten those to
“attention-score swap” or “value swap” without a separate realizability test.

## Minimal reusable tests

The model-free test
`test_induction_centered_factor_intervention_derivation.py` contains four
checks:

1. the exact bilinear finite-difference identity and literal zero self-replay;
2. an explicit native contraction error showing that centered and literal
   replacement differ by $C_x-B_x$;
3. a negative normalized-attention fixture where swapping only the two selected
   coefficients changes total coefficient mass from $1$ to $1.3$, so the output
   cannot be called a literal full attention-pattern swap; and
4. a role-permutation fixture showing that semantic alignment is an external
   authority, not an algebraic consequence.

Test SHA-256:
`0d945d9d3a6995b8fc2bfa1a2a0619fa3f741626bc2eaaf63de238ac5f4de56b`.
Result: `4 passed`.

## Implication for later circuit agents

There are at least four distinct valid counterfactuals here: a partial
output-factor swap, a mass-compensated partial score swap, a complete attention
pattern swap, and a realizable query/key-state swap. They answer different
questions and need separate names. The centered partial factor is the cleanest
test of whether the proposed $E\times U$ mediator controls downstream behavior;
the full-state and literal-removal arms test whether that mediator is sufficient
and selectively removable in the native computation.

The current derivation can therefore proceed conditionally after R591 only if a
future preregistration preserves this scope, binds role/site alignment, saves the
actual inserted tensors, and retains full-state, active-control, held-out, and
literal-removal tests. Claiming native score/value interchange from the centered
arm alone would be blocked.

