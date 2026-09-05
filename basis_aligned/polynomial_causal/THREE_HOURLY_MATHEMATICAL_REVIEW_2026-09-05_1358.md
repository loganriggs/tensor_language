# Three-hour mathematical circuit review — 2026-09-05 13:58 UTC

## Decision

The highest-information mathematical action is the already licensed Task14 experiment, not a
new decomposition or rank calculation. The question is whether a particular source term inside
layer-11 attention head 3 separates into two causally meaningful parts:

- the attention probability placed on the grammatical subject; and
- the subject's vector after the value and output matrices.

The fresh authority changes exactly the subject token while holding the preceding eight tokens
fixed. It separately supplies an opposite-number version of the same noun and a different noun
with the same number. Native accuracy passed all 24 registered cells, so these are meaningful
counterfactuals. The causal runner must still show that the intervention itself is exact and
that replacing the complete head can change the answer; capability alone is not causal evidence.

## Exact computation

At the final position, write the exact output of head 3 as

$$
H=\sum_j p_j u_j\in\mathbb R^{1152},
$$

where $p_j$ is the post-softmax attention probability on source position $j$, and $u_j$ is that
source's value after the head's value and output matrices. For the subject at position 8, the
exact source term is

$$
t_8=p_8u_8.
$$

Calling $p_8$ a “score” in file names is historical shorthand; computationally it is the
normalized attention probability, not the raw query-key dot product. The intervention leaves
every other native source term untouched and installs one of four subject terms:

$$
\begin{aligned}
t_{00}&=p_{\mathrm{same}}u_{\mathrm{same}}, &
t_{10}&=p_{\mathrm{opposite}}u_{\mathrm{same}},\\
t_{01}&=p_{\mathrm{same}}u_{\mathrm{opposite}}, &
t_{11}&=p_{\mathrm{opposite}}u_{\mathrm{opposite}}.
\end{aligned}
$$

Thus the head installed in the recipient run is exactly

$$
H_{ab}=H_{\mathrm{recipient}}-p_{\mathrm{same}}u_{\mathrm{same}}+t_{ab}.
$$

The corresponding interaction in the installed residual vector is not estimated or fitted:

$$
t_{11}-t_{01}-t_{10}+t_{00}
=\left(p_{\mathrm{opposite}}-p_{\mathrm{same}}\right)
 \left(u_{\mathrm{opposite}}-u_{\mathrm{same}}\right).
$$

This is the exact multiplicative interaction between how strongly the head uses the subject and
what vector that subject contributes. It is invariant to a change of basis inside the contracted
value/output matrices because $u_j$ is measured after that contraction, in the residual stream.
It does not identify the query and key branches separately; that stronger interpretation remains
closed.

For a downstream task outcome $Y$, the causal interaction is

$$
I_Y=Y(H_{11})-Y(H_{01})-Y(H_{10})+Y(H_{00}).
$$

Because the rest of the network is nonlinear, $I_Y$ need not equal a linear readout of the
vector interaction above. That difference is scientifically useful: it tells us whether later
components actually compose the attention probability and value for grammatical agreement.
Vaidyanathan et al. show why single-component activation-patching effects generally include such
state-dependent interactions and why factorial interventions are needed to expose them
([primary paper](https://arxiv.org/abs/2606.27510)).

## Two sign conventions that must not be mixed

The pre-outcome code review found and repaired an internal contradiction before model access.
There are two legitimate measurements:

$$
M_{\mathrm{fixed}}=\operatorname{logit}(\text{`` are''})-
\operatorname{logit}(\text{`` is''}),
$$

which should move positively for singular-to-plural and negatively for plural-to-singular; and

$$
M_{\mathrm{donor}}=\operatorname{logit}(\text{counterfactual answer})-
\operatorname{logit}(\text{recipient answer}),
$$

which must move positively in **both** directions when an intervention helps the counterfactual
answer. Full-vocabulary CE is also answer-directed:

$$
\Delta\mathrm{CE}_{\mathrm{donor}}
=\mathrm{CE}_{\mathrm{native}}(\text{donor answer})-
 \mathrm{CE}_{\mathrm{patched}}(\text{donor answer}),
$$

so positive always means improvement. The fixed margin tests signed number discrimination; the
donor margin and donor CE test task usefulness. Keeping them separate prevents the old error in
which multiplying CE by a direction sign could label a worse counterfactual answer as success.

## Identification and simplicity

The candidate subcomputation is identified operationally only if all of the following separate
claims survive:

1. Exact replay, source-term algebra, a no-op installation, and complete-head replacement make
   the intervention capable of passing or failing honestly.
2. Changing $p_8$ between matched opposite-number nouns has the registered signed effect under
   both value states, and the sign reverses when the value state reverses.
3. Changing $p_8$ between different nouns with the same number has at most one quarter of the
   opposite-number effect. This tests number specificity rather than generic token identity.
4. With the opposite-number value installed, changing to the opposite-number probability helps
   the counterfactual answer by both donor margin and donor CE in both grammatical directions.

These claims are intentionally independent. A live number-sensitive probability can still fail
to help the task, and a task-use result can still fail lexical selectivity. If complete-head
replacement is incapable, all three scientific conclusions remain descriptive rather than being
reported as nulls.

This is a useful notion of simplicity for the project because the proposed high-level program
has two reusable variables and one multiplication, and it makes held-out causal predictions.
It is not “simple” merely because it has low rank or few stored numbers. Distributed Alignment
Search likewise defines internal variables by successful counterfactual interchange in a learned
basis, rather than by reconstruction alone
([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html)). Here no rotated basis
is needed yet because the tensor program exposes the exact $p_8u_8$ contraction directly.

## Concrete continuation

The frozen seven-arm HOLDOUT experiment is being implemented now. It uses 16 untouched rows,
three model forwards, 208 endpoint evaluations, no gradients, and no parameter updates. A
same-number/different-noun probability is the active lexical control, while complete
opposite-head replacement is the capability control. The result will be recorded as separate
number-discrimination, lexical-selectivity, and bidirectional-task-use outcomes in the canonical
Task14 dossier. No result will be collapsed into a single success label, and no threshold, row,
or direction will be repaired after outcomes.

