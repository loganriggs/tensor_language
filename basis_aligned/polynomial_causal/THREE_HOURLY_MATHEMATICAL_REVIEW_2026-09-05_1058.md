# Three-hour mathematical circuit review — 2026-09-05 10:58 UTC

## Decision and circuit target

The current question is not whether layer-8 attention has a low-rank representation. It is
whether heads 3 and 7 contain a **shared cached number value** that is reused by numbered-list
successor and digit-sequence successor, while each task supplies its own attention routing.
This directly targets cross-head grouping, within-head splitting, held-out prediction,
selective manipulation, composition/reuse, and gauge-stable identification.

The descriptive v3 result is unusually specific: replacing the cached-value part transfers
between the list and digit formats at $0.932$--$1.080$ times the within-format effect in all
eight split-by-format-by-direction cells, and every cached or joint intervention moves the
answer toward the donor. Replacing only the attention score is small and unstable. This is not
yet an identified circuit because the synthetic control authority was not reliably solved by
the native model.

## The exact Theseus contraction

The residual vector is $x_{\ell,t}\in\mathbb{R}^{1152}$. Each attention block has nine heads,
and each head has width 128. For $h\in\{3,7\}$ at layer 8, the unnormalized attention score from
the final query position $q$ to a source position $j$ is

$$
s_h(q,j)=
\frac{\langle \widehat Q_hx_{8,q},\widehat K_hx_{8,j}\rangle}{128}
\frac{\langle \widehat Q'_hx_{8,q},\widehat K'_hx_{8,j}\rangle}{128}.
$$

The hats include the model's per-head normalization and rotary position operation. Thus the
score is the product of two bilinear forms after normalization. The attention pattern is the
causal masked softmax $p_h(q,j)=\operatorname{softmax}_j s_h(q,j)$. The exact contribution from
source $j$ is

$$
t_h(q,j)=p_h(q,j)W_{O,h}
\left((1-\lambda_8)V_hx_{8,j}+\lambda_8 V_hx_{0,j}\right)
\in\mathbb{R}^{1152}.
$$

The second term is the cached early value. Define

$$
u_h(n,j)=W_{O,h}\lambda_8V_hx_{0,j}(n),
\qquad
U_c(n)=\sum_{h\in\{3,7\}}p_h(c,j_n)u_h(n,j_n),
$$

where $n$ is the visible final number and $c$ is a recipient context, including its format,
query, and other source tokens. The shared-payload/private-router hypothesis says the reusable
object is the family $u_h(n,j_n)$ (or an operational equivalence class of it), while the
recipient supplies $p_h(c,j_n)$. It does **not** say that the complete output of head 3 or head 7
is one semantic unit.

Internal changes of coordinates in a contracted $V/O$ or $Q/K$ bond are gauges when the inverse
change is made on the paired map. They change the coordinates of $u_h$ but not $t_h$, the logits,
or an exact installed-term intervention. Therefore raw vector similarity is not the correct
criterion for saying that two payloads are the same.

## A gauge-invariant definition of “the same computation”

Let $P$ be a set of candidate payload values, $C$ a set of validated recipient contexts, and
$A$ a set of registered downstream outcomes. For payload $z\in P$, define its causal-response
fingerprint

$$
R(z)_{c,a}=
Y_a\!\left(\operatorname{do}[u(c)\leftarrow z]\right)
-Y_a\!\left(\operatorname{do}[u(c)\leftarrow u_{\mathrm{native}}(c)]\right).
$$

$Y_a$ includes answer-minus-foil margin and answer CE for the target tasks, plus the same
quantities on independently capable unrelated tasks. Two payloads are equivalent on the tested
intervention algebra when

$$
z\sim_{C,A}z' \quad\Longleftrightarrow\quad
R(z)_{c,a}=R(z')_{c,a}\quad\text{for every }(c,a)\in C\times A,
$$

after applying the registered high-level relabeling, such as mapping the number 12 in one
surface format to the number 12 in another. In finite precision, equality becomes a frozen
task-scale tolerance plus directional tests. This definition is invariant to internal gauge
changes because it depends only on exact interventions and downstream behavior.

This is the interaction-determined basis the user proposed: components are grouped when all
validated downstream readers treat them as the same variable, and split when some reader can
distinguish them.

## Closest exact mathematics

### Constructive causal abstraction

Geiger et al.'s constructive causal abstraction is the closest direct formalism. A low-level
intervention realizes a high-level intervention when the intervention-and-readout diagram
commutes. Their value-merge operation explicitly permits different low-level values to become
one high-level value when the high-level causal model cannot distinguish them. Interchange
interventions are the operational test. See
[Geiger et al., *Causal Abstraction*, JMLR 2025](https://jmlr.org/papers/volume26/23-0058/23-0058.pdf).

Object mapping:

- low-level variable: the exact cached H3/H7 contribution before recipient-specific scaling;
- low-level intervention: replace that term using a donor with the same semantic number in a
  different format, or an opposite number under a matched recipient;
- high-level variable: the visible number value $n$;
- high-level contexts: list successor, digit successor, and capable unrelated controls;
- high-level output: registered answer/foil margins and CE;
- abstraction condition: cross-format and within-format swaps produce the same high-level
  intervention response, while unrelated behaviors are preserved.

For a finite table, checking a proposed value merge costs $O(|P||C||A|)$ model outcomes once the
interventions are available. The guarantee is exact only for the declared intervention algebra.
It does not infer the right counterfactual dataset, and a finite test cannot prove equality on
all language contexts. Held-out templates, values, and independently authored tasks are therefore
part of identification rather than optional decoration.

### Minimal realization and Hankel equivalence

Weighted-automaton realization gives a stronger theorem under stronger assumptions. For a
series $f$, the Hankel matrix $H_f(r,c)=f(rc)$ has rank equal to the number of states in a minimal
linear weighted-automaton realization; minimal realizations are equivalent up to an invertible
change of state coordinates. See
[Kiefer, *Notes on Equivalence and Minimization of Weighted Automata* (2020)](https://arxiv.org/abs/2009.01217)
and [Balle, Carreras, and Luque, *A Canonical Form for Weighted Automata* (2015)](https://arxiv.org/abs/1501.06841).

The useful part here is the observability principle: two states are distinct only if some legal
continuation gives a different output. Our $R(z)$ table is an interventional analogue of rows of
an observation/Hankel matrix.

The theorem does **not** exactly solve the current model. Theseus has RMS normalization, softmax,
quadratic MLPs, context-dependent routing, and finite semantic interventions rather than a
time-homogeneous linear transition indexed by a free monoid. There is no established
concatenation law $f(rc)$ for our intervention table. Consequently Hankel rank or SVD would again
be only a probe/compression basis. The exact consequence we retain is row equivalence under
validated downstream tests, not rank truncation.

## Executable consequence and opposing predictions

The next valid experiment should reuse independently established, natively capable authorities
rather than repair the failed synthetic rows after seeing outcomes. It will construct a response
tensor

$$
\mathcal R[	ext{donor format},\text{recipient task},\text{number},
\text{factor},\text{outcome}],
$$

where factor is cached value, attention score, or their exact joint term. Target outcomes are
margin and CE; unrelated-task outcomes use the same quantities. No factorization or rank cutoff
is part of the verdict.

The frozen opposing accounts should be:

1. **Shared payload, private routing.** On held-out numbers and text templates, a same-number
   cached payload imported across formats is behaviorally equivalent to the within-format cached
   payload, while importing the score is not. The cached intervention changes both successor
   tasks in the registered direction and preserves independently capable copy/non-successor
   controls.
2. **Broad numeral/copy path.** Cross-format cached transfer remains strong on the successor
   tasks but materially changes at least one independently capable copy or non-successor task.
3. **Matched-dataset artifact.** Cross-format equivalence disappears when whole validated task
   authorities are imported without selecting rows using v2/v3 outcomes.
4. **Whole-head artifact.** Cached and score replacements cannot be separated once the test uses
   fresh capable authorities; the previous split does not reproduce.

Evidence for account 1 would define a gauge-invariant shared subcomputation below the native-head
boundary and identify recipient-specific routing as a separate component. Accounts 2--4 would
preserve an informative null and prevent premature grouping.

## Route decision

This response-equivalence test is higher information than a weight SAE, Tucker decomposition, or
rank screen because it can directly change the grouping/splitting, OOD, selective manipulation,
reuse, and stable-identification ledgers. It is also cheaper than searching a rotated basis: the
exact cached and score terms already exist in the model's contraction. The immediate engineering
step is to bind two explicitly separate evidence legs. The v2/v3 matched target endpoints are
natively capable and are the only current authority with the same visible numerals in list and
digit formats. The canonical R567 list data use numerals 21--54, whereas its digit sequences use
8--19, so pairing them by ordinal would not test identity of the same numeral payload. R567 can
instead supply its independently capable copy and $+2$ families wholesale for the
collateral/removal leg. The target-equivalence and collateral legs must be reported separately
rather than presented as one homogeneous row authority. A within-behavior action-router
factorial on R567 can test structural reuse, but not absolute cross-format numeral identity. If
these two evidence legs cannot be bound without outcome-selected row edits, this route stops as
a descriptive hypothesis and the next test moves to a different validated circuit pair.
