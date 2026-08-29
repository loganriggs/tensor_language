# Selective compilation risk certificate v1

## Object and purpose

S1913 establishes that the deployed compiled program's logits, and hence its top-two
margin, are functions of the current token alone to numerical tolerance. This makes a
token-keyed cascade executable:

$$
F_\tau(c,t)=
\begin{cases}
F_{\mathrm{compiled}}(t),&s(t)\geq\tau,\\
F_{\mathrm{fallback}}(c,t),&s(t)<\tau.
\end{cases}
$$

The score table $s(t)$ and one scalar threshold are charged to literal storage. The
fallback may initially be native for measurement, but no candidate earns standalone
compression credit until the fallback is itself native-free and priced.

## Finite-sample certificate

Thresholds must be frozen from fit-token scores before the calibration role is opened.
For each independent calibration document $d$ and threshold $\tau$, define

$$
a_d(\tau)=\frac{\#\{j:s(t_{dj})\geq\tau\}}{L},\qquad
e_d(\tau)=\frac{\sum_j \mathbf 1[s(t_{dj})\geq\tau]\ell_{dj}}{L},
$$

where every document has the same fixed $L$ scored positions and
$0\leq\ell_{dj}\leq1$. Primary losses are top-one disagreement with native and task
error. Any CE/KL loss must be clipped and rescaled by a bound frozen before calibration;
the certificate then applies only to the clipped quantity.

For $K$ frozen thresholds and failure probability $\delta$, set

$$
\epsilon=\sqrt{\frac{\log(2K/\delta)}{2n}}.
$$

Hoeffding's inequality and a union bound give simultaneous bounds

$$
\mathbb E[e(\tau)]\leq \bar e(\tau)+\epsilon,
\qquad
\mathbb E[a(\tau)]\geq \bar a(\tau)-\epsilon
$$

for every threshold with probability at least $1-\delta$. Therefore

$$
R(\tau)=\frac{\mathbb E[e(\tau)]}{\mathbb E[a(\tau)]}
\leq
\frac{\min(1,\bar e+\epsilon)}{\max(0,\bar a-\epsilon)}.
$$

Because all $K$ bounds hold simultaneously, the highest-coverage passing threshold may
be selected after calibration. Tokens within a document need not be independent.

## Falsification and interpretation

The first real run must use a fit-derived fixed threshold grid, selection-natural for
calibration, and final-natural plus code-OOD only after threshold publication. It fails
if no threshold achieves both the preregistered risk upper bound and minimum accepted
mass, if token-score lookup is incomplete, or if final/OOD violates the frozen gate.

A passing gate would certify when a simpler token program may be used and quantify
expected fallback cost. It would not by itself explain the rejected tokens, prove OOD
exchangeability, or license native fallback as a compressed whole model.

The implementation is `selective_compilation_risk.py`. Its simultaneous bound is a
simple cluster-level specialization of risk-control ideas in
[Angelopoulos et al., *Conformal Risk Control*](https://arxiv.org/abs/2208.02814),
chosen because the selective conditional-risk ratio is not assumed monotone in the
threshold.
