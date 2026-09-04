# Task 14 head-11.3 causal projector — execution addendum

Frozen at 2026-09-04 16:45 UTC, before any projector fit or inner-SELECT score. This addendum closes constants and corrects prices
that the 16:14 preregistration left underspecified. It does not change the scientific success bars, ranks, data split, or outer
validation boundary.

## Fit objective

For an answer-changing relation $i$, retain the preregistered finite complete-head and projected effects

$$
E_{h,i}=s_{h,i}-s_{b,i},\qquad E_{U,i}=s_{U,i}-s_{b,i},\qquad r_i=\frac{E_{U,i}}{E_{h,i}}-1.
$$

Every $E_{h,i}$ must be finite and greater than $10^{-6}$. The robust target loss uses the Huber function with transition
$\delta=0.5$,

$$
\rho_{0.5}(r)=
\begin{cases}
\tfrac12r^2,& |r|\le 0.5,\\
0.5\left(|r|-0.25\right),& |r|>0.5.
\end{cases}
$$

Let $\tau$ be the detached median of the positive FIT-target $E_h$ values. For a same-answer control $j$, let

$$
c_j=\frac{s_{U,j}-s_{b,j}}{\tau}.
$$

The literal objective is

$$
L=L_{\rm target}+L_{\rm control},
$$

where $L_{\rm target}$ is the mean of $\rho_{0.5}(r_i)$ after first averaging within each exact target cell, and
$L_{\rm control}$ is the mean of $c_j^2$ after first averaging within each exact control cell. Both coefficients are exactly 1.0.
The full-vocabulary control score is

$$
\frac{\sqrt{|V|^{-1}\sum_{v\in V}(\ell_{U,j,v}-\ell_{b,j,v})^2}}{\tau},
$$

using the post-softcap logits and all 50,304 output coordinates. The same $\tau$ normalizes answer-margin control movement and the
complete-head control comparator. No normalizer receives gradients.

## Optimizer and deterministic batches

Use Adam on only the unconstrained parameters underlying PyTorch's built-in Householder orthogonal parametrization:

$$
\eta_t=0.03\,\frac{1+\cos(\pi t/99)}2,\qquad t=0,\ldots,99,
$$

with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, and zero weight decay. Model parameters remain frozen.

Each 32-relation update contains 16 target and 16 control draws. A draw chooses an exact cell uniformly and then a relation within
that cell uniformly, with replacement. The entire schedule is generated on CPU before model execution using seed

$$
14114000+100k+q+10000p,
$$

where $k$ is rank, $q$ is start number, and $p=0$ for ordinary fits or $1+$ the permutation-null ID. This is an unbiased stochastic
estimate of the registered equal-cell objective and prevents large cells from dominating.

For permutation-null fit $p\in\{0,1\}$, controls are unchanged. Within every target cell, a deterministic SHA-256 order assigns the
desired normalized target response $+1$ to $\lceil n/2\rceil$ rows and $-1$ to the rest, using label
`task14-head11.3-permutation|p|record_id`; the second null reverses that order. The donor state and denominator stay unchanged. This
tests whether the optimizer actually follows the target labels; it is not a semantic donor remapping.

## Corrected execution price

The earlier primary price omitted some setup and random-control scoring. With batch size 32 and shared endpoint caches, the corrected
primary ceiling is:

- 1,199 forwards;
- 902 backwards;
- 37,491 example evaluations;
- 141,824 raw tensor bytes for nine float32 fitted frames plus the float64 $128\times128$ analytic operator.

If rank 8 is legally opened, add 395 forwards, 300 backwards, 12,355 example evaluations, and 12,288 fitted-frame bytes. If a rank is
provisionally selected, the two confirmation and two permutation fits together add 420 forwards, 400 backwards, 13,380 evaluations,
and $2,048k$ fitted-frame bytes. These are ceilings, not targets to consume. A backend must report actual counts and refuse to exceed
the compatible path ceiling.

## Leakage and stopping rules unchanged

Program A may consume only the committed DISCOVERY endpoint shard with file SHA
`1e3b9a204c08a9c6af4ea7f5668abba719fd1943a8a7e7df0dc488f3183f4e1b`. It cannot read the full token-bearing authority or derive
VALIDATION tokens. Program B remains unopened unless Program A passes exactly. Failure of the fit-health gates is invalid
instrumentation; failure of every legal healthy projector is the registered small-linear-subspace null.
