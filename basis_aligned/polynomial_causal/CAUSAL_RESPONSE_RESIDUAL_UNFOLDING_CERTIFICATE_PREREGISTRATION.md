# Causal-response residual unfolding certificate — FIT-only preregistration

Status: prospective before inspecting any singular value or owner-conditioned residual
tensor. The already published facts that pooled shared-rank-32 MSE is 0.016047438 and
that every fitted candidate's worst owner pair is `m16 -> m16` motivate this analysis;
they are not outcomes of it.

## Question

Does `m16` fail because it contains genuinely higher-rank residual interaction
structure, or merely because its six sources have unusually large amplitude and are
underweighted by pooled MSE?

For each of the three frozen shared-rank-32 programs, form the exact FIT residual

$$
E^{(q)}=R-\widehat R^{(q)}
$$

and, for owner $g$, the block

$$
E_g^{(q)}\in\mathbb R^{2\times |S_g|\times49\times229}.
$$

No validation or EVAL artifact is in scope. The analysis fails closed rather than
imputing if any FIT response cell is invalid.

## Certificate

For every mode-$j$ unfolding $M_j(E_g)$ and proposed owner-private CP correction of
rank $r$, Eckart--Young gives

$$
\inf_{\operatorname{rank}(A)\le r}\|M_j(E_g)-A\|_F^2
=\sum_{i>r}\sigma_i(M_j(E_g))^2.
$$

A CP-rank-$r$ tensor has matrix rank at most $r$ in every unfolding. Therefore

$$
\inf_{\operatorname{CP-rank}(X)\le r}\|E_g-X\|_F^2
\ge \max_j\sum_{i>r}\sigma_i(M_j(E_g))^2.
$$

This is a lower bound, not another local fit. It can rule out a cheap CP correction;
it cannot prove that a correction attaining the bound exists. We report normalized
lower-bound tails for $r\in\{1,2,4,8,16,32\}$, mode-wise 95% and 99% energy ranks,
stable rank, effective rank, and residual energy per cell. The same spectra are
reported for the raw owner blocks so amplitude and post-fit complexity are separable.

The rank lower bound follows the same unfolding logic used in classical tensor-rank
work, while block-specific alternatives are instances of block-term decomposition:

- Kruskal, *Three-way arrays: rank and uniqueness of trilinear decompositions*
  (1977), <https://doi.org/10.1016/0024-3795(77)90069-6>.
- De Lathauwer, *Decompositions of a Higher-Order Tensor in Block Terms—Part II*
  (2008), <https://doi.org/10.1137/070690729>.

## Frozen decision fork

All residual summaries use the median across the three seeds and preserve the full
range.

1. **Rank-complexity support:** `m16` residual energy per cell is at least 1.5 times
   the next owner *and* its rank-16 unfolding lower-bound tail is at least 1.5 times
   the next owner. The next model may use an asymmetric `m16` block-term rank.
2. **Amplitude/weighting support:** the energy condition holds, but the rank-tail
   ratio is at most 1.25. More CP rank is not licensed; use a prospectively balanced
   owner loss or minimax interface objective.
3. **Mixed/inconclusive:** everything else. Neither topology change is licensed from
   this analysis.

The 1.25--1.5 gap is deliberately an indifference region. A result inside it is not
rounded into a conclusion.

## Cheapest falsifier and toy gate

The asymmetric-rank hypothesis is falsified if `m16` is not simultaneously exceptional
in normalized rank tail and residual energy. Before touching FIT, a planted CPU toy
must show that a rank-2 four-way CP tensor has negligible unfolding tail after rank 2,
while a planted higher-rank tensor has a positive rank-2 lower bound. Source, test,
input artifact hashes, all three seed artifacts, elapsed time, and false validation/EVAL
flags are bound into a create-only receipt.

## What this cannot claim

This analysis cannot select a validation candidate, establish semantic atoms, certify
Kruskal uniqueness, move the whole-model ledger, or establish prediction, composition,
extraction, selective removal, or OOD transport. It only chooses between two
mathematically different explanations of the already observed FIT residual.
