# Three-hour mathematical review — 2026-08-29 07:50 UTC

## Executive update

Three conclusions change the work queue.

1. **The next compression should be chosen by downstream observability, not local
   reconstruction energy.** Family F exposed the need cleanly: retaining native Down
   has worse local write NRMSE (`0.86957`) but better downstream teacher KL (`0.05772`)
   than refitting Down (NRMSE `0.70275`, KL `0.08476`). A reachable/observable balanced
   port is now implemented and tested on CPU. It is the leading new experiment.
2. **The saved Family-F quadratic programs are already exact product-minimal inside a
   restricted but relevant grammar.** The K256 candidates have exact quadratic product
   rank 256 and the K512 candidates rank 512 under a robust unfolding certificate.
   This prunes exact same-depth CP/HOSVD refactorization of those candidates; it does
   not prune approximation, deeper reuse, or functional equivalence.
3. **Generic token-Hankel and deterministic information-bottleneck ideas remain too
   weak.** The worthwhile version is a finite, intervention-conditional causal quotient
   at a named residual interface. It must predict unseen edit compositions and support
   extraction/removal, or it is only another reconstruction coordinate system.

The live compiled-program work also closed an important interpretation ambiguity.
On positions where both native model and program are wrong, they choose the same wrong
token at `6.10 / 6.24 / 6.50` times a marginal-matched permutation null. A second build
with about three times the covered vocabulary gives `6.23 / 6.18 / 6.49` times. Thus
the program is reproducing model-specific behavior, including mistakes; agreement is
not merely concentrated on easy positions. This strengthens functional extraction
evidence but does not move the strict semantic/removal ledgers.

## Current measured state

| Currency | Strict result |
|---|---:|
| Replaceable write interfaces | 36/36 |
| Original storage with consequence-certified removal | 5.3481% |
| Named causal CE headroom recovered | 10.923% |
| Remaining unnamed causal CE | 4.72714 nat |
| Completed extraction/removal/OOD action cells | 0/68 |

The largest gap is still an **autonomous composable state**. A rank-512 map works much
better when fed a native upstream stream than when fed the recursively compiled stream:
the closed version loses `1.09--1.27` nat, and self-consistent refitting loses about
`5.5` nat. We can imitate many outputs, but we cannot yet carry the sufficient state
through the whole simplified program without native calls.

## Ranked mathematical moves

### 1. Reachable/observable balancing at a causal residual port

**Exact object in bilin18.** At the MLP3 Family-F output—or another named residual
cut—collect two kinds of row vectors:

- reachable vectors $X\in\mathbb R^{n\times1152}$: actual residual-write differences
  caused by selected gates, component removals, transplants, or finite upstream edits;
- observable vectors $O\in\mathbb R^{m\times1152}$: gradients of capped logits, KL/CE,
  or fixed downstream response tests with respect to that same residual cut.

“Reachable” means upstream computation can actually produce the direction.
“Observable” means changing that direction affects a downstream quantity we care
about. PCA uses only the first property; a gradient basis uses only the second.

Normalize

$$
R=\frac{X^\top}{\sqrt n},\qquad Q=\frac{O}{\sqrt m},
$$

and form the cross-response matrix

$$
H=QR.
$$

If $H=U\Sigma V^\top$, define primal and dual state bases

$$
\Phi=RV_r\Sigma_r^{-1/2},\qquad
\Psi=Q^\top U_r\Sigma_r^{-1/2},
$$

so that $\Psi^\top\Phi=I$ and the rank-$r$ physical projection is
$P_r=\Phi\Psi^\top$. The singular values measure directions that are simultaneously
reachable and observable. This is the finite-snapshot analogue of balanced model
reduction: classic linear balancing is due to Moore, with nonlinear snapshot/subspace
extensions by Lall, Marsden and Glavaški and modern state/gradient covariance balancing
in CoBRAS.

**Useful theorem/operational guarantee.** By Eckart--Young,
$\sum_{i>r}\sigma_i^2$ is the minimum squared Frobenius error of a rank-$r$ linear
approximation to the measured cross-response matrix. More importantly for our gauge
problem, under any invertible residual coordinate change $x'=Tx$,

$$
X'=XT^\top,\qquad O'=OT^{-1},
$$

so $H$ and its singular values are invariant, while
$\Phi'=T\Phi$, $\Psi'=T^{-\top}\Psi$, and $P'_r=TP_rT^{-1}$. The score is about the
causal interface, not an arbitrary residual basis.

**Assumptions that can fail.** Bilin18 is nonlinear, RMSNorm changes local geometry,
the chosen edits may not span deployment behavior, and a logit gradient is only an
infinitesimal observable. The classical balanced-truncation $H_\infty$ error bound does
not automatically apply. We must validate finite secants and unseen compositions.

**Prediction beyond reconstruction.** At the same literal rank/storage, balanced modes
should beat write-PCA or local reduced-rank regression on held-out suffix KL, CE, and
finite response prediction. A retained state coordinate should also support a targeted
edit with less collateral than a high-variance but downstream-inert coordinate.

**Cheapest falsifier.** Add a fit-only balanced basis to the already planned fresh
native-Down behavioral-port measurement. At ranks 16/32/64, compare it with PCA and
local RRR at identical price on sealed finite edits and document roles. Prune if it
does not reduce held-out KL/response error, or if any gain disappears under finite
secants or produces CE collateral. No claim should be made from the singular spectrum
alone.

**Work completed now.** `causal_port_balancing.py` implements the construction and
checks support and biorthogonality. Four tests establish nuisance-direction removal,
coordinate-gauge covariance, the response-tail identity, and fail-closed rank handling.
It is implementation and a proof check, not model evidence: the required aligned
Family-F suffix gradients/responses are not present in the saved artifact.

Primary sources: [Moore 1981](https://algos.inesc-id.pt/projects/mor4less/Moore_81.pdf),
[Lall, Marsden and Glavaški 2002](https://www.cds.caltech.edu/~marsden/bib/2002/06-LaMaGl2002/),
and [CoBRAS](https://arxiv.org/abs/2207.14387).

### 2. Exact partially symmetric arithmetic-rank certificates

**Exact object in bilin18.** The saved Family-F MLP3 candidates compute

$$
q(x)=b+\sum_{g=1}^{K}d_g(\ell_g^\top x)(r_g^\top x).
$$

This is a vector-valued quadratic polynomial, or equivalently a partially symmetric
order-three coefficient tensor. Product count is a concrete executable-complexity
measure: it counts scalar multiplications after the linear forms have been produced.

**Theorem/operational definition.** The rank of any matrix unfolding lower-bounds
tensor/CP product rank. For diagonal quadratic evaluation, the correct factors are
$\operatorname{sym}(\ell_g\otimes r_g)$, not arbitrary bilinear factors. If the decoder
columns and these symmetric products are each linearly independent, the output-mode
unfolding has rank $K$; the displayed $K$-product representation gives the matching
upper bound. Therefore the exact product rank is $K$.

**Measured consequence.** Robust float64 Gram certificates give exact rank 256 for
both K256 real-F programs and rank 512 for both K512 programs; the smallest positivity
margin is over $1.9\times10^8$. The Family-A K512 comparator also has rank 512. Full
values and limits are in `QUADRATIC_PRODUCT_RANK_CERTIFICATE_2026-08-29.md`.

**Assumptions that can fail.** This fixes the same depth-two quadratic grammar and exact
equality for every real $x$. It cannot exclude approximation, equality only on the
data manifold, a deeper circuit that reuses products, or a different program with the
same downstream behavior. The native K4608 MLP also cannot be certified above output
dimension 1152 by this unfolding.

**Prediction beyond reconstruction.** The result is a certified negative prediction:
an exact same-depth gate-merging search cannot compress these selected programs at
all. This saves search/GPU time and provides a gauge-invariant executable-cost floor.

**Cheapest falsifier.** The certificate itself would fail if either Gram matrix lost
positive definiteness under a verified error allowance. It did not. The next relevant
test is no longer another exact CP/HOSVD fit; it is whether an approximate or deeper
program beats this count at fixed downstream CE/KL and edit behavior.

Background on tensor rank and bilinear complexity: [JáJá 1979](https://doi.org/10.1137/0208037)
and the [Kolda--Bader tensor decomposition survey](https://www.math.ucdavis.edu/~saito/data/tensor/kolda-bader_tensor-decomp-siamrev.pdf).

### 3. Finite intervention-conditional causal quotient

**Exact object in bilin18.** At a named cut, treat a document context $c$, a physical
edit word $u$ (removal, transplant, or finite residual perturbation), a suffix test $v$,
and output statistic $\phi_k$ as an empirical response table

$$
H[(c,u),(v,k)]
=\phi_k\!\left(G_v(T_u(s_c))\right)-\phi_k\!\left(G_v(s_c)\right).
$$

Here $s_c$ is the actual student state, $T_u$ applies the edit sequence, and $G_v$
runs the suffix. Two states are approximately equivalent only if every registered
action leads to equivalent future response distributions. This is an empirical
Nerode/bisimulation quotient, not a clustering by activation distance.

**Theorem/operational definition.** For a complete rational series, Hankel rank equals
the dimension of a minimal weighted automaton, and shifted Hankel blocks recover its
transitions. For an MDP, exact bisimulation equates states with the same immediate
rewards and transition probabilities into equivalence classes; bisimulation metrics
relax this continuously. Interchange interventions operationalize whether a proposed
high-level variable really realizes the same causal behavior.

**Assumptions that can fail.** The transformer is nonlinear and layer-dependent; the
action/test alphabet will be incomplete; approximate low rank need not close under
composition; and RMSNorm can make finite actions amplitude-dependent. Therefore no
minimal-automaton theorem may be claimed from one SVD.

**Prediction beyond reconstruction.** A valid quotient must predict a sealed two-edit
composition without calling the native intermediate state, and its abstract removal
or state swap must reproduce native logit/KL effects with bounded collateral. This is
directly useful for extraction, selective removal, and OOD transport.

**Cheapest falsifier.** Finish the existing finite L8 to L11 to L14 triangle on the
384 unique-document cache, including matched PCA/RRR, shuffle/null, gauge, and lifecycle
controls. It must predict L8-to-L14 finite responses from composed fitted maps. If that
fails, do not enlarge to a general Hankel model. If it passes, add a 2-by-2 edit diamond
at native Down and test shift consistency.

Primary sources: [weighted automata, tensor networks and spectral learning](https://arxiv.org/abs/2010.10029),
[Ferns, Panangaden and Precup on bisimulation metrics](https://aaai.org/Papers/AAAI/2004/AAAI04-124.pdf),
[interchange intervention training](https://proceedings.mlr.press/v162/geiger22a.html),
and [distributed causal alignment](https://proceedings.mlr.press/v236/geiger24a).

## Near-term fourth move: fixed-projector quadratic closure

A cheap CPU diagnostic can test whether MLP0--2 split into two approximately
independent quadratic subcircuits in a previously frozen causal subspace $P$ and its
complement $Q=I-P$. For quadratic tensor $T$, measure

$$
\epsilon(P)=
\frac{\|T-[P\,T(P,P)+Q\,T(Q,Q)]\|_F^2}{\|T\|_F^2}.
$$

Exact zero means the mixed $P/Q$ Hessian blocks vanish, so the two programs compose as
a direct sum and can in principle be edited separately. The first falsifier is a
Gaussian contraction estimate for MLP1/2 against matched Haar projectors. Reject if
leakage exceeds 0.25 or does not beat two Haar controls across layers and seeds. This
is more useful than unconstrained HOSVD because $P$ is fixed by a causal/downstream
object rather than selected to minimize local tensor error.

## What was pruned, and why

| Mathematical family | Decision now |
|---|---|
| Norm minimization before HOSVD | Closed for the honest reciprocal gate gauge: it only balances factor norms and leaves the folded tensor/HOSVD unchanged to numerical precision. A general hidden GL transform is not a symmetry of elementwise products. |
| Plain CP/HOSVD of saved Family F | Exact lower bound now meets K; only approximate/deeper/causal alternatives remain. |
| One shared output dictionary | Rank-512 shared and typed hierarchies lost to equal-price private maps. Keep only a bounded tight-budget 64/128 question. |
| Generic SAE/dictionary learning | No causal projector passed at the large budget; sparse rotation alone cannot restore discarded predictive directions. Revisit only after a balanced/causal subspace passes. |
| Generic token Hankel or automaton | Existing token-Hankel evidence is OOD and recursive closure failed. Only the finite intervention-conditional form remains. |
| MDL or prequential coding | Useful only as a tie-breaker among programs that already pass CE/KL, composition, and edits. A short code is not evidence of the right abstraction. |
| Deterministic information bottleneck | Mutual information with a deterministic continuous state is ill-behaved or trivial without noise/quantization. Use a successive-refinement rate/distortion curve only after defining causal distortion. |
| Sparse program synthesis | Promising after a causal quotient exists. Constrain a typed DSL to selectors, equality/match tests, value writes, residual addition, and late gates; score causal validity before description length. |
| Approximation certificates | Local Lipschitz/MSE bounds are too loose across RMSNorm/residual composition. First certify measured finite ports; pursue formal downstream bounds only for a passed program. |

## Priority after this review

1. Instrument and run the fresh native-Down port with balanced versus PCA/RRR bases.
2. Preserve the quadratic-rank certificate as a hard floor and stop exact same-depth
   refactor searches for the saved K256/K512 programs.
3. Complete the controlled finite transport triangle; expand to an edit diamond only
   if unseen composition succeeds.
4. Run the fixed-projector MLP1/2 mixed-block CPU diagnostic.
5. Use the new mechanism evidence to test uneven per-site table-rank allocation. The
   frontier to beat is already frozen; allocation must be selected on fit-only roles
   and evaluated in CE/KL at exact literal price.

The review produced real mathematics and tested code, but it did not itself move the
strict explanation ledger. The decisive next result is whether causal balancing turns
the Family-F local-error/downstream-KL reversal into a reproducible matched-price win.

