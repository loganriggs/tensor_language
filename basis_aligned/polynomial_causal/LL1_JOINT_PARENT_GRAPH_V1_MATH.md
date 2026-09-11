# Executable graphs with several shared parents

The earlier shared-parent census supplied overlapping proposals. We now have a joint executor that computes every shared reader once and accounts for interactions once. It gives modest storage savings, but the first projection rule loses too much of the fitted function. A discriminating control exposed a specific flaw; the corrected rule reduces that loss by 56–61% at unchanged size. The remaining error bar still fails. These are weight-only graph proposals, not identified behavioral circuits.

## Joint computation

Write an existing LL1 group as a scalar quadratic followed by its output vector:

$$
F_g(x)=c_g x^TQ_gx,
\qquad Q_g=\sum_{j=1}^{16}s_{gj}a_{gj}a_{gj}^T.
$$

Let the rows of $R_g$ be all proposed shared readers consumed by group $g$. These are drawn from one global bank, evaluated once. They need not be orthogonal. Let $V_g$ contain private orthonormal readers perpendicular to their joint span. The executable group is

$$
z=Rx,\qquad y_g=V_gx,
$$

$$
\widehat F_g(x)=c_g\left[
z_g^TH_gz_g+2z_g^TB_gy_g+
\sum_j\lambda_{gj}y_{gj}^2\right].
$$

Here $z_g$ selects this group's parents from $z$. $H_g$ is a symmetric shared/shared interaction matrix; $B_g$ holds shared/private interactions. The private block is diagonalized, preserving its function. Shared/shared products that occur in several groups are also computed once in the global product bank. This is a concrete arithmetic DAG with multiple parents and multiple consumers. It avoids adding overlapping single-parent components, which would double-count interactions.

For a group with $k$ parents we initially keep $16-k$ private readers. Let $P$ project onto the shared span. With a chosen private span, the resulting group is the exact projection of $Q_g$ into their combined span. It is generally an approximation of the old group because the proposed parents are only approximately inside its original input space. No continuous refitting has yet been applied to this graph.

If $R_g^T=UT$ is a thin QR decomposition and $\widetilde V=V_g^T$, the stored cores are

$$
H_g=T^{-1}(U^TQ_gU)T^{-T},\qquad
B_g=T^{-1}U^TQ_g\widetilde V.
$$

This conversion preserves the original shared-reader identities rather than replacing them by independently rotated readers in each group. The executor is checked against an independent signed-square expansion of its full cores.

## Why the first private-space rule failed

V1 selected the strongest eigenvectors of the private/private restriction

$$
(I-P)Q_g(I-P).
$$

This misses private directions whose importance comes from their interaction with a shared reader. For orthogonal unit $u,v$ and

$$
Q=uv^T+vu^T,\qquad P=uu^T,
$$

the private/private restriction is zero, even though the private input $v$ is essential. The new rule selects the leading eigenvectors of

$$
(I-P)Q_g^2(I-P),
$$

which measures the full squared response of $Q_g$ to a private input. In the example it equals $vv^T$. The retained private/private core is then diagonalized, and all shared/private terms retained. This is a proposal rule, not a theorem that it globally minimizes the final tensor projection error at the fixed rank.

The executed mixed-only two-group control gives 0.4163 relative coefficient error with V1 and $2.86\times10^{-16}$ with V2. The zero spectrum makes V1 numerically ambiguous; the result demonstrates a flaw in the rule, not merely a bad optimizer. A separate nonorthogonal, multiple-parent planted example passes the original executor check below $9\times10^{-16}$.

## Native pilot results

Both graphs start from the same frozen, unconverged LL1 pilots. Capture means $1-\|T-\widehat T\|^2/\|T\|^2$, with the entire folded unembedding included in the coefficient metric. The numbers below concern this weight objective, not text accuracy.

| Measurement | Spectral start | Native-product start |
|---|---:|---:|
| Original capture | 11.6604% | 11.6396% |
| V1 graph capture | 11.3485% | 11.2897% |
| V2 graph capture | 11.5391% | 11.4866% |
| V2 reduction of V1 capture loss | 61.12% | 56.28% |
| V2 squared graph change / native energy | 0.001197 | 0.001511 |
| Stored floating coefficients saved | 1.4335% | 1.5214% |
| Linear readers, before → after | 1,024 → 1,008 | 1,024 → 1,007 |
| Variable products, before → after | 1,024 → 1,051 | 1,024 → 1,049 |

Both executors agree with their coefficient representations below $2.2\times10^{-15}$. V1 passes its 1% storage-saving bar but fails the 0.001 coefficient-change and capture-loss bars. V2 passes its registered 25% improvement bar but still fails its 0.001 capture-loss bar. The graph saves dense reader arithmetic while adding variable products; it is not cheaper on every operation count.

Logical graph costs include 131/129 int64 indices and 1,236,418/1,235,315 floating coefficients, versus 1,254,400 old coefficients. V2 records scalar additions and coefficient multiplications separately. Bias, metric-to-physical output conversion, native unembedding, residuals and nonlinear tail are common background, not eliminated costs. Saved graph artifacts include the output whitener and bias. A physical MLP replacement must invert the whitener on the graph output and add the bias; these artifacts alone are not full-model replacements.

## Remaining proposal problem

Each parent individually had at least 0.95 squared membership in a consumer's input space. This does not ensure that their joint span fits there. Two nearly parallel parents can have a difference pointing outside the group. After orthogonalization, the minimum squared membership of their joint span falls to **0.0862 / 0.0615** in the worst groups. Four/two affected groups fall below 0.90. Forcing all those directions into a rank-16 group can consume capacity with the wrong private directions even after the marginal repair.

The next discriminating graph move should test compatible parent sets jointly, or refit shared readers and cores together. Simply lowering an individual-reader cosine threshold or calling these misses an absence of DAG structure would not address this measured failure.

Sources: [V1 executor and proposal](ll1_joint_parent_graph_v1.py), [V1 receipt](LL1_JOINT_PARENT_GRAPH_V1_AUDIT.json), [V2 proposal](ll1_joint_parent_graph_v2.py), [V2 controls and diagnosis](ll1_joint_parent_graph_v2_audit.py), [V2 receipt](LL1_JOINT_PARENT_GRAPH_V2_AUDIT.json). Durable graphs: `LL1_JOINT_PARENT_GRAPH_V1.pt` and `LL1_JOINT_PARENT_GRAPH_V2.pt`.

## Separate longer LL1 fit

The projected spectral arm finished its 1,200-second budget with 11.8891% capture. Numerical checks pass, but its normalized gradient $1.32\times10^{-5}$ and 20-step relative progress $4.37\times10^{-6}$ miss the registered $10^{-7}$ and $10^{-6}$ local-convergence criteria. Exact output elimination has not made this a converged fit. The second start is running; no cross-start verdict is available yet. [Completed spectral receipt](PROJECTED_LL1_CONVERGENCE_V3_SPECTRAL.json).
