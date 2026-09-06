# Three-hour mathematical circuit review — 2026-09-06 02:30 UTC

## Decision

The most useful new object is an exact finite causal factorization of an attention write into module, head, and source terms. For the
aspectual-anchor task, the result is not a direct token-to-head edge. The temporal preposition changes causal-prefix states at later
tokens; L9H1 and L9H4 integrate principally the `last`, period, and determiner contributions at the final subject query. The next
mathematical question is the depth at which that contextual source-state bank is written.

This complements, rather than replaces, the v12 transparent margin program. Task14/bracket currently has the stronger predictive
release; aspectual anchor currently has the sharper internal causal factorization but lacks prospective construction validation.

## Exact attention factorization

For layer 9, head \(h\), query position \(q\), and source position \(k\), the checkpoint uses two normalized rotary score factors,

\[
s^{(1)}_{hqk}=\frac{\langle Rq_{hq},Rk_{hk}\rangle}{d_h},\qquad
s^{(2)}_{hqk}=\frac{\langle Rq'_{hq},Rk'_{hk}\rangle}{d_h},
\]

and the unnormalized signed bilinear attention coefficient

\[
p_{hqk}=s^{(1)}_{hqk}s^{(2)}_{hqk}.
\]

With the checkpoint's mixed value state \(v_{hk}\), the preprojection head output is exactly

\[
y_{hq}=\sum_{k\le q} c_{hqk},\qquad c_{hqk}=p_{hqk}v_{hk}.
\]

The source-term intervention for a paired base \(b\) and donor \(d\) replaces only one summand,

\[
\widetilde y_{hq}^{(k)}=y^b_{hq}-c^b_{hqk}+c^d_{hqk},
\]

then applies the native projection and exact downstream suffix. A source bank \(K\) replaces all named summands simultaneously,

\[
\widetilde y_{hq}^{(K)}=y^b_{hq}+\sum_{k\in K}(c^d_{hqk}-c^b_{hqk}).
\]

This is a finite intervention on executed tensor terms, not an attribution gradient, attention-probability visualization, or linearized
logit estimate. The manual path matched native scored logits exactly; independent summation reconstructed the native H1/H4 vectors to
maximum absolute error \(3.8\times10^{-6}\).

## Module lattice and exact Shapley attribution

Let \(M=\{A_8,F_8,A_9,F_9\}\) denote the attention and MLP writes of blocks 8 and 9, and let \(G(S)\) be mean signed donor recovery after
exactly replacing modules in \(S\subseteq M\). All 16 subsets were executed. The factorial Shapley value is

\[
\phi_i=\sum_{S\subseteq M\setminus\{i\}}
\frac{|S|!(|M|-|S|-1)!}{|M|!}\bigl(G(S\cup\{i\})-G(S)\bigr).
\]

The measured values are 0.19247 for attention 8, 0.03466 for MLP8, 0.40272 for attention 9, and 0.05345 for MLP9. The complete bank
recovers 0.68330; removing attention 9 reduces it to 0.29606, a loss of 0.38724. Thus attention 9 is dominant both by global factorial
allocation and by the preregistered full-bank removal contrast. These numbers describe endpoint recovery under exact replacement, not
ambient activation variance or weight importance.

## Head and source resolution

Among all nine non-adaptively tested L9 heads, H1 and H4 individually pass the fixed partial-carrier rule. Their joint complete-output
replacement recovers 0.38390 of the task effect with perfect direction in both constructions.

The direct-cue hypothesis fails decisively:

- `since/by` source term: 0.02420 mean recovery, only 0.06305 of the complete pair;
- `last`: 0.08964;
- period: 0.15192;
- self: -0.01091.

The exact multi-source result then gives

\[
\frac{G(\{\text{period},\text{the}\})}{G(\text{full H1/H4})}=0.71995,
\]

\[
\frac{G(\{\text{last},\text{period},\text{the}\})}{G(\text{full H1/H4})}=0.96034,
\]

whereas cue+self retains only 0.03443. Because causal masking prevents a later cue from changing earlier prefix states, and because all
changed source terms from cue through self numerically close to the complete head-pair intervention at BF16 resolution, this supports a
specific computational picture: the cue is transformed into contextual states at the next three positions, which are then read by the
two heads at the final subject.

The conclusion is operational. It does not say the period token has an intrinsic aspect feature, that H1/H4 are necessary under every
intervention, or that source terms are independently additive at the final logits. The intervention is additive before projection; the
suffix remains nonlinear.

## Numerical boundary

Repeated BF16 additions and direct complete-head assignment are mathematically equivalent source closures but not bitwise identical
execution paths. Their maximum scored-logit difference was 0.020951. The initial 1e-4 behavioral closure test was therefore invalid.
The sole corrected run used a predeclared 0.125 BF16-scale logit quantum and preserved all causal bars, rows, heads, and arms. This
distinction should be retained in future instruments: tensor-identity checks can use tight reconstruction tolerances, while two BF16
execution paths need a dtype-aware behavioral tolerance.

## Predictive boundary

A frozen two-construction attempt produced strong, directionally correct intervention movement, including 0.74199 recovery for the
module bank and 0.94341 retention for its H1/H4 reduction, but native A1/A2 capability failed. Those figures are invalid diagnostics,
not transfer evidence. Hence the internal factorization is established only on the valid discovery authority. No new prompt may be
selected because it makes the circuit look good.

## Immediate theorem-to-experiment consequence

For residual boundary \(\ell\) and source bank \(K=\{\text{last},\text{period},\text{the}\}\), define the exact state intervention

\[
x^{(\ell,K)}_{b,k}\leftarrow x^{(\ell)}_{d,k}\quad(k\in K),
\]

with all other positions and the suffix native to the base. Measure signed donor recovery separately in A1 and A2 for boundaries
\(0,\ldots,9\). Register cue-only and final-subject swaps as controls. The earliest boundary crossing 0.50 recovery in both constructions
is the causal onset of a sufficient contextual source bank under this operator.

If that onset precedes the known final-subject onset at resid:10, the next exact path test should replace the source bank and clamp the
L9H1/H4 contribution, yielding a two-factor mediation lattice. If no boundary passes, the source-term factorization remains valid but
its upstream representation must be treated as distributed across positions or modules. Either outcome is more informative than a
generic SVD: it changes the executable circuit graph directly.
