# MLP0 token/context tensor-factorial findings

**Executed:** 2026-08-30 03:50 UTC

**Runtime:** 44.69 seconds

**Primary receipt:** `mlp0_token_context_tensor_factorial_discovery.json`

**Receipt SHA-256:** `0d586445478829e29669415a196fead77294a2b060eee4e6cbfff658b7a26010`

## What was computed

Write the normalized MLP0 input as a token-derived part $p$ plus a contextual part
$q$. For the bilinear MLP numerator

$$
T(u,v)=D\big((Lu)\odot(Rv)\big),
$$

the exact output, apart from the native bias and bf16 roundoff, is

$$
T(p,p)+T(p,q)+T(q,p)+T(q,q).
$$

We grouped this into three fixed tensor branches:

$$
\mathrm{TT}=T(p,p),\qquad
\mathrm{X}=T(p,q)+T(q,p),\qquad
\mathrm{CC}=T(q,q).
$$

`TT` is token-by-token computation, `X` is token-by-context interaction, and `CC`
is context-by-context computation. Unlike the earlier TopK program, these are fixed
tensor contractions with no input-dependent discrete router.

For every one of the eight subsets of these three branches, the intervention computed
native MLP0 and subtracted the omitted analytical branches. This is therefore an
oracle-assisted causal census, not yet a cheap replacement. `EMPTY` means native bias
plus numerical residual; it does not mean the mean MLP0 write.

## Main result

The two independent 96-document roles agree on the ordering and sign of every Shapley
contribution. A Shapley contribution is the average CE benefit of adding one branch
over every possible set of the other branches.

| Branch | FIT CE benefit | 95% document interval | SELECT CE benefit | 95% document interval |
|---|---:|---:|---:|---:|
| `CC` | 1.1651 | [1.1142, 1.2165] | 1.1778 | [1.1277, 1.2287] |
| `TT` | 0.9348 | [0.8756, 0.9948] | 0.9281 | [0.8761, 0.9816] |
| `X` | 0.4124 | [0.3759, 0.4494] | 0.4008 | [0.3634, 0.4390] |

All three branches matter. The continuous context quadratic is largest, but the token
quadratic is also large, and the cross branch supplies a substantial conditional
effect. Full MLP0 improves CE over `EMPTY` by 2.5122 nats on FIT and 2.5067 on SELECT.

## Why independent compression will not simply add

The eight-arm factorial exposes interaction dividends. On SELECT:

- `TT` plus `X` has a **+1.7216 nat synergy** beyond their separate effects;
- `TT` plus `CC` has a **-1.1537 nat overlap**;
- `X` plus `CC` has a **-1.0328 nat overlap**;
- the remaining three-way dividend is only +0.0244 nat.

Thus the algebraic decomposition is exact, but downstream CE is nonlinear in the
branches. The large `TT`–`X` synergy says that lexical information and contextual
refinement should probably share a joint interface or dictionary. Separately fitting
three low-MSE compressors and adding them is not justified. The near-zero three-way
remainder is encouraging: pairwise coupling may be sufficient for the next grammar.

The empirical branch writes are also not orthogonal. FIT/SELECT correlations are
stable at approximately

$$
\operatorname{corr}(\mathrm{TT},\mathrm{X})=-0.339,\quad
\operatorname{corr}(\mathrm{TT},\mathrm{CC})=0.408,\quad
\operatorname{corr}(\mathrm{X},\mathrm{CC})=-0.387.
$$

## Numerical checks and claim boundary

- The analytical branch sum matches a direct float32 MLP0 quadratic at relative MSE
  about $3.11\times10^{-13}$ on both roles.
- Its difference from deployed bf16 native output is about $5.48\times10^{-6}$ relative
  MSE and is kept as a numerical residual, not assigned semantic meaning.
- Every arm made exactly 24 forwards per role, 432 attention calls, 24 MLP0 calls, and
  the expected 408 later-MLP calls. No component was silently skipped.
- FINAL was not opened. This result does not establish OOD behavior, lexical classes,
  executable compression, extraction, or selective removal.

## Decision

This closes the immediate decomposition question. MLP0 is neither only a token table
nor only a continuous context map. Its fixed tensor form contains all three substantial
branches, with dominant pairwise interactions. The next MLP0 program should use a
mixture of fixed structures—overlapping lexical/DAG factors for `TT`, continuous
low-rank quadratic factors for `CC`, and shared block terms coupling `TT` to `X`—and
must be optimized and evaluated jointly through downstream CE.

The project now pivots to a panel of behavior circuits. Those terminal readers will
provide identifiable downstream endpoints for deciding which parts of these early
branches are shared, behavior-specific, extractable, and safely removable.
