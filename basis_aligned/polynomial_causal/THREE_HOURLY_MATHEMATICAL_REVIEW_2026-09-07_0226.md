# Three-hour mathematical tensor-network review — 2026-09-07 02:26 UTC

## Decision

The complete downstream response lattice is exactly a vector-valued pseudo-Boolean function. Its
Möbius transform gives a unique, gauge-aware interaction algebra over circuit sites; no fitted rank
or approximate tensor decomposition is needed. Applying that transform to the 1,024 executed arms
shows that `99.9994%` of nonconstant squared coefficient mass is degree one and only
`5.49e-6` is degree two. This does **not** say the native transformer is additive. It says that once
the serial modules' complete live/base responses are cached and installed, their endpoint program is
almost additive. The three-site program MLP13 + MLP15 + MLP16 plus the direct residual background is
therefore a mathematically coherent response program, but it must be frozen on fresh text and then
replaced by activation-conditioned weight computations before adoption.

## Exact current object

The model has residual width `d=1152`, 18 blocks, nine attention heads per block, head width 128,
and vocabulary size 50,257. Let the already-identified upstream writer intervention be

\[
 W=\{\mathrm{L7H8},\mathrm{L9H1},\mathrm{L9H4},\mathrm{MLP7},\mathrm{MLP9}\}.
\]

For each of 28 temporal/is-was base/donor rows, replaying the donor responses of `W` defines a live
trajectory. The downstream candidate set is

\[
 P=(\mathrm{L13H6},\mathrm{L15H1},\mathrm{L15H5},\mathrm{L17H2},
     \mathrm{MLP12},\ldots,\mathrm{MLP17}),\qquad |P|=10.
\]

It is fixed by the prospective rule “absolute pooled signed necessity at least `.005` in behavior or
either canonical mode” from the complete 60-site atlas. At every one of all 60 downstream sites, a
binary response intervention installs the cached live value for a selected member of `P` and the
cached native-base value otherwise. For mask `z in {0,1}^10`, row `r`, and output index
`a in {behavior, mode 1, mode 2}`, define

\[
 Y_{zra}=L_a\!\left(F_r(z;W) - F_r(0;0)\right).
\]

`L_behavior` is the donor-answer minus donor-foil logit margin. `L_1,L_2` contract final residual
state with the two physical canonical reader covectors. The full truth-table tensor has shape
`1024 x 28 x 3`; scoring separates the two tasks, producing six vector cells. The model inside
`F` remains nonlinear because of RMS normalization, attention normalization, MLP products, and the
output nonlinearity. Only the binary endpoint function is claimed to be pseudo-Boolean.

The tied parameters are the transformer's literal weights and scalars. The installed response values
are row-specific cached tensors, not learned parameters. Permuting site labels merely permutes Boolean
indices. A canonical reader change `Q -> QG`, `G in O(2)`, rotates the last two output coordinates;
the joint Frobenius norm of their interaction tensor is invariant, while individual named-mode
coefficients require the already-frozen physical gauge.

The executed price is 1,036 model forwards and 29,008 example evaluations, with zero fitting,
updates, or backwards passes. Exhaustive subset search is exact at `2^10=1024`; this is preferable to
a heuristic greedy path at the present size.

## Exact Möbius representation

Every set function on the Boolean lattice has a unique Möbius inverse. Coordinatewise here,

\[
 M_{Sra}=\sum_{T\subseteq S}(-1)^{|S|-|T|}Y_{Tra},\qquad
 Y_{Sra}=\sum_{T\subseteq S}M_{Tra}.
\]

Thus `M_{i}` is a complete-response main effect, `M_{ij}` is the response interaction not explained
by either singleton, and higher-order terms are exact conditional interactions. This is the Boolean
lattice specialization of Rota's incidence-algebra inversion
([Rota 1964](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf)). The pseudo-Boolean
literature states the corresponding unique multilinear polynomial representation and explains why
general subset optimization is hard
([Boros and Hammer 2002](https://citeseerx.ist.psu.edu/document?doi=2a036b38d574afedf2de277ee3c7f38dd9066675&repid=rep1&type=pdf)).
Set-function interaction transforms provide related invertible representations, but do not change
the exact Möbius coefficients used here
([Grabisch, Marichal, and Roubens 2000](https://pubsonline.informs.org/doi/10.1287/moor.25.2.157.12225)).

The in-place subset transform costs `O(n 2^n m)` arithmetic for `m` response coordinates and
`O(2^n m)` storage. A tested CPU helper, `ops/boolean_lattice_mobius.py`, now implements the forward
and inverse transforms, degree energy, and top interaction terms. Exact round-trip error on the
measured six-coordinate table is `2.22e-16`.

For the actual lattice, excluding the constant/direct-path term, degree-energy fractions are:

\[
 E_1=0.9999943567,\quad E_2=5.4890\times10^{-6},\quad
 E_3=9.87\times10^{-8},
\]

with every higher degree individually below `1.6e-8`. The largest main effects are MLP17, MLP15,
MLP16, MLP13, L15H5, and MLP12. The largest pair term, MLP16 x MLP17, has six-cell norm `.00106`,
far below the main-effect norms `.287-.325`. This exactly verifies that cached endpoint responses
compose almost linearly.

## What the theorem does and does not solve

Möbius inversion exactly solves interaction attribution for the finite intervention cube and gives a
canonical answer at the fixed site/output gauges. It also provides an exact falsifier for additive
composition: nonzero higher-order coefficients quantify every failure. It does not identify the
input-dependent function that generates a cached MLP response, prove transfer to unseen rows, or
make the transformer itself degree one. Caching absorbs native serial dependencies into each site's
live value; that is why this algebra can be nearly additive even though the earlier native recomputing
writer factorial was strongly non-additive.

General pseudo-Boolean minimization has no theorem making greedy subset selection exact here; the
objective is neither established monotone nor submodular, and signed MLP17 effects violate an
obvious monotonicity premise. Exhaustive search is the exact algorithm at ten sites. Tensor-train,
CP, or low-rank compression would add gauge and approximation choices without improving the current
causal grouping decision, so they are demoted.

The selected mask `416` is MLP13 + MLP15 + MLP16. It has worst six-cell squared residual `.1073`,
100% behavioral direction, and behavioral signed projection `.990/.991` on temporal/is-was.
The empty downstream mask already preserves much behavior through the direct residual path but fails
temporal mode 2 with residual `1.062`; the three MLP responses restore the missing canonical state.
The full ten-site pool has worst residual `.000640`. Therefore the selected object is explicitly

\[
 \text{direct writer residual} + \Delta\mathrm{MLP13}
 +\Delta\mathrm{MLP15}+\Delta\mathrm{MLP16},
\]

under base-clamped excluded downstream responses, rather than a claim that the three MLPs are the
only native causal modules.

## Executable consequence and next comparison

The immediate consequence is to freeze mask 416 without reselection on a genuinely fresh temporal
and is/was construction population. Measure all six vector cells, full-vocabulary centered effects,
and selective complement/task collateral. Also remove each of the three retained responses from the
fresh program; Möbius near-additivity predicts each decrement from its frozen main-effect tensor.
Failure of the mask or predicted decrement kills stable identification while preserving the current
discovery receipt.

On a pass, replace each cached `Delta MLPk` with its literal bilinear computation

\[
 \Delta m_k=W^{Down}_k\left[(L_kx'_k)\odot(R_kx'_k)
 -(L_kx_k)\odot(R_kx_k)\right],\qquad k\in\{13,15,16\},
\]

using the validated writer-induced inputs. Contract these writes through the physical canonical
reader tensors and finite suffix sensitivity. That test distinguishes a real reusable tensor program
from an input-indexed response cache. It directly advances computational specification,
composition/reuse, fresh prediction, and selective manipulation; coefficient sparsity alone receives
no circuit credit.

For DAS, the mathematical correction is separate: solve a feasibility/Pareto problem that first
requires multi-family target performance above a frozen DIM floor and only then minimizes complement
and cross-task effects. A weighted scalar regularizer has no guarantee against trading away target
sufficiency. This follows the current circuit program rather than displacing it.

Next mathematical review due around **2026-09-07 05:26 UTC**.
