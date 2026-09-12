# A compact executable branch of the regional shared component

12 September2026,14:51UTC. The whole shared component has a reproducible regional
removal effect. Its dominant private consumer can now be executed as a separate
666,656-scalar package, given explicit query/source inputs. This is local
extraction, not a token-to-logit circuit or completion of the four-property goal.

## What changed the decision

Approximating all consumers' normalization was unsuccessful. Constants, spectral
readers with an isotropic remainder, and kernel-preserving truncations at32/64
missed native removal-effect preservation. The last comparison retained64
directions but still gave19.3–24.2%effect error; its full128 control passed.
See [kernel-preserving result](REGIONAL_TRUNCATED_NORMALIZERS_V1_RESULT.json).
Preserving native null directions fixes one mathematical flaw but is insufficient
for these behavioral bars. Further rank variants were demoted at the
[hourly review](HOURLY_STRATEGIC_REVIEW_2026-09-12_1439.md).

Instead, the [weight-only consumer Gram](SHARED_PRIVATE_CONSUMER_GRAM_V1_RESULT.json)
computes inner products of the nine private query-dependent component functions,
including cross-head terms. For cubic source-atom Gram G, dual rows
B=(G^{-1})[-2:,:], selected source Gram S=G[-2:,-2:], and full cross-head
query/output kernel K, the exact reference-normalized coefficient Gram is

$$
J_{hk}=\sum_{i,j=1}^{2}\sum_{r,t=1}^{16}
S_{ij}B_{ir}B_{jt}K_{hr,kt}.
$$

For omitted heads O, coefficient error squared is
$\sum_{h,k\in O}J_{hk}/\sum_{h,k}J_{hk}$. Thus cross-head cancellation is counted.
At fixed relative positions0,1,2,4,8,16, head2 carries99.71–99.83%of diagonal
energy; head4 is next. This selected consumers before native effect measurement.
Reference-normalized coefficient dominance alone is not a causal claim.

The [native selection test](REGIONAL_PRIVATE_CONSUMERS_V1_RESULT.json) retains
the exact norms. On48 previously frozen geographic prompts, keeping only head2's
shared-component branch gives3.03–3.92%write error and0.404–0.434%removal-effect
error. Heads2+4 give0.173–0.272%effect error. Both pass the10%bar in everyfamily.
Full=head2+remainder write partition error is1.50e-16. These interventions do
not remove entire native heads: other computations within those heads remain.
The native test took4.06seconds after the shared runner finished a peer's job.

## Explicit executable program

Inputs are query $q\in\mathbb R^{1152}$, source
$s=(x,x_{\mathrm{first}})\in\mathbb R^{2304}$ and the relative rotary matrix R.
For native use these are the normalized states at the declared attention boundary.
Their generators are not part of this package.

Keep the four original head2 Q/K maps. They supply query projections, current-
source key projections and the exact four head-normalization scalars. The two
shared cubic functions are a common quadratic parent times separate child readers:

$$
p(s)=(a^\top s)(b^\top s),\qquad
z_j(s)=p(s)c_j^\top s,\qquad t_r(s)=\sum_jz_j(s)B_{jr}.
$$

Precompute K1/K2 applied to all16atoms' three source vectors, and OV applied to
their signed current/first-value combinations. No full cubic coefficient tensor
is materialized. With atom factors $u_{ri}$ and six permutations $\pi$,

$$
w(q,s)=g(q,x)\sum_{r=1}^{16}t_r(s)\frac16
\sum_{\pi\in S_3}
[(Q_1q)^\top R K_1u_{r,\pi(1)}]
[(Q_2q)^\top R K_2u_{r,\pi(2)}]
OVu_{r,\pi(3)}.
$$

Key maps here read the current-source1152coordinates, with implicit zeros on
the first-value coordinates. V includes the actual signed native value mixture.
The gate g uses native epsilon in all four mean-square norms and the128² score
denominator. The earlier reference scale g0 cancels between folded writers and
the gate ratio; it is not an extra fitted parameter. The inverse of G is folded
once into B and is not solved during execution.

[Compiler/executor](compiled_shared_head_v1.py) and
[identity check](check_compiled_shared_head_v1.py) use the existing factor algebra.
Six relative-position checks on16 independent real query/source pairs agree
within4.64e-15. These probes establish compilation equivalence to the selected
branch, not new natural-text or native integration evidence. The selected branch's
native removal test and this compilation control are distinct receipts.

## Literal price and remaining requirements

| Stored part | Scalars |
|---|---:|
| Four source readers | 9,216 |
| Two dual rows over16atoms | 32 |
| Four native head Q/K matrices | 589,824 |
| Two key-projected atom banks | 12,288 |
| Folded physical output writers | 55,296 |
| Total | 666,656 |

The saved artifact is FP64 (about5.33MB of tensor values). Nominal FP32 storage
would be2.67MB; FP32 execution has not been validated. Relative rotary operators,
input-state generation, final MLP/RMS/unembedding and their caches remain outside
this price. Comparing norm maps alone gives589,824 versus5,308,416 scalars for
all nine consumers; that is not a whole-model saving.

The full shared component has fresh-context removal and narrow control evidence.
Its head2 approximation preserves measured removal effects, but broader selective
manipulations, changed contexts and combination with other extracted components
still require validation. The earlier approximate-routing cancellation failure
is not repaired by this result. Next integration should execute this exact package
on native states and compare both writes and interventions, before claiming it
replaces the existing component implementation. Upstream closure remains the
largest unresolved extraction dependency.
