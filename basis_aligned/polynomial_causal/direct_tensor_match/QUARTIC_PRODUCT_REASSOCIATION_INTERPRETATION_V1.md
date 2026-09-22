# Product reassociation exposes additional exact reuse

22 September 2026, 03:50 UTC.

The existing approximate reader-sharing edit made a small additional **exact** graph simplification possible. Reassociating quartic products reduces the two exported candidates to **1,527 and 1,525 distinct multiplications**, versus 1,536 for their original unshared CP evaluations. The untouched native-fit CP dictionaries offer no reduction under this pass. This is a successful second-stage graph rewrite of a fitted candidate, not improved reconstruction of the native model.

## The computation being simplified

Each fitted quartic atom multiplies four learned linear readings of the MLP16 input and writes through a 16-dimensional output vector. A typical original evaluation is

$$
\phi_j(x)=\bigl(a_j(x)b_j(x)\bigr)\bigl(c_j(x)d_j(x)\bigr).
$$

There are 512 atoms. The earlier reader edit made some linear readings exactly shared in the exported candidate. This pass changes only the arrangement and reuse of multiplications; it neither changes those reader vectors nor refits output coefficients.

For example, two terms can contain the same pair without computing it in the original arrangement:

$$
(ac)(bd)+(ae)(bf)=q(cd)+q(ef),\qquad q=ab.
$$

A cubic can also be shared:

$$
abc\,d_i=t\,d_i,\qquad q=ab,\quad t=qc.
$$

The latter is available to a general computation graph, even though a fixed quadratic/quartic tree may duplicate work.

## Compiler and exactness

The pass uses the existing scalar arithmetic DAG implementation. It canonicalizes each atom as a multiset of reader IDs, merges repeated roots with rational output coefficients, and removes exact cancellations. For each root it enumerates binary product trees; four deterministic root orders provide bounded greedy choices about which intermediates to compute first. Existing nodes are reused and charged once. This is not a globally optimal circuit search, nor a detector of arbitrary linear identities between different reader vectors.

Five controls check exact rational polynomial equality: independent products, fourth powers, shared quadratic factors, shared cubic factors, and duplicate cancellation. The quadratic test reduces12to9 products, and the cubic test9to6. Other controls correctly give no extra saving beyond existing fixed-tree interning.

A tie-break refinement avoids gratuitously introducing cubic chains when balanced products cost the same. Both native candidate exports retain nonlinear product depth2 and use only quadratic sharing; cubic reuse is demonstrated by the planted control, not found in these native candidates.

| Candidate | Naive CP products | Already-interned fixed tree | Reassociated DAG | Extra saving versus fixed tree |
| --- | ---: | ---: | ---: | ---: |
| Untouched CP seed1001 | 1,536 | 1,536 | 1,536 | 0 |
| Untouched CP seed1002 | 1,536 | 1,536 | 1,536 | 0 |
| Reader-sharing seed1001 | 1,536 | 1,536 | 1,527 | 9 |
| Reader-sharing seed1002 | 1,536 | 1,535 | 1,525 | 10 |

Seed1002 already saved one product through ordinary fixed-tree interning, so only10of its total11saved products are attributable to reassociation. The new graphs have1,015and1,013quadratic nodes, followed by512quartic roots. Both retain8,192output coefficients and8,176output additions.

The dense1,920-by1,152linear reader bank is unchanged and priced separately; it dominates arithmetic and storage. These logical product savings are only about0.6–0.7%of the naive nonlinear count. No wall-clock speedup or further floating-coefficient compression is asserted.

## Export and relation to circuit evidence

Both reachable graphs are exported in the existing rational JSON review format, with a hash-checked reference to their unchanged reader-bank artifact. Reloaded graphs reproduce their source candidates on256opened text states within $7.5\times10^{-16}$ relative error in float64. This is an algebra/serialization check, not new held-out validation. The JSON representation is not a claim of compact deployment bytes.

The earlier reader edit was approximate relative to its CP parent. This subsequent product rewrite is exact relative to that edited candidate. It inherits the candidate's native approximation and intervention failures. It supplies a concrete example of edits composing usefully: sharing linear features can expose shared higher-degree computations, which a later graph pass can cache. It does not identify their semantics or establish causal adoption.

[Plan and tie-break refinement](QUARTIC_PRODUCT_REASSOCIATION_PLAN_V1.md) · [Controls and native prices](QUARTIC_PRODUCT_REASSOCIATION_V1.json) · [Compiler](quartic_product_reassociation.py) · [Export checks](QUARTIC_PRODUCT_DAG_EXPORT_V1.json) · [Seed1001graph](QUARTIC_PRODUCT_DAG_SEED1001_V1.graph.json) · [Seed1002graph](QUARTIC_PRODUCT_DAG_SEED1002_V1.graph.json).
