# Product allocation matters more than simply widening the output basis

21 September 2026, 02:35 UTC. Completes the experiment described in the [in-progress report](research_update_2026-09-21_0228_allocation_in_progress.md).

The wider-output hypothesis failed: spreading a small product budget across more output directions worsened native conditional fidelity. A subsequent adaptive allocation worked better when allowed to assign up to four products to important directions. It nearly matches a uniform1,024-product program using512products, but remains substantially larger than our earlier compact graph and still has high context-only error.

## Controlled uniform comparison

Each candidate uses the original centered operator's output basis and weighted per-output matrix SVDs. It stores the exact native constant and first-order terms around calibration means. These terms alone cost2,654,208 weights and are included below. All512 matrix decompositions and native evaluation finished in approximately521seconds through the managed runner.

| Candidate | Products | Weight coefficients | Context error: FineWeb / code |
|---|---:|---:|---:|
| Earlier compact graph | 512 | 1,291,264 | 57.12% / 42.13% |
| 256directions × rank2 | 512 | 4,128,768 | 63.33% / 59.93% |
| 512directions × rank1 | 512 | 4,423,680 | 73.15% / 67.00% |
| 256directions × rank4 | 1,024 | 5,308,416 | 54.57% / 38.80% |
| 512directions × rank2 | 1,024 | 5,603,328 | 62.94% / 59.71% |

Both registered substantive predictions fail:512×1 does not beat256×2 at matched512products, and512×2 does not beat the earlier compact512-product graph. Mean-expansion and executable-difference controls pass below $4.1\times10^{-15}$ relative error.

The output projection oracle improved when widened, but those new coordinates were too poorly computed with only one or two products each. A good output space is not sufficient; it consumes capacity that must also be allocated to its interactions.

## Adaptive ranks under a fixed product budget

For fixed orthogonal output directions and fixed per-output singular values $\sigma_{gk}$, the weighted error reduction from retaining term $k$ of output $g$ is $\sigma_{gk}^2$. At a fixed number of products, selecting the largest available squared singular values gives the optimal allocation within that factor pool. Each slice keeps a prefix because its singular values decrease.

This is a restricted optimum: it does not optimize the output basis, arbitrary arithmetic DAGs, factor reuse, coefficient storage or native intervention error.

We first allowed at most two terms per output. This retains288output directions but improves retained fitting energy by only0.018% over256×2. Native context error is slightly worse, so its registered superiority prediction fails.

An exploratory second pool also includes the already-computed third and fourth terms for the first256directions. It requires no new SVDs and uses no diagnostic outcomes to fit coefficients. At512products, the allocation becomes:

| Products assigned to a direction | Number of directions |
|---|---:|
| 0 | 316 |
| 1 | 56 |
| 2 | 41 |
| 3 | 22 |
| 4 | 77 |

That is196active output directions and512products. Retained fitting energy increases10.94% over the two-term adaptive pool. The exported mixed program matches an explicit masked-factor computation to relative error $5.48\times10^{-16}$.

## Native result and cost

| Metric | Uniform256×4,1,024products | Mixed adaptive512products |
|---|---:|---:|
| Weight coefficients | 5,308,416 | 4,423,680 |
| FineWeb context error | 54.57% | 55.37% |
| Code context error | 38.80% | 39.55% |
| FineWeb source-only error | 25.28% | 25.67% |
| Code source-only error | 20.30% | 20.69% |
| FineWeb replacement CE added | 0.00500 | 0.00551 |
| Code replacement CE added | 0.02187 | 0.02367 |

The adaptive graph uses half the products and16.7% fewer weight coefficients than this uniform1,024-product graph. Across removal, whole-contribution swap, source-only swap and context-only swap, its largest relative error increase is4.25%. This comparison is exploratory, not a preregistered adoption gate or fresh confirmation.

Output writers for two-or-more-product groups are currently duplicated explicitly in the adaptive export and fully charged. Sharing them could save storage, but that saving is not claimed here. The adaptive graph remains about3.43times the weight count of the earlier compact512-product graph because the exact first-order maps are expensive. Equal product count is not equal total cost.

## What this adds to the research direction

This is a concrete instance of useful structural sparsity: some output directions receive no interactions, while others need several. A single uniform rank can spend products poorly. It also shows why the oracle coverage improvement did not automatically yield a better decomposition.

The next useful comparison should keep this interaction graph fixed while separately evaluating compression or sharing of the expensive first-order terms. Context-only effects cancel those terms algebraically, so their accuracy can be protected while testing the full-function and source-only cost of that additional approximation.

All results use the existing diagnostic panels. No stable semantic meanings, task-selective interventions, cross-module composition or broad external OOD evidence have been established. The absolute conditional errors remain large; the complete circuit goal is unfinished.

Evidence: `MIDPOINT_CENTERED_ALLOCATION_V1.json`, `MIDPOINT_ADAPTIVE_ALLOCATION_NATIVE_V1.json`, `MIDPOINT_ADAPTIVE_MIXED_V1.json`, their executable graph exports and raw native records, and the successor CPU comparison `MIDPOINT_ALLOCATION_TRADEOFF_V1.json` under `direct_tensor_match`.
