# Correcting the metric recovers the smaller shared program

2026-09-20 20:52 UTC

The same eight-product bank that failed under coefficient loss succeeds after
refitting with exact noncentral Gaussian bank function loss. All three
registered checks pass. No rank or root structure was added.

| Program | Stored coefficients | Products | Fresh64 error | Fresh256 error |
|---|---:|---:|---:|---:|
| Original quartic | 47,312 | 26 | 17.44% | 17.90% |
| Shared eight-product bank, corrected metric | 28,912 | 18 | 17.54% | 17.95% |
| Shared twelve-product bank, corrected metric | 38,144 | 22 | 17.50% | 17.95% |

The selected eight-product bank reduces storage by 38.9% and products by 30.8%.
Its weighted Gaussian bank function error falls from 24.75% to 0.75%; its
component-energy ratio is 0.246, without the earlier large cancellation.
The arithmetic DAG computes each primitive product once and reuses it across
four bank features, which feed ten root products. Replay is below 4e-16.

The smaller program also passes the same native input-response screen:
learned-direction response errors 18.44–19.94%, versus 18.68–20.36% originally.
Random-direction errors remain 48–52%. This is fidelity of the isolated folded
function, not full-model or semantic causal manipulation. The panels are now
reused diagnostics; these refinements are not a new untouched validation set.

Feature identity remains unresolved. Across four width 8 fits, centered bank
function cosines are at least 0.99949, yet the worst matched centered primitive
feature cosine is 0.44. Matching removes sign, scale and permutation, and the
centered comparison excludes mean-dominated similarity. A stable aggregate
function is therefore not evidence for unique elementary features.

The next bounded comparison lowers the bank dictionary to four or six products,
using the corrected metric. Rank ceilings distinguish full-function energy
from centered variation and discourage treating a mean-dominated small rank
as a discovery. More compact functions remain candidates until identity,
selectivity, OOD and full-path integration are supported.

[Refit](../../direct_tensor_match/NONCENTRAL_BANK_REFIT_V1.json),
[response validation](../../direct_tensor_match/DIRECTIONAL_RESPONSE_SHARED_V1.json),
[identity audit](../../direct_tensor_match/SHARED_BANK_IDENTITY_AUDIT_V1.json),
[rank ceilings](../../direct_tensor_match/BANK_FUNCTION_CAPACITY_V1.json),
[next width comparison](../../direct_tensor_match/BANK_WIDTH_FRONTIER_PLAN_V1.md).
