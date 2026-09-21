**Shared quadratic features help compact restricted quartics, with an important scaling caveat**

All five planted multi-output controls recover their supplied shared widths below1e-9 in all three starts. Exact core fitting in the planted span replays below1.3e-15. This verifies the common-bank search and conditional core solver, not feature identifiability or general convergence.

The native follow-up jointly fits four outputs in each of16saved five-input contexts. It preserves native relative output scales. Results use the best exact coefficient objective of two starts:

| Shared quadratic features | Median relative coefficient error | Worst error | Stored coefficients | Distinct nonlinear products |
|---|---:|---:|---:|---:|
| 4 | 2.568% | 13.091% |100|25|
| 6 | 0.989% | 5.562% |174|36|
| 8 | 0.289% | 1.420% |264|51|

Both registered all-context criteria fail: width4below5% and width8below1%. Each graph's actual two-stage polynomial execution agrees with its coefficient representation below1e-10. No program is adopted or exported for native normalized evaluation.

The computation first forms15shared quadratic monomials, then r linear combinations of those monomials, then r(r+1)/2distinct products reused by four output readouts. Storage is15r+4r(r+1)/2. Coefficient multiplications and additions still cost arithmetic; nonlinear-product count is not total compute. A dense quartic coefficient representation has280values in this four-output restriction. These comparisons do not price or replace the full model's1152-dimensional computation.

**Red-team successor: is the apparent freedom an artifact of the restriction?**

For a fixed15-feature quadratic dictionary, there are120symmetric root coefficients per output. The CPU audit explicitly symmetrizes all24input permutations and computes the rank of the root-to-quartic coefficient map. Random orthonormal dictionaries give:

| Input dimension | Rank | Nullity |
|---|---:|---:|
|5|70|50|
|6|120|0|
|8|120|0|
|12|120|0|

At dimension5the dictionary spans all quadratic monomials, and50directions preserve the polynomial exactly. With the same dictionary size in the larger tested spaces, that freedom disappears. This is a numerical control at one seed per dimension, not a generic-rank theorem or a measurement of trained model structure.

Therefore successful affine Gram search on amplitude restrictions cannot be assumed to transfer to full-dimensional learned banks. Before investing in large-scale equivalent-representative search, measure exact or approximate dependencies among products of the actual quadratic features. If that map is well-conditioned and injective, this particular freedom is absent within the frozen dictionary; changing the feature dictionary remains a separate possibility.

The result advances a candidate decomposition initialization and identifies its scope. It supplies no fresh OOD, semantic, selective-removal, or extraction evidence. The full circuit goal remains open.

[Toy controls](SHARED_QUARTIC_GRAM_TOYS_V1.json) · [Native results](NATIVE_SHARED_QUARTIC_GRAM_V1.json) · [Dimensional audit](GRAM_NULLSPACE_SCALING_V1.json) · [Plan](SHARED_QUARTIC_GRAM_PLAN_V1.md).
