# Uneven table-rank allocation — preregistration

**Frozen before running real allocations or evaluation:** 2026-08-29 08:05 UTC

## Question

Can the settled 36-site context-free compiler improve its CE/storage frontier by
giving different table ranks to different sites, while retaining a standalone learned
fallback map at every site?

This is distinct from the older experiment that chose six modules to remain native.
Every site here remains compiled and context-free; only its table and fallback-map
ranks change.

## Frozen construction

- Fit corpus: the existing 96-row `skip80` role with exactly 5,419 covered token IDs.
- Evaluation roles: existing `skip7000`, `skip11000`, and `skip1200` discovery roles.
  These roles are spent; the result is discovery-only.
- Per-site centered table SVD is fitted only on the covered-token table.
- Allowed table ranks are multiples of 64 from 64 through 1152. The rank floor of 64
  prevents the retracted “all MLPs are inert” claim from forcing zero-capacity sites.
- The learned embedding-to-row map rank at a site is $\min(r,512)$, matching the
  previously measured rank law.
- Literal site price is

  $$
  C(r)=r(5419+1152)+2(1152)+2(1152)\min(r,512).
  $$

- Total price cannot exceed the uniform rank-512/map-512 price of 163.6669 million
  stored real values.
- The primary allocation maximizes the sum, across sites, of the fraction of centered
  table energy captured. Dynamic programming is deterministic; ties prefer lower cost
  and then lexicographically smaller ranks.
- A raw-energy allocation is reported as a non-promotive diagnostic.
- A type-preserving shifted null cyclically moves the primary MLP ranks among MLP
  sites and attention ranks among attention sites. It preserves the exact rank
  multiset and price while breaking rank/site pairing.

No S1897/S1898 evaluation statistic selects a rank. Those results motivate the
question only.

## Predictions

1. The primary normalized-energy allocation improves all-position CE by at least
   `0.005` nat over uniform rank 512 on all three roles.
2. It improves by at least `0.005` nat over the shifted null on all three roles.
3. Uniform rank-512/map-512 reproduces `5.98100 / 5.94957 / 5.96977` all-position CE
   within `0.002`, coverage remains 5,419, and every arm stays within the exact budget.
4. Covered and uncovered CE are reported separately. No claim may be rescued by an
   aggregate improvement confined to already memorized covered tokens.

Failure prunes fit-table spectral energy as an allocator. It does not prune uneven
allocation based on suffix gradients, balanced causal ports, or separately frozen
causal sensitivities.

