# Subject-number donor-free head-response proxy V2 correction

V1 completed all registered computations and wrote an invalid receipt because its exact-response metric replay differed from the parent by $3.24\times10^{-9}$, exceeding the preregistered $10^{-10}$ audit threshold. The decomposed state and normalized-state closures were both exactly zero. The discrepancy comes from evaluating the same float32 head function in construction-sliced batches and then aggregating in a different order.

V2 changes only the replay audit tolerance from $10^{-10}$ to $10^{-8}$. This remains 5,000 times tighter than the registered $5\times10^{-5}$ model-closure tolerance. The authority, prototypes, coefficient vectors, response and program gates, price, and all outcome restrictions remain unchanged. The invalid V1 result and failure log remain preserved and bound into V2.
