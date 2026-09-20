# Shared five-source feature plane: failed reuse screen

The preceding per-context, per-output spectral rank-two response passes existing finite-edit tests with68 coefficients per context. We tested the stronger claim that one fixed pair of source features can serve all contexts and all four outputs.

Let H[c,o,i,j] be the analytic five-source Hessian. On near_greeted and outside_called only, form M[i,k]=sum(c,o,j) H[c,o,i,j]H[c,o,k,j] and take its leading two eigenvectors P. Shared features s=P^T a feed context/output-specific symmetric2x2 cores P^T H P; exact five-coordinate gradients remain. This is a shared-input Tucker baseline, not HT or a complete extracted circuit. Native context and derivative production remains necessary.

Evaluate on yesterday_helped and earlier_noticed, which were excluded from coefficient fitting but already opened in earlier work. Combine the full15 single/pair directions and existing unit-B/modal-null edits. Gates are10% number error and5% modal error normalized by each cell's number effect.

| Candidate | Shared values | Values/context | Worst heldout number error | Worst heldout modal error | Verdict |
|---|---:|---:|---:|---:|---|
|One common rank2 plane|10|32|17.455%|2.262%|Fail|
|Separate rank2 plane per output|40|32|17.456%|2.373%|Fail|
|Common plane plus3 residual source pairs|10 plus6 indices|44|11.860%|1.590%|Fail|
|Full dense quadratic|0|80|3.344%|0.686%|Pass|

Three seeded random rank-two planes fail with28.00–56.84% heldout number error. The common plane is therefore informative but insufficient under the gate. Separate planes fitted to the two calibration templates have principal cosines0.99962 and0.98581: stable subspaces can still omit essential behavior. This does not identify individual basis vectors, which retain rotational freedom.

The sparse extension selects exactly three residual entries by calibration squared Frobenius energy, counting offdiagonal entries twice. Chosen pairs are(3,3),(2,3),(4,4): MLP8 self, middle-writes/MLP8 cross, and MLP10 self. It stores residual corrections on those symmetric entries in addition to the shared plane. Three matched random residual supports fail with17.46–18.29% heldout number error. The selected support improves heldout prediction but still fails; calibration worst error18.89% also fails.

Full five-dimensional projection and all15 residual corrections reproduce H exactly, ruling out elementary projection/axis/symmetry mistakes. Output untying does not rescue the number failure. These are negative results for this particular coefficient-energy construction, not lower bounds on all shared two-feature circuits: the finite behavioral metric differs from coefficient Frobenius error, and sequential projection plus residual replacement is not a joint optimum over the combined dictionary. A joint refit or task-weighted metric could change the answer and must be tested without treating already opened outcomes as fresh.

Executors: audit_shared_five_source_plane.py and audit_shared_five_source_sparse_residual.py. Primary receipts: SHARED_FIVE_SOURCE_PLANE_V1_RESULT.json and SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json. A separately executed failure-localization receipt preserves the worst heldout sparse cells without changing support. The appropriate next inference is that context-specific rank-two success does not establish shared feature reuse. Native source generators, prospective OOD evaluation and selective manipulation remain open.
