# MLP0 × MLP2 interaction geometry V1 — post-outcome analysis lock

## Status and claim boundary

The eight-arm physical outcome is already open.  This is therefore a frozen
**post-outcome diagnostic**, not a prospective scientific preregistration and not a
strict-ledger experiment.  The exact computations and thresholds below are committed
before reading document-level interaction values from the ledger.

Input ledger SHA-256:
`969aa29c58ad2ee860bb0d486a44bcc20792f5c1d966cbb48ddba38f49a8ae0b`.
Input receipt SHA-256:
`22026cd77420e8cf739796e2283782bbe971be1852eaa1996902aaf7e0bab30e`.

## Objects

For document $d$, program $P\in\{FULL,CONTINUE,ROBUST\}$, and C512 intervention
$C$, define per-token document loss and interaction

$$
\ell_d(A)=\frac{NLL_d(A)-NLL_d(NATIVE)}{n_d},
\qquad
i_{P,d}=\ell_d(C+P)-\ell_d(C)-\ell_d(P).
$$

Compute the analogous teacher-KL interaction.  For each interaction vector report:

- mean, standard deviation, median, positive/negative fractions, and a deterministic
  10,000-draw document-bootstrap 95% interval for the mean (seed 2026082943);
- shares of absolute interaction mass in the largest 1, 5, 10, 20, and 25 percent of
  documents;
- Gini coefficient and effective participation
  $(\sum_d|i_d|)^2/\sum_d i_d^2$.

Across the three programs, report pairwise Pearson/Spearman correlations and singular-
value energy fractions for both raw and document-centered $192\times3$ interaction
matrices.  Correlate each interaction with native document NLL, C512 standalone dCE,
the corresponding MLP2 standalone dCE, and composed dCE.  Also analyze the reductions
$i_{FULL}-i_{CONTINUE}$ and $i_{FULL}-i_{ROBUST}$.

## Frozen diagnostic rules

- `diffuse_all_programs`: every CE interaction has effective participation at least
  48 documents and no top-10%-of-documents absolute-mass share above 0.60.
- `shared_document_mode`: minimum pairwise CE-interaction Pearson correlation at least
  0.70 and the first centered singular mode explains at least 70% of energy.
- `simple_difficulty_predictor`: at least one absolute Pearson correlation with native
  NLL or standalone C512 dCE is at least 0.50.
- `robust_reduction_targets_large_full_interactions`: the mean FULL-minus-ROBUST
  reduction is positive and its Pearson correlation with $|i_{FULL}|$ is at least
  0.50.
- `sparse_gate_candidate`: some program places at least 75% of its absolute interaction
  mass in the largest 10% of documents.

Interpretation is limited: diffuse/shared geometry motivates a common observable-
weighted correction; strong concentration motivates a gate or mixture; predictability
by native difficulty suggests a nuisance rather than a named circuit.  No result by
itself establishes semantic meaning, OOD transport, or selective editability.
