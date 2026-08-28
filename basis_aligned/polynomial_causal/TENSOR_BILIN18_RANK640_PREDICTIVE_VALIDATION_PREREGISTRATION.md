# Preregistration: rank640 complete-program predictive validation

Date: 2026-08-28

## Purpose

The prospective 16-intervention bank selected the complete shared-QK rank640 program
as a robust causal candidate. Its predictive behavior has not been measured. This gate
tests the same rank640 construction on the hash-registered skip31000 and skip35000
cross-task FineWeb roles. Those rows have been opened for rank512, but never evaluated
at rank640; this is cross-candidate replication rather than a claim of globally untouched
data.

## Frozen construction and price

- shared-QK rank: 640 at every attention site;
- fitting: unchanged activation covariance on skip80 only;
- MLPs, embedding, residual coefficients, RMSNorm sequence, unembedding, and softcap:
  exact standalone tensors;
- exact stored values: 516,707,766, saving 29,196,288 or 5.3481% from dense.

## Gates

1. Complete ownership, disjoint storage, checkpoint collection, total input support,
   zero native calls/modules/tables, and the exact price must pass.
2. On each role, all-position and seen-current CE harm must be at most 0.020 nat;
   unseen-current harm must be at most 0.025 nat.
3. Each all-position and seen-current harm may be no more than 0.002 nat worse than the
   already measured rank512 result on the identical role.
4. Seen-current harm must replicate across roles within 0.010 nat.
5. The immutable causal-bank parent must identify rank640 as a robust pass.

If all gates pass, rank640 is admitted as the first compressed complete program whose
same parameters possess exact ownership, heldout predictive, unseen-support, and
prospective distributional causal certificates. If prediction fails, capacity repaired
causal transport at an unacceptable predictive price and the next move is joint
rank/objective selection rather than mixing certificates across ranks.
