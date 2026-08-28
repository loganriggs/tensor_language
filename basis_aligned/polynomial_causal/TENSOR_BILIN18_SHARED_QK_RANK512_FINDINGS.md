# Shared-QK-512: first admitted compressed complete tensor program

Date: 2026-08-28

Status: pass on the already-opened composition roles and frozen prefix intervention.
Fresh OOD and intervention validation remain required.

All preregistered gates pass. The checkpoint-independent program combines shared-QK
rank-512 attention with exact dense MLPs and the complete owned shell. The checkpoint
model is destroyed before scoring, storage is disjoint, total token support is true, and
native calls, native module references, and fitted lookup tables are zero.

## Complete predictive result

| role | all-position CE harm | covered harm | unseen-current harm |
|---|---:|---:|---:|
| skip7000 | +0.008660 | +0.009243 | +0.006823 |
| skip11000 | +0.009745 | +0.010412 | +0.007788 |

Harm improves by about 0.010 nat relative to rank384 on both roles and replicates within
0.00117 nat.

## Causal transport result

- context-delta recovery: 0.914855;
- delta cosine: 0.956512;
- delta norm ratio: 0.964252;
- recovery gain over rank384: +0.068031;
- cosine gain over rank384: +0.035404;
- native/program maximum downstream changes: 3.497423 / 3.522475.

The program crosses the frozen 0.90 recovery and 0.95 cosine gates. The improvement from
rank384 exceeds both registered rank-limitation bars, so ordinary shared-routing capacity
was the immediate cause of the rank384 failure for this intervention.

## Price

The complete program stores 503,436,726 float32 values, 92.2207% of dense. It saves
42,467,328 values, or 7.7793%, while preserving exact MLPs and shell.

This is the first admitted compressed complete-model point under simultaneous ownership,
prediction, all-position support, and causal-transport criteria. It is not yet a fresh
OOD result because rank and gates were developed using these roles and this intervention.
It also does not semantically explain the routing coordinates. The next promotion gate
must use untouched FineWeb rows and a bank of new prefix interventions.
