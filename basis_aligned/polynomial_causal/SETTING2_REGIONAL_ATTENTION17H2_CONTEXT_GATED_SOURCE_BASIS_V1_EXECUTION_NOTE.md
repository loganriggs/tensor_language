# Execution note: context-gated head17.2 source basis V1

The bound V1 runner and its scientific result remain authoritative.  A
post-run source audit found one accounting defect at runner lines 246--247:
the identical complete-support removal suffix call is made twice, and the
second result overwrites the first before any metric is computed.

Consequently the recorded behavioral tensors and every preregistered verdict
are unaffected, but the literal suffix price in the preregistration and result
understates execution by 48 suffix sequences.  The actual suffix price was:

- 864 candidate suffix sequences (nine candidates, 48 rows, install/remove);
- 144 complete-support suffix sequences (48 install and two identical 48-row
  removal calls), not the recorded 96;
- 1,008 suffix sequences total, not 960.

The higher-level model-trace price remains exactly 30 physical executions and
192 full-model sequences.  This note does not authorize a scientific-result
switch: eliminating the redundant deterministic call changes cost only.  In
particular, the null is independently supported by the unprojected comparator,
which fails preservation at `1.400769403459618` despite retaining the complete
tested source addition.
