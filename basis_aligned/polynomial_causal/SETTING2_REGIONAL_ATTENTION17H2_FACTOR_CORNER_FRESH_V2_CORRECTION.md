# Precision correction for the fresh head17.2 factor corner

V1 completed all 12 full-model executions and every scientific raw metric passed,
but the run was instrument-invalid. The independent expansion of
`native-QK1 × (edited-QK2/value - native-QK2/value)` was accumulated as three
separate BF16 projected writes and differed from the one-shot corner subtraction
by relative error `2.28544e-6`, narrowly above the frozen `2e-6` audit bar. The
direct native/full corner endpoints were exact; attention17 reconstruction was
`2.08926e-7`; fresh response, installation, and removal errors were
`.02136/.02099/.02090`; capability, controls, and equal-norm random nulls all
passed. Therefore V1 is an execution-only precision failure, not evidence for or
against transfer.

V2 changes only offline factor-corner arithmetic: promote captured QK1, QK2,
value, and head17.2 output weights to FP64 before the three corner contractions
and independent expansion. The resulting writes are cast back to the model's
native dtype before MLP17 and final-readout execution. Rows, frozen corner,
readers, random seeds, scientific thresholds, price, model execution, and all
other code remain unchanged. The `2e-6` audit bar is not relaxed.

As a result-switching guard, V2 must reproduce V1's raw aggregate response,
installation, and removal errors and each family's response/install/removal
relative errors within absolute `.005`. The invalid V1 result and runner hashes
are bound. If that guard or any original gate fails, V2 cannot promote the
candidate.
