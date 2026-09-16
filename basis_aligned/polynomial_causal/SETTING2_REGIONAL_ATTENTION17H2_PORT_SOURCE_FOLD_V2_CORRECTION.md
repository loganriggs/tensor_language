# BF16 recurrence-residual correction for the head17.2 port-source fold

V1 completed its frozen execution and wrote an invalid receipt. Five independent
audits passed at `0–1.91e-7`, but the mathematical 16-term propagated-write sum
differed from the actual BF16 layer-17 residual-state change by `4.83294e-6`,
above the frozen `2e-6` closure bar. This is expected non-distributivity from
repeated BF16 residual multiply/add rounding: a sum of separately propagated
write differences need not reproduce the difference of the two rounded native
recurrences bit-for-bit.

V2 adds the observed recurrence-rounding remainder as an explicit bookkeeping
term for the exact source identity. It is not a selectable scientific source,
does not count toward the three-edge support, and is not supplied to any hybrid,
head factor, suffix execution, reader, or control. The original raw remainder
norm remains reported. After adding this exact remainder, source identity closure
is zero, so the unchanged `2e-6` bar applies to the corrected identity. This is
the same treatment already used for explicit BF16 projection residuals elsewhere
in the attention-fold lineage.

Rows, 16 selectable sources, all 697 supports, chosen support, random-support
nulls, factor computations, suffix executions, scientific thresholds, and price
are unchanged. V2 must reproduce every V1 scientific scalar and all family
response/install/removal/control values exactly (JSON equality); otherwise it is
invalid. The V1 invalid receipt and exact V1 runner bytes are bound. A corrected
instrument may therefore produce a valid selectivity null, but cannot turn a
failed scientific gate into a pass.
