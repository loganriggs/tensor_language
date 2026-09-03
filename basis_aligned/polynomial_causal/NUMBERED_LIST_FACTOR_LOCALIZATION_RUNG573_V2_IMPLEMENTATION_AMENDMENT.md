# Rung 573 v2 implementation amendment: forward-price correction

## Why v1 is invalid

The supervised v1 process completed its in-memory computations and then raised:

```text
RuntimeError: forward price exceeded: 280 > 278
```

No v1 result JSON was written. The original price formula counted the two one-time native-versus-custom replay checks
on FIT but overlooked that the shared evaluator performs the same two checks again when conditional SELECT opens. The
failure therefore reveals one outcome bit: at least one FIT arm passed and SELECT opened. V2 is an implementation-only
replay of a protocol frozen before that leak; it is not an independent outcome-blind replication.

## What is unchanged

V2 uses the same R567 FIT and SELECT rows, R573 semantic label positions, L8H7/L8H3 site, eight intervention arms,
fixed arm order, target and control families, effect definitions, thresholds, bootstrap count, seed, checkpoint, and
conditional split rule as the original R573 preregistration. FINAL_TEST and OOD remain closed. No model weight is
updated.

## The only repair

The maximum execution price is corrected from 278 to 280 model forwards and zero backwards. V2 writes to
`numbered_list_factor_localization_rung573_v2_results.json`. Any other scientific or computational change invalidates
V2.
