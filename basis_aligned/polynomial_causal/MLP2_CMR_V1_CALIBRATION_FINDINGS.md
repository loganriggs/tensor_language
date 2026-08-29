# MLP2 CMR v1 FIT_SELECTOR calibration findings

## Status

Receipt-backed fit-only calibration complete. Validation and replication remain
unopened, no finite MLP2 candidate was constructed, and no compression claim is
authorized.

The source-closed runner executed on the RTX 5090 in bfloat16 after three independent
audit rounds. It used only the role-only FIT_SELECTOR artifact: 192 documents,
31,505 eligible positions, 191 supporting documents, and all-false ordinal 82.

## Numerical result

- 48 complete native forwards and 48 returns;
- exactly 48 attention and 48 MLP calls at each of 18 sites;
- 864 attention calls, 864 MLP calls, zero backwards;
- internal measured runtime 13.3966 seconds; shell wall time 16.13 seconds;
- checkpoint hash identical before and after strict state loading;
- failure artifact absent and receipt published last.

The native top-1 margin is the largest final logit minus the second largest. Its
mean is `1.58057`, maximum is `12.3125`, and the registered quantiles include:

| Quantile | Margin |
|---:|---:|
| 0.02 | 0 |
| 0.05 | 0.0625 |
| 0.10 | 0.125 |
| 0.50 | 0.9375 |
| 0.90 | 3.9375 |
| 0.99 | 8.375 |

The exact epsilon grid contains 28 positive values: the fixed dyadics
$2^{-10},\ldots,2^5$ plus positive empirical margin quantiles divided by two.
The zero 2% quantile is a warning that a nontrivial lower tail has tied top logits
at bfloat16 resolution; the finite certificate may therefore be informative only
for a subset of positions. The actual validator, rather than this observation, must
measure the bound.

The token-only copy partition is:

| Cell | Positions | Fraction |
|---|---:|---:|
| copy-positive | 2,420 | 7.681% |
| repeat-negative | 11,224 | 35.626% |
| nonrepeat | 17,861 | 56.693% |

These masks were not published. Only their counts and hashes were published; they
replay deterministically from the protected role-only rows.

## What this does and does not establish

This calibration freezes reporting geometry for the physical K=512 validation:
native-margin thresholds, FIT_SELECTOR target-frequency counts, and copy/repeat cell
definitions. It prevents validation outcomes from selecting convenient thresholds
or strata.

It does not test SUFFIX, LOCAL, RMS, MASS, DERANGED, or HASH_RANDOM finite programs.
It changes no strict causal/storage ledger. The next scientific result must come from
physically executing the equal-price 512-channel candidates through layers 3--17 and
measuring final CE, teacher KL, centered-logit NRMSE, top-1 agreement, registered
cells, signed endpoints, and the margin certificate.

## Provenance

- source commit: `cee5eb58305853df1b51f4fed2e8fbf5d9b1a7d8`
- authority SHA-256: `7be8c2e42b6995fafcd478cee50437408c3f2b6ba7745d5189acc8925bc8141d`
- bundle SHA-256: `3f9aa5ff69530a099c7859b454298eca48cd0789413b412f599746793fd6c1fa`
- result SHA-256: `e30ae749d59dedad4c17159d5b29af1c4c0c79e3f620e794e3a590f3b049c08c`
- receipt-file SHA-256: `08267122572157203ccf87f9d901d9c4efdfb41c9bb3b4f0d34f1f1f4e669b52`

