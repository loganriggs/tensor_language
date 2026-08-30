# Numeric, unit, and date formatting

## CURRENT tier: 1

A candidate localization and exact intervention identity exist, but the behavioral
sample is underpowered and its null failed.  Ordered successor cells are excluded.

## Behavior and tensor program

Endpoint: CE on units or registered formatting delimiters immediately after a numeral.
Matched negatives are numeral contexts followed by non-units.

Fixed tensor form: MLP0 quadratic number gates factored into magnitude, surface-form,
and layout factors plus unit/delimiter readout.  Router: native factor scores may select
formatting branches; decoded unit/date labels may not route the executor.  Extraction
installs a minimal native-gate subset or paired-product program; removal deletes only
formatting service.

## Evidence

- [`number_word_verify.py`](../../number_word_verify.py) and
  [`number_word_verify_results.json`](../../number_word_verify_results.json): identity
  delta `1.84e-7`, only `77` targets, and failed null.
- [`digit_copy_split.py`](../../digit_copy_split.py): copy/fresh numeral masks.
- [`year_succ.py`](../../year_succ.py): successor exclusion control only.

## Terminal gates

Collateral includes words, punctuation, successor digits, and copied numbers.  OOD
holds out magnitudes, decimals, grouping, percentages, currencies, units, and date
layouts.  Before default terminal gates, require a shuffled-label null and at least 200
target positions per SELECT and OOD role.

Shared-owner caveat: copy and successor circuits can make numeric tokens easy without
implementing formatting; retain them as exclusions/covariates.

**Next experiment:** fresh powered numeral-to-unit/delimiter screen with matched nonunit,
shuffled-label, successor-exclusion, and identity controls.  Another null failure
retires the candidate.
