# Rung 525 terminal receipt — task-free token-by-context grouping closes

**Completed:** 2026-09-03 09:52 UTC  
**Audited:** 2026-09-03 09:55 UTC  
**Decision:** registered strong null; no physical downstream substitution licensed

## What was tested

For each of the 50,257 real tokens, rung 525 represented the exact linear map from an attention0 context
deviation to MLP0's centered token-by-context output. Bank A chose a functionally nearest donor token whose raw
token vector had cosine at most 0.50. A disjoint document bank B tested whether that donor remained close. Ordinary
nearest-token donors, far-random donors, and independently coordinate-scrambled candidates were fixed controls.

This was a function-level grouping screen using exact MLP0 weights. It used no circuit labels or downstream loss,
and it was never sufficient by itself to identify a circuit.

## Frozen result

- Prediction A, exact lawful instrument: **pass**. The scalar contraction identity had relative squared error
  `7.29e-13`; FIT and SELECT roles differed; no downstream model, circuit, FINAL, or sealed outcome was opened.
- Prediction B, operator grouping transfers: **fail**. Median held-out candidate distance was `1.1641`, versus
  `0.9423` for the ordinary nearest-token control. The candidate/raw ratio was therefore **1.2353**, where the
  frozen pass bar was at most `0.75` and the strong-null boundary was at least `0.95`.
- The operator candidate did beat a far-random donor (`0.5862` ratio) and the scrambled control (`0.7290` ratio),
  and `96.47%` of receivers beat their own far-random fifth percentile. This says the operator sketch carries
  stable token structure, but not structure beyond what ordinary token similarity predicts.
- Prediction C, repeated stable groups: **pass**. There were 1,444 repeated donors covering 3,252 receivers;
  A/B distance Spearman correlation was `0.6733`; the two half-sketch medians differed by only `0.169%`.
- Overall: **strong null**. Stable repeated matches are real, but task-free operator similarity did not improve
  on the raw-token baseline. The registered physical substitution is blocked.

The descriptive representative table would use `16.40%` as many bits as caching every token's two MLP0 factor
vectors, but that is not compression of the native MLP0 weights and carries no simplicity or adoption credit.
Execution used 72 attention0 capture batches, 4.91 seconds, 3.84 GB peak GPU memory, zero downstream forwards,
and zero deployed values.

## Independent audit

The terminal auditor recomputed every registered B/C/strong-null statistic from the 10,052 receiver-pair artifact,
checked the donor/receiver split and artifact hashes, rejected test mutations, and confirmed zero downstream calls
and no physical license. Four focused mutation tests pass.

- Result SHA: `34714559df04b966c503321b78fbbabd2f6150dac5e1354ed9070b1dc9e86a0b`
- Pair-artifact SHA: `11e295fb744bde435158578fffad6a7db994bdb3df28201c4df39354f31d8d4b`
- Audit SHA: `571b07b6715bf9289de7c925016b4afcdf17f4efa16383f795de0b41e784783c`

## Consequence

Do not tune the sketch width, cosine ceiling, split, or thresholds, and do not run the blocked physical successor.
The next object must incorporate downstream use. The leading option is to replace the random MLP0-output probes
with the exact MLP1 native-state response reader already validated for the MLP0 interaction branch. Candidate token
groups would then mean “these token-by-context writes have the same predicted downstream computation,” with disjoint
document selection and validation followed by finite full-suffix swaps only if the downstream-conditioned grouping
beats raw-token and shuffled controls. That changes the measurement, rather than relaxing this null.
