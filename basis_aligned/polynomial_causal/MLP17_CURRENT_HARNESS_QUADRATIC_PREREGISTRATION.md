# Rung 415 preregistration — physical current-harness MLP17 quadratic surrogate

## Prior-work distinction

The MLP17 dossier already contains four different facts: four output directions recover about75% of its loss benefit;
each chosen scalar coefficient is almost rank2 quadratic; an activation-conditioned rank4 factorization of Down
recovers about83%; and current product/output spectra are concentrated. These do not yet provide a clean-split,
physically factored current-harness whole-layer replacement.

Rung415 ports only the historical whole-layer object: four output directions, each coefficient computed by two signed
squared input projections. It does not claim that Down rank, product width, or output rank is four.

## Frozen object and controls

- Fit output directions, input second moment, four rank2 forms, and constant on FineWeb fit-B rows [24,48).
- Score disjoint fit-A [0,24), function-fresh [48,72), census, FineWeb fresh, and WikiText.
- Physical clean program retains only directions [4,1152], form vectors [4,2,1152], form values [4,2], and constant
  [1152]: exactly14,984 float32 values with no dense form.
- Seed415 random orthonormal output directions receive the same rank2 construction and exact price.
- Reproduce the historical overlapping-row CE3.557555 only as an implementation identity.

## Frozen predictions

- pred_a: legacy CE within .005; exact clean/random shapes,14,984 count,no dense forms,fit/evaluation splits,live hooks.
- pred_b: clean heldout and fresh local function R2 at least .65; census damage at most .130 nat and at least5 of62
  certificates; WikiText mean/p95/max at most .15/.25/.40; fresh mean/max at most .15/.30.
- pred_c: clean census damage within .05 of the historical .101811 nat and beat random by at least .30 heldout R2
  and .05 census nat.
- pred_d: clean normalized certificate vector has frozen-ray cosine at least .90, vector R2 at least .50, and predicted
  certificate-count error at most5.

Strong null: heldout R2 below .40, census damage at least .20, zero certificates, inert hook, or random within .05
heldout R2 and .01 census nat.

A complete pass licenses one original-native signed intervention gate for this exact physical object. No output rank,
form rank, metric, layer, split, or threshold tuning.
