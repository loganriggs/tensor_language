# Frozen common-factor reuse across native rows — 2026-09-20 16:03 UTC

Use the best training-error common-factor model from row0 of each of16 native groups. Freeze its common quadratic q. Inspect all11 other rows in each group,176 heldout coefficient targets. Preserve the original row0 scale; do not normalize the heldout target into agreement.

Measure (1) unchanged full-program error, (2) oracle scalar-rescaling error, (3) optimal output-specific quadratic quotient refit with q fixed. Only(1) is a prediction from the original program. (2)/(3) inspect reuse capacity using heldout target weights and are not predictive validation. Report them separately. Compare quotient refit to a fixed random common quadratic baseline with the same normalization and cost.

Predictions: all row0 reconstruction metrics agree with fit receipts within1e-8; median frozen-full-program error across176rows<50%; fitted common factor beats random-factor quotient refit on at least75% of rows. Preserve the null that the common factor is context/row specific. Also independently check all16 selected programs plus planted control by exact-degree Gauss–Hermite quadrature.

Amplitude slots retain the same named source roles, but their native directions/readers vary by row. Thus this is local polynomial transfer in those coordinates, not native model behavioral OOD. Frozen-q quotient refits directly test whether a reusable quadratic component exists in that stated interface.
