# Induction early-MLP finite bilinear factor V1

The exact induction selector/payload edit is live, but direct residual axes and
per-row suffix-VJP axes fail to separate answer-preserving response from
collateral.  This test uses the exact finite algebra of the first downstream
bilinear consumers, MLP8--12.

For each joint-edited normalized MLP input `x1`, freeze the matching recipient
native input `x0` and let `d=x1-x0`.  Decompose the current MLP response exactly
as left cross `D[(Ld)*(Rx0)]`, right cross `D[(Lx0)*(Rd)]`, and quadratic
interaction `D[(Ld)*(Rd)]`.  At every MLP8--12, remove one of: quadratic,
left-cross, right-cross, both cross terms, or the full three-term response.
The decomposition is recomputed online from the intervened trajectory, so full
removal returns each affected write to its native-input value rather than using
a first-order approximation.

Use the frozen 48-group induction authority and its fixed DISCOVERY/CONFIRM
partition.  Select the first DISCOVERY arm in order quadratic, left, right,
cross that preserves the recipient answer in every cell (mean CE damage at
most `.10`, correct fraction at least `.75`) and reduces median full-vocabulary
RMS by at least `.25` versus the exact joint edit in every cell.  Require the
same arm on CONFIRM.  Full removal is a reference and cannot establish compact
factor identification.  Native replay, finite bilinear closure, active edits,
and exact price gate interpretation.  No axis, SVD, fit, coefficient, gain,
subgroup, threshold change, gradient, update, or quantization is allowed.
Price: 24 forwards, 768 sequences, zero backwards and zero fits.
