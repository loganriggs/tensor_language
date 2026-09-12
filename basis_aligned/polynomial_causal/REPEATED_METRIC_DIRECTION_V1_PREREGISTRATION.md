# Exact repeated-input metric: direction viability pilot

Use the existing locally converged formal rank128 frame, arm0. This tests whether
the exact coefficient metric after substituting MLP15 provides a reproducible
new descent direction. It does not repeat the formal fit or use text statistics.
Background and normalizers remain explicit external ports.

For each producer degree1–4, use the exact mixed contraction and full-U writer
Gram. Normalize squared errors by the reference energy in the independent
2048-probe MIXED_REPEATED_NATIVE_V1 artifact. These fixed estimates define the
objective weights; uncertainty in them is not included in conditional test SEs.
Compute two gradients from independent512-probe sets per degree, seeds73220/21.
Project to the Grassmann tangent. Use only the first gradient to propose QR
steps of Frobenius lengths0.01,0.05,0.15. Independent2048 probes per degree,
seed73222, compare each candidate with the start on paired rows.

- A: native directional central difference relative discrepancy <=1e-4;
  candidate frame orthogonality error <=1e-8; finite values.
- B: independent tangent gradient cosine >=0.5.
- C: at least one held-out step improves balanced squared loss by >0.001
  and >3 standard errors of its paired improvement; runtime <=240 seconds.

Null: finite synthetic batches do not offer reliable descent at this budget.
Three steps are a predeclared exploratory screen, not a confirmatory statistical
claim. A failed gradient cosine motivates a variance audit, not an absence claim.
No candidate is adopted or declared converged here. A successful direction still
requires optimization and frozen native effect, extraction, removal and reuse
validation. All existing failed circuit criteria remain failed.

Price: zero body forwards; 4096 total gradient contractions (reference plus
candidate for512 probes x4 grades x2 sets), plus paired validation of three
steps on8192 probes. Batch64, FP64, one managed GPU job,300 second hard timeout.
Native weights and all background ports remain charged. Save frames/gradients
and per-probe validation differences; no vocabulary tensor is materialized.
