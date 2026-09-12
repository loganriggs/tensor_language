# Optimize full-input interaction coordinates

Full-input fixed spectral frames give4.56%single-layer and8.18%producer capture
with4096edges. The earlier128input restriction scarcelychanges best256edge
capture. Test the remaining fixed-frame restriction using the complete1152frame.

Reuse exact fixed-support gradient and Armijo from sparse_core_stiefel_v1;
stream all edge energies for exact reselection. All output directions retained.
Support optimization is piecewise smooth; a positive support gap plus relative
projected-gradient norm<=1e-6 is local stationarity, not global recovery.
No rank/output reduction or data fitting. Native implementation is in progress;
no job is claimed queued by this document.

Planned producer-metric native comparison: spectral and seeded independent Haar
frames,4096edges each, at most200accepted updates/600seconds per start. Preserve
time-limit outcomes as unconverged and checkpoints for result-dependent decisions.
Instrument checks<=1e-8; predict both starts converge under the stated criterion,
best capture>=1.25times8.1838618358%, and full function cosine>=.95 across starts.
Score each prediction separately. No generic absent-structure conclusion on a miss.
Native exact4608products use nonorthogonal/overcomplete readers; this orthogonal
family can remain inefficient even when a small arithmetic representation exists.

Circuit consequence: a stable interacting-reader graph becomes a candidate
for frozen extraction and joint edge/node intervention; it is not a semantic
circuit until held-out/OOD, removal and composition tests succeed. Conditional
cost remains6,045,696floats plus8192edge indices and required native producer,
normalization, common-output and background computation. Any future causal panel
must distinguish new held-out data from reused developmental morphology examples.
