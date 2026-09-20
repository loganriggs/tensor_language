# Failed-square optimizer control — 2026-09-20 19:28 UTC

The known-topology graph check failed both random starts on a square of an affine quadratic (best fresh error39.92%). Its oracle and gradient checks pass. Diagnose optimization before adding topology.

Teacher is the same planted square graph. Student has four affine linear readers, q=ab+cd, two outputs linear in(q²,1). Compare Adam/Muon, rates.005/.03, seeds0/1, joint writer fitting versus analytic least-squares writers,1500steps. All arms start from paired identical reader/writer draws. Matrix parameter blocks explicit; Muon grouping is part of the optimizer specification. Fixed5-point-per-axis Gaussian quadrature (625points) integrates the degree-eight squared loss exactly. Independent7-point rule checks it;4096 independent Gaussian probes assess prediction. This isolates optimizer/readout elimination rather than noisy probe generalization.

pred_a: compiler teacher replay and5vs7rule agreement<1e-10; no nonfinite fits.
pred_b: at least one analytic-writer arm reaches<.001 exact Gaussian relative error.
pred_c: analytic-writer arm has at least as many<.01 recoveries as joint fitting over the paired grid.

Keep failures and final vs best checkpoints. No native circuit/general optimizer superiority claim. Analytic writer uses a centered two-feature regression; variance denominator floor1e-24, counted if active. Finite Gaussian quadrature is a toy instrument, not a scalable native proposal.
