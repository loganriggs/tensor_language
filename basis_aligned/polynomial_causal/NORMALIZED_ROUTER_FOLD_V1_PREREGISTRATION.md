# Normalized router folding: mathematical instrument audit

Frozen scope: existing bilin18 checkpoint, layer 0 head 0; no selection from effects.
This tests weight-analysis machinery, not a language circuit or a model replacement.
Original handoff/pilot authority. No calibration or is/was refinement.

For head width d and normalized residual inputs x,y, retain two bilinear
numerators A_i=Q_i^T R_t^T R_s K_i/d and four quadratic forms
G_Qi=Q_i^T Q_i/d, G_Ki=K_i^T K_i/d. The exact real-arithmetic router is
product_i(x^T A_i y) divided by the square root of the product of
(x^T G_Qi x+eps)(y^T G_Ki y+eps), with deployed FP32 eps.
R uses the actual native FP32-angle/BF16-cosine-and-sine convention.

Freeze seed 9111260, 64 independent Gaussian pairs, each normalized to RMS 1,
and positions (0,0), (1,0), (31,7), (127,32). These are local continuous inputs,
not tokens or established naturally reachable states. All 256 pairs are counted.
Also check x=y at (0,0) separately; no distribution fitting.

Controls, fixed before outcomes:

1. Direct projection/norm/rotation versus folded evaluation: FP64 relative L2
   <=1e-10 and maximum absolute error <=1e-10; FP32 versus folded FP64
   relative L2 <=1e-5 and max absolute <=1e-5 in every position cell.
2. Scale half the rotary planes by 2 and half by 1/2 in Q1, and reciprocal
   scales in K1. Because each scale is repeated over the paired halves, it
   commutes with every R_t, even with rounded cosine/sine. Both numerator
   matrices must remain equal within relative L2 1e-10, while normalized
   router relative L2 must change by >=.01 in every cell. A failure is retained.
3. Repeat using paired signs +1/-1 instead of scales: numerator and complete
   normalized router must remain equal within relative L2 1e-10.
4. Native same-token folded replay uses the same FP64 bars. Its score changes
   under the scaling control are descriptive, not an additional fitted gate.

Opposing interpretation: if control 2's numerator equality holds but routing
does not change, this selected control did not demonstrate the claimed false
sharing on the tested inputs. A pass proves that matching numerator tensors
alone is an inadequate general criterion. It does not prove two actual heads
share no operation. Sign control verifies a valid nontrivial reparameterization.

CPU, two threads, mmap checkpoint; no model forwards/backwards, GPU, training
or checkpoint writes. Each dense analysis matrix is 1152^2 FP64 (10.125 MiB),
six per folded signature (60.75 MiB), below the per-tensor cap. These are
temporary diagnostic matrices; runtime can retain factored maps. All original
545902902 parameters remain charged. No structural saving is claimed.
Results immutable; hash checkpoint, source, helper and protocol. No head, gain,
sample, tolerance or threshold sweep after observing outcomes.
