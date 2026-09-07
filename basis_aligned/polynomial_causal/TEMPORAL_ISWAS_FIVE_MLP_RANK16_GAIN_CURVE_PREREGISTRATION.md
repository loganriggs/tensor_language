# Temporal/is-was five-MLP rank-16 gain curve — preregistration

Fit the same all-position rank-16 bases as the sealed ladder and multiply every projected source delta
by one shared scalar from the fixed grid `1.00, 1.10, 1.15, 1.20, 1.25, 1.30`. Run targets and controls
for every gain. Select the lowest gain with response projection at least `.8`, response RSE at most `.2`,
behavior projection at least `.75` on both tasks, median control KL at most `.02`, and zero top-one flips.

Predictions: A authority/basis/full replay/finiteness passes; B some registered gain is functional; C the
selected gain is selective; D selected gain is at most `1.25`; E selected response projection remains at
most `1.20` on both tasks. Maximum 32 forwards, no gradients or parameter updates. Selection occurs on
the discovery fresh direction and requires a separate reverse-direction OOD validation.
