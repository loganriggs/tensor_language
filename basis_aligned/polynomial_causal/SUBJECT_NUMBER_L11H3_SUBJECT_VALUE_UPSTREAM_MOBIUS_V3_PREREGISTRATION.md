# Subject-number L11H3 subject-value upstream Möbius V3

Registered after V2 fixed the V1 integer-layer grouping bug but failed its
absolute component-reconstruction gate (`0.00146484375 > 1e-5`). V2 otherwise
replayed the native subject value within `9.54e-5` and closed the Möbius sum
within `3.06e-5` absolute / `6.57e-8` relative. V1 and V2 remain `invalid`.

V3 tests the specific prospective hypothesis that V2's residual mismatch is
only float32 summation association. It keeps the same authority, panels, five
ports, 32 corners, readouts, greedy rule, scientific thresholds, and price.
The four residual-stream groups are converted to float64; the late group is
defined as the native raw state minus embedding, early, and middle groups.
The explicit independently propagated late group is retained only to measure
the numerical gauge correction.

Instrumentation passes only if all prior replay, Möbius, synthetic, price, and
checkpoint gates pass, the closed float64 reconstruction error is at most
`1e-10`, and the late-gauge correction has relative L2 norm at most `1e-5`.
The absolute correction is reported but is not thresholded because the test is
whether it is negligible relative to the component it repairs. The scientific
gates remain term-4 relative L2 at most `0.25` and target advantage over the
median of 16 orthogonal random writer readouts at least `0.10`.

Passing the instrument licenses interpretation of the unchanged scientific
tests. Failure leaves the upstream decomposition unresolved rather than
negative. Passing compactness but failing writer specificity means the five-port
decomposition is numerically real and compact for this readout, but does not
identify a behavior-specific reusable atom set.
