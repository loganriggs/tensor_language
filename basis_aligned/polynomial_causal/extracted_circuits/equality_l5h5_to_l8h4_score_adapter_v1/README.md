# L5H5 → L8H4 equality-score adapter V1

This package exposes the frozen one-scalar adapter from L5H5's equality score
to the score port of the extracted L8H4 equality edge.

The scalar was fitted once on natural text before code-OOD evaluation.  Inside
the exact-order L8H4 node it recovers `.97287` of the code-OOD copy effect,
compared with `-1.84448` for the preregistered L7H3 wrong-score donor.  The
package therefore contains one charged calibration scalar and no newly learned
model parameters.
