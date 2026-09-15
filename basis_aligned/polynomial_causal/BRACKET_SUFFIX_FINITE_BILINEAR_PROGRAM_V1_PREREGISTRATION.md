# Bracket suffix finite bilinear program V1

The donor-free rank-two L13H8 source generator transfers, while fixed and
one-feature suffix scalar laws fail across constructions. This experiment
tests an architecture-defined source-by-background interaction program in the
native suffix.

Use the exact donor L13H8 source term so source approximation cannot confound
the suffix test. At each query-position MLP13--17, compare the current
normalized state `x1` with the matching recipient-native state `x0`, set
`d=x1-x0`, and exactly split the finite MLP response into
`D[(Ld)*(Rx0)]`, `D[(Lx0)*(Rd)]`, and `D[(Ld)*(Rd)]`. Recompute this split
online after every retained factor. Test programs that retain no MLP response,
only quadratic, only left-cross, only right-cross, or both cross terms; all
attention modules and native background remain active.

Select on the sixth embedded-pending construction in fixed order source-only,
quadratic, left, right, cross. Against the exact-source behavioral effect,
require overall cosine `.95`, relative L2 `.25`, sign `.90`, norm ratio
`.75`--`1.25`; every ordered pair must have cosine `.85`, relative L2 `.35`,
and sign `.85`. Exact effects must be positive on `.90` of every pair and
candidate control RMS must be at most `.50` of target exact RMS. Only a passing
sixth-construction program may open the seventh cascade construction, where it
must pass unchanged. Source-only success is recorded separately; a retained
bilinear factor supports interaction compression.

Native replay must be within `1e-5`, the independent FP64 three-term identity
within `1e-10`, every removed/retained response must be active, and full exact
source replay is the reference. No coefficient, fit, scalar law, rank change,
gain, construction feature, subgroup, gradient, update, or quantization is
allowed. Maximum conditional price is 12 forwards and 1,728 sequences; a
sixth-construction null stops at 8 forwards and 1,152 sequences.
