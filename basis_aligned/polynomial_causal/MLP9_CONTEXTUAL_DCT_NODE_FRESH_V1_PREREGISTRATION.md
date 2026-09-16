# Extracted MLP9 contextual DCT node fresh test V1

## Question

The contextual range experiment found four behavior-blind MLP9 input factors
whose native RMS-conditioned curvature is stable across panels and highly
compressible, but its diagonal-only rank-eight basis was not a CrossFirst
reader.  Turn that positive model-level result into a standalone reusable node.

For `f(x)=Down[(Left x)*(Right x)] / s(x)` with
`s(x)=mean(x^2)+epsilon`, the exact mixed derivative for frozen directions
`v,w` is

`D2f_x(v,w) = q_vw/s - q_v*s_w/s^2 - q_w*s_v/s^2`

`                 + q_0*(2*s_v*s_w/s^3 - s_vw/s^2)`,

where `q_0=Down[(Left x)*(Right x)]`,
`q_v=Down[(Left v)*(Right x)+(Left x)*(Right v)]`,
`q_vw=Down[(Left v)*(Right w)+(Left w)*(Right v)]`,
`s_v=2*mean(x*v)`, and `s_vw=2*mean(v*w)`.

On the earlier 48-row discovery contexts only, collect the complete symmetric
4×4 response table for the four independently stable raw-DCT inputs.  Choose
the smallest output rank in `{8,16,32}` retaining at least `.99` of response
norm; if none does, freeze rank 32.  Package full `Left`/`Right`, projected
`Down`, the chosen output basis, the four directions, and their precomputed
left/right projections.  The sole activation port is native pre-MLP9 state
`z9`.

Test without refitting on 64 score-blind cross-domain prefixes absent from all
repository row authorities at construction.  Compare the standalone analytic
node to direct mixed JVP for all 16 ordered direction pairs.  Also test eight
fixed pairs of arbitrary linear combinations by contracting the node's 4×4
table and comparing to direct mixed JVP.  Random equal-rank output subspaces and
the earlier raw-DCT rank-eight output span are frozen comparators.

## Predictions

- **A — analytic instrument:** the uncompressed closed form matches direct
  mixed JVP within `2e-5` relative L2 overall, every family, and every ordered
  direction pair; swapped response symmetry is within `2e-5`.
- **B — sparse discovery:** selected output rank is at most 16 and retains at
  least `.99` response norm on discovery.
- **C — fresh response prediction:** the standalone node has relative L2 at
  most `.05` and cosine at least `.995` overall and in every fresh context
  family, with every ordered pair at most `.10` relative L2.
- **D — compositional reuse:** all eight arbitrary coefficient-pair contractions
  have fresh relative L2 at most `.08` and cosine at least `.99` against direct
  mixed JVP.
- **E — compression specificity:** node relative L2 is at least `.20` lower
  than the median of 16 equal-rank random output-subspace controls and at least
  `.20` lower than the raw-DCT rank-eight comparator.
- **F — standalone extraction:** isolated CPU replay is at most `2e-6`; package
  parameter count is at most `.70` of the native MLP9 bilinear matrices; package
  has one activation port, no model/suffix/tokenizer/readout dependency, and
  serializes complete weights, fixture, metrics, hashes, and price.

If A fails, repair formula or differentiation mechanics before interpretation.
B failure means the full pairwise response is not sparse at the registered
ranks.  C failure rejects response-level OOD transfer.  D failure rejects the
node as a compositional tensor interface.  E failure means compression is not
specific.  Passing A--F establishes a transferable extracted and compositional
response node, not behavioral relevance, selective removal, or a complete
circuit.

## Price

One checkpoint load; 48 opened discovery prefixes; 64 untouched score-blind
cross-domain prefixes; 16 ordered stable-factor mixed JVPs per prefix plus eight
mixture checks on fresh prefixes; one eigendecomposition and fixed rank choice;
16 random output controls; one isolated CPU package replay; no suffix, answer
logit, behavioral outcome, fitting to model outputs, gradient optimization,
parameter update, or causal removal intervention.
