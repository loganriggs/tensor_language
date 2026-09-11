# Joint QK/value source ports — 11 September 2026

Freeze the compact shared-input component from
JOINT_SHARED_READER_RANK16_V1_RESULT.json. Its 17 downstream input readers define
per-head source spaces through native attention17 OV and signed current/base
value mixing. No task labels, activation samples or text corpus enter this test.

Question: does the complete QK1*QK2 numerator use these source directions more
than matched downstream frames? This differs from whole-head proportionality,
separate QK1/QK2 assignment, or task-derived quadratic feature-space angles.

For each of nine heads, define the source projector P onto row(FO_hV_h).
The joint numerator N(q,s)=(q^T A s)(q^T B s) splits into N(q,Ps),
N(q,(I-P)s), and the two crossing products. Their separately symmetric
biquadratic coefficient tensors are orthogonal. Report inside/mixed/outside
energies and touch=(inside+mixed)/total. The source tuple has current/base
width2304; keys read only the current block. Query position8, distinct source
positions7 and0. No self-position result: that needs tied query/source slots.

Three Gaussian orthonormal downstream frames of dimension17, seeds1213,1217,1223,
pass through the SAME native OV maps before defining their source projectors.
They control generic native OV/key alignment; an isotropic source-space null
would conflate this alignment with selection of the MLP component.

Predictions, fixed before native measurement:

- A: CPU dense/planted controls pass; native normalized-score replay <=1e-10;
  orthonormal source bases and energy fractions satisfy 1e-10 checks.
- B: across the18head/position rows, the learned mean touch fraction is at least
  twice the matched-random mean AND at least .02 greater in absolute fraction.
  This is an alignment screen, not circuit identification.
- C: heads17.2 and17.3 each have touch fraction >=.10 at BOTH source positions.
  This tests whether the old pair's routing substantially reads this particular
  frozen component's value space, not whether the old two behaviors share QK.

The null is weak or generic alignment. Any miss is scoped to this fixed candidate,
source geometry and numerator. Execute the head-resolved/null comparison before
interpreting it; do not infer absence of task-specific sharing. No optimization
is performed, so there is no convergence claim or time-limit negative.

All normalized numerator components share the original full four RMS factors,
including epsilon. This is an exact additive routing decomposition when those
normalizers are retained; projecting source states and recomputing the normalizers
is a different operation. Do not remove normalizer dependencies from a circuit.

Price: 0 model-body forwards, no corpus, 4 downstream frames, 9 heads, 2 positions;
CPU two threads. Only small Gram matrices are contracted; no large fourth-order
tensor is stored. Input/output maps and full routing remain native background.
