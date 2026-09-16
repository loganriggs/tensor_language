# Subject-number embedding-coordinate removal V1

Registered after freezing a third, disjoint 16-pair lexical authority and its
64 fixed prompts, but before opening any behavior on those prompts.

The frozen decoder score is `s(x) = a^T x - t`, where `x` is the model's
RMS-normalized checkpoint embedding at the subject token. This experiment asks
whether that coordinate is causally used, rather than merely decodable.

At the subject position, the target intervention moves `x` to the unique point
in the span of `a` and the component of `x` orthogonal to `a` that has the same
L2 norm as `x` and satisfies `a^T x' = t`. Thus it removes the centered decoder
coordinate without changing embedding RMS. It uses no number label or answer
outcome. Thirty-two frozen Gaussian controls move each same embedding by exactly
the same L2 distance on its constant-norm sphere, along independently seeded
random tangent directions at the same token position.

The authority has eight irregular and eight regular singular/plural pairs absent
from both decoder authorities. Each form occurs in two fixed prepositional
templates with an opposite-number attractor: 64 equal-length prompts. No row may
be filtered after behavior is opened.

Instrumentation requires:

- the checkpoint, decoder, fresh-decoder result, authority, and registration
  hashes to match;
- the frozen decoder to classify all 64 new subjects correctly with strictly
  positive minimum signed margin;
- native number accuracy at least `.75` in every number/template and stratum
  cell;
- a manual input-state forward to replay the observed facade within `1e-5` logits;
- target post-edit decoder residual at most `1e-5`, target RMS preservation
  within `1e-5`, and every random edit's distance and RMS matching within `1e-5`;
- exactly 35 physical forwards over 2,240 sequences, with no fit, backward, or
  parameter update.

Define removal damage as the baseline native-minus-opposite answer margin minus
the intervened margin. Selective causal removal requires:

1. target damage RMS at least `.02` and at least `.05` of native-margin RMS;
2. positive target damage on at least `.75` of rows overall and separately in
   every number, template, and regularity stratum;
3. target damage RMS greater than all 32 equal-distance random controls and at
   least twice their median RMS;
4. target `can`-minus-`will` control-change RMS at most `.50` of target number
   damage RMS.

Passing establishes selective causal use of the frozen embedding coordinate on
fresh vocabulary and prompt text. It does not by itself prove that this exact
coordinate is the unique upstream source of the frozen L11H3 writer, or that the
model internally performs the external threshold/lookup program.
